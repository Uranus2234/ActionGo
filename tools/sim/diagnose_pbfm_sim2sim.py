import argparse
import gzip
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

from pbfm_depth import DepthHistory, MujocoDepthCamera
from pbfm_smoke import render_video


PUBLIC_REVISION = "671fbafe71a5f58bab77eeb5d969051fd2d283f4"


def public_file(source, path):
    return subprocess.check_output(["git", "-C", str(source), "show",
                                    f"{PUBLIC_REVISION}:interactive/pbfm/{path}"])


def motor_torque(target, position, velocity, kp, kd, effort, velocity_limit):
    limited_velocity = np.clip(velocity, -2 * velocity_limit, 2 * velocity_limit)
    lower = np.maximum(-effort, effort * (-1 - limited_velocity / velocity_limit))
    upper = np.minimum(effort, effort * (1 - limited_velocity / velocity_limit))
    return np.clip(kp * (target - position) - kd * velocity, lower, upper)


def run_case(args, config, motion, session, case):
    scene_path = (args.output / "public_robot/scene.xml" if args.scene == "public" else
                  args.output / "public_robot/g1_29dof.xml" if args.scene == "public_robot_plane" else
                  args.source / "pbfm/data/robots/g1_mjlab/g1_29dof.xml")
    scene = mujoco.MjSpec.from_file(str(scene_path))
    if args.scene != "public":
        scene.worldbody.add_geom(name="diagnostic_floor",
                             type=mujoco.mjtGeom.mjGEOM_BOX if args.scene == "box" else mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[50, 50, 0.5], pos=[0, 0, -0.5] if args.scene == "box" else [0, 0, 0],
                             group=0, friction=[1, 0.005, 0.0001],
                             rgba=[0.3, 0.35, 0.4, 1])
    scene.worldbody.add_light(pos=[0, -3, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    model = scene.compile()
    model.opt.timestep = config["physicsDt"]
    options = ET.fromstring(public_file(args.source, "robot/scene.xml")).find("option").attrib
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = int(options["iterations"])
    model.opt.ls_iterations = int(options["ls_iterations"])
    model.opt.tolerance = float(options["tolerance"])
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    if case == "legacy_physics":
        model.opt.timestep = 0.001
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                          for name in config["jointNames"]])
    qpos_adr = model.jnt_qposadr[joint_ids]
    dof_adr = model.jnt_dofadr[joint_ids]
    actuators = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0] for joint in joint_ids])
    model.dof_armature[dof_adr] = config["armature"]
    model.dof_frictionloss[dof_adr] = config["frictionLoss"]
    effort = np.asarray(config["effortLimit"])
    model.actuator_ctrllimited[actuators] = True
    model.actuator_ctrlrange[actuators] = np.stack([-effort, effort], axis=1)
    data = mujoco.MjData(model)
    if case == "refresh_constants":
        mujoco.mj_setConst(model, data)
    initial = motion["initialPose"]
    initial_state = motion["initialState"]
    data.qpos[:3] = initial["rootPosition"]
    data.qpos[3:7] = initial["rootQuaternionWxyz"]
    data.qpos[qpos_adr] = initial["jointPosition"]
    data.qvel[:3] = initial_state["rootLinearVelocityWorld"]
    data.qvel[3:6] = initial_state["rootAngularVelocityBody"]
    data.qvel[dof_adr] = initial_state["jointVelocity"]
    if args.scene == "public" or case == "aligned_reset":
        spawn_origin = [0, 0, 0]
        if args.scene == "public":
            terrain = json.loads(public_file(args.source, "robot/terrain.json"))
            spawn_origin = next(region["origin"] for region in terrain["spawnRegions"]
                                if region["family"] == "flat" and region["row"] == 4)
        quaternion = data.qpos[3:7].copy()
        yaw = np.arctan2(2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                         1 - 2 * (quaternion[2]**2 + quaternion[3]**2))
        alignment = np.array([np.cos(-yaw / 2), 0, 0, np.sin(-yaw / 2)])
        mujoco.mju_mulQuat(data.qpos[3:7], alignment, quaternion)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, alignment)
        data.qvel[:3] = rotation.reshape(3, 3) @ data.qvel[:3]
        data.qpos[:3] = np.asarray(spawn_origin) + [0, 0, initial["rootPosition"][2]]
    mujoco.mj_forward(model, data)
    camera = MujocoDepthCamera(model)
    if case == "occluded_camera":
        camera.housing_geom_ids = np.empty(0, dtype=np.int32)
    depth_history = DepthHistory()
    history = {key: np.zeros((4, size), dtype=np.float32) for key, size in
               [("actions", 29), ("base_ang_vel", 3), ("dof_pos", 29),
                ("dof_vel", 29), ("projected_gravity", 3)]}
    last_action = np.zeros(29, dtype=np.float32)
    defaults = np.asarray(config["defaultJointPosition"], dtype=np.float32)
    kp, kd = np.asarray(config["kp"]), np.asarray(config["kd"])
    velocity_limit = np.asarray(config["velocityLimit"])
    scale = np.asarray(config["actionScale"], dtype=np.float32)
    ranges = np.asarray(config["jointRanges"])
    initial_yaw = None
    previous_target = data.qpos[qpos_adr].copy()
    positions, velocities, actions, targets, heights, tilts, heading_errors = [], [], [], [], [], [], []
    low_height_episodes = 0
    was_low = False
    clamped_steps = 0
    max_clamp = 0.0
    frames = motion["frames"][:round(args.duration / config["controlDt"])]
    substeps = round(config["controlDt"] / model.opt.timestep)
    for frame in frames:
        quaternion = data.qpos[3:7]
        quat_w, quat_x, quat_y, quat_z = quaternion
        yaw = np.arctan2(2 * (quat_w * quat_z + quat_x * quat_y), 1 - 2 * (quat_y**2 + quat_z**2))
        if initial_yaw is None:
            initial_yaw = yaw
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        gravity = -rotation.reshape(3, 3)[2].astype(np.float32)
        values = {"actions": last_action, "base_ang_vel": data.qvel[3:6].astype(np.float32) * 0.25,
                  "dof_pos": data.qpos[qpos_adr].astype(np.float32) - defaults,
                  "dof_vel": data.qvel[dof_adr].astype(np.float32), "projected_gravity": gravity}
        state = np.concatenate([values[key] for key in ["dof_pos", "dof_vel", "projected_gravity", "base_ang_vel"]])
        heading_angle = initial_yaw + np.arctan2(*frame["headingRelativeXY"][::-1]) - yaw
        heading_errors.append(float(np.rad2deg(np.arctan2(np.sin(heading_angle), np.cos(heading_angle)))))
        heading = np.array([1 - np.cos(heading_angle), np.sin(heading_angle)], dtype=np.float32)
        if not frame["headingValid"]:
            heading.fill(0)
        vx, vy, valid = frame["velocityHeading0"]
        angle = initial_yaw - yaw
        velocity = np.array([np.cos(angle) * vx - np.sin(angle) * vy,
                             np.sin(angle) * vx + np.cos(angle) * vy, valid], dtype=np.float32)
        feeds = {"state": state[None], "last_action": last_action[None],
                 "history_actor": np.concatenate([history[key].reshape(-1) for key in sorted(history)])[None],
                 "heading": heading[None], "velocity_command": velocity[None],
                 "depth_image": depth_history.append(camera.capture(model, data))[None],
                 "z": np.asarray(frame["z"], dtype=np.float32)[None]}
        action = session.run(["action"], feeds)[0][0]
        for key, value in values.items():
            history[key][1:] = history[key][:-1].copy()
            history[key][0] = value
        last_action = np.clip(action * np.float32(config["actionMultiplier"]),
                              -config["actionClip"], config["actionClip"])
        target = defaults + scale * last_action
        clipped = np.clip(target, ranges[:, 0], ranges[:, 1])
        max_clamp = max(max_clamp, float(np.max(np.abs(target - clipped))))
        clamped_steps += bool(np.any(np.abs(target - clipped) > 1e-6))
        if case == "joint_clamp":
            target = clipped
        for substep in range(substeps):
            control_target = target
            if case == "interpolated":
                fraction = substep / substeps
                control_target = previous_target + fraction * (target - previous_target)
            data.ctrl[actuators] = motor_torque(control_target, data.qpos[qpos_adr], data.qvel[dof_adr],
                                               kp, kd, effort, velocity_limit)
            previous_time = data.time
            mujoco.mj_step(model, data)
            if data.time <= previous_time or not np.isfinite(data.qpos).all():
                raise RuntimeError(f"{case}: unstable simulation")
            heights.append(float(data.qpos[2]))
        mujoco.mj_forward(model, data)
        was_low_now = data.qpos[2] < 0.45
        low_height_episodes += bool(was_low_now and not was_low)
        was_low = was_low_now
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        actions.append(action.copy())
        targets.append(target.copy())
        tilts.append(float(np.rad2deg(np.arccos(np.clip(-gravity[2], -1, 1)))))
        previous_target = target.copy()
    output = args.output / case
    output.mkdir()
    np.savez_compressed(output / "rollout.npz", qpos=positions, qvel=velocities, action=actions, target=targets)
    summary = {"case": case, "motion": motion["id"], "policy_steps": len(frames),
               "simulated_duration": float(data.time), "min_pelvis_height": min(heights),
               "low_height_episodes_below_045m": low_height_episodes, "max_tilt_deg": max(tilts),
               "final_position": data.qpos[:3].tolist(), "steps_with_target_outside_joint_range": clamped_steps,
               "max_target_outside_joint_range_rad": max_clamp,
               "scene": args.scene,
               "max_abs_heading_error_deg": float(np.max(np.abs(heading_errors))),
               "rms_heading_error_deg": float(np.sqrt(np.mean(np.square(heading_errors)))),
               "scope": "native Python reproduction of public browser ONNX/PD contract; no DDS"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if args.video:
        video_positions = [positions[min(round(index * 50 / 30), len(positions) - 1)]
                           for index in range(round(len(positions) * 30 / 50))]
        render_video(model, video_positions, output / "pbfm.mp4")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Diagnose native MuJoCo P-BFM contract differences")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--motion", default="lafan-full-run1-subject5")
    parser.add_argument("--duration", type=float, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["official"],
                        choices=["official", "occluded_camera", "interpolated", "joint_clamp", "legacy_physics",
                                 "refresh_constants", "aligned_reset"])
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--scene", choices=["plane", "box", "public", "public_robot_plane"], default="plane")
    parser.add_argument("--idle-context", type=Path)
    parser.add_argument("--movement-context", type=Path)
    parser.add_argument("--walk-window", type=float, nargs=2, action="append", default=[])
    parser.add_argument("--speed", type=float, default=1.2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(public_file(args.source, "config.json"))
    if args.scene in ("public", "public_robot_plane"):
        files = json.loads(public_file(args.source, "robot/files.json"))
        for name in [*files, "files.json", "terrain.json"]:
            destination = args.output / "public_robot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(public_file(args.source, f"robot/{name}"))
    motion = json.loads(gzip.decompress(public_file(args.source, f"tracking/{args.motion}.json.gz")))
    if args.idle_context:
        idle = yaml.safe_load(args.idle_context.read_text())
        movement = yaml.safe_load(args.movement_context.read_text()) if args.movement_context else None
        movement_frames = movement.get("frames", [movement]) if movement else None
        reference_step = 0
        sequence = motion["frames"]
        frames = []
        for step in range(round(args.duration / config["controlDt"])):
            walking = any(begin <= step * config["controlDt"] < end for begin, end in args.walk_window)
            frame = dict(sequence[reference_step % len(sequence)])
            frame.update(headingRelativeXY=[1, 0], headingValid=True, velocityHeading0=[args.speed if walking else 0, 0, 1])
            if walking:
                if movement_frames:
                    frame["z"] = movement_frames[reference_step % len(movement_frames)]["z"]
                reference_step += 1
            else:
                frame["z"] = idle["z"]
            frames.append(frame)
        motion["frames"] = frames
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    model = Path(__file__).resolve().parents[2] / "actions/perception/pbfm/model/policy.onnx"
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    summaries = [run_case(args, config, motion, session, case) for case in args.cases]
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
