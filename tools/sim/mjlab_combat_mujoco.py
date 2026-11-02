"""Run the mjlab combat ActionGo variant through loopback DDS and validate parity."""

import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

import mujoco
import numpy as np
import torch
import yaml

from allmimic_mujoco import load_file


ROOT = Path(__file__).resolve().parents[2]


def yaw_quaternion(angle):
    return np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])


def multiply_quaternion(left, right):
    result = np.empty(4)
    mujoco.mju_mulQuat(result, np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    return result


def reference_alignment(rows, state, enabled):
    if not enabled:
        return rows.copy(), 0.0
    waist = multiply_quaternion(yaw_quaternion(state[7 + 12]),
        multiply_quaternion([np.cos(state[7 + 13] / 2), np.sin(state[7 + 13] / 2), 0, 0],
                            [np.cos(state[7 + 14] / 2), 0, np.sin(state[7 + 14] / 2), 0]))
    torso = multiply_quaternion(state[3:7], waist)
    def heading(quaternion):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=np.float64))
        return np.arctan2(rotation[3], rotation[0])
    offset = heading(torso) - heading(rows[0, 61:65])
    offset = np.arctan2(np.sin(offset), np.cos(offset))
    rotation = np.array([[np.cos(offset), -np.sin(offset)], [np.sin(offset), np.cos(offset)]])
    result = rows.copy()
    for start in (58, 65, 68, 71, 74, 77, 80):
        origin = rows[0, 58:60] if start == 58 or start >= 71 else np.zeros(2)
        result[:, start:start + 2] = (rows[:, start:start + 2] - origin) @ rotation.T + origin
    for frame in range(len(rows)):
        result[frame, 61:65] = multiply_quaternion(yaw_quaternion(offset), rows[frame, 61:65])
    return result, float(offset)


@torch.inference_mode()
def validate(args, records, velocities, commands):
    exporter = load_file("mjlab_export", ROOT / "tools/export/mjlab_combat_export.py")
    manifest = json.loads((args.package / "manifest.json").read_text())
    if exporter.sha256(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the exported model")
    if exporter.sha256(args.package / "policy.onnx") != manifest["policy_sha256"]:
        raise ValueError("Policy hash does not match the export manifest")
    env, runner = exporter.load_environment(args.training_repo, args.checkpoint, Path(manifest["motion"]), args.device)
    try:
        from mjlab.utils.lab_api.math import matrix_from_quat

        robot = env.scene["robot"]
        motion = env.command_manager.get_term("motion")
        actor = copy.deepcopy(runner.alg._raw_actor).cpu().eval().as_onnx(verbose=False)
        policy = exporter.MjlabCombatPolicy(actor, robot.data.default_joint_pos[0].cpu())
        rows = exporter.reference_rows(motion.bank, motion.motion_anchor_body_index, motion.task_body_indexes)
        np.testing.assert_array_equal(rows.numpy(), np.loadtxt(args.package / "reference.csv", delimiter=",", dtype=np.float32))
        aligned_rows, yaw_offset = reference_alignment(rows.numpy(), records[0], args.yaw_alignment == "initial")
        canonical_rows = rows
        rows = torch.tensor(aligned_rows, dtype=torch.float32)
        events = [json.loads(line) for line in (args.log_dir / "player.jsonl").read_text().splitlines()]
        events = [event for event in events if event.get("event") == "mjlab_combat_inference"][:len(commands)]
        if len(events) != len(commands):
            raise ValueError("Missing C++ inference trace")
        params = yaml.safe_load((args.package / "MjlabCombat.yaml").read_text())["robot"]
        previous = torch.zeros(1, 29)
        errors = {key: 0.0 for key in ("raw_input", "actions", "target", "dds_target", "mjlab_observation")}
        sample_indices = {0, min(25, len(commands) - 1), len(commands) // 2, len(commands) - 1}
        for index, event in enumerate(events):
            def values(key):
                return torch.tensor(np.fromstring(event[key], sep=","), dtype=torch.float32)[None]

            frame = min(index, len(rows) - 1)
            if event["reference_frame"] != frame or event["step"] != index:
                raise ValueError("Reference time differs from DDS simulation time")
            raw = torch.cat((values("q_csv"), values("dq_csv"), values("quat_csv"), values("gyro_csv"), previous,
                             rows[frame][None], rows[min(frame + 2, len(rows) - 1)][None],
                             torch.tensor([[float(frame + 2 < len(rows))]])), dim=1)
            action = values("actions_csv")
            target = torch.tensor(params["default_angles"])[None] + torch.tensor(params["action_scale"])[None] * action
            for key, expected, actual, tolerance in (
                ("raw_input", raw, values("obs_csv"), 3e-6),
                ("actions", policy(raw), action, 1e-4),
                ("target", target, values("target_csv"), 1e-6),
            ):
                errors[key] = max(errors[key], float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=1e-4)
            if not args.hardware_safety:
                expected = values("target_csv")
                actual = torch.tensor(commands[index]["q"], dtype=torch.float32)[None]
                errors["dds_target"] = max(errors["dds_target"], float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
            if index in sample_indices:
                canonical_state = np.array(records[index], copy=True)
                canonical_state[3:7] = multiply_quaternion(yaw_quaternion(-yaw_offset), canonical_state[3:7])
                canonical_velocity = np.array(velocities[index], copy=True)
                inverse_rotation = np.array([[np.cos(yaw_offset), np.sin(yaw_offset)],
                                             [-np.sin(yaw_offset), np.cos(yaw_offset)]])
                canonical_state[:2] = inverse_rotation @ (canonical_state[:2] - canonical_rows[0, 58:60].numpy()) + canonical_rows[0, 58:60].numpy()
                canonical_velocity[:2] = inverse_rotation @ canonical_velocity[:2]
                state = torch.tensor(canonical_state, dtype=torch.float32, device=args.device)[None]
                velocity = torch.tensor(canonical_velocity, dtype=torch.float32, device=args.device)[None]
                angular_world = (matrix_from_quat(state[:, 3:7]) @ velocity[:, 3:6, None]).squeeze(-1)
                robot.write_joint_state_to_sim(state[:, 7:], velocity[:, 6:])
                robot.write_root_state_to_sim(torch.cat((state[:, :7], velocity[:, :3], angular_world), dim=1))
                motion.motion_ids.zero_()
                motion.time_steps.fill_(frame)
                env.action_manager.action.copy_(previous.to(args.device))
                env.sim.forward()
                env.scene.update(env.step_dt)
                motion._update_command()
                expected = env.observation_manager.compute(update_history=True)["actor"].cpu()
                actual = policy.observation(raw)
                errors["mjlab_observation"] = max(errors["mjlab_observation"], float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-4)
            previous = action
        return {"passed": True, "samples": len(events), "mjlab_state_samples": len(sample_indices),
                "yaw_offset_deg": float(np.rad2deg(yaw_offset)),
                "dds_target_compared": not args.hardware_safety,
                "max_abs_error": errors}
    finally:
        env.close()


def render(args, tracked, truth):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    helpers = load_file("reference_renderer", ROOT / "tools/sim/uni_supermimic_mujoco.py")
    model = mujoco.MjModel.from_binary_path(str(args.package / "scene.mjb"))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    actual, reference = mujoco.MjData(model), mujoco.MjData(model)
    geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == model.jnt_bodyid[0])
    for geom in geoms:
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            model.geom_group[geom] = 3
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    scene = mujoco.MjvScene(model, maxgeom=max(1000, model.ngeom * 2))
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 4.0, 135, -16
    output = args.log_dir / "preview.mp4"
    with mujoco.Renderer(model, height=720, width=1280) as renderer, imageio.get_writer(
            output, fps=50, codec="libx264", quality=7, macro_block_size=2,
            ffmpeg_params=["-movflags", "+faststart"]) as writer:
        for frame in range(len(tracked) - 1):
            actual.qpos[:] = tracked[frame]
            reference.qpos[:] = truth[frame]
            reference.qpos[1] += 1.6
            mujoco.mj_forward(model, actual)
            mujoco.mj_forward(model, reference)
            camera.lookat[:] = (actual.qpos[:3] + reference.qpos[:3]) / 2
            camera.lookat[2] = max(0.7, camera.lookat[2])
            renderer.update_scene(actual, camera=camera, scene_option=options)
            mujoco.mjv_updateScene(model, reference, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            helpers.append_reference_geoms(mujoco, renderer.scene, scene, geoms, np.array([0, 0.8, 1, 0.45], dtype=np.float32))
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 1280, 64), fill=(16, 20, 26))
            draw.text((16, 8), f"mjlab combat | 4 GPU checkpoint 92000 | ActionGo C++ ONNX | {frame / 50:.2f}s", fill="white", font_size=18)
            draw.text((16, 35), "Solid: policy simulation | Cyan: reference shifted 1.6m | 1x", fill=(90, 225, 255), font_size=18)
            writer.append_data(np.asarray(image))
    return str(output)


def run(args):
    helpers = load_file("supermimic_dds", ROOT / "tools/sim/supermimic_mujoco.py")
    args.log_dir.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(args.package / "scene.mjb"))
    data = mujoco.MjData(model)
    with np.load(args.package / "motion.npz") as archive:
        reference, velocity = archive["qpos"], archive["qvel"]
    data.qpos[:], data.qvel[:] = reference[0], velocity[0]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    data.qvel[3:6] = rotation.reshape(3, 3).T @ velocity[0, 3:6]
    if args.initial_pose == "default":
        params = yaml.safe_load((args.package / "MjlabCombat.yaml").read_text())["robot"]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7:] = params["default_angles"]
        data.qvel[:] = 0
    data.qpos[3:7] = multiply_quaternion(yaw_quaternion(np.deg2rad(args.initial_yaw_deg)), data.qpos[3:7])
    initial_heading = np.deg2rad(args.initial_yaw_deg)
    data.qvel[:2] = np.array([[np.cos(initial_heading), -np.sin(initial_heading)],
                             [np.sin(initial_heading), np.cos(initial_heading)]]) @ data.qvel[:2]
    mujoco.mj_forward(model, data)
    decimation = round(0.02 / model.opt.timestep)
    if decimation <= 0 or not np.isclose(decimation * model.opt.timestep, 0.02):
        raise ValueError("Simulation timestep must divide the trained 20ms policy timestep")
    indices = {"q": np.arange(7, 36), "v": np.arange(6, 35), "root_q": 0, "root_v": 0,
               "actuators": np.array([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0] for joint in range(1, 30)])}
    initializer = helpers.load_sdk(args)
    domain = secrets.randbelow(140) + 50
    initializer(domain, "lo")
    bridge = helpers.DDSBridge(model, data, indices)
    spec = yaml.safe_load((args.package / "generic_mjlab_combat.yaml").read_text())
    spec["robot"]["params"] = str(args.package / "MjlabCombat.yaml")
    spec["model"]["path"] = str(args.package / "policy.onnx")
    spec["reference"]["path"] = str(args.package / "reference.csv")
    spec["reference"]["yaw_alignment"] = args.yaw_alignment
    if args.hardware_safety:
        spec.pop("safety", None)
        spec["handoff"]["activation_blend_sec"] = 0.1
    spec_path = args.log_dir / "simulation_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    ready, start = args.log_dir / "ready", args.log_dir / "start"
    command = [str(args.player), "--net", "lo", "--domain-id", str(domain), "--spec", str(spec_path),
               "--ready-file", str(ready), "--start-file", str(start),
               "--event-log", str(args.log_dir / "player.jsonl"), "--no-damping-on-stop"]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(filter(None, [args.library_path, environment.get("LD_LIBRARY_PATH")]))
    records, velocities, commands = [], [], []
    process = None
    with (args.log_dir / "player.log").open("w") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            deadline = time.monotonic() + 30
            while not ready.exists() or not bridge.hold_messages:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"ActionGo startup failed; see {log.name}")
                bridge.publish()
                time.sleep(0.002)
            start.touch()
            for index in range(args.steps):
                deadline, current = time.monotonic() + 10, None
                while current is None:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"ActionGo command timeout; see {log.name}")
                    bridge.publish()
                    current = bridge.pop()
                    if current is None:
                        time.sleep(0.001)
                if current["sequence"] != index + 1:
                    raise ValueError("Nonconsecutive DDS policy sequence")
                records.append(data.qpos.copy())
                velocities.append(data.qvel.copy())
                commands.append(current)
                bridge.apply(current)
                for substep in range(decimation):
                    mujoco.mj_step(model, data)
                    bridge.publish()
                    time.sleep(0.0005)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Nonfinite MuJoCo state")
            records.append(data.qpos.copy())
        finally:
            helpers.stop_player(process)
    actual = np.asarray(records)
    truth = reference[np.arange(len(actual)).clip(max=len(reference) - 1)].copy()
    rows = np.loadtxt(args.package / "reference.csv", delimiter=",", dtype=np.float32)
    _, yaw_offset = reference_alignment(rows, records[0], args.yaw_alignment == "initial")
    heading_rotation = np.array([[np.cos(yaw_offset), -np.sin(yaw_offset)],
                                 [np.sin(yaw_offset), np.cos(yaw_offset)]])
    truth[:, :2] = (truth[:, :2] - rows[0, 58:60]) @ heading_rotation.T + rows[0, 58:60]
    for frame in range(len(truth)):
        truth[frame, 3:7] = multiply_quaternion(yaw_quaternion(yaw_offset), truth[frame, 3:7])
    np.savez_compressed(args.log_dir / "trace.npz", actual=actual, reference=truth, qvel=np.asarray(velocities))
    metrics = {"steps": len(commands), "seconds": len(commands) / 50, "sim_timestep": model.opt.timestep,
               "initial_pose": args.initial_pose, "initial_yaw_deg": args.initial_yaw_deg,
               "yaw_alignment": args.yaw_alignment, "hardware_safety": args.hardware_safety,
               "joint_rmse_rad": float(np.sqrt(np.mean((actual[1:, 7:] - truth[1:, 7:]) ** 2))),
               "root_position_mean_error_m": float(np.linalg.norm(actual[1:, :3] - truth[1:, :3], axis=1).mean()),
               "minimum_pelvis_height_m": float(actual[:, 2].min()), "finite": True, "automatic_resets": 0}
    metrics["maximum_pelvis_tilt_deg"] = float(np.rad2deg(np.arccos(
        (1 - 2 * (actual[:, 4] ** 2 + actual[:, 5] ** 2)).clip(-1, 1))).max())
    (args.log_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    parity = validate(args, records[:-1], velocities, commands)
    (args.log_dir / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    video = render(args, actual, truth) if args.video else None
    print(json.dumps({"metrics": metrics, "parity": parity, "video": video}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--player", type=Path, default=ROOT.parent / "ActionGo-local/build-bundled-ort/generic_motion_tracking_player")
    parser.add_argument("--python-deps", type=Path, default=ROOT.parent / "ActionGo-local/python313")
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--library-path", default="/opt/conda/lib")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--initial-pose", choices=("reference", "default"), default="reference")
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
    parser.add_argument("--yaw-alignment", choices=("none", "initial"), default="initial")
    parser.add_argument("--hardware-safety", action="store_true", help="Use hardware command filtering and 0.1s blend; DDS remains lockstep")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    args.package, args.log_dir = args.package.resolve(), args.log_dir.resolve()
    torch.set_num_threads(2)
    run(args)
