import argparse
import gzip
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import yaml


PUBLIC_REVISION = "671fbafe71a5f58bab77eeb5d969051fd2d283f4"


def periodic(values, harmonics=8):
    spectrum = np.fft.rfft(values, axis=0)
    spectrum[harmonics + 1:] = 0
    return np.fft.irfft(spectrum, n=len(values), axis=0)


def derivative(values, dt):
    result = (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (2 * dt)
    return gaussian_filter1d(result, 2, axis=0, mode="wrap", radius=8)


def symmetric_gait(joints, names):
    half_cycle = len(joints) // 2
    for suffix, sign in [("hip_pitch", 1), ("hip_roll", -1), ("hip_yaw", -1),
                         ("knee", 1), ("ankle_pitch", 1), ("ankle_roll", -1),
                         ("shoulder_pitch", 1), ("shoulder_roll", -1), ("shoulder_yaw", -1),
                         ("elbow", 1), ("wrist_roll", -1), ("wrist_pitch", 1), ("wrist_yaw", -1)]:
        left = names.index("left_" + suffix + "_joint")
        right = names.index("right_" + suffix + "_joint")
        joints[:, left] = (joints[:, left] + sign * np.roll(joints[:, right], -half_cycle)) / 2
        joints[:, right] = sign * np.roll(joints[:, left], -half_cycle)
    for suffix, sign in [("yaw", -1), ("roll", -1), ("pitch", 1)]:
        index = names.index("waist_" + suffix + "_joint")
        joints[:, index] = (joints[:, index] + sign * np.roll(joints[:, index], -half_cycle)) / 2
    return joints


def level_swing_feet(config, model, joints, root_height, root_rotation, target_pitch):
    data = mujoco.MjData(model)
    addresses = [model.joint(name).qposadr[0] for name in config["jointNames"]]
    corrected = joints.copy()
    for step, position in enumerate(joints):
        data.qpos[:3] = [0, 0, root_height[step]]
        data.qpos[3:7] = root_rotation[step].as_quat()[[3, 0, 1, 2]]
        data.qpos[addresses] = position
        mujoco.mj_forward(model, data)
        for side in ("left", "right"):
            foot = data.body(side + "_ankle_roll_link")
            rotation = foot.xmat.reshape(3, 3)
            pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
            weight = np.clip((foot.xpos[2] - 0.065) / 0.07, 0, 1)
            weight = weight * weight * (3 - 2 * weight)
            index = config["jointNames"].index(side + "_ankle_pitch_joint")
            corrected[step, index] -= weight * (pitch - target_pitch)
    corrected = periodic(corrected)
    for side in ("left", "right"):
        name = side + "_ankle_pitch_joint"
        index = config["jointNames"].index(name)
        corrected[:, index] = np.clip(corrected[:, index], *model.joint(name).range)
    return corrected


def encode_reference(config, model, joints, root_height, root_rotation, speed, encoder, yaw_rate=0.0):
    dt = config["controlDt"]
    count = len(joints)
    defaults = np.asarray(config["defaultJointPosition"])
    addresses = [model.joint(name).qposadr[0] for name in config["jointNames"]]
    links = ["pelvis", *["torso_link" if name == "waist_pitch_joint" else
                         name.removesuffix("_joint") + "_link" for name in config["jointNames"]]]
    data = mujoco.MjData(model)
    positions, rotations = [], []
    for step in range(count):
        data.qpos[:3] = [0, 0, root_height[step]]
        data.qpos[3:7] = root_rotation[step].as_quat()[[3, 0, 1, 2]]
        data.qpos[addresses] = joints[step]
        mujoco.mj_forward(model, data)
        position = np.array([data.body(name).xpos.copy() for name in links])
        rotation = np.array([data.body(name).xmat.reshape(3, 3).copy() for name in links])
        torso = links.index("torso_link")
        positions.append(np.concatenate([position, [position[torso] + rotation[torso] @ [0, 0, 0.35]]]))
        rotations.append(np.concatenate([rotation, [rotation[torso]]]))
    positions, rotations = np.asarray(positions), np.asarray(rotations)
    linear_velocity = derivative(positions, dt)
    linear_velocity[:, :, 0] += speed
    angular_velocity = Rotation.from_matrix(
        (np.roll(rotations, -1, axis=0) @ np.swapaxes(rotations, -1, -2)).reshape(-1, 3, 3)
    ).as_rotvec().reshape(count, 31, 3) / dt
    angular_velocity = gaussian_filter1d(angular_velocity, 2, axis=0, mode="wrap", radius=8)
    relative_position = positions - positions[:, :1]
    linear_velocity[:, :, 0] -= yaw_rate * relative_position[:, :, 1]
    linear_velocity[:, :, 1] += yaw_rate * relative_position[:, :, 0]
    angular_velocity[:, :, 2] += yaw_rate
    joint_velocity = (np.roll(joints, -1, axis=0) - joints) / dt
    state = np.concatenate([joints - defaults, joint_velocity, -rotations[:, 0, 2, :],
                            np.einsum("nji,nj->ni", rotations[:, 0], angular_velocity[:, 0]) * config["angularVelocityScale"]], axis=1)
    privileged = np.concatenate([root_height[:, None],
                                 (positions[:, 1:] - positions[:, :1]).reshape(count, -1),
                                 rotations[:, :, :, [0, 2]].transpose(0, 1, 3, 2).reshape(count, -1),
                                 linear_velocity.reshape(count, -1), angular_velocity.reshape(count, -1)], axis=1)
    latents = []
    for observation, privileged_observation in zip(state, privileged):
        latents.append(encoder.run(["z"], {"state": observation[None].astype(np.float32),
                                           "privileged_state": privileged_observation[None].astype(np.float32)})[0][0])
    latent = np.mean([np.roll(latents, -offset, axis=0) for offset in range(1, 9)], axis=0)
    latent *= 16 / np.linalg.norm(latent, axis=1, keepdims=True)
    return latent, positions


def main():
    parser = argparse.ArgumentParser(description="Encode a straight, periodic LaFAN gait using the official backward encoder")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=682)
    parser.add_argument("--length", type=int, default=54)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--turn-rate", type=float, default=0.6)
    parser.add_argument("--turn-gait-amplitude", type=float, default=0.6)
    parser.add_argument("--symmetric", action="store_true", help="Mirror-average the two half cycles to remove left/right bias")
    parser.add_argument("--swing-foot-pitch", type=float, help="Target world foot pitch during swing, in radians")
    args = parser.parse_args()
    if args.start < 0 or args.length < 18 or not np.isfinite(args.speed) or args.speed == 0:
        parser.error("start must be nonnegative, length at least 18, and speed finite and nonzero")
    if args.swing_foot_pitch is not None and not np.isfinite(args.swing_foot_pitch):
        parser.error("swing foot pitch must be finite")
    if not np.isfinite(args.turn_rate) or args.turn_rate <= 0:
        parser.error("turn rate must be finite and positive")
    if not 0 < args.turn_gait_amplitude <= 1:
        parser.error("turn gait amplitude must be within (0,1]")
    package = args.source / "model/public_demo"
    config = json.loads((package / "config.json").read_text())
    asset = "interactive/pbfm/tracking/lafan-full-walk1-subject5.json.gz"
    motion = json.loads(gzip.decompress(subprocess.check_output([
        "git", "-C", str(args.source), "show", f"{PUBLIC_REVISION}:{asset}"])))
    if config["checkpointStep"] != 364822528 or motion["checkpointStep"] != config["checkpointStep"] or motion["dt"] != 0.02:
        raise ValueError("Expected the public P-BFM checkpoint and 50 Hz reference")
    if motion["referencePoseTiming"] != "post-action" or args.start + args.length > len(motion["frames"]):
        raise ValueError("Unsupported reference timing or frame range")
    frames = motion["frames"][args.start:args.start + args.length]
    joints = periodic(np.asarray([frame["referencePose"]["jointPosition"] for frame in frames]))
    root_height = periodic(np.asarray([frame["referencePose"]["rootPosition"][2] for frame in frames]))
    quaternion = np.asarray([frame["referencePose"]["rootQuaternionWxyz"] for frame in frames])
    rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
    yaw = np.arctan2(rotation.as_matrix()[:, 1, 0], rotation.as_matrix()[:, 0, 0])
    root_rotation = Rotation.from_rotvec(periodic((Rotation.from_euler("z", -yaw) * rotation).as_rotvec()))
    if args.symmetric:
        if args.length % 2:
            parser.error("symmetric gait requires an even frame count")
        joints = symmetric_gait(joints, config["jointNames"])
        root_height = (root_height + np.roll(root_height, -args.length // 2)) / 2
        tilt = root_rotation.as_euler("xyz")
        tilt[:, 0] = (tilt[:, 0] - np.roll(tilt[:, 0], -args.length // 2)) / 2
        tilt[:, 1] = (tilt[:, 1] + np.roll(tilt[:, 1], -args.length // 2)) / 2
        tilt[:, 2] = 0
        root_rotation = Rotation.from_euler("xyz", tilt)
    model = mujoco.MjModel.from_xml_path(str(args.source / "pbfm/data/robots/g1_mjlab/g1_29dof.xml"))
    if args.swing_foot_pitch is not None:
        joints = level_swing_feet(config, model, joints, root_height, root_rotation, args.swing_foot_pitch)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    encoder = ort.InferenceSession(str(package / "backward_encoder.onnx"), options, providers=["CPUExecutionProvider"])
    latent, positions = encode_reference(config, model, joints, root_height, root_rotation, args.speed, encoder)
    output = {"contract": "tracking_heading0_yaw_velocity_v1", "dt": config["controlDt"],
              "source": f"P-BFM {PUBLIC_REVISION}:{asset}", "source_frame_start": args.start,
              "source_frame_count": len(frames), "checkpoint_step": config["checkpointStep"],
              "reference_forward_speed_mps": args.speed,
              "feature_source": f"P-BFM {PUBLIC_REVISION}:interactive/assets/motion-import-uImOWE3z.js backwardFeatures/motionKinematics",
              "adaptation": "periodic Fourier smoothing; root yaw and lateral translation removed; forward joint/angular differences and central linear differences with periodic boundaries; link velocities Gaussian-smoothed (sigma=2, radius=8); future-eight latent mean normalized to 16",
              "symmetric_half_cycles": args.symmetric,
              "swing_foot_pitch_rad": args.swing_foot_pitch,
              "frames": [{"z": value.tolist(), "heading_relative_xy": [1, 0], "heading_valid": True,
                          "reference_velocity_heading0_xy": [args.speed, 0], "velocity_valid": True} for value in latent]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "context.yaml").write_text(yaml.safe_dump(output, sort_keys=False))
    idle_joints = np.asarray(config["defaultJointPosition"])[None].repeat(args.length, axis=0)
    for side in ("left", "right"):
        idle_joints[:, config["jointNames"].index(side + "_elbow_joint")] = 1.2
        idle_joints[:, config["jointNames"].index(side + "_shoulder_roll_joint")] = 0.1 if side == "left" else -0.1
    idle, _ = encode_reference(config, model, idle_joints, np.full(args.length, 0.793),
                               Rotation.identity(args.length), 0, encoder)
    (args.output / "idle.yaml").write_text(yaml.safe_dump({"contract": "reward_goal_yaw_command_v1",
        "z": idle[0].tolist(), "heading_relative_xy": [1, 0], "heading_valid": True,
        "velocity_command": [0, 0, 1], "target_joint_position": idle_joints[0].tolist(),
        "source": "official backward encoder, default legs and lowered forearms (elbows 1.2 rad)"}, sort_keys=False))
    turn_joints = idle_joints + args.turn_gait_amplitude * (joints - idle_joints)
    turn_height = 0.793 + args.turn_gait_amplitude * (root_height - 0.793)
    turn_rotation = Rotation.from_rotvec(root_rotation.as_rotvec() * args.turn_gait_amplitude)
    for direction, sign in (("left", 1), ("right", -1)):
        turn, _ = encode_reference(config, model, turn_joints, turn_height, turn_rotation,
                                    0, encoder, yaw_rate=sign * args.turn_rate)
        turning = {**output, "root_yaw_rate_rad_s": sign * args.turn_rate,
                   "turn_gait_amplitude": args.turn_gait_amplitude,
                   "adaptation": "scaled gait about lowered-arm idle; zero root translation; yaw angular velocity and omega-cross-relative-position link velocities encoded in the root-yaw frame",
                   "frames": [{"z": value.tolist(), "heading_relative_xy": [1, 0], "heading_valid": True,
                               "reference_velocity_heading0_xy": [0, 0], "velocity_valid": True} for value in turn]}
        (args.output / f"turn_{direction}.yaml").write_text(yaml.safe_dump(turning, sort_keys=False))
    np.savez_compressed(args.output / "reference.npz", joint_position=joints, root_height=root_height,
                        root_quaternion_wxyz=root_rotation.as_quat()[:, [3, 0, 1, 2]], link_position=positions)
    print(json.dumps({"frames": len(frames), "duration": len(frames) * config["controlDt"],
                      "latent_norm": np.linalg.norm(latent, axis=1).min().item(),
                      "left_ankle_z_range": np.ptp(positions[:, 6, 2]).item(),
                      "right_ankle_z_range": np.ptp(positions[:, 12, 2]).item()}))


if __name__ == "__main__":
    main()
