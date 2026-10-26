#!/usr/bin/env python3
"""Export a Uni-SuperMimic checkpoint and one native training motion for ActionGo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import torch
import yaml


OBS_DIM = 99
HISTORY_LENGTH = 10
GOAL_FRAME_DIM = 73
GOAL_FRAMES = 10
GOAL_DIM = GOAL_FRAME_DIM * GOAL_FRAMES
ACTION_DIM = 29
PACKED_DIM = OBS_DIM * HISTORY_LENGTH + GOAL_DIM


class PackedPolicy(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        history = packed[:, : OBS_DIM * HISTORY_LENGTH].reshape(-1, HISTORY_LENGTH, OBS_DIM)
        goal = packed[:, OBS_DIM * HISTORY_LENGTH :]
        return self.policy(history, goal)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *arguments: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def load_training_modules(training_repo: Path):
    sys.path.insert(0, str(training_repo / "src"))
    from unisupermimic.rl.algos.supermimic import networks
    from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP
    from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib
    from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import (
        G1_JOINT_NAMES,
        action_scale,
        default_joint_pos_list,
    )
    return networks, G1Kinematics, MDP, MotionLib, G1_JOINT_NAMES, action_scale, default_joint_pos_list


def load_actor(training_repo: Path, checkpoint_path: Path):
    networks, *_ = load_training_modules(training_repo)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    arguments = checkpoint["args"]
    if int(arguments.get("goal_frames", -1)) != GOAL_FRAMES:
        raise ValueError("Checkpoint goal_frames must be 10")
    if int(arguments.get("hist", -1)) != HISTORY_LENGTH:
        raise ValueError("Checkpoint hist must be 10")
    if int(arguments.get("z_dim", -1)) != 32:
        raise ValueError("Checkpoint z_dim must be 32")
    if int(checkpoint.get("iteration", -1)) != 200000:
        raise ValueError(f"Expected the 200000-iteration checkpoint, got {checkpoint.get('iteration')}")
    networks.AMP_DTYPE = None
    policy = networks.Policy(
        OBS_DIM,
        GOAL_DIM,
        hist=HISTORY_LENGTH,
        z_dim=int(arguments["z_dim"]),
        num_joints=ACTION_DIM,
        hidden=tuple(arguments["policy_hidden"]),
    ).cpu().float().eval()
    policy.load_state_dict(checkpoint["pi"], strict=True)
    for name, value in policy.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite actor tensor: {name}")
    return checkpoint, arguments, policy


def load_motion(training_repo: Path, motion_path: Path, source_fps: float):
    _, G1Kinematics, MDP, MotionLib, *_ = load_training_modules(training_repo)
    scene_path = training_repo / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"
    mjcf_path = training_repo / "src/unisupermimic/assets/robots/g1/supermimic.xml"
    kin = G1Kinematics(str(mjcf_path), "cpu")
    motion = MotionLib([str(motion_path)], kin, "cpu", src_fps=source_fps)
    default_q = torch.tensor(
        load_training_modules(training_repo)[-1](), dtype=torch.float32
    )
    mdp = MDP(kin, motion, default_q)
    model = __import__("mujoco").MjModel.from_xml_path(str(scene_path))
    return motion, mdp, model


def robot_parameters(model, joint_names, get_action_scale, get_default_q):
    import mujoco

    qpos = []
    qpos_lower = []
    qpos_upper = []
    kp = []
    kd = []
    effort = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise ValueError(f"Training scene is missing joint or actuator {name}")
        qpos.append(int(model.jnt_qposadr[joint_id]))
        qpos_lower.append(float(model.jnt_range[joint_id, 0]))
        qpos_upper.append(float(model.jnt_range[joint_id, 1]))
        kp.append(float(model.actuator_gainprm[actuator_id, 0]))
        kd.append(float(-model.actuator_biasprm[actuator_id, 2]))
        effort.append(float(model.actuator_forcerange[actuator_id, 1]))
    if abs(float(model.opt.timestep) - 0.005) > 1e-9:
        raise ValueError(f"Expected the training physics step of 5 ms, got {model.opt.timestep}")
    defaults = get_default_q()
    scales = [get_action_scale(name) for name in joint_names]
    return {
        "schema_version": 1,
        "robot": {
            "motor_count": ACTION_DIM,
            "policy_joint_order": list(joint_names),
            "joint_map": list(range(ACTION_DIM)),
            "default_angles": defaults,
            "kp": kp,
            "kd": kd,
            "action_scale": scales,
            "torque_limits": effort,
            "joint_pos_lower_limit": qpos_lower,
            "joint_pos_upper_limit": qpos_upper,
        },
        "policy": {"num_obs": PACKED_DIM, "num_actions": ACTION_DIM},
        "observation": {},
        "action": {},
        "safety": {},
        "_qpos_addresses": qpos,
    }


def validation_inputs(policy, mdp, motion):
    indices = torch.tensor(sorted({0, motion.num_frames // 2, motion.num_frames - 1}), dtype=torch.long)
    states = motion.ref_state(indices)
    observations = mdp.obs(states, torch.zeros(len(indices), ACTION_DIM),
                           torch.zeros(len(indices), 33), indices)
    history = observations.unsqueeze(1).repeat(1, HISTORY_LENGTH, 1)
    goals = motion.goal(indices, GOAL_FRAMES)
    samples = [torch.cat((history[i:i + 1].flatten(1), goals[i:i + 1]), dim=1)
               for i in range(len(indices))]
    generator = torch.Generator().manual_seed(0)
    random_history = policy.obs_norm.mean + policy.obs_norm.std() * torch.randn(
        1, HISTORY_LENGTH, OBS_DIM, generator=generator)
    random_goal = policy.goal_norm.mean + policy.goal_norm.std() * torch.randn(
        1, GOAL_DIM, generator=generator)
    samples.append(torch.cat((random_history.flatten(1), random_goal), dim=1))
    return samples


def export_and_verify(packed_policy, samples, output_path):
    torch.onnx.export(packed_policy, samples[0], str(output_path), input_names=["obs"],
                      output_names=["actions"], opset_version=17, do_constant_folding=True,
                      dynamo=False)
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model, full_check=True)
    for values, name, expected_shape in ((model.graph.input, "obs", [1, PACKED_DIM]),
                                         (model.graph.output, "actions", [1, ACTION_DIM])):
        value = values[0]
        shape = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        if len(values) != 1 or value.name != name or shape != expected_shape:
            raise ValueError(f"Unexpected ONNX tensor: {value.name}, {shape}")
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"ONNX tensor {name} must be FP32")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    runner = ort.InferenceSession(str(output_path), options, providers=["CPUExecutionProvider"])
    errors = []
    for sample in samples:
        expected = packed_policy(sample).detach().numpy()
        actual = runner.run(["actions"], {"obs": sample.numpy()})[0]
        if actual.shape != (1, ACTION_DIM) or not np.isfinite(actual).all():
            raise ValueError("Invalid ONNX policy output")
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        errors.append(float(np.max(np.abs(actual - expected))))
    return {"onnx_checker": "passed", "backend": "onnxruntime_cpu", "sample_count": len(samples),
            "samples": "first/middle/final training-reference frames plus seeded normalized random input",
            "max_abs_error": max(errors), "atol": 1e-5, "rtol": 1e-4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=50.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    training_repo = args.training_repo.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    motion_path = args.motion.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (training_repo / "src/unisupermimic/tasks/motion_tracking/g1/supermimic/mdp.py").is_file():
        parser.error("--training-repo must point to the Uni-SuperMimic source checkout")
    if not checkpoint_path.is_file() or not motion_path.is_file():
        parser.error("--checkpoint and --motion must be existing files")
    if not np.isfinite(args.source_fps) or args.source_fps <= 0:
        parser.error("--source-fps must be finite and positive")
    torch.set_num_threads(2)
    checkpoint, training_args, policy = load_actor(training_repo, checkpoint_path)
    networks, _, _, _, joint_names, get_action_scale, get_default_q = load_training_modules(training_repo)
    del networks
    motion, mdp, mujoco_model = load_motion(training_repo, motion_path, args.source_fps)
    if len(joint_names) != ACTION_DIM or mdp.obs_dim != OBS_DIM or mdp.goal_frame_dim != GOAL_FRAME_DIM:
        raise ValueError("Uni-SuperMimic training MDP dimensions do not match the ActionGo adapter")
    params = robot_parameters(mujoco_model, joint_names, get_action_scale, get_default_q)
    qpos_addresses = params.pop("_qpos_addresses")
    packed_policy = PackedPolicy(policy).cpu().float().eval()
    samples = validation_inputs(policy, mdp, motion)
    frames = torch.arange(motion.num_frames)
    from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat

    reference = torch.cat((motion.goal(frames, GOAL_FRAMES), motion.root_R.reshape(-1, 9)), dim=1).numpy()
    if reference.shape != (motion.num_frames, 739) or not np.isfinite(reference).all():
        raise ValueError("Reference cache must have 739 finite values per training control frame")
    first = motion.ref_state(torch.tensor([0]))
    first_quaternion = mat_to_quat(first[1])[0]
    training_files = (
        "src/unisupermimic/rl/algos/supermimic/networks.py",
        "src/unisupermimic/tasks/motion_tracking/g1/supermimic/mdp.py",
        "src/unisupermimic/tasks/motion_tracking/g1/supermimic/motion.py",
        "src/unisupermimic/tasks/motion_tracking/g1/supermimic/robot.py",
        "src/unisupermimic/tasks/motion_tracking/g1/supermimic/manager_terms.py",
        "src/unisupermimic/assets/robots/g1/supermimic.xml",
        "src/unisupermimic/assets/robots/g1/scene_supermimic.xml",
    )
    manifest = {
        "schema_version": 1,
        "format": "action_go_supermimic_v1",
        "variant": "uni_supermimic",
        "policy": {
            "input": {"name": "obs", "shape": [1, PACKED_DIM], "dtype": "float32"},
            "output": {"name": "actions", "shape": [1, ACTION_DIM], "dtype": "float32"},
            "history": {"shape": [HISTORY_LENGTH, OBS_DIM], "order": "oldest_to_newest"},
            "goal": {"shape": [GOAL_FRAMES, GOAL_FRAME_DIM], "future_offsets": list(range(1, GOAL_FRAMES + 1))},
            "normalization": "checkpoint obs_norm and goal_norm embedded in ONNX; do not normalize externally",
        },
        "provenance": {
            "training_repo": str(training_repo),
            "training_repo_commit": git_output(training_repo, "rev-parse", "HEAD"),
            "training_repo_dirty": bool(git_output(training_repo, "status", "--porcelain")),
            "training_files_sha256": {name: sha256(training_repo / name) for name in training_files},
            "checkpoint": {"path": str(checkpoint_path), "name": checkpoint_path.name,
                           "sha256": sha256(checkpoint_path), "iteration": int(checkpoint["iteration"])},
            "training_config": {key: training_args.get(key) for key in
                                ("goal_frames", "hist", "z_dim", "policy_hidden", "amp", "seed")},
            "training_run_config": str(checkpoint_path.parent / "run_config.json"),
            "motion": {"path": str(motion_path), "name": motion_path.name,
                       "sha256": sha256(motion_path), "source_fps": args.source_fps},
            "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "numpy": np.__version__},
            "export_precision": "cpu_float32",
            "training_amp": training_args.get("amp"),
        },
        "robot_contract": {"joint_names": list(joint_names), "control_dt": 0.02,
                           "physics_dt": float(mujoco_model.opt.timestep),
                           "default_q": get_default_q(), "action_scale": [get_action_scale(name) for name in joint_names],
                           "xml_path": "src/unisupermimic/assets/robots/g1/scene_supermimic.xml",
                           "qpos_addresses": qpos_addresses},
        "reference": {"clip": motion.names[0], "num_frames": motion.num_frames, "fps": 50,
                      "last_frame_time_sec": (motion.num_frames - 1) * 0.02,
                      "end_behavior": "hold_last_reference_and_continue_inference"},
        "initial_state": {"root_position": first[0][0].tolist(),
                          "root_quaternion_wxyz": first_quaternion.tolist(),
                          "root_linear_velocity_world": first[2][0].tolist(),
                          "root_angular_velocity_world": first[3][0].tolist(),
                          "joint_position": first[4][0].tolist(), "joint_velocity": first[5][0].tolist()},
        "runtime": {"inference_hz": 50, "calibration": [0.0] * ACTION_DIM,
                    "yaw_alignment": "none", "end_behavior": "hold", "policy_step_source": "lowstate_tick",
                    "history_initialization": "repeat_first_observation_10_times; previous_action_zero",
                    "action_decode": "default_angles + calibration + action_scale * actions",
                    "automatic_fall_reset": False, "automatic_fall_pause": False},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".uni-supermimic-export-", dir=output_dir) as temp_dir:
        staging = Path(temp_dir)
        manifest["validation"] = export_and_verify(packed_policy, samples, staging / "policy.onnx")
        np.savetxt(staging / "reference.csv", reference, delimiter=",", fmt="%.9g")
        recovered = np.loadtxt(staging / "reference.csv", delimiter=",", dtype=np.float32, ndmin=2)
        np.testing.assert_array_equal(recovered, reference)
        (staging / "SuperMimic.yaml").write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
        names = ("policy.onnx", "reference.csv", "SuperMimic.yaml")
        manifest["artifacts"] = {name: {"sha256": sha256(staging / name),
                                        "bytes": (staging / name).stat().st_size} for name in names}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in (*names, "manifest.json"):
            (staging / name).replace(output_dir / name)
    print(json.dumps({"output_dir": str(output_dir), "checkpoint": manifest["provenance"]["checkpoint"],
                      "reference": manifest["reference"], "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
