#!/usr/bin/env python3
"""Package a local SuperMimic legacy99 policy and one motion for ActionGo."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import torch
import yaml


HISTORY_LENGTH = 10
OBS_DIM = 99
GOAL_FRAMES = 10
GOAL_FRAME_DIM = 73
GOAL_DIM = GOAL_FRAMES * GOAL_FRAME_DIM
HISTORY_DIM = HISTORY_LENGTH * OBS_DIM
PACKED_DIM = HISTORY_DIM + GOAL_DIM
ACTION_DIM = 29


class PackedPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        history = obs[:, :HISTORY_DIM].reshape(-1, HISTORY_LENGTH, OBS_DIM)
        return self.policy(history, obs[:, HISTORY_DIM:])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def select_motion(path, clip):
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files or any(not file.is_file() or file.suffix.lower() != ".csv" for file in files):
        raise ValueError("--motion must be an existing CSV or a directory containing CSV clips")
    if clip is None:
        if len(files) != 1:
            raise ValueError("Multiple clips found; use --clip with a clip stem or zero-based index")
        return files[0]
    matches = [file for file in files if clip in (file.stem, file.name)]
    if matches:
        return matches[0]
    if clip.isdecimal() and int(clip) < len(files):
        return files[int(clip)]
    raise ValueError(f"Unknown clip {clip!r}; available clips: {', '.join(file.stem for file in files)}")


def load_policy(source_repo, checkpoint_path):
    sys.path.insert(0, str(source_repo))
    import supermimic.networks as networks
    from supermimic.robot_contract import assert_contract_compatible, load_contract
    from supermimic.runtime import network_shapes

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    arguments = checkpoint["args"]
    expected = {
        "hist": HISTORY_LENGTH,
        "goal_frames": GOAL_FRAMES,
        "observation_mode": "legacy99",
        "profile": "upstream",
        "network_size": "paper",
        "robot": "g1_29dof",
    }
    for key, value in expected.items():
        if arguments.get(key) != value:
            raise ValueError(f"Unsupported checkpoint {key}: expected {value!r}, got {arguments.get(key)!r}")
    contract = load_contract(arguments["profile"], robot=arguments["robot"])
    assert_contract_compatible(checkpoint.get("contract"), contract)
    if len(contract.joint_names) != ACTION_DIM or abs(contract.control_dt - 0.02) > 1e-9:
        raise ValueError("Expected 29 SDK-order joints and a 50 Hz control contract")
    networks.AMP_DTYPE = None
    policy = networks.Policy(
        OBS_DIM, GOAL_DIM, hist=HISTORY_LENGTH, hidden=network_shapes(arguments["network_size"])[1]
    ).cpu().float().eval()
    policy.load_state_dict(checkpoint["pi"], strict=True)
    for key, value in policy.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite policy tensor: {key}")
    for name in ("obs_norm", "goal_norm"):
        if (getattr(policy, name).var < 0).any():
            raise ValueError(f"Negative normalization variance: {name}")
    return checkpoint, contract, policy


def robot_parameters(contract, kinematics):
    return {
        "schema_version": 1,
        "robot": {
            "motor_count": ACTION_DIM,
            "policy_joint_order": list(contract.joint_names),
            "joint_map": list(range(ACTION_DIM)),
            "default_angles": list(contract.default_q),
            "kp": list(contract.kp),
            "kd": list(contract.kd),
            "action_scale": list(contract.action_scale),
            "torque_limits": list(contract.effort),
            "joint_pos_lower_limit": kinematics.q_lo.tolist(),
            "joint_pos_upper_limit": kinematics.q_hi.tolist(),
        },
        "policy": {"num_obs": PACKED_DIM, "num_actions": ACTION_DIM},
        "observation": {},
        "action": {},
        "safety": {},
    }


def validation_inputs(policy, contract, kinematics, motion):
    from supermimic.mdp import MDP, XI_DIM

    mdp = MDP(kinematics, motion, torch.tensor(contract.default_q), observation_mode="legacy99")
    samples = []
    for frame in sorted({0, motion.num_frames // 2, motion.num_frames - 1}):
        indices = torch.tensor([frame], dtype=torch.long)
        observation = mdp.obs(
            motion.ref_state(indices), torch.zeros(1, ACTION_DIM), torch.zeros(1, XI_DIM), indices
        )
        history = observation.unsqueeze(1).repeat(1, HISTORY_LENGTH, 1)
        samples.append(torch.cat((history.flatten(1), motion.goal(indices, GOAL_FRAMES)), dim=1))
    generator = torch.Generator().manual_seed(0)
    history = policy.obs_norm.mean + policy.obs_norm.std() * torch.randn(
        1, HISTORY_LENGTH, OBS_DIM, generator=generator
    )
    goal = policy.goal_norm.mean + policy.goal_norm.std() * torch.randn(1, GOAL_DIM, generator=generator)
    samples.append(torch.cat((history.flatten(1), goal), dim=1))
    return samples


def export_and_verify(packed_policy, samples, output_path):
    torch.onnx.export(
        packed_policy,
        samples[0],
        str(output_path),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model, full_check=True)
    for values, expected_name, expected_shape in (
        (model.graph.input, "obs", [1, PACKED_DIM]),
        (model.graph.output, "actions", [1, ACTION_DIM]),
    ):
        if len(values) != 1:
            raise ValueError("Expected exactly one ONNX input and one output")
        value = values[0]
        shape = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        if value.name != expected_name or shape != expected_shape:
            raise ValueError(f"Unexpected ONNX tensor: {value.name}, {shape}")
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"ONNX tensor {value.name} must be FP32")
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        runner = ort.InferenceSession(str(output_path), options, providers=["CPUExecutionProvider"])
        backend = "onnxruntime_cpu"
    else:
        from onnx.reference import ReferenceEvaluator

        runner = ReferenceEvaluator(model)
        backend = "onnx_reference_evaluator"
    maximum_error = 0.0
    for sample in samples:
        expected = packed_policy(sample).numpy()
        actual = runner.run(["actions"], {"obs": sample.numpy()})[0]
        if actual.shape != (1, ACTION_DIM) or not np.isfinite(actual).all():
            raise ValueError("Invalid ONNX policy output")
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
    return {
        "onnx_checker": "passed",
        "backend": backend,
        "sample_count": len(samples),
        "samples": "first/middle/final reference states and seeded normalized random history/goal",
        "max_abs_error": maximum_error,
        "atol": 1e-5,
        "rtol": 1e-4,
    }


def manifest_schema():
    return {
        "policy": {
            "input": {"name": "obs", "shape": [1, PACKED_DIM], "dtype": "float32"},
            "output": {"name": "actions", "shape": [1, ACTION_DIM], "dtype": "float32"},
            "history": {"columns": [0, HISTORY_DIM], "shape": [HISTORY_LENGTH, OBS_DIM], "order": "oldest_to_newest"},
            "goal": {"columns": [HISTORY_DIM, PACKED_DIM], "shape": [GOAL_FRAMES, GOAL_FRAME_DIM]},
            "normalization": "checkpoint obs_norm and goal_norm embedded in ONNX; do not normalize externally",
        },
        "observation_frame": {
            "joint_position_minus_default_minus_calibration": [0, 29],
            "joint_velocity_unscaled": [29, 58],
            "pelvis_local_angular_velocity": [58, 61],
            "pelvis_projected_gravity": [61, 64],
            "previous_raw_action": [64, 93],
            "pelvis_relative_reference_rotation_6d": [93, 99],
            "rotation_6d_order": ["R00", "R01", "R10", "R11", "R20", "R21"],
        },
        "goal_frame": {
            "reference_position_delta_in_current_reference_heading": [0, 3],
            "reference_rotation_in_current_reference_heading_6d": [3, 9],
            "reference_linear_velocity_in_future_reference_body": [9, 12],
            "reference_angular_velocity_in_future_reference_body": [12, 15],
            "reference_joint_position": [15, 44],
            "reference_joint_velocity_times_0_1": [44, 73],
            "future_frame_offsets": list(range(1, GOAL_FRAMES + 1)),
        },
        "reference_csv": {
            "type": "supermimic_csv_v1",
            "header": False,
            "delimiter": ",",
            "columns": GOAL_DIM + 9,
            "goal_columns": [0, GOAL_DIM],
            "current_reference_root_rotation_columns": [GOAL_DIM, GOAL_DIM + 9],
            "rotation_order": "row_major",
            "rotation_frame": "reference pelvis body to world",
            "index_ranges": "zero_based_half_open",
        },
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True, help="Local 36-column CSV or directory of CSV clips")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip", help="Clip stem, filename, or zero-based index in sorted CSV filenames")
    parser.add_argument("--source-fps", type=float, default=30.0, help="Input CSV frame rate (default: 30 Hz)")
    arguments = parser.parse_args()
    source_repo = arguments.source_repo.expanduser().resolve()
    if not (source_repo / "supermimic/networks.py").is_file():
        parser.error("--source-repo must point to the local SuperMimic-MuJoCo source checkout")
    checkpoint_path = arguments.checkpoint.expanduser().resolve()
    motion_path = select_motion(arguments.motion.expanduser().resolve(), arguments.clip)
    output_dir = arguments.output_dir.expanduser().resolve()
    torch.set_num_threads(2)
    checkpoint, contract, policy = load_policy(source_repo, checkpoint_path)
    from supermimic.kinematics import G1Kinematics
    from supermimic.motion import MotionLib
    from supermimic.rotations import mat_to_quat

    kinematics = G1Kinematics("cpu", xml_path=contract.xml_path)
    motion = MotionLib([str(motion_path)], kinematics, "cpu", src_fps=arguments.source_fps)
    frames = torch.arange(motion.num_frames)
    reference = torch.cat((motion.goal(frames, GOAL_FRAMES), motion.root_R.flatten(1)), dim=1).numpy()
    if reference.shape != (motion.num_frames, GOAL_DIM + 9) or not np.isfinite(reference).all():
        raise ValueError("Reference cache must have 739 finite columns per frame")
    packed_policy = PackedPolicy(policy).cpu().float().eval()
    samples = validation_inputs(policy, contract, kinematics, motion)
    source_files = (
        "supermimic/networks.py", "supermimic/mdp.py", "supermimic/motion.py",
        "supermimic/kinematics.py", "supermimic/rotations.py", "supermimic/robot_contract.py",
        "supermimic/robot.py", "supermimic/runtime.py",
    )
    contract_metadata = contract.metadata()
    contract_metadata["xml_path"] = Path(contract.xml_path).relative_to(source_repo).as_posix()
    manifest = {
        "schema_version": 1,
        "format": "action_go_supermimic_v1",
        "schema": manifest_schema(),
        "provenance": {
            "source_repo_commit": git_output(source_repo, "rev-parse", "HEAD"),
            "source_repo_dirty": bool(git_output(source_repo, "status", "--porcelain")),
            "source_files_sha256": {name: sha256(source_repo / name) for name in source_files},
            "exporter_sha256": sha256(__file__),
            "checkpoint": {"name": checkpoint_path.name, "sha256": sha256(checkpoint_path), "iteration": checkpoint.get("iteration")},
            "original_training_checkpoint_sha256": checkpoint.get("source_checkpoint_sha256"),
            "motion": {"name": motion_path.name, "sha256": sha256(motion_path), "source_fps": arguments.source_fps},
            "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "numpy": np.__version__},
            "export_precision": "cpu_float32",
            "training_amp": checkpoint["args"].get("amp"),
        },
        "robot_contract": contract_metadata,
        "reference": {
            "clip": motion.names[0], "num_frames": motion.num_frames, "fps": 50,
            "last_frame_time_sec": (motion.num_frames - 1) * contract.control_dt,
            "end_behavior": "hold_last_reference_and_continue_inference",
        },
        "initial_state": {
            "root_position": motion.root_p[0].tolist(),
            "root_quaternion_wxyz": mat_to_quat(motion.root_R[:1])[0].tolist(),
            "root_linear_velocity_world": motion.root_v[0].tolist(),
            "root_angular_velocity_world": motion.root_w[0].tolist(),
            "joint_position": motion.q[0].tolist(),
            "joint_velocity": motion.qd[0].tolist(),
        },
        "runtime": {
            "inference_hz": 50, "calibration": [0.0] * ACTION_DIM,
            "yaw_alignment": "initial", "end_behavior": "hold",
            "policy_step_source": "wall_clock",
            "history_initialization": "repeat_first_observation_10_times; previous_action_zero",
            "action_decode": "default_angles + calibration + action_scale * actions",
            "action_clip": contract.action_clip,
            "automatic_fall_reset": False, "automatic_fall_pause": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".supermimic-export-", dir=output_dir) as temporary:
        staging = Path(temporary)
        manifest["validation"] = export_and_verify(packed_policy, samples, staging / "policy.onnx")
        np.savetxt(staging / "reference.csv", reference, delimiter=",", fmt="%.9g")
        recovered = np.loadtxt(staging / "reference.csv", delimiter=",", dtype=np.float32, ndmin=2)
        np.testing.assert_array_equal(recovered, reference)
        manifest["validation"]["reference_csv_float32_roundtrip"] = "passed"
        (staging / "SuperMimic.yaml").write_text(
            yaml.safe_dump(robot_parameters(contract, kinematics), sort_keys=False), encoding="utf-8"
        )
        artifact_names = ("policy.onnx", "reference.csv", "SuperMimic.yaml")
        manifest["artifacts"] = {
            name: {"sha256": sha256(staging / name), "bytes": (staging / name).stat().st_size}
            for name in artifact_names
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in (*artifact_names, "manifest.json"):
            (staging / name).replace(output_dir / name)
    print(json.dumps({"output_dir": str(output_dir), "reference": manifest["reference"], "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
