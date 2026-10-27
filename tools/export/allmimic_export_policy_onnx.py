#!/usr/bin/env python3
"""Export a deployable AllMimic actor and reference clips for ActionGo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import mujoco
import numpy as np
import onnx
import torch
import yaml


OBS_DIM = 93
HISTORY = 10
GOAL_DIM = 730
PACKED_DIM = 1660


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training(training_repo, checkpoint):
    sys.path.insert(0, str(training_repo))
    from allmimic import networks
    from allmimic.task import task_cfg

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = task_cfg(saved["cfg"], train=False)
    actual = (config.goal_pos, config.goal_rot, config.heading, config.rot_repr,
              config.action, config.res_legs, config.history, config.offsets)
    expected = ("ref", "robot", "xaxis", "6d", "res", "abs", 10, tuple(range(1, 11)))
    if actual != expected:
        raise ValueError(f"Unsupported AllMimic deployment contract: {actual}")
    if int(saved["cfg"].get("latent", 32)) != 128:
        raise ValueError("This AllMimic adapter requires latent=128")
    networks.AMP["enabled"] = False
    policy = networks.Policy(OBS_DIM * HISTORY, GOAL_DIM, latent=128).cpu().float().eval()
    policy.load_state_dict(saved["policy"], strict=True)
    if not all(torch.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError("Nonfinite AllMimic policy weights")
    return saved, config, policy


def training_robot(config):
    from allmimic.kinematics import TreeKinematics
    from allmimic.task import Task
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg
    from mjlab.terrains import TerrainEntityCfg

    scene = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1,
                           entities={"robot": get_g1_robot_cfg()}), device="cpu")
    model = scene.compile()
    MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20).apply(model)
    for geom in range(model.ngeom):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").split("/")[-1]
        if re.fullmatch(r"(left|right)_foot[1-7]_collision", name):
            model.geom_friction[geom, 0] = 1.0
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint).split("/")[-1]
              for joint in range(1, model.njnt)]
    if len(joints) != 29 or model.nq != 36 or model.nv != 35:
        raise ValueError("AllMimic requires a floating-base G1 with 29 joints")
    scales = [next(value for pattern, value in G1_ACTION_SCALE.items() if re.fullmatch(pattern, name))
              for name in joints]
    defaults = torch.tensor(model.key_qpos[0, 7:].copy(), dtype=torch.float32)
    limits = model.jnt_range[1:]
    midpoint, half = limits.mean(axis=1), np.diff(limits, axis=1)[:, 0] * 0.45
    task = Task(TreeKinematics(model, "cpu", root_body=int(model.jnt_bodyid[0])), defaults,
                torch.tensor(np.stack((midpoint - half, midpoint + half), axis=1)).float(),
                torch.tensor(scales), config, "cpu")
    task.set_joint_names(joints)
    actuators = [int(np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0])
                 for joint in range(1, model.njnt)]
    params = {
        "schema_version": 1,
        "robot": {"motor_count": 29, "policy_joint_order": joints, "joint_map": list(range(29)),
                  "default_angles": defaults.tolist(), "action_scale": scales,
                  "kp": model.actuator_gainprm[actuators, 0].tolist(),
                  "kd": (-model.actuator_biasprm[actuators, 2]).tolist(),
                  "torque_limits": model.actuator_forcerange[actuators, 1].tolist(),
                  "joint_pos_lower_limit": limits[:, 0].tolist(),
                  "joint_pos_upper_limit": limits[:, 1].tolist()},
        "policy": {"num_obs": PACKED_DIM, "num_actions": 29},
        "action": {"residual_scale": task.res_w.tolist(), "residual_reference": task.res_ref.tolist()},
        "observation": {}, "safety": {},
    }
    return model, task, params


class PackedPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        return self.policy(obs[:, :930], obs[:, 930:])


def export_policy(policy, samples, destination):
    import onnxruntime as ort

    packed = PackedPolicy(policy).eval()
    torch.onnx.export(packed, samples[0], str(destination), input_names=["obs"],
                      output_names=["actions"], opset_version=17, dynamo=False)
    exported = onnx.load(str(destination))
    onnx.checker.check_model(exported, full_check=True)
    for tensors, name, expected in ((exported.graph.input, "obs", [1, PACKED_DIM]),
                                     (exported.graph.output, "actions", [1, 29])):
        if len(tensors) != 1 or tensors[0].name != name or tensors[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("Unexpected AllMimic ONNX signature")
        if [dimension.dim_value for dimension in tensors[0].type.tensor_type.shape.dim] != expected:
            raise ValueError("Unexpected AllMimic ONNX dimensions")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    runner = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
    errors = []
    for sample in samples:
        expected = packed(sample).detach().numpy()
        actual = runner.run(["actions"], {"obs": sample.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
        errors.append(float(np.max(np.abs(actual - expected))))
    return {"onnx_checker": "passed", "backend": "onnxruntime_cpu", "sample_count": len(samples),
            "max_abs_error": max(errors), "atol": 2e-5, "rtol": 1e-4}


@torch.inference_mode()
def export(args):
    from allmimic.motion_lib import MotionLibrary

    saved, config, policy = load_training(args.training_repo, args.checkpoint)
    model, task, params = training_robot(config)
    library = MotionLibrary(str(args.library), "cpu")
    if not set(args.clip).issubset(library.names) or library.fps != 50:
        raise ValueError("All requested clips must exist in the 50 Hz training library")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "AllMimic.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    mujoco.mj_saveModel(model, str(args.output_dir / "scene.mjb"))
    samples = []
    manifests = []
    for clip_index, clip_name in enumerate(library.names):
        if clip_name not in args.clip:
            continue
        clip_dir = args.output_dir / Path(clip_name).stem
        clip_dir.mkdir(exist_ok=True)
        start, length = int(library.starts[clip_index]), int(library.lengths[clip_index])
        frames = torch.arange(start, start + length)
        future = (frames[:, None] + torch.tensor(config.offsets)).clamp(max=start + length - 1)
        state, reference = library.frame(frames), library.frame(future)
        identity = torch.eye(3).expand(length, 3, 3)
        cache = task.goal({"rot": identity}, reference)
        rows = torch.cat((cache, state["rot"].reshape(-1, 9)), dim=1).numpy()
        np.savetxt(clip_dir / "reference.csv", rows, delimiter=",", fmt="%.9g")
        np.testing.assert_array_equal(np.loadtxt(clip_dir / "reference.csv", delimiter=",", dtype=np.float32), rows)
        for frame in sorted({0, length // 2, length - 1}):
            current = {key: value[frame:frame + 1] for key, value in state.items()}
            observation = task.obs_clean(current, torch.zeros(1, 29), torch.zeros(1, 29))
            goal = task.goal(current, {key: value[frame:frame + 1] for key, value in reference.items()})
            samples.append(torch.cat((observation.repeat(1, HISTORY), goal), dim=1))
        np.savez_compressed(clip_dir / "motion.npz", qpos=np.column_stack((
            state["pos"].numpy(), state["quat"].numpy(), state["q"].numpy())),
            qvel=np.column_stack((state["v"].numpy(), state["w"].numpy(), state["qd"].numpy())))
        manifest = {
            "schema_version": 1, "variant": "allmimic", "format": "allmimic_csv_v1",
            "checkpoint": {"name": args.checkpoint.name, "iteration": int(saved["it"]),
                           "sha256": sha256(args.checkpoint)},
            "training": {"repo_commit": subprocess.check_output(
                ["git", "-C", str(args.training_repo), "rev-parse", "HEAD"], text=True).strip(),
                "config": saved["cfg"], "source_sha256": {name: sha256(args.training_repo / name)
                    for name in ("allmimic/task.py", "allmimic/networks.py", "allmimic/sim_env.py",
                                 "allmimic/motion_lib.py", "allmimic/rotations.py")}},
            "reference": {"clip": clip_name, "num_frames": length, "fps": 50,
                          "last_frame_time_sec": (length - 1) / 50,
                          "library_sha256": sha256(args.library),
                          "csv_sha256": sha256(clip_dir / "reference.csv")},
            "policy": {"input": [1, PACKED_DIM], "output": [1, 29], "history": [10, 93],
                       "future": [10, 73], "normalization": "embedded in ONNX"},
            "robot": params["robot"], "action": params["action"],
        }
        manifests.append((clip_dir, manifest))
    generator = torch.Generator().manual_seed(0)
    samples.append(torch.cat((policy.obs_norm.mean + policy.obs_norm.std * torch.randn(1, 930, generator=generator),
                              policy.goal_norm.mean + policy.goal_norm.std * torch.randn(1, 730, generator=generator)), dim=1))
    validation = export_policy(policy, samples, args.output_dir / "policy.onnx")
    for clip_dir, manifest in manifests:
        manifest["validation"] = validation
        manifest["policy"]["sha256"] = sha256(args.output_dir / "policy.onnx")
        (clip_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"iteration": int(saved["it"]), "clips": args.clip,
                      "output_dir": str(args.output_dir), "validation": validation}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--clip", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.training_repo = args.training_repo.resolve()
    sys.path.insert(0, str(args.training_repo))
    torch.set_num_threads(2)
    export(args)


if __name__ == "__main__":
    main()
