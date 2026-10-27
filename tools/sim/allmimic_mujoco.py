#!/usr/bin/env python3
"""Validate the ActionGo AllMimic actor in native MuJoCo through loopback DDS."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import mujoco
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_trace(args, records, commands):
    exporter = load_file("allmimic_export", ROOT / "tools/export/allmimic_export_policy_onnx.py")
    manifest = json.loads((args.package / args.clip / "manifest.json").read_text())
    if exporter.sha256(args.checkpoint) != manifest["checkpoint"]["sha256"]:
        raise ValueError("Validation checkpoint differs from the exported policy source")
    if exporter.sha256(args.package / "policy.onnx") != manifest["policy"]["sha256"]:
        raise ValueError("Exported policy hash differs from the manifest")
    saved, config, policy = exporter.load_training(args.training_repo, args.checkpoint)
    _, task, _ = exporter.training_robot(config)
    from allmimic.rotations import quat_to_mat

    with np.load(args.package / args.clip / "motion.npz") as archive:
        reference_qpos, reference_qvel = archive["qpos"], archive["qvel"]
    ref = {"pos": torch.tensor(reference_qpos[:, :3], dtype=torch.float32),
           "rot": quat_to_mat(torch.tensor(reference_qpos[:, 3:7], dtype=torch.float32)),
           "q": torch.tensor(reference_qpos[:, 7:], dtype=torch.float32),
           "v": torch.tensor(reference_qvel[:, :3], dtype=torch.float32),
           "w": torch.tensor(reference_qvel[:, 3:6], dtype=torch.float32),
           "qd": torch.tensor(reference_qvel[:, 6:], dtype=torch.float32)}
    events = [json.loads(line) for line in (args.log_dir / "player.jsonl").read_text().splitlines()]
    events = [event for event in events if event.get("event") == "allmimic_inference"][:len(records)]
    if len(events) != len(records):
        raise ValueError("C++ inference trace does not cover all consumed DDS commands")
    history = None
    previous = torch.zeros(1, 29)
    errors = {"obs": 0.0, "actions": 0.0, "target": 0.0, "dds_target": 0.0}
    for index, event in enumerate(events):
        def values(key):
            return torch.tensor(np.fromstring(event[key], sep=","), dtype=torch.float32)[None]
        rotation = quat_to_mat(values("quat_csv"))
        state = {"q": values("q_csv"), "qd": values("dq_csv"), "rot": rotation,
                 "w": (rotation @ values("gyro_csv").unsqueeze(-1)).squeeze(-1)}
        obs = task.obs_clean(state, previous, torch.zeros(1, 29))
        history = obs.repeat(1, 10) if history is None else torch.cat((history[:, 93:], obs), dim=1)
        frame = min(index, len(reference_qpos) - 1)
        indices = (frame + torch.tensor(config.offsets)).clamp(max=len(reference_qpos) - 1)
        future = {key: value[indices][None] for key, value in ref.items()}
        goal = task.goal(state, future)
        packed = torch.cat((history, goal), dim=1)
        raw_action = values("actions_csv")
        expected_action = policy(history, goal)
        expected_target = task.q0 + task.action_scale * task.setpoint_action(raw_action, future["q"][:, 0])
        for key, expected, actual, tolerance in (
            ("obs", packed, values("obs_csv"), 5e-5),
            ("actions", expected_action, raw_action, 1e-4),
            ("target", expected_target, values("target_csv"), 5e-5),
            ("dds_target", values("target_csv"), torch.tensor(commands[index]["q"], dtype=torch.float32)[None], 1e-6),
        ):
            error = float((expected - actual).abs().max())
            errors[key] = max(errors[key], error)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=1e-4)
        previous = raw_action
    return {"passed": True, "samples": len(events), "iteration": int(saved["it"]), "max_abs_error": errors}


@torch.inference_mode()
def run(args):
    helpers = load_file("supermimic_dds", ROOT / "tools/sim/supermimic_mujoco.py")
    args.package = args.package.resolve()
    args.log_dir.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(args.package / "scene.mjb"))
    data = mujoco.MjData(model)
    with np.load(args.package / args.clip / "motion.npz") as archive:
        reference, velocity = archive["qpos"], archive["qvel"]
    data.qpos[:] = reference[0]
    data.qvel[:] = velocity[0]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    data.qvel[3:6] = rotation.reshape(3, 3).T @ velocity[0, 3:6]
    mujoco.mj_forward(model, data)
    indices = {"q": np.arange(7, 36), "v": np.arange(6, 35), "root_q": 0, "root_v": 0,
               "actuators": np.array([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]
                                      for joint in range(1, 30)])}
    initializer = helpers.load_sdk(argparse.Namespace(python_deps=args.python_deps, sdk_root=args.sdk_root))
    domain = secrets.randbelow(140) + 50
    initializer(domain, "lo")
    bridge = helpers.DDSBridge(model, data, indices)
    spec = yaml.safe_load((ROOT / "actions/allmimic/config/generic_allmimic_paoxiao.yaml").read_text())
    spec["robot"]["params"] = str(args.package / "AllMimic.yaml")
    spec["model"]["path"] = str(args.package / "policy.onnx")
    spec["reference"].update(path=str(args.package / args.clip / "reference.csv"),
                             policy_step_source="lowstate_tick", sim_lockstep_sequence=True,
                             yaw_alignment="none", end_behavior="hold", trace=True)
    spec["safety"] = {"profile": "simulation_parity", "lowstate_timeout_sec": 5.0}
    spec_path = args.log_dir / "simulation_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    ready, start = args.log_dir / "ready", args.log_dir / "start"
    command = [str(args.player), "--net", "lo", "--domain-id", str(domain), "--spec", str(spec_path),
               "--ready-file", str(ready), "--start-file", str(start), "--activation-blend-sec", "0",
               "--event-log", str(args.log_dir / "player.jsonl"), "--no-damping-on-stop"]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(filter(None, [args.library_path, environment.get("LD_LIBRARY_PATH")]))
    records, commands = [], []
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
            steps = args.steps or len(reference) - 1
            for index in range(steps):
                deadline = time.monotonic() + 10
                current = None
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
                commands.append(current)
                bridge.apply(current)
                for substep in range(4):
                    mujoco.mj_step(model, data)
                    bridge.publish()
                    time.sleep(0.0005)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Nonfinite MuJoCo state")
            records.append(data.qpos.copy())
        finally:
            helpers.stop_player(process)
    actual = np.asarray(records)
    reference_indices = np.arange(len(actual)).clip(max=len(reference) - 1)
    truth = reference[reference_indices]
    np.savez_compressed(args.log_dir / "trace.npz", actual=actual, reference=truth)
    metrics = {"steps": len(commands), "seconds": len(commands) / 50,
               "joint_rmse_rad": float(np.sqrt(np.mean((actual[1:, 7:] - truth[1:, 7:]) ** 2))),
               "root_position_mean_error_m": float(np.linalg.norm(actual[1:, :3] - truth[1:, :3], axis=1).mean()),
               "finite": True, "automatic_resets": 0, "automatic_pauses": 0}
    (args.log_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    parity = validate_trace(args, records[:-1], commands)
    (args.log_dir / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "parity": parity}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--python-deps", type=Path, default=ROOT.parent / "ActionGo-local/python")
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--library-path", default="")
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    args.log_dir = args.log_dir.resolve()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
