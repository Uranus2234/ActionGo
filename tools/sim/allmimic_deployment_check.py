#!/usr/bin/env python3
"""Exercise the AllMimic hardware spec with asynchronous DDS in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import subprocess
import time

import mujoco
import numpy as np
import yaml

import supermimic_mujoco as helpers


ROOT = Path(__file__).resolve().parents[2]


class DeploymentBridge(helpers.DDSBridge):
    def receive(self, message):
        motors = message.motor_cmd[:29]
        command = {key: np.asarray([getattr(motor, key) for motor in motors], dtype=np.float64)
                   for key in ("q", "dq", "tau", "kp", "kd")}
        with self.condition:
            if not all(np.isfinite(values).all() for values in command.values()):
                self.error = "Nonfinite DDS command"
            self.messages += 1
            self.hold_messages += 1
            self.latest = command

    def command(self):
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            return getattr(self, "latest", None)


def run(args):
    args.log_dir.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(args.package / "scene.mjb"))
    data = mujoco.MjData(model)
    with np.load(args.package / args.clip / "motion.npz") as archive:
        reference = archive["qpos"].copy()
        velocity = archive["qvel"].copy()
    data.qpos[:] = reference[0] if args.initial == "reference" else model.key_qpos[0]
    data.qvel[:] = velocity[0] if args.initial == "reference" else 0
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    data.qvel[3:6] = rotation.reshape(3, 3).T @ data.qvel[3:6]
    yaw_quat = np.array([np.cos(args.yaw / 2), 0, 0, np.sin(args.yaw / 2)])
    aligned = np.empty(4)
    mujoco.mju_mulQuat(aligned, yaw_quat, data.qpos[3:7])
    data.qpos[3:7] = aligned
    mujoco.mj_forward(model, data)
    indices = {"q": np.arange(7, 36), "v": np.arange(6, 35), "root_q": 0, "root_v": 0,
               "actuators": np.array([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]
                                       for joint in range(1, 30)])}
    initializer = helpers.load_sdk(args)
    domain = secrets.randbelow(140) + 50
    initializer(domain, "lo")
    bridge = DeploymentBridge(model, data, indices)
    actuator_ids = indices["actuators"]
    if args.pd == "explicit":
        model.actuator_gaintype[actuator_ids] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator_ids] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[actuator_ids, :] = 0
        model.actuator_gainprm[actuator_ids, 0] = 1
        model.actuator_biasprm[actuator_ids, :] = 0
        model.actuator_ctrllimited[actuator_ids] = False
    spec_path = args.log_dir / "spec.yaml"
    motion_name = "wbst_motive" if args.clip == "12_wbst_motive" else args.clip
    spec = yaml.safe_load((ROOT / f"actions/allmimic/config/generic_allmimic_{motion_name}.yaml").read_text())
    spec["robot"]["params"] = str(args.package / "AllMimic.yaml")
    spec["model"]["path"] = str(args.package / "policy.onnx")
    spec["reference"].update(path=str(args.package / args.clip / "reference.csv"), trace=True)
    if args.safety_profile:
        spec["safety"]["profile"] = args.safety_profile
    if args.preserve_pd_torque is not None:
        spec["safety"]["preserve_pd_torque"] = args.preserve_pd_torque
    if args.blend_sec is not None:
        spec["handoff"]["activation_blend_sec"] = args.blend_sec
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    ready, start, complete, release = [args.log_dir / name for name in ("ready", "start", "complete", "release")]
    command = [str(args.player), "--net", "lo", "--domain-id", str(domain), "--spec", str(spec_path),
               "--ready-file", str(ready), "--start-file", str(start),
               "--handoff-complete-file", str(complete), "--handoff-release-file", str(release),
               "--event-log", str(args.log_dir / "player.jsonl"), "--no-damping-on-stop"]
    records, applied, feedforward = [], [], []
    process = None
    completed_at = None
    with (args.log_dir / "player.log").open("w") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while not ready.exists() or bridge.command() is None:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Player failed before ready; inspect {log.name}")
                bridge.publish()
                time.sleep(0.002)
            start.touch()
            next_step = time.monotonic()
            while data.time < (args.seconds or len(reference) / 50 + 10):
                if process.poll() is not None:
                    if complete.exists() and release.exists() and process.returncode == 0:
                        break
                    raise RuntimeError(f"Player failed: {process.returncode}; inspect {log.name}")
                current = bridge.command()
                if args.pd == "implicit":
                    bridge.apply(current)
                else:
                    model.actuator_forcerange[actuator_ids] = bridge.effort
                    model.actuator_forcelimited[actuator_ids] = True
                    data.ctrl[actuator_ids] = (current["tau"] + current["kp"] * (current["q"] - data.qpos[7:])
                                               + current["kd"] * (current["dq"] - data.qvel[6:]))
                mujoco.mj_step(model, data)
                bridge.publish()
                records.append(data.qpos.copy())
                applied.append(current["q"].copy())
                feedforward.append(current["tau"].copy())
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Nonfinite physics state")
                if complete.exists():
                    if completed_at is None:
                        completed_at = data.time
                    if data.time - completed_at >= 0.30:
                        release.touch()
                next_step += model.opt.timestep
                time.sleep(max(0, next_step - time.monotonic()))
        finally:
            helpers.stop_player(process)
    events = [json.loads(line) for line in (args.log_dir / "player.jsonl").read_text().splitlines()]
    inference = [event for event in events if event.get("event") == "allmimic_inference"]
    actual = np.asarray(records)
    target = np.asarray(applied)
    params = yaml.safe_load((args.package / "AllMimic.yaml").read_text())["robot"]
    reference_indices = np.minimum((np.arange(len(actual)) + 1) // 4, len(reference) - 1)
    compare_count = min(len(actual), 4 * (len(reference) - 1))
    truth = reference[reference_indices[:compare_count]]
    policy_frames = np.asarray([event["reference_frame"] for event in inference], dtype=int)
    policy_positions = np.asarray([np.fromstring(event["q_csv"], sep=",") for event in inference])
    state_indices = np.clip(np.rint(np.asarray([event["source_tick_ms"] for event in inference]) /
                                   (1000 * model.opt.timestep)).astype(int) - 1, 0, len(actual) - 1)
    runtime_metrics = [event for event in events if event.get("event") == "runtime_metrics"]
    tilts = []
    for quat in actual[:, 3:7]:
        mujoco.mju_quat2Mat(rotation, quat)
        tilts.append(float(np.arccos(np.clip(rotation[8], -1, 1)) * 180 / np.pi))
    metrics = {"initial": args.initial, "pd": args.pd, "yaw": args.yaw,
               "safety_profile": spec["safety"].get("profile", "hardware"),
               "preserve_pd_torque": spec["safety"].get("preserve_pd_torque", False),
               "seconds": float(data.time), "min_height_m": float(actual[:, 2].min()),
               "first_second_min_height_m": float(actual[:200, 2].min()),
               "max_tilt_deg": max(tilts), "first_second_max_tilt_deg": max(tilts[:200]),
               "complete_seen": complete.exists(), "release_sent": release.exists(),
               "release_acknowledged": any(event.get("event") == "handoff_hold_stopped" and
                                           event.get("released") is True for event in events),
               "inference_steps": len(inference), "dds_messages": bridge.messages,
               "joint_rmse_rad": float(np.sqrt(np.mean((actual[:compare_count, 7:] - truth[:, 7:]) ** 2))),
               "height_rmse_m": float(np.sqrt(np.mean((actual[:compare_count, 2] - truth[:, 2]) ** 2))),
               "policy_phase_joint_rmse_rad": float(np.sqrt(np.mean((policy_positions - reference[policy_frames, 7:]) ** 2))),
               "policy_phase_height_rmse_m": float(np.sqrt(np.mean((actual[state_indices, 2] - reference[policy_frames, 2]) ** 2))),
               "policy_deadline_miss": max((event.get("policy_deadline_miss", 0) for event in runtime_metrics), default=0),
               "max_feedforward_nm": float(np.abs(feedforward).max()),
               "targets_within_joint_limits": bool(np.all(target >= np.asarray(params["joint_pos_lower_limit"]) - 1e-6)
                    and np.all(target <= np.asarray(params["joint_pos_upper_limit"]) + 1e-6)),
               "max_published_target_step_rad": float(np.abs(np.diff(target, axis=0)).max()),
               "events": [event for event in events if event.get("event") in
                          ("activation_blend_started", "safety_stop", "runtime_metrics", "handoff_hold_stopped")]}
    np.savez_compressed(args.log_dir / "trace.npz", actual=actual, targets=target, feedforward=np.asarray(feedforward))
    (args.log_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metrics.items() if key != "events"}, indent=2), flush=True)
    if any(event.get("event") == "safety_stop" for event in events):
        raise RuntimeError("Player reported a safety stop; inspect player.jsonl")
    if not args.seconds and not (metrics["complete_seen"] and metrics["release_sent"] and metrics["release_acknowledged"]):
        raise RuntimeError("Full motion did not complete the complete/release handshake")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--clip", choices=("paoxiao", "12_wbst_motive"), default="paoxiao")
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--initial", choices=("reference", "stand"), default="stand")
    parser.add_argument("--pd", choices=("implicit", "explicit"), default="implicit")
    parser.add_argument("--yaw", type=float, default=0.8)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--safety-profile", choices=("hardware", "simulation_parity"))
    parser.add_argument("--preserve-pd-torque", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--blend-sec", type=float)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--python-deps", type=Path, default=ROOT.parent / "ActionGo-local/python")
    args = parser.parse_args()
    args.package, args.log_dir, args.player = args.package.resolve(), args.log_dir.resolve(), args.player.resolve()
    if not np.isfinite(args.seconds) or args.seconds < 0 or not np.isfinite(args.yaw):
        parser.error("seconds must be finite and nonnegative; yaw must be finite")
    if args.blend_sec is not None and (not np.isfinite(args.blend_sec) or not 0 <= args.blend_sec <= 5):
        parser.error("blend-sec must be finite and within [0, 5]")
    run(args)


if __name__ == "__main__":
    main()
