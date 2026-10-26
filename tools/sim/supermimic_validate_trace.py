#!/usr/bin/env python3
"""Compare ActionGo DDS traces with the original SuperMimic MDP and FP32 actor."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_repo.resolve()))
    exporter_path = Path(__file__).resolve().parents[1] / "export/supermimic_export_policy_onnx.py"
    module_spec = importlib.util.spec_from_file_location("supermimic_export", exporter_path)
    exporter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(exporter)
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP, XI_DIM
    from supermimic.motion import MotionLib
    from supermimic.rotations import quat_to_mat

    torch.set_num_threads(2)
    _, contract, policy = exporter.load_policy(args.source_repo.resolve(), args.checkpoint.resolve())
    manifest = json.loads((args.package / "manifest.json").read_text())
    kinematics = G1Kinematics("cpu", xml_path=contract.xml_path)
    source_fps = float(manifest["provenance"]["motion"]["source_fps"])
    motion = MotionLib.from_path(str(args.motion.resolve()), kinematics, "cpu", src_fps=source_fps)
    clip_id = motion.names.index(manifest["reference"]["clip"])
    first_frame = int(motion.clip_start[clip_id])
    last_frame = int(motion.clip_end[clip_id])
    mdp = MDP(kinematics, motion, torch.tensor(contract.default_q), observation_mode="legacy99")
    trace = np.load(args.log_dir / "trace.npz")
    cache = np.loadtxt(args.package / "reference.csv", delimiter=",", dtype=np.float32, ndmin=2)
    maximum = dict(observation=0.0, action=0.0, target=0.0, dds_state=0.0, dds_target=0.0, goal=0.0)
    inference_times = []
    compared = 0

    def compare(label, actual, expected, atol=1e-5, rtol=1e-5):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)
        maximum[label] = max(maximum[label], float(np.max(np.abs(actual - expected))))

    for reset_index in np.unique(trace["reset_index"]):
        events = [json.loads(line) for line in (args.log_dir / f"player-{reset_index}.jsonl").read_text().splitlines()]
        events = {int(event["command_sequence"]): event for event in events
                  if event.get("event") == "supermimic_inference"}
        history = None
        previous = torch.zeros(1, 29)
        indices = np.flatnonzero(trace["reset_index"] == reset_index)
        for row in indices:
            sequence = int(trace["sequence"][row])
            event = events[sequence]
            frame_index = min(first_frame + sequence - 1, last_frame)
            if int(event["reference_frame"]) != frame_index - first_frame:
                raise AssertionError("C++ reference frame and DDS sequence differ")
            values = {key: np.fromstring(event[key + "_csv"], sep=",", dtype=np.float32)
                      for key in ("q", "dq", "quat", "gyro", "obs", "actions", "target")}
            for key, trace_key in (("q", "q"), ("dq", "qd"), ("quat", "root_quat"), ("gyro", "root_w_local")):
                compare("dds_state", values[key], trace[trace_key][row], atol=2e-6, rtol=1e-6)
            compare("dds_target", values["target"], trace["target_q"][row], atol=1e-7, rtol=1e-7)
            rotation = quat_to_mat(torch.from_numpy(values["quat"]).unsqueeze(0))
            gyro = torch.from_numpy(values["gyro"]).reshape(1, 3, 1)
            state = (torch.zeros(1, 3), rotation, torch.zeros(1, 3), (rotation @ gyro).squeeze(-1),
                     torch.from_numpy(values["q"]).unsqueeze(0), torch.from_numpy(values["dq"]).unsqueeze(0))
            frame = torch.tensor([frame_index])
            obs = mdp.obs(state, previous, torch.zeros(1, XI_DIM), frame)
            history = obs.unsqueeze(1).repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)
            goal = motion.goal(frame, 10)
            compare("goal", cache[frame_index - first_frame, :730], goal[0].numpy(), atol=2e-6)
            packed = torch.cat((history.flatten(1), goal), dim=1)
            compare("observation", values["obs"], packed[0].numpy(), atol=3e-5, rtol=2e-5)
            with torch.inference_mode():
                action = policy(history, goal)[0].numpy()
            compare("action", values["actions"], action, atol=5e-5, rtol=5e-5)
            expected_target = np.asarray(contract.default_q) + np.asarray(contract.action_scale) * values["actions"]
            compare("target", values["target"], expected_target, atol=2e-6)
            previous = torch.from_numpy(values["actions"]).unsqueeze(0)
            inference_times.append(float(event["inference_ms"]))
            compared += 1
    if compared == 0:
        raise AssertionError("No consumed inference commands to validate")
    report = dict(passed=True, samples=compared, max_abs_errors=maximum,
                  inference_ms=dict(median=float(np.median(inference_times)),
                                    p95=float(np.percentile(inference_times, 95)),
                                    maximum=float(np.max(inference_times))))
    (args.log_dir / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
