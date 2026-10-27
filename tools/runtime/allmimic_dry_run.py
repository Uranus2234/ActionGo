#!/usr/bin/env python3
"""Measure AllMimic receive-to-command latency without publishing LowCmd."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("state_age_ms", "preprocess_ms", "onnx_ms", "postprocess_ms",
          "publish_wait_ms", "command_ms", "pipeline_ms", "e2e_ms")


def distribution(values):
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"mean": statistics.fmean(ordered), "p50": percentile(0.50),
            "p95": percentile(0.95), "p99": percentile(0.99), "max": ordered[-1]}


def summarize(events, warmup_steps):
    samples = [event for event in events if event.get("event") == "dry_run_latency"]
    finished = [event for event in events if event.get("event") == "dry_run_latency_finished"]
    exits = [event for event in events if event.get("event") == "process_exit"]
    if not finished or not exits or not exits[-1].get("ok"):
        raise RuntimeError("Dry run did not finish successfully; inspect player.log and player.jsonl")
    if exits[-1].get("publish_count") != 0:
        raise RuntimeError("Dry run recorded unexpected active LowCmd publications")
    if not samples or any(sample.get("lowcmd_sent") is not False for sample in samples):
        raise RuntimeError("Missing dry-run latency evidence")
    sequences = [sample["sequence"] for sample in samples]
    if sequences != sorted(set(sequences)) or len(samples) != finished[-1]["latency_count"]:
        raise RuntimeError("Latency samples contain duplicate or inconsistent sequences")
    for sample in samples:
        if any(not math.isfinite(sample[field]) or sample[field] < 0 for field in FIELDS):
            raise RuntimeError("Invalid latency sample")
    measured = [sample for sample in samples if sample["sequence"] > warmup_steps]
    if not measured:
        raise RuntimeError("No samples remain after warmup; increase duration or reduce --warmup-steps")
    return {
        "definition": "DDS LowState callback entry to first prepared LowCmd including CRC; DDS Write is skipped",
        "excludes": ["sensor acquisition", "inbound network/DDS delivery before callback",
                     "outbound DDS Write/network delivery", "motor response"],
        "lowcmd_sent": False,
        "synthetic_state": any(sample.get("synthetic_state", False) for sample in samples),
        "warmup_steps": warmup_steps,
        "sample_count": len(measured),
        "inference_count": finished[-1]["inference_count"],
        "unmeasured_count": finished[-1]["unmeasured_count"],
        "first_sample": samples[0],
        "latency_ms": {field[:-3]: distribution([sample[field] for sample in measured])
                       for field in FIELDS},
        "e2e_over_20ms": sum(sample["e2e_ms"] > 20 for sample in measured),
        "pipeline_over_20ms": sum(sample["pipeline_ms"] > 20 for sample in measured),
        "runtime_metrics": [event for event in events if event.get("event") == "runtime_metrics"
                            and event.get("phase") == "final"],
    }


def run_model(args, name):
    directory = args.log_dir / name
    directory.mkdir(parents=True, exist_ok=False)
    suffix = "_no_omni" if name == "no-omni" else ""
    spec = ROOT / f"actions/allmimic/config/generic_allmimic_paoxiao{suffix}.yaml"
    model = ROOT / f"actions/allmimic/model/policy{suffix}.onnx"
    command = [str(args.player), "--spec", str(spec), "--net", args.net,
               "--domain-id", str(args.domain_id), "--dry-run", "--dry-run-latency",
               "--duration-sec", str(args.duration_sec), "--no-damping-on-stop",
               "--event-log", str(directory / "player.jsonl")]
    if args.onnx_threads is not None:
        command += ["--onnx-threads", str(args.onnx_threads)]
    print(f"[{name}] subscribing LowState on {args.net}, domain {args.domain_id}; no LowCmd", flush=True)
    with (directory / "player.log").open("w") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                timeout=args.duration_sec + 30)
    if result.returncode:
        raise RuntimeError(f"{name}: player exited {result.returncode}; inspect {directory / 'player.log'}")
    events = [json.loads(line) for line in (directory / "player.jsonl").read_text().splitlines()]
    summary = summarize(events, args.warmup_steps)
    started = next(event for event in events if event.get("event") == "dry_run_latency_started")
    summary.update(model=name, model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                   spec=str(spec), command=command, platform=platform.platform(),
                   machine=platform.machine(), cpu_count=os.cpu_count(),
                   onnx_threads=started["onnx_threads"], inference_hz=50,
                   publish_hz=started["publish_hz"], net=args.net, domain_id=args.domain_id)
    report = directory / "latency.json"
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{name}] {summary['sample_count']} samples after warmup; latency in ms")
    print(f"{'stage':<18} {'mean':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9}")
    for stage, metrics in summary["latency_ms"].items():
        print(f"{stage:<18}" + "".join(f" {metrics[key]:9.3f}" for key in ("mean", "p50", "p95", "p99", "max")))
    print(f"e2e >20ms: {summary['e2e_over_20ms']}; unmeasured: {summary['unmeasured_count']}; report: {report}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("omni", "no-omni", "both"), default="both")
    parser.add_argument("--player", type=Path, default=ROOT / "build/generic_motion_tracking_player")
    parser.add_argument("--net", required=True, help="Interface receiving the robot or simulation LowState")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--duration-sec", type=float, default=10)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--onnx-threads", type=int)
    parser.add_argument("--log-dir", type=Path,
                        default=ROOT / "outputs/allmimic_dry_run" / str(time.time_ns()))
    args = parser.parse_args()
    if not math.isfinite(args.duration_sec) or args.duration_sec <= 0:
        parser.error("--duration-sec must be finite and positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.duration_sec * 50:
        parser.error("--warmup-steps must leave at least one inference sample")
    if args.domain_id < 0 or args.domain_id > 232:
        parser.error("--domain-id must be in [0, 232]")
    if args.onnx_threads is not None and args.onnx_threads <= 0:
        parser.error("--onnx-threads must be positive")
    args.player = args.player.resolve()
    args.log_dir = args.log_dir.resolve()
    if not args.player.is_file():
        parser.error(f"Player not found: {args.player}; build generic_motion_tracking_player first")
    names = ("omni", "no-omni") if args.model == "both" else (args.model,)
    summaries = [run_model(args, name) for name in names]
    (args.log_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
