import argparse
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def run_case(player, case):
    with tempfile.TemporaryDirectory(prefix="pbfm-realtime-") as temporary:
        directory = Path(temporary)
        original = ROOT / "actions/pbfm/config/generic_pbfm_pico.yaml"
        spec = yaml.safe_load(original.read_text())
        for section, key in [("model", "path"), ("robot", "params")]:
            spec[section][key] = str((original.parent / spec[section][key]).resolve())
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        spec["context"].update(udp_bind="127.0.0.1", udp_port=port, startup_timeout_sec=0.7)
        np.save(directory / "depth.npy", np.ones((8, 18, 32), dtype=np.float32))
        spec["sensor"].update(depth_source="file", path=str(directory / "depth.npy"))
        spec_path = directory / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(spec))
        ready, start, trace = [directory / name for name in ("ready", "start", "trace.jsonl")]
        command = [str(player), "--spec", str(spec_path), "--dry-run", "--offline-dry-run-state",
                   "--duration-sec", "0.5", "--ready-file", str(ready), "--start-file", str(start),
                   "--trace-file", str(trace)]
        with (directory / "output").open("w+") as output, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            sequence = 0
            ready_time = None
            frozen = None
            origin = time.monotonic()
            try:
                while process.poll() is None and time.monotonic() - origin < 8:
                    now = time.monotonic()
                    if ready.exists() and ready_time is None:
                        ready_time = now
                    active = case in {"normal", "active_stale", "frozen_sequence", "frozen_timestamp", "turning"}
                    if ready_time is not None and active:
                        start.touch()
                    send = case != "missing"
                    if ready_time is not None and case in {"standby_stale", "active_stale"}:
                        send = now - ready_time < 0.1
                    if send:
                        heading = np.pi / 2
                        velocity = [0, 0.4, 1]
                        if ready_time is not None and case == "turning" and now - ready_time > 0.1:
                            heading += 0.25
                        context = {"contract": "bfm_realtime_context_v1", "sequence": sequence,
                                   "capture_unix_sec": time.time(), "z": [16.0] + [0.0] * 255,
                                   "heading_relative_xy": [float(np.cos(heading)), float(np.sin(heading))],
                                   "heading_valid": True, "velocity_command": velocity}
                        if case == "bad_latent":
                            context["z"] = [1.0] * 255
                        if case == "old_source":
                            context["capture_unix_sec"] -= 5
                        if ready_time is not None and case in {"frozen_sequence", "frozen_timestamp"}:
                            if frozen is None:
                                frozen = context.copy()
                            if case == "frozen_sequence":
                                context["sequence"] = frozen["sequence"]
                            else:
                                context["capture_unix_sec"] = frozen["capture_unix_sec"]
                        sender.sendto(json.dumps(context).encode(), ("127.0.0.1", port))
                        sequence += 1
                    time.sleep(0.01)
                if process.poll() is None:
                    raise RuntimeError(f"{case}: runtime timed out")
                output.seek(0)
                logs = output.read()
                frames = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
                if case in {"normal", "turning"}:
                    if process.returncode != 0 or len(frames) < 10:
                        raise RuntimeError(f"{case}: no realtime inference\n{logs}")
                    np.testing.assert_allclose(frames[0]["heading"], [0, 0], atol=1e-6)
                    for frame in frames:
                        assert frame["motion_state"] == "pico_realtime"
                        np.testing.assert_allclose(frame["velocity_command"], [0.4, 0, 1], atol=1e-6)
                        assert np.isfinite(frame["target_q_before_safety"]).all()
                    if case == "turning":
                        np.testing.assert_allclose(frames[-1]["heading"], [1 - np.cos(0.25), np.sin(0.25)], atol=1e-6)
                else:
                    if process.returncode == 0:
                        raise RuntimeError(f"{case}: failure was not detected\n{logs}")
                    if case in {"missing", "bad_latent", "old_source"}:
                        assert ready_time is None and not frames, logs
                    elif case == "standby_stale":
                        assert ready_time is not None and not frames and not ready.exists(), logs
                    else:
                        assert frames and "stale" in logs, logs
                print(f"P-BFM realtime lifecycle passed: {case}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    args = parser.parse_args()
    for name in ("normal", "turning", "missing", "old_source", "bad_latent", "standby_stale",
                 "active_stale", "frozen_sequence", "frozen_timestamp"):
        run_case(args.player.resolve(), name)
