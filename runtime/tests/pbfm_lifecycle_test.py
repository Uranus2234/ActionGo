import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/sim"))
from pbfm_depth import DepthPublisher


def run_case(player, name):
    successful = {"normal", "sequence", "remote", "idle", "directional", "joystick", "handoff"}
    with tempfile.TemporaryDirectory(prefix="pbfm-lifecycle-") as raw:
        folder = Path(raw)
        ready = folder / "ready"
        start = folder / "start"
        complete = folder / "complete"
        release = folder / "release"
        trace = folder / "trace"
        shm_name = "pbfm_test_" + folder.name.replace("-", "_")
        publisher = None if name == "missing" else DepthPublisher(shm_name)
        command = [str(player), "--spec", str(ROOT / "actions/pbfm/config/generic_pbfm.yaml"),
                   "--dry-run", "--offline-dry-run-state", "--duration-sec", "1.2",
                   "--ready-file", str(ready), "--start-file", str(start),
                   "--depth-shm-name", shm_name, "--trace-file", str(trace)]
        sequence_frames = []
        if name == "joystick":
            command[2] = str(ROOT / "actions/pbfm/config/generic_pbfm_joystick.yaml")
        if name == "handoff":
            command += ["--handoff-complete-file", str(complete), "--handoff-release-file", str(release)]
        if name in {"sequence", "remote", "idle", "directional", "handoff"}:
            original = ROOT / "actions/pbfm/config/generic_pbfm.yaml"
            spec = yaml.safe_load(original.read_text())
            for section, key in [("model", "path"), ("robot", "params"), ("context", "path")]:
                spec[section][key] = str((original.parent / spec[section][key]).resolve())
            fixed = yaml.safe_load(Path(spec["context"]["path"]).read_text())
            for index in range(3):
                sequence_frames.append({"z": [value + index * 0.01 for value in fixed["z"]],
                                        "heading_relative_xy": [0.0, 1.0], "heading_valid": True,
                                        "reference_velocity_heading0_xy": [0.1 * index, 0.2],
                                        "velocity_valid": True})
            context = folder / "context.yaml"
            context.write_text(yaml.safe_dump({"contract": "tracking_heading0_yaw_velocity_v1",
                                                "dt": 0.02, "frames": sequence_frames}))
            spec["context"]["path"] = str(context)
            if name in {"remote", "idle", "directional"}:
                spec["context"].update(command_source="wireless_remote", max_forward_speed=0.8,
                                       max_backward_speed=0.3, max_lateral_speed=0.4, max_yaw_rate=1.0)
            if name in {"idle", "directional"}:
                spec["context"].update(idle_path=str(ROOT / "actions/pbfm/config/context_idle.yaml"),
                                       command_deadband=0.05)
            if name == "directional":
                spec["context"]["path"] = str(ROOT / "actions/pbfm/config/keyboard_walk/context.yaml")
                spec["context"]["directional_paths"] = {
                    direction: str(ROOT / f"actions/pbfm/config/context_keyboard_{direction}.yaml")
                    for direction in ("forward", "backward", "left", "right")}
                spec["context"]["directional_paths"]["forward"] = spec["context"]["path"]
            sequence_spec = folder / "spec.yaml"
            sequence_spec.write_text(yaml.safe_dump(spec))
            command[2] = str(sequence_spec)
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        origin = time.monotonic()
        ready_time = None
        complete_time = None
        try:
            while process.poll() is None and time.monotonic() - origin < 15:
                now = time.monotonic()
                if ready.exists() and ready_time is None:
                    ready_time = now
                if publisher and (ready_time is None or name in successful or now - ready_time < 0.15):
                    publisher.write(np.full((36, 64), 2.5, dtype=np.float32))
                    if name == "range" and ready_time is not None:
                        publisher.array.fill(255)
                if ready_time is not None and (name in successful or name == "active_stale"):
                    start.touch()
                if complete.exists():
                    if complete_time is None:
                        complete_time = now
                    if now - complete_time >= 0.15:
                        release.touch()
                time.sleep(0.02)
            if process.poll() is None:
                raise RuntimeError(f"{name}: timed out")
            output = process.communicate(timeout=5)[0]
            if name in successful:
                if process.returncode != 0 or not trace.read_text().strip():
                    raise RuntimeError(f"{name}: inference did not finish\n{output}")
                if name == "handoff" and (complete_time is None or not release.exists()):
                    raise RuntimeError("tracking did not wait for handoff release")
                if name in {"sequence", "remote", "handoff"}:
                    actual = [json.loads(line) for line in trace.read_text().splitlines()]
                    if len(actual) != len(sequence_frames):
                        raise RuntimeError("tracking did not stop at the final frame")
                    for index, (frame, expected) in enumerate(zip(actual, sequence_frames)):
                        if frame["step"] != index:
                            raise RuntimeError("tracking frame index advanced before start")
                        np.testing.assert_allclose(frame["z"], expected["z"], atol=1e-6, rtol=1e-6)
                        np.testing.assert_allclose(frame["heading"], [0, 0] if name == "remote" else [1, 1], atol=1e-6)
                        np.testing.assert_allclose(frame["velocity_command"],
                                                   [0, 0, 1] if name == "remote" else
                                                   expected["reference_velocity_heading0_xy"] + [1], atol=1e-6)
                if name in {"idle", "directional", "joystick"}:
                    actual = [json.loads(line) for line in trace.read_text().splitlines()]
                    idle_path = "joystick_walk/idle.yaml" if name == "joystick" else "context_idle.yaml"
                    idle = yaml.safe_load((ROOT / "actions/pbfm/config" / idle_path).read_text())
                    if len(actual) <= len(sequence_frames):
                        raise RuntimeError("idle reference incorrectly exhausted the walking cache")
                    for frame in actual:
                        if frame["motion_state"] != "idle" or frame["reference_step"] != 0:
                            raise RuntimeError("neutral remote did not pause walking reference")
                        np.testing.assert_allclose(frame["z"], idle["z"], atol=1e-6, rtol=1e-6)
                        np.testing.assert_allclose(frame["velocity_command"], [0, 0, 1], atol=1e-6)
            else:
                expected = "missing" if name == "missing" else "normalized" if name == "range" else "stale"
                if process.returncode == 0 or expected not in output:
                    raise RuntimeError(f"{name}: expected {expected} failure\n{output}")
                if name == "missing" and ready.exists():
                    raise RuntimeError("missing depth incorrectly signaled ready")
                if name in {"standby_stale", "range"} and trace.read_text().strip():
                    raise RuntimeError(f"{name}: ran active inference before start")
                if name == "active_stale" and not trace.read_text().strip():
                    raise RuntimeError("active dropout did not exercise inference")
            print(f"P-BFM lifecycle passed: {name}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            if publisher:
                publisher.close()


def reject_file_publishing(player):
    with tempfile.TemporaryDirectory(prefix="pbfm-file-") as raw:
        folder = Path(raw)
        original = ROOT / "actions/pbfm/config/generic_pbfm.yaml"
        spec = yaml.safe_load(original.read_text())
        for section, key in [("model", "path"), ("robot", "params"), ("context", "path")]:
            spec[section][key] = str((original.parent / spec[section][key]).resolve())
        np.save(folder / "depth.npy", np.ones((8, 18, 32), dtype=np.float32))
        spec["sensor"].update(depth_source="file", path=str(folder / "depth.npy"))
        path = folder / "spec.yaml"
        path.write_text(yaml.safe_dump(spec))
        result = subprocess.run([str(player), "--spec", str(path)], cwd=ROOT, capture_output=True,
                                text=True, timeout=15)
        if result.returncode == 0 or "only allowed" not in result.stderr:
            raise RuntimeError(f"file source allowed publishing: {result.stdout}\n{result.stderr}")
        print("P-BFM lifecycle passed: file cannot publish")


def reject_missing_idle(player, directional=False):
    with tempfile.TemporaryDirectory(prefix="pbfm-idle-") as raw:
        folder = Path(raw)
        original = ROOT / "actions/pbfm/config/generic_pbfm.yaml"
        spec = yaml.safe_load(original.read_text())
        for section, key in [("model", "path"), ("robot", "params"), ("context", "path")]:
            spec[section][key] = str((original.parent / spec[section][key]).resolve())
        spec["context"].update(command_source="wireless_remote", max_forward_speed=0.8,
                               max_backward_speed=0.3, max_lateral_speed=0.4, max_yaw_rate=1.0,
                               idle_path=str(folder / "missing_idle.yaml"), command_deadband=0.05)
        missing = "missing_idle.yaml"
        if directional:
            spec["context"]["idle_path"] = str(ROOT / "actions/pbfm/config/context_keyboard_idle.yaml")
            spec["context"]["directional_paths"] = {
                name: str(ROOT / f"actions/pbfm/config/context_keyboard_{name}.yaml")
                for name in ("forward", "backward", "left", "right")}
            missing = "missing_forward.yaml"
            spec["context"]["directional_paths"]["forward"] = str(folder / missing)
        path = folder / "spec.yaml"
        path.write_text(yaml.safe_dump(spec))
        ready = folder / "ready"
        result = subprocess.run([str(player), "--spec", str(path), "--dry-run", "--ready-file", str(ready)],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 or ready.exists() or missing not in result.stdout + result.stderr:
            raise RuntimeError(f"{missing} did not fail before readiness")
        print(f"P-BFM lifecycle passed: {missing} cannot become ready")


def reject_missing_turn(player):
    with tempfile.TemporaryDirectory(prefix="pbfm-turn-") as raw:
        folder = Path(raw)
        original = ROOT / "actions/pbfm/config/generic_pbfm_joystick.yaml"
        spec = yaml.safe_load(original.read_text())
        for section, key in [("model", "path"), ("robot", "params"), ("context", "path"), ("context", "idle_path")]:
            spec[section][key] = str((original.parent / spec[section][key]).resolve())
        for field in ("directional_paths", "turn_paths"):
            spec["context"][field] = {name: str((original.parent / path).resolve())
                                      for name, path in spec["context"][field].items()}
        spec["context"]["turn_paths"]["left"] = str(folder / "missing_turn.yaml")
        path = folder / "spec.yaml"
        path.write_text(yaml.safe_dump(spec))
        ready = folder / "ready"
        result = subprocess.run([str(player), "--spec", str(path), "--dry-run", "--ready-file", str(ready)],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 or ready.exists() or "missing_turn.yaml" not in result.stdout + result.stderr:
            raise RuntimeError("missing turn context did not fail before readiness")
        print("P-BFM lifecycle passed: missing turn cannot become ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    args = parser.parse_args()
    for case in ["missing", "normal", "sequence", "remote", "idle", "directional", "joystick", "handoff", "standby_stale", "active_stale", "range"]:
        run_case(args.player.resolve(), case)
    reject_file_publishing(args.player.resolve())
    reject_missing_idle(args.player.resolve())
    reject_missing_idle(args.player.resolve(), directional=True)
    reject_missing_turn(args.player.resolve())
