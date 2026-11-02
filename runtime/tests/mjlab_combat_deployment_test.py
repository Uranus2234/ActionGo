import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "actions/mjlab_combat"


def wait_file(path, process, log):
    deadline = time.monotonic() + 20
    while not path.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            raise AssertionError(log.read_text())
        time.sleep(0.005)


def run(player):
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    bindings = yaml.safe_load((ROOT / "config/remote_actions.yaml").read_text())["bindings"]
    actions = yaml.safe_load((PACKAGE / "mjlab_combat.yaml").read_text())["actions"]["generic_motion_tracking"]
    registry = yaml.safe_load((ROOT / "config/motion_actions.yaml").read_text())
    assert registry["action_files"]["mjlab_combat"] == "actions/mjlab_combat/mjlab_combat.yaml"
    assert not any("allmimic" in action for action in bindings.values())
    assert manifest["iteration"] == 92000
    for asset in (manifest["model"], manifest["params"], *manifest["motions"].values()):
        path = PACKAGE / asset["path"]
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
    for number, binding, count in (("017", "Right+A", 824), ("034", "Right+B", 876)):
        item = manifest["motions"][number]
        assert item["binding"] == binding and bindings[binding] == item["action"]
        assert item["frames"] == count and item["fps"] == 50
        assert item["parity"]["passed"] and item["metrics"]["steps"] == count
        rows = (PACKAGE / item["path"]).read_text().splitlines()
        assert len(rows) == count and all(len(row.split(",")) == 83 for row in rows)
        action = actions[item["action"]]
        assert action["safety_enable"] and action["switch_back"] and action["prestart_before_userctrl"]
        assert action["postrun_handoff_hold"] and action["switch_back_mode"] == "walkrun"
        assert "ActionGo-local" not in action["command"] and "{action_go_root}/build/" in action["command"]
        spec_path = PACKAGE / action["params"]
        spec = yaml.safe_load(spec_path.read_text())
        for section, key, expected in (("model", "path", manifest["model"]["path"]),
                                       ("robot", "params", manifest["params"]["path"]),
                                       ("reference", "path", item["path"])):
            assert (spec_path.parent / spec[section][key]).resolve() == PACKAGE / expected
        assert spec["reference"]["yaw_alignment"] == "none"
        assert spec["reference"]["end_behavior"] == "stop" and spec["reference"]["policy_step_source"] == "wall_clock"
        assert spec["observation"]["obs_dim"] == 261 and spec["action"]["action_dim"] == 29
        checked = subprocess.run([str(player), "--spec", str(spec_path), "--check-only"],
                                 capture_output=True, text=True, check=True, timeout=30)
        assert "inference_check=passed" in checked.stdout
        with tempfile.TemporaryDirectory(prefix="mjlab-combat-") as temporary:
            folder = Path(temporary)
            ready, start, complete, release = [folder / name for name in ("ready", "start", "complete", "release")]
            events_path, log_path = folder / "events.jsonl", folder / "player.log"
            command = [str(player), "--spec", str(spec_path), "--dry-run", "--dry-run-latency",
                       "--offline-dry-run-state", "--max-frames", "3", "--duration-sec", "0",
                       "--lowstate-timeout-sec", "10", "--event-log", str(events_path),
                       "--ready-file", str(ready), "--start-file", str(start),
                       "--handoff-complete-file", str(complete), "--handoff-release-file", str(release)]
            with log_path.open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    wait_file(ready, process, log_path)
                    assert not complete.exists()
                    start.touch()
                    wait_file(complete, process, log_path)
                    assert process.poll() is None
                    release.touch()
                    assert process.wait(timeout=10) == 0, log_path.read_text()
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            samples = [event for event in events if event["event"] == "dry_run_latency"]
            assert len(samples) == 3 and all(not event["lowcmd_sent"] and event["synthetic_state"] for event in samples)
            assert any(event["event"] == "handoff_hold_stopped" and event["released"] for event in events)
            assert not any(event["event"] in ("first_command_published", "first_active_command") for event in events)
        print(f"PASS DuoBox {number}: assets, bindings, ONNX and handoff; no LowCmd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    run(parser.parse_args().player.resolve())
