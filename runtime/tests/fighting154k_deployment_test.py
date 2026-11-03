import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "actions/tracking/fighting154k"
MODEL_KEYS = ("supermimic_omni", "supermimic_no_omni", "allmimic_omni", "allmimic_no_omni")


def wait_file(path, process, log, timeout=20):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None or time.monotonic() >= deadline:
            raise AssertionError(f"No {path.name}: {log.read_text()}")
        time.sleep(0.005)


def run(player):
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    archived = json.loads((PACKAGE / "validation.json").read_text())
    bindings = yaml.safe_load((ROOT / "config/remote_actions.yaml").read_text())["bindings"]
    registered = yaml.safe_load((PACKAGE / "fighting154k.yaml").read_text())["actions"]["generic_motion_tracking"]
    config = yaml.safe_load((ROOT / "config/motion_actions.yaml").read_text())
    assert config["action_files"]["fighting154k"] == "actions/tracking/fighting154k/fighting154k.yaml"
    assert len(registered) == len(manifest["actions"]) == 10
    assert bindings["R1+A"] == "fighting154k_supermimic_omni_wbst_motive"
    assert bindings["R1+Down"] == "fighting154k_supermimic_omni_yewendun"
    for asset in list(manifest["models"].values()) + list(manifest["motions"].values()):
        assert hashlib.sha256((PACKAGE / asset["path"]).read_bytes()).hexdigest() == asset["sha256"]
    for model in manifest["models"].values():
        assert model["iteration"] == 154000
        assert hashlib.sha256((PACKAGE / model["params"]).read_bytes()).hexdigest() == model["params_sha256"]
    with tempfile.TemporaryDirectory(prefix="fighting154k-") as temporary:
        for key in MODEL_KEYS:
            motions = ("051_ma", "side_kick")
            if key == "supermimic_omni":
                motions += ("wbst_motive", "yewendun")
            for motion in motions:
                name = f"fighting154k_{key}_{motion}"
                item = manifest["actions"][name]
                if item["binding"] is None:
                    assert name not in bindings.values()
                else:
                    assert bindings[item["binding"]] == name
                action = registered[name]
                assert action["prestart_before_userctrl"] and action["postrun_handoff_hold"]
                assert action["safety_enable"] and action["switch_back"]
                assert action["switch_back_mode"] == "walkrun" and action["switch_back_overlap_sec"] == 0.30
                for placeholder in ("ready_file", "start_file", "handoff_complete_file", "handoff_release_file"):
                    assert "{" + placeholder + "}" in action["command"]
                spec_path = PACKAGE / action["params"]
                assert action["params"] == item["spec"]
                spec = yaml.safe_load(spec_path.read_text())
                assert (spec_path.parent / spec["model"]["path"]).resolve() == PACKAGE / manifest["models"][key]["path"]
                assert (spec_path.parent / spec["robot"]["params"]).resolve() == PACKAGE / manifest["models"][key]["params"]
                assert (spec_path.parent / spec["reference"]["path"]).resolve() == PACKAGE / manifest["motions"][item["motion"]]["path"]
                assert spec["safety"]["preserve_pd_torque"]
                assert spec["reference"]["end_behavior"] == "stop"
                assert spec["reference"]["yaw_alignment"] == "initial"
                assert spec["reference"]["policy_step_source"] == "wall_clock"
                assert spec["handoff"]["activation_blend_sec"] == 0.10
                assert not spec["handoff"]["damping_on_stop"]
                evidence = archived["runs"][name]
                assert evidence["native_parity"]["passed"] and evidence["metrics"]["full_duration"]
                assert evidence["model_sha256"] == manifest["models"][key]["sha256"]
                assert evidence["reference_sha256"] == manifest["motions"][item["motion"]]["sha256"]
                result = subprocess.run([str(player), "--spec", str(spec_path), "--check-only"],
                                        capture_output=True, text=True, check=True, timeout=30)
                assert "inference_check=passed" in result.stdout
                directory = Path(temporary) / name
                directory.mkdir()
                ready, start, complete, release = [directory / value for value in ("ready", "start", "complete", "release")]
                events_path, log_path = directory / "events.jsonl", directory / "player.log"
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
                        time.sleep(0.03)
                        assert process.poll() is None, "Player must hold until release"
                        release.touch()
                        assert process.wait(timeout=10) == 0, log_path.read_text()
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                events = [json.loads(line) for line in events_path.read_text().splitlines()]
                assert any(event["event"] == "activation_blend_started" for event in events)
                assert any(event["event"] == "handoff_hold_stopped" and event["released"] for event in events)
                assert not any(event["event"] in ("first_command_published", "first_active_command") for event in events)
                samples = [event for event in events if event["event"] == "dry_run_latency"]
                assert len(samples) == 3, log_path.read_text()
                assert all(not sample["lowcmd_sent"] and sample["synthetic_state"] and math.isfinite(sample["e2e_ms"]) for sample in samples)
                print(f"PASS {name}: ONNX, dry-run, ready/start/complete/release; no LowCmd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    run(parser.parse_args().player.resolve())
