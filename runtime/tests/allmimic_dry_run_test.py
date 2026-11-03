import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
MODULE_SPEC = importlib.util.spec_from_file_location("allmimic_dry_run", ROOT / "tools/runtime/allmimic_dry_run.py")
BENCHMARK = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(BENCHMARK)


def run(player):
    with tempfile.TemporaryDirectory(prefix="allmimic-dry-run-") as temporary:
        for model in ("paoxiao", "paoxiao_no_omni"):
            events_path = Path(temporary) / f"{model}.jsonl"
            command = [str(player), "--spec", str(ROOT / f"actions/tracking/allmimic/config/generic_allmimic_{model}.yaml"),
                       "--dry-run", "--dry-run-latency", "--offline-dry-run-state",
                       "--duration-sec", "0.12", "--lowstate-timeout-sec", "1",
                       "--event-log", str(events_path)]
            subprocess.run(command, check=True, cwd=ROOT, timeout=30, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            summary = BENCHMARK.summarize(events, 1)
            assert summary["sample_count"] >= 3
            assert summary["synthetic_state"] is True
            assert summary["lowcmd_sent"] is False
            assert summary["unmeasured_count"] == 0
            assert any(event["event"] == "activation_blend_started" for event in events)
            assert not any(event["event"] in ("first_command_published", "first_active_command") for event in events)
            samples = [event for event in events if event["event"] == "dry_run_latency"]
            for sample in samples:
                assert isinstance(sample["command_crc"], int)
                stages = sum(sample[field] for field in BENCHMARK.FIELDS[:6])
                assert math.isclose(stages, sample["e2e_ms"], abs_tol=1e-4), sample
                assert math.isclose(sample["pipeline_ms"] + sample["state_age_ms"], sample["e2e_ms"], abs_tol=1e-4)
            invalid = command.copy()
            invalid.remove("--dry-run")
            result = subprocess.run(invalid, cwd=ROOT, timeout=30, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert result.returncode != 0 and b"requires --dry-run" in result.stdout
            try:
                BENCHMARK.summarize(events, 1000)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Empty samples after warmup must be rejected")
    assert BENCHMARK.distribution([1, 2, 3, 4, 5]) == {
        "mean": 3, "p50": 3, "p95": 4.8, "p99": 4.96, "max": 5}
    print("AllMimic dry-run latency: both models, no publications, complete stage timings, argument guards passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=Path, required=True)
    run(parser.parse_args().player.resolve())
