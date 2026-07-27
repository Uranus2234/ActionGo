#!/usr/bin/env python3

import argparse
import json
import pathlib
import subprocess
import tempfile
import time


def wait_for_file(path: pathlib.Path, process: subprocess.Popen, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"player exited before preview file (code={process.returncode})\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        time.sleep(0.01)
    process.terminate()
    stdout, stderr = process.communicate(timeout=5)
    raise RuntimeError(
        f"timeout waiting for preview file\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )


def read_events(path: pathlib.Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_transaction(
    repo: pathlib.Path,
    player: pathlib.Path,
    spec: str,
    decision: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="actiongo_preview_test_") as raw_tmp:
        tmp = pathlib.Path(raw_tmp)
        preview = tmp / "preview"
        commit = tmp / "commit"
        discard = tmp / "discard"
        event_log = tmp / "events.jsonl"
        command = [
            str(player),
            "--spec",
            spec,
            "--dry-run",
            "--offline-dry-run-state",
            "--duration-sec",
            "0.02",
            "--candidate-topic",
            f"action_go/test/{pathlib.Path(spec).stem}",
            "--preview-file",
            str(preview),
            "--commit-file",
            str(commit),
            "--discard-file",
            str(discard),
            "--preview-timeout-sec",
            "5",
            "--event-log",
            str(event_log),
        ]
        process = subprocess.Popen(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        wait_for_file(preview, process, 20.0)
        (commit if decision == "commit" else discard).write_text(decision + "\n")
        stdout, stderr = process.communicate(timeout=20)
        if process.returncode != 0:
            raise RuntimeError(
                f"{pathlib.Path(spec).stem} {decision} failed (code={process.returncode})\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )

        events = read_events(event_log)
        names = [event["event"] for event in events]
        if "policy_entry_contract" not in names or "preview_ready" not in names:
            raise RuntimeError(f"missing preview contract/readiness events for {spec}: {names}")
        if decision == "commit":
            if "preview_committed" not in names:
                raise RuntimeError(f"missing preview_committed for {spec}: {names}")
        else:
            if "preview_discarded" not in names or "preview_state_restored" not in names:
                raise RuntimeError(f"discard did not restore state for {spec}: {names}")
        if events[-1]["event"] != "process_exit" or not events[-1]["ok"]:
            raise RuntimeError(f"transaction did not exit cleanly for {spec}: {events[-1]}")


def test_stale_commit_rejected(repo: pathlib.Path, player: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="actiongo_preview_stale_") as raw_tmp:
        tmp = pathlib.Path(raw_tmp)
        preview = tmp / "preview"
        commit = tmp / "commit"
        commit.write_text("stale\n")
        result = subprocess.run(
            [
                str(player),
                "--spec",
                "actions/bfm0/config/generic_bfm_zero.yaml",
                "--dry-run",
                "--offline-dry-run-state",
                "--preview-file",
                str(preview),
                "--commit-file",
                str(commit),
                "--candidate-topic",
                "action_go/test/stale_target",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=20,
        )
        if result.returncode == 0 or "stale preview lifecycle input" not in result.stderr:
            raise RuntimeError(
                "stale commit barrier was not rejected\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )


def test_real_topic_rejected(repo: pathlib.Path, player: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="actiongo_preview_real_topic_") as raw_tmp:
        tmp = pathlib.Path(raw_tmp)
        result = subprocess.run(
            [
                str(player),
                "--spec",
                "actions/bfm0/config/generic_bfm_zero.yaml",
                "--dry-run",
                "--offline-dry-run-state",
                "--preview-file",
                str(tmp / "preview"),
                "--commit-file",
                str(tmp / "commit"),
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=20,
        )
        if result.returncode == 0 or "requires a private --candidate-topic" not in result.stderr:
            raise RuntimeError(
                "preview was allowed to publish to the real LowCmd topic\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--policy-player", type=pathlib.Path, required=True)
    parser.add_argument("--motion-player", type=pathlib.Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    policy_player = args.policy_player.resolve()
    motion_player = args.motion_player.resolve()
    run_transaction(
        repo,
        policy_player,
        "actions/bfm0/config/generic_bfm_zero.yaml",
        "commit",
    )
    run_transaction(
        repo,
        policy_player,
        "actions/beyondmimic/config/generic_beyondmimic.yaml",
        "commit",
    )
    run_transaction(
        repo,
        motion_player,
        "actions/protomotions/config/generic_protomotions.yaml",
        "commit",
    )
    run_transaction(
        repo,
        policy_player,
        "actions/bfm0/config/generic_bfm_zero.yaml",
        "discard",
    )
    test_stale_commit_rejected(repo, policy_player)
    test_real_topic_rejected(repo, policy_player)
    print("generic_preview_integration: PASS")


if __name__ == "__main__":
    main()
