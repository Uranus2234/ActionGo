"""Shared resources and diagnostics for the single-scene simulator."""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


MOTOR_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def walkrun_pose_accepted(height: float, tilt: float, minimum_height: float, maximum_tilt: float) -> bool:
    return math.isfinite(height) and math.isfinite(tilt) and height >= minimum_height and tilt <= maximum_tilt


def motor_layout(joint_names: list[str], actuator_joints: list[int]) -> list[int]:
    mapped = []
    for name in MOTOR_NAMES:
        joints = [index for index, candidate in enumerate(joint_names)
                  if candidate.split("/")[-1] in {name, "robot_" + name}]
        actuators = [index for index, joint in enumerate(actuator_joints)
                     if len(joints) == 1 and joint == joints[0]]
        if len(joints) != 1 or len(actuators) != 1:
            raise RuntimeError(f"scene requires exactly one joint and actuator for {name}")
        mapped.append(actuators[0])
    return mapped


def resource_overlay(source: Path, assets: Path | None, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = set(entry.name for entry in source.iterdir()) if source.is_dir() else set()
    if assets and assets.is_dir():
        names.update(entry.name for entry in assets.iterdir())
    for name in sorted(names):
        original = source / name
        fallback = assets / name if assets else None
        target = destination / name
        if original.is_dir() or fallback and fallback.is_dir() and not original.is_file():
            resource_overlay(original, fallback, target)
            continue
        selected = original if original.is_file() else fallback
        if selected and selected.is_file():
            if original.is_file() and original.stat().st_size < 1024 and fallback and fallback.is_file():
                if original.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                    selected = fallback
            if selected.suffix.lower() in {".yaml", ".yml"}:
                shutil.copyfile(selected, target)
            else:
                target.symlink_to(selected.resolve())


def signal_exists(path: str) -> bool:
    try:
        info = os.lstat(path)
        import stat
        return stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1
    except OSError:
        return False


def write_signal(path: str, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(content)


class SimSession:
    def __init__(self, root: Path, mode: str):
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d_%H%M%S_", time.gmtime()), dir=root.resolve()))
        (self.directory / "actions").mkdir()
        self.console = sys.stdout
        self.mode = mode
        self.drawn = False
        self.last_render = 0.0
        self.last_line = 0.0
        self.last_state = None
        self.result = 0
        self.session_fields = {}
        self.events = (self.directory / "events.jsonl").open("a", buffering=1)
        self.readable = (self.directory / "events.log").open("a", buffering=1)
        self.output = (self.directory / "runtime.output.log").open("a", buffering=1)
        self.redirects = contextlib.ExitStack()
        self.old_environment = {key: os.environ.get(key) for key in ("ACTION_GO_RUN_ID", "ACTION_GO_SESSION_DIR")}
        os.environ["ACTION_GO_RUN_ID"] = self.directory.name
        os.environ["ACTION_GO_SESSION_DIR"] = str(self.directory)
        latest = root / "latest"
        temporary = root / (".latest_" + self.directory.name)
        temporary.symlink_to(self.directory, target_is_directory=True)
        temporary.replace(latest)

    def __enter__(self):
        self.console.write(f"ActionGo | SIM2SIM\nLogs: {self.directory}\n")
        self.console.flush()
        if self.mode != "debug":
            self.redirects.enter_context(contextlib.redirect_stdout(self.output))
            self.redirects.enter_context(contextlib.redirect_stderr(self.output))
        return self

    def event(self, event: str, level: str = "INFO", **fields):
        record = dict(event=event, level=level, pid=os.getpid(), run_id=self.directory.name,
                      t_unix_ns=time.time_ns(), t_steady_ns=time.monotonic_ns(), **fields)
        self.events.write(json.dumps(record) + "\n")
        self.readable.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) +
                            f" {level} {event} " + json.dumps(fields) + "\n")

    def notice(self, message: str, level: str = "INFO"):
        self.event("console_notice", level, message=message)
        if self.drawn:
            self.console.write("\033[4A\033[J")
            self.drawn = False
        self.console.write(f"{level} {message}\n")
        self.console.flush()

    def render(self, state: str, policy: str, timing_path: str):
        now = time.monotonic()
        if now - self.last_render < 0.2:
            return
        self.last_render = now
        changed = (state, policy) != self.last_state
        if changed:
            self.event("display_state_changed", control_state=state, policy=policy)
        self.last_state = state, policy
        if self.mode != "normal":
            return
        latency = "--"
        if policy and timing_path:
            for suffix in (".latest.json", ".dds.jsonl.latest.json"):
                try:
                    timing = json.loads(Path(timing_path + suffix).read_text())
                    age = time.monotonic_ns() - timing["t_steady_ns"]
                    latency = f"{timing['e2e_ms']:.1f} ms" if 0 <= age < 1_000_000_000 else "STALE"
                    break
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        if self.console.isatty():
            if self.drawn:
                self.console.write("\033[4A")
            self.console.write("\033[2K-----------------------------------------------\n"
                               f"\033[2K  STATE    {state}\n"
                               f"\033[2K  POLICY   \033[1;36m{policy or '--'}\033[0m    E2E  {latency}\n"
                               "\033[2K-----------------------------------------------\n")
            self.drawn = True
        elif changed or policy and now - self.last_line >= 1.0:
            self.console.write(f"STATE {state} | POLICY {policy or '--'} | E2E {latency}\n")
            self.last_line = now
        self.console.flush()

    def metadata(self, **fields):
        self.session_fields.update(fields)
        record = dict(application="ActionGo", kind="sim2sim", pid=os.getpid(), **self.session_fields)
        (self.directory / "session.json").write_text(json.dumps(record, indent=2) + "\n")

    def __exit__(self, exception_type, exception, traceback):
        if exception:
            if not self.result:
                self.result = 1
            self.notice(str(exception), "ERROR")
        ok = exception is None and self.result == 0
        self.event("session_finished", "INFO" if ok else "ERROR", ok=ok, exit_code=self.result)
        (self.directory / "summary.json").write_text(json.dumps(dict(clean_shutdown=ok, exit_code=self.result)) + "\n")
        self.redirects.close()
        if self.drawn:
            self.console.write("\n")
            self.console.flush()
        for stream in (self.events, self.readable, self.output):
            stream.close()
        for key, value in self.old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
