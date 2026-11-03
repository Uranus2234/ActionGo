"""Shared resources and diagnostics for the single-scene simulator."""

from __future__ import annotations

import contextlib
from collections import deque
import json
import math
import os
import queue
import shutil
import sys
import tempfile
import threading
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


KEYBOARD_REMOTE_BUTTONS = (
    "R+1", "L+1", "Shift+S / Enter", "T", "R+2", "L+2", "F+1", "F+2",
    "Shift+A", "B", "X", "Y", "Up", "Right", "Down", "Left",
)


def keyboard_combo_name(mask: int) -> str:
    return "+".join("Shift+S" if button == 2 else name
                    for button, name in enumerate(KEYBOARD_REMOTE_BUTTONS)
                    if mask & (1 << button)) or "None"


def keyboard_remote_input(keys: set[int]) -> tuple[int, list[float]]:
    keys = {ord(chr(key).lower()) if ord("A") <= key <= ord("Z") else key for key in keys}
    shifted = bool(keys & {340, 344})
    button_keys = {
        ord("b"): 9, ord("x"): 10, ord("y"): 11,
        265: 12, 262: 13, 264: 14, 263: 15,
        257: 2, ord("t"): 3,
    }
    mask = sum(1 << button for key, button in button_keys.items() if key in keys)
    if shifted and ord("a") in keys:
        mask |= 1 << 8
    if shifted and ord("s") in keys:
        mask |= 1 << 2
    if ord("r") in keys and ord("1") in keys:
        mask |= 1 << 0
    if ord("l") in keys and ord("1") in keys:
        mask |= 1 << 1
    if ord("r") in keys and ord("2") in keys:
        mask |= 1 << 4
    if ord("l") in keys and ord("2") in keys:
        mask |= 1 << 5
    if ord("f") in keys and ord("1") in keys:
        mask |= 1 << 6
    if ord("f") in keys and ord("2") in keys:
        mask |= 1 << 7
    movement = keys - {ord("a"), ord("s")} if shifted else keys
    forward = float(ord("w") in movement) - 0.3 * float(ord("s") in movement)
    lateral = float(ord("d") in movement) - float(ord("a") in movement)
    yaw = float(ord("e") in movement) - float(ord("q") in movement)
    if ord("w") in movement and ord("s") in movement:
        forward = 0.0
    if ord("a") in movement and ord("d") in movement:
        lateral = 0.0
    if ord("q") in movement and ord("e") in movement:
        yaw = 0.0
    return mask, [lateral, 0.0, yaw, forward]


class KeyboardRemoteInput:
    def __init__(self):
        self.keys: set[int] = set()
        self.transitions: deque[int] = deque(maxlen=64)

    def set_key(self, key: int, pressed: bool):
        if ord("A") <= key <= ord("Z"):
            key += ord("a") - ord("A")
        previous, _ = self.sample()
        if pressed:
            self.keys.add(key)
        else:
            self.keys.discard(key)
        current, _ = self.sample()
        if current != previous:
            self.transitions.append(current)

    def sample(self) -> tuple[int, list[float]]:
        return keyboard_remote_input(self.keys)

    def consume_buttons(self) -> int:
        return self.transitions.popleft() if self.transitions else self.sample()[0]

    def clear(self):
        self.keys.clear()
        self.transitions.clear()


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
        self.pending = queue.Queue()
        self.writer_error = None
        self.writer = threading.Thread(target=self.write_pending, name="sim2sim-logs")
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
        self.writer.start()
        return self

    def event(self, event: str, level: str = "INFO", **fields):
        record = dict(event=event, level=level, pid=os.getpid(), run_id=self.directory.name,
                      t_unix_ns=time.time_ns(), t_steady_ns=time.monotonic_ns(), **fields)
        if threading.current_thread() is self.writer:
            self.write_event(record)
        else:
            self.pending.put(("event", record))

    def write_event(self, record):
        fields = {key: value for key, value in record.items()
                  if key not in {"event", "level", "pid", "run_id", "t_unix_ns", "t_steady_ns"}}
        self.events.write(json.dumps(record) + "\n")
        self.readable.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record["t_unix_ns"] / 1e9)) +
                            f" {record['level']} {record['event']} " + json.dumps(fields) + "\n")

    def write_pending(self):
        while True:
            item = self.pending.get()
            if item is None:
                return
            kind, payload = item
            try:
                if kind == "event":
                    self.write_event(payload)
                elif kind == "notice":
                    self.write_notice(*payload)
                else:
                    self.write_display(*payload)
            except Exception as error:
                self.writer_error = error

    def notice(self, message: str, level: str = "INFO"):
        self.event("console_notice", level, message=message)
        self.pending.put(("notice", (message, level)))

    def write_notice(self, message: str, level: str):
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
        self.pending.put(("display", (state, policy, timing_path)))

    def write_display(self, state: str, policy: str, timing_path: str):
        now = time.monotonic()
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
        self.pending.put(None)
        self.writer.join()
        if self.writer_error is not None:
            ok = False
            self.result = self.result or 1
            self.console.write(f"ERROR Session log writer failed: {self.writer_error}\n")
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
