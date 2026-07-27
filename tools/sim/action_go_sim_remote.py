#!/usr/bin/env python3
"""Unified action_go remote interface for unitree_mujoco simulation.

The real robot entrypoint (build/action_go_remote_test) switches between the
Unitree official walkrun/passive controller and action_go LowCmd players.  In
unitree_mujoco there is no official walkrun service, so this simulator keeps the
same button/shell logic and replaces official walkrun with the requested
unitree_rl_mjlab velocity policy player.

Default walkrun policy:
actions/unitree_rl_mjlab/model/policy.onnx
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")

import mujoco
import numpy as np
import yaml

try:
    import mujoco.viewer
except Exception:  # headless users may not have a display backend
    mujoco_viewer = None
else:
    mujoco_viewer = mujoco.viewer

ACTION_GO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WALKRUN_MODEL = ACTION_GO_ROOT / "actions/unitree_rl_mjlab/model/policy.onnx"

# Instinct parkour is disabled in the unified simulation entry for now.  The
# real-robot config stays intact, but sim should not launch it from joystick /
# keyboard shortcuts because the current policy is not safe in MuJoCo.
SIM_DISABLED_ACTION_TYPES = {"instinct_parkour"}

# These action_go players all expose --publish-hz.  In the unified MuJoCo
# entrypoint they share one Python bridge whose lowstate/control cadence is
# 1 / --sim-dt, so normalize their LowCmd publication cadence there instead of
# letting some actions silently run at hard-coded real-robot 500 Hz while
# others run at a different MuJoCo cadence.
SIM_PUBLISH_HZ_CAPABLE_ACTION_TYPES = {
    "generic_policy",
    "generic_motion_tracking",
    "beyondmimic",
    "bfm_zero",
    "instinct_parkour",
    "holosoma_wbt",
    "protomotions_tracker",
    "omnixtreme",
    "steadytray",
    "unitree_rl_mjlab",
    "twist",
}

STOP = threading.Event()


def _signal_handler(_signum, _frame):
    STOP.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# Same bit order as action_go_remote_test / Unitree wireless_remote.
BUTTONS = {
    "R1": 0, "L1": 1, "Start": 2, "Select": 3,
    "R2": 4, "L2": 5, "F1": 6, "F2": 7,
    "A": 8, "B": 9, "X": 10, "Y": 11,
    "Up": 12, "Right": 13, "Down": 14, "Left": 15,
}
BUTTON_NAMES = [None] * 16
for _name, _idx in BUTTONS.items():
    BUTTON_NAMES[_idx] = _name


def bit(button: int) -> int:
    return 1 << int(button)


def parse_button(name: str) -> int:
    n = str(name).strip().lower()
    aliases = {
        "r1": "R1", "rb": "R1", "l1": "L1", "lb": "L1",
        "start": "Start", "select": "Select", "back": "Select",
        "r2": "R2", "rt": "R2", "l2": "L2", "lt": "L2",
        "f1": "F1", "f2": "F2", "a": "A", "b": "B", "x": "X", "y": "Y",
        "up": "Up", "right": "Right", "down": "Down", "left": "Left",
    }
    if n not in aliases:
        raise ValueError(f"unknown remote button: {name}")
    return BUTTONS[aliases[n]]


def parse_combo(text: str) -> int:
    mask = 0
    for item in str(text).split("+"):
        item = item.strip().strip("'\"")
        if item:
            mask |= bit(parse_button(item))
    if mask == 0:
        raise ValueError(f"empty combo: {text}")
    return mask


def parse_optional_combo(text: str) -> int:
    s = str(text).strip().strip("'\"").lower()
    if s in {"", "none", "null", "off", "disable", "disabled", "-"}:
        return 0
    return parse_combo(text)


def combo_name(mask: int) -> str:
    parts = [BUTTON_NAMES[i] for i in range(16) if mask & bit(i)]
    return "+".join(parts) if parts else "None"


def combo_rising(now: int, prev: int, combo: int) -> bool:
    return (now & combo) == combo and (prev & combo) != combo


def normalize_mode(mode: str) -> str:
    m = str(mode or "walkrun").strip().lower().replace("-", "_")
    if m in {"walk", "run", "walk_run", "walkrun", "2"}:
        return "walkrun"
    if m in {"damp", "damping", "passive", "1"}:
        return "passive"
    if m in {"last", "0"}:
        return "last"
    return "walkrun"


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "on"}:
        return True
    if s in {"false", "0", "no", "off"}:
        return False
    return default


def is_numeric_literal(value: str) -> bool:
    try:
        float(str(value).strip())
        return True
    except Exception:
        return False


def projected_gravity_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-9:
        q = np.asarray([1.0, 0.0, 0.0, 0.0])
    else:
        q = q / n
    w, x, y, z = [float(v) for v in q]
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.asarray([-r20, -r21, -r22], dtype=np.float64)


def tilt_deg_from_quat_wxyz(q: np.ndarray) -> float:
    g = projected_gravity_from_quat_wxyz(q)
    return float(np.degrees(np.arccos(np.clip(-float(g[2]), -1.0, 1.0))))


@dataclass
class Binding:
    combo: int
    action: str


@dataclass
class SafetyConfig:
    enable: bool = True
    action: str = "ufo_stand"
    return_mode: str = "walkrun"
    fall_angle_deg: float = 57.3
    recover_angle_deg: float = 20.0
    fall_hold_ms: int = 60
    recover_hold_ms: int = 1200
    cooldown_ms: int = 1000


@dataclass
class RemoteConfig:
    bindings: list[Binding] = field(default_factory=list)
    emergency_combo: int = bit(BUTTONS["R2"]) | bit(BUTTONS["B"])
    walkrun_exit_combo: int = bit(BUTTONS["R2"]) | bit(BUTTONS["Start"])
    damping_exit_combo: int = bit(BUTTONS["R2"]) | bit(BUTTONS["X"])
    shell_start_combo: int = bit(BUTTONS["Start"])
    shell_start_alt_combo: int = bit(BUTTONS["Left"]) | bit(BUTTONS["B"])
    shell_start_action: str = "bfm_zero_stand"
    shell_start_alt_action: str = "ufo_stand"
    shell_damping_combo: int = bit(BUTTONS["Select"]) | bit(BUTTONS["Start"])
    shell_walkrun_combo: int = bit(BUTTONS["R2"]) | bit(BUTTONS["A"])
    lie_down_combo: int = bit(BUTTONS["L2"]) | bit(BUTTONS["Down"])
    lie_action: str = "lie_down_beyondmimic"
    safety: SafetyConfig = field(default_factory=SafetyConfig)


def load_remote_config(path: Path | None) -> RemoteConfig:
    cfg = RemoteConfig()
    if path is None or not path.exists():
        cfg.bindings = [Binding(bit(BUTTONS["R2"]) | bit(BUTTONS["Down"]), "kick_right")]
        return cfg
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg.bindings = []
    for key, value in (data.get("bindings") or {}).items():
        cfg.bindings.append(Binding(parse_combo(str(key)), str(value)))
    if not cfg.bindings:
        cfg.bindings = [Binding(bit(BUTTONS["R2"]) | bit(BUTTONS["Down"]), "kick_right")]
    if "emergency" in data:
        cfg.emergency_combo = parse_optional_combo(data["emergency"])
    if "normal_exit" in data:
        cfg.walkrun_exit_combo = parse_optional_combo(data["normal_exit"])
    if "damping_exit" in data:
        cfg.damping_exit_combo = parse_optional_combo(data["damping_exit"])
    if "shell_start" in data:
        cfg.shell_start_combo = parse_optional_combo(data["shell_start"])
    if "shell_start_alt" in data:
        cfg.shell_start_alt_combo = parse_optional_combo(data["shell_start_alt"])
    if "shell_start_action" in data:
        cfg.shell_start_action = str(data["shell_start_action"])
    if "shell_start_alt_action" in data:
        cfg.shell_start_alt_action = str(data["shell_start_alt_action"])
    if "shell_damping" in data:
        cfg.shell_damping_combo = parse_optional_combo(data["shell_damping"])
    if "shell_walkrun" in data:
        cfg.shell_walkrun_combo = parse_optional_combo(data["shell_walkrun"])
    if "lie_down" in data:
        cfg.lie_down_combo = parse_combo(data["lie_down"])
    if "lie_action" in data:
        cfg.lie_action = str(data["lie_action"])
    if "safety_enable" in data:
        cfg.safety.enable = parse_bool(data["safety_enable"], cfg.safety.enable)
    if "safety_action" in data:
        cfg.safety.action = str(data["safety_action"])
    if "safety_return_mode" in data:
        cfg.safety.return_mode = normalize_mode(data["safety_return_mode"])
    for key, attr, typ in [
        ("fall_angle_deg", "fall_angle_deg", float),
        ("recover_angle_deg", "recover_angle_deg", float),
        ("fall_hold_ms", "fall_hold_ms", int),
        ("recover_hold_ms", "recover_hold_ms", int),
        ("safety_cooldown_ms", "cooldown_ms", int),
    ]:
        if key in data:
            setattr(cfg.safety, attr, typ(data[key]))
    return cfg


@dataclass
class ActionConfig:
    name: str = ""
    type: str = "beyondmimic"
    model: str = ""
    params: str = ""
    trajectory: str = ""
    command: str = ""
    resource_base_dir: str = ""
    network_interface: str = "lo"
    duration_sec: float = 0.0
    max_frames: int = 0
    switch_back: bool = True
    switch_back_mode: str = "walkrun"
    safety_enable: bool = True
    prestart_before_userctrl: bool = False
    prestart_ready_timeout_sec: float = 3.0
    ready_file: str = ""
    start_file: str = ""


def apply_action_fields(action: ActionConfig, fields: dict[str, Any]):
    for key, value in (fields or {}).items():
        if value is None:
            continue
        if key == "type":
            action.type = str(value)
        elif key == "model":
            action.model = str(value)
        elif key == "params":
            action.params = str(value)
        elif key == "trajectory":
            action.trajectory = str(value)
        elif key == "command":
            action.command = str(value)
        elif key in {"network_interface", "net"}:
            action.network_interface = str(value)
        elif key == "duration_sec":
            action.duration_sec = float(value)
        elif key in {"max_frames", "max_steps", "motion_end_step", "motion_end_frame"}:
            action.max_frames = int(value)
        elif key == "switch_back":
            action.switch_back = parse_bool(value, True)
        elif key in {"switch_back_mode", "internal_ctrl_mode", "back_mode"}:
            action.switch_back_mode = normalize_mode(str(value))
        elif key in {"safety_enable", "safety", "enable_safety", "safety_mode"}:
            action.safety_enable = parse_bool(value, True)
        elif key in {"prestart_before_userctrl", "start_before_userctrl"}:
            action.prestart_before_userctrl = parse_bool(value, False)
        elif key in {"prestart_ready_timeout_sec", "ready_timeout_sec"}:
            action.prestart_ready_timeout_sec = float(value)
        elif key == "ready_file":
            action.ready_file = str(value)
        elif key == "start_file":
            action.start_file = str(value)


def merge_action(name: str, typ: str, base_dir: Path, global_defaults: dict[str, Any], type_defaults: dict[str, Any], fields: dict[str, Any]) -> ActionConfig:
    action = ActionConfig(name=name, type=typ, resource_base_dir=str(base_dir))
    apply_action_fields(action, global_defaults)
    if typ in type_defaults:
        old_name, old_type = action.name, action.type
        apply_action_fields(action, type_defaults[typ] or {})
        action.name, action.type = old_name, old_type
    apply_action_fields(action, fields or {})
    if action.type in type_defaults and action.type != typ:
        old_name, old_type = action.name, action.type
        merged = ActionConfig(name=old_name, type=old_type, resource_base_dir=str(base_dir))
        apply_action_fields(merged, global_defaults)
        apply_action_fields(merged, type_defaults[action.type] or {})
        apply_action_fields(merged, fields or {})
        merged.name, merged.type = old_name, old_type
        action = merged
    return action


def iter_action_file_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(v) for v in value.values() if v]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return [str(value)]


def load_action_registry(config_path: Path, net: str) -> dict[str, ActionConfig]:
    config_path = config_path.expanduser().resolve()
    config_dir = config_path.parent
    action_go_root = config_dir.parent if config_dir.name == "config" else config_dir
    loaded: set[Path] = set()
    actions: dict[str, ActionConfig] = {}

    main_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    global_defaults = dict(main_data.get("defaults") or {})
    type_defaults = dict(main_data.get("type_defaults") or {})

    def resolve_include(base_dir: Path, raw: str) -> Path:
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p
        for candidate in (base_dir / p, action_go_root / p, config_dir / p):
            if candidate.exists():
                return candidate.resolve()
        return (action_go_root / p).resolve()

    def parse_file(path: Path):
        path = path.expanduser().resolve()
        if path in loaded:
            return
        loaded.add(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        local_actions = data.get("actions") or {}
        for key, value in local_actions.items():
            if isinstance(value, dict) and key in type_defaults:
                typ = key
                for action_name, fields in value.items():
                    if not isinstance(fields, dict):
                        fields = {}
                    actions[action_name] = merge_action(action_name, typ, path.parent, global_defaults, type_defaults, fields)
            elif isinstance(value, dict):
                fields = value
                typ = str(fields.get("type", key if key in type_defaults else "beyondmimic"))
                actions[key] = merge_action(key, typ, path.parent, global_defaults, type_defaults, fields)
        includes = data.get("action_files") or data.get("actions_files") or data.get("include_actions") or data.get("includes")
        for inc in iter_action_file_values(includes):
            parse_file(resolve_include(path.parent, inc))

    parse_file(config_path)

    def resolve_resource(raw: str, base_dir: str) -> str:
        if not raw:
            return ""
        p = Path(str(raw)).expanduser()
        if p.is_absolute():
            return str(p)
        for candidate in (Path(base_dir) / p, action_go_root / p, config_dir / p):
            if candidate.exists():
                return str(candidate.resolve())
        return str((action_go_root / p).resolve())

    for action in actions.values():
        action.network_interface = net
        action.model = resolve_resource(action.model, action.resource_base_dir)
        action.params = resolve_resource(action.params, action.resource_base_dir)
        if not (action.type == "beyondmimic" and is_numeric_literal(action.trajectory)):
            action.trajectory = resolve_resource(action.trajectory, action.resource_base_dir)
        if action.prestart_before_userctrl:
            action.ready_file = action.ready_file or f"/tmp/action_go_sim_{action.name}_ready"
            action.start_file = action.start_file or f"/tmp/action_go_sim_{action.name}_start"
    return actions


def replace_all(text: str, pairs: dict[str, str]) -> str:
    for key, value in pairs.items():
        text = text.replace(key, value)
    return text


def default_player_path() -> str:
    return str(ACTION_GO_ROOT / "build" / "generic_policy_player")


def expand_command(action: ActionConfig) -> str:
    cmd = action.command
    values = {
        "{name}": action.name,
        "{action_dir}": action.resource_base_dir,
        "{resource_base_dir}": action.resource_base_dir,
        "{action_go_root}": str(ACTION_GO_ROOT),
        "{config_dir}": str(ACTION_GO_ROOT / "config"),
        "{player}": default_player_path(),
        "{model}": action.model,
        "{params}": action.params,
        "{trajectory}": action.trajectory,
        "{net}": action.network_interface,
        "{network_interface}": action.network_interface,
        "{duration_sec}": str(action.duration_sec),
        "{max_frames}": str(action.max_frames),
        "{max_steps}": str(action.max_frames),
        "{motion_end_step}": str(action.max_frames),
        "{ready_file}": action.ready_file,
        "{start_file}": action.start_file,
        "{event_log_arg}": "",
    }
    return replace_all(cmd, values)


def sim_cadence_hz(sim_dt: float) -> int:
    dt = float(sim_dt)
    if dt <= 0.0:
        raise ValueError(f"--sim-dt must be positive, got {sim_dt}")
    return max(1, int(round(1.0 / dt)))


def effective_publish_hz(configured_hz: int | float, sim_dt: float) -> int:
    hz = int(round(float(configured_hz or 0)))
    return hz if hz > 0 else sim_cadence_hz(sim_dt)


def _set_cli_option(tokens: list[str], flag: str, value: str) -> bool:
    """Set --flag value / --flag=value in a shlex token list."""
    found = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == flag:
            if i + 1 < len(tokens):
                tokens[i + 1] = value
            else:
                tokens.append(value)
            found = True
            i += 2
        elif tok.startswith(flag + "="):
            tokens[i] = f"{flag}={value}"
            found = True
            i += 1
        else:
            i += 1
    return found


def _shell_join(tokens: list[str]) -> str:
    # shlex.join() quotes shell operators such as &&, which would turn a
    # compound command into argv text.  Keep common shell operators bare while
    # still quoting normal argv tokens safely.
    operators = {"&&", "||", ";", "|", "&", "(", ")"}
    return " ".join(tok if tok in operators else shlex.quote(tok) for tok in tokens)


def adapt_command_for_sim(
    command: str,
    action_type: str = "",
    publish_hz: int = 0,
    lowstate_timeout_sec: float = 0.0,
    normalize_publish_hz: bool = False,
    map_user_lowcmd: bool = True,
) -> str:
    if map_user_lowcmd:
        # Legacy unitree_mujoco bridge consumes only rt/lowcmd directly; real
        # UserCtrl players use rt/user_lowcmd.  The native action_go bridge
        # below subscribes to both topics and should *not* use this mapping.
        command = command.replace("--cmd-topic rt/user_lowcmd", "--cmd-topic rt/lowcmd")
        command = command.replace("--cmd-topic=rt/user_lowcmd", "--cmd-topic rt/lowcmd")
    if normalize_publish_hz and publish_hz > 0 and action_type in SIM_PUBLISH_HZ_CAPABLE_ACTION_TYPES:
        try:
            tokens = shlex.split(command)
            found = _set_cli_option(tokens, "--publish-hz", str(int(publish_hz)))
            if not found:
                tokens += ["--publish-hz", str(int(publish_hz))]
            command = _shell_join(tokens)
        except ValueError as e:
            print(
                f"[action_go_sim] WARNING: failed to normalize --publish-hz for "
                f"type={action_type}: {e}",
                flush=True,
            )
    if lowstate_timeout_sec > 0.0 and action_type in SIM_PUBLISH_HZ_CAPABLE_ACTION_TYPES:
        tokens = shlex.split(command)
        found = _set_cli_option(tokens, "--lowstate-timeout-sec", str(float(lowstate_timeout_sec)))
        if not found:
            tokens += ["--lowstate-timeout-sec", str(float(lowstate_timeout_sec))]
        command = _shell_join(tokens)
    return " ".join(command.split())


def start_shell_command(command: str) -> subprocess.Popen:
    print(f"[action_go_sim] launch: {command}", flush=True)
    return subprocess.Popen(command, shell=True, preexec_fn=os.setsid)


def start_argv(argv: list[str]) -> subprocess.Popen:
    print("[action_go_sim] launch: " + " ".join(shlex.quote(a) for a in argv), flush=True)
    return subprocess.Popen(argv, preexec_fn=os.setsid)


def stop_process(proc: subprocess.Popen | None, name: str = "process", force: bool = False):
    if proc is None or proc.poll() is not None:
        return
    try:
        sig = signal.SIGKILL if force else signal.SIGTERM
        os.killpg(os.getpgid(proc.pid), sig)
    except Exception:
        try:
            proc.kill() if force else proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=1.5 if not force else 0.5)
    except subprocess.TimeoutExpired:
        if not force:
            stop_process(proc, name=name, force=True)
    print(f"[action_go_sim] stopped {name}", flush=True)


@dataclass
class ImuSample:
    valid: bool = False
    tilt_deg: float = 0.0
    quat: np.ndarray = field(default_factory=lambda: np.asarray([1, 0, 0, 0], dtype=np.float32))
    stamp: float = 0.0


@dataclass
class SimLowCommand:
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    mode_machine: int = 0
    stamp: float = 0.0
    seq: int = 0


class NativeActionGoBridge:
    """Synchronous MuJoCo <-> Unitree DDS bridge with a UserCtrl-style gate.

    The stock unitree_mujoco Python bridge has one rt/lowcmd subscriber and
    writes data.ctrl directly from the DDS callback.  That is fine for simple
    demos, but it cannot mimic the real action_go handoff:

      official/walkrun controller -> rt/lowcmd
      action_go UserCtrl players  -> rt/user_lowcmd

    This bridge subscribes to both command topics and applies exactly one of
    them from the main MuJoCo loop.  It also publishes LowState once per sim
    step, avoiding an extra wall-time lowstate thread racing with mj_step().
    """

    def __init__(self, model, data, joystick_type: str = "xbox", joystick_device: int = 0):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

        self.model = model
        self.data = data
        self.num_motor = int(model.nu)
        if self.num_motor < 29:
            raise RuntimeError(f"native bridge expects at least 29 actuators, got {self.num_motor}")
        self.dim_motor_sensor = 3 * self.num_motor
        self.have_imu_sensor = False
        self.have_frame_sensor = False
        for i in range(self.dim_motor_sensor, self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name == "imu_quat":
                self.have_imu_sensor = True
            if name == "frame_pos":
                self.have_frame_sensor = True

        self.low_state = LowState_default()
        self.lock = threading.Lock()
        self.mode = "none"  # none/passive, lowcmd/walkrun, user/action
        self.lowcmd: SimLowCommand | None = None
        self.usercmd: SimLowCommand | None = None
        self.seq = 0
        self.joystick = None
        self.axis_id: dict[str, int] = {}
        self.button_id: dict[str, int] = {}
        self.joystick_type = joystick_type
        self.joystick_device = int(joystick_device)

        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()
        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(lambda msg: self._cmd_handler("lowcmd", msg), 10)
        self.user_low_cmd_suber = ChannelSubscriber("rt/user_lowcmd", LowCmd_)
        self.user_low_cmd_suber.Init(lambda msg: self._cmd_handler("user", msg), 10)

        # The C++ players in this repo read wireless_remote from LowState.  We
        # still publish rt/wirelesscontroller for tools that inspect it.
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
        self.wireless_controller_puber.Init()

        self.actuator_limited = np.asarray(self.model.actuator_ctrllimited[: self.num_motor], dtype=bool)
        self.ctrl_low = np.asarray(self.model.actuator_ctrlrange[: self.num_motor, 0], dtype=np.float64)
        self.ctrl_high = np.asarray(self.model.actuator_ctrlrange[: self.num_motor, 1], dtype=np.float64)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        import pygame
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count <= 0:
            print("No gamepad detected.")
            raise SystemExit(1)
        self.joystick = pygame.joystick.Joystick(int(device_id))
        self.joystick.init()
        self.joystick_type = js_type
        self.joystick_device = int(device_id)
        if js_type == "xbox":
            self.axis_id = {"LX": 0, "LY": 1, "RX": 3, "RY": 4, "LT": 2, "RT": 5}
            self.button_id = {"X": 2, "Y": 3, "B": 1, "A": 0, "LB": 4, "RB": 5, "SELECT": 6, "START": 7}
        elif js_type == "switch":
            self.axis_id = {"LX": 0, "LY": 1, "RX": 2, "RY": 3, "LT": 5, "RT": 4}
            self.button_id = {"X": 3, "Y": 4, "B": 1, "A": 0, "LB": 6, "RB": 7, "SELECT": 10, "START": 11}
        else:
            raise RuntimeError(f"unsupported joystick type: {js_type}")

    def set_command_mode(self, mode: str):
        m = str(mode).lower()
        if m in {"walkrun", "lowcmd", "normal"}:
            m = "lowcmd"
        elif m in {"user", "action", "userctrl"}:
            m = "user"
        elif m in {"none", "passive", "damping"}:
            m = "none"
        else:
            raise ValueError(f"unknown native bridge mode: {mode}")
        with self.lock:
            if self.mode != m:
                print(f"[action_go_sim][bridge] command gate: {self.mode} -> {m}", flush=True)
            self.mode = m

    def _extract_cmd(self, msg) -> SimLowCommand:
        n = self.num_motor
        q = np.zeros(n, dtype=np.float64)
        dq = np.zeros(n, dtype=np.float64)
        tau = np.zeros(n, dtype=np.float64)
        kp = np.zeros(n, dtype=np.float64)
        kd = np.zeros(n, dtype=np.float64)
        mc = msg.motor_cmd
        for i in range(min(n, len(mc))):
            q[i] = float(mc[i].q)
            dq[i] = float(mc[i].dq)
            tau[i] = float(mc[i].tau)
            kp[i] = float(mc[i].kp)
            kd[i] = float(mc[i].kd)
        self.seq += 1
        return SimLowCommand(q, dq, tau, kp, kd, int(msg.mode_machine), time.monotonic(), self.seq)

    def _cmd_handler(self, topic: str, msg):
        cmd = self._extract_cmd(msg)
        with self.lock:
            if topic == "user":
                self.usercmd = cmd
            else:
                self.lowcmd = cmd

    def _joystick_mask_and_axes(self) -> tuple[int, list[float]]:
        if self.joystick is None:
            return 0, [0.0, 0.0, 0.0, 0.0]
        import pygame
        pygame.event.get()
        b = self.button_id
        a = self.axis_id
        mask = 0
        def set_btn(name: str, pressed: bool):
            nonlocal mask
            if pressed:
                mask |= bit(BUTTONS[name])
        set_btn("R1", bool(self.joystick.get_button(b["RB"])))
        set_btn("L1", bool(self.joystick.get_button(b["LB"])))
        set_btn("Start", bool(self.joystick.get_button(b["START"])))
        set_btn("Select", bool(self.joystick.get_button(b["SELECT"])))
        set_btn("R2", bool(self.joystick.get_axis(a["RT"]) > 0))
        set_btn("L2", bool(self.joystick.get_axis(a["LT"]) > 0))
        set_btn("A", bool(self.joystick.get_button(b["A"])))
        set_btn("B", bool(self.joystick.get_button(b["B"])))
        set_btn("X", bool(self.joystick.get_button(b["X"])))
        set_btn("Y", bool(self.joystick.get_button(b["Y"])))
        hat = self.joystick.get_hat(0)
        set_btn("Up", bool(hat[1] > 0))
        set_btn("Right", bool(hat[0] > 0))
        set_btn("Down", bool(hat[1] < 0))
        set_btn("Left", bool(hat[0] < 0))
        axes = [
            float(self.joystick.get_axis(a["LX"])),
            float(self.joystick.get_axis(a["RX"])),
            float(-self.joystick.get_axis(a["RY"])),
            float(-self.joystick.get_axis(a["LY"])),
        ]
        return mask, axes

    @staticmethod
    def _write_float_bytes(buf, offset: int, value: float):
        bs = struct.pack("f", float(value))
        for i, b in enumerate(bs):
            buf[offset + i] = b

    def _fill_remote(self):
        remote = self.low_state.wireless_remote
        for i in range(len(remote)):
            remote[i] = 0
        mask, axes = self._joystick_mask_and_axes()
        remote[2] = mask & 0xFF
        remote[3] = (mask >> 8) & 0xFF
        self._write_float_bytes(remote, 4, axes[0])   # LX
        self._write_float_bytes(remote, 8, axes[1])   # RX
        self._write_float_bytes(remote, 12, axes[2])  # RY
        self._write_float_bytes(remote, 20, axes[3])  # LY

        self.wireless_controller.keys = int(mask)
        self.wireless_controller.lx = axes[0]
        self.wireless_controller.rx = axes[1]
        self.wireless_controller.ry = axes[2]
        self.wireless_controller.ly = axes[3]

    def publish_lowstate(self):
        if self.data is None:
            return
        n = min(self.num_motor, len(self.low_state.motor_state))
        for i in range(n):
            self.low_state.motor_state[i].q = float(self.data.sensordata[i])
            self.low_state.motor_state[i].dq = float(self.data.sensordata[i + self.num_motor])
            self.low_state.motor_state[i].tau_est = float(self.data.sensordata[i + 2 * self.num_motor])
        if self.have_imu_sensor and self.data.sensordata.size >= self.dim_motor_sensor + 10:
            for i in range(4):
                self.low_state.imu_state.quaternion[i] = float(self.data.sensordata[self.dim_motor_sensor + i])
            for i in range(3):
                self.low_state.imu_state.gyroscope[i] = float(self.data.sensordata[self.dim_motor_sensor + 4 + i])
                self.low_state.imu_state.accelerometer[i] = float(self.data.sensordata[self.dim_motor_sensor + 7 + i])
        else:
            for i in range(4):
                self.low_state.imu_state.quaternion[i] = float(self.data.qpos[3 + i])
            for i in range(3):
                self.low_state.imu_state.gyroscope[i] = float(self.data.qvel[3 + i])
        self._fill_remote()
        self.low_state_puber.Write(self.low_state)
        self.wireless_controller_puber.Write(self.wireless_controller)

    def apply_command(self):
        with self.lock:
            cmd = self.usercmd if self.mode == "user" else self.lowcmd if self.mode == "lowcmd" else None
        if cmd is None:
            self.data.ctrl[: self.num_motor] = 0.0
            return
        q_now = np.asarray(self.data.sensordata[: self.num_motor], dtype=np.float64)
        dq_now = np.asarray(self.data.sensordata[self.num_motor : 2 * self.num_motor], dtype=np.float64)
        ctrl = cmd.tau + cmd.kp * (cmd.q - q_now) + cmd.kd * (cmd.dq - dq_now)
        if self.actuator_limited.size:
            limited = self.actuator_limited
            ctrl[limited] = np.clip(ctrl[limited], self.ctrl_low[limited], self.ctrl_high[limited])
        self.data.ctrl[: self.num_motor] = ctrl


class RemoteMonitor:
    def __init__(self, topic: str = "rt/lowstate"):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        self.lock = threading.Lock()
        self.mask_value = 0
        self.imu_sample = ImuSample()
        self.connected = False
        self.sub = ChannelSubscriber(topic, LowState_)
        self.sub.Init(self._handler, 1)

    def _handler(self, msg):
        remote = list(msg.wireless_remote)
        mask = 0
        if len(remote) >= 4:
            mask = int(remote[2]) | (int(remote[3]) << 8)
        quat = np.asarray(list(msg.imu_state.quaternion), dtype=np.float32)
        tilt = tilt_deg_from_quat_wxyz(quat)
        with self.lock:
            self.mask_value = mask
            self.imu_sample = ImuSample(True, tilt, quat, time.monotonic())
            self.connected = True

    def mask(self) -> int:
        with self.lock:
            return int(self.mask_value)

    def imu(self) -> ImuSample:
        with self.lock:
            return ImuSample(self.imu_sample.valid, self.imu_sample.tilt_deg, self.imu_sample.quat.copy(), self.imu_sample.stamp)


class SimActionManager:
    def __init__(self, args: argparse.Namespace, actions: dict[str, ActionConfig], bridge=None):
        self.args = args
        self.actions = actions
        self.bridge = bridge
        self.native_bridge = isinstance(bridge, NativeActionGoBridge)
        self.sim_publish_hz = effective_publish_hz(getattr(args, "sim_action_publish_hz", 0), args.sim_dt)
        self.walkrun_publish_hz = effective_publish_hz(getattr(args, "walkrun_publish_hz", 0), args.sim_dt)
        self.walkrun_proc: subprocess.Popen | None = None
        self.active_proc: subprocess.Popen | None = None
        self.active_action: str = ""
        self.active_switch_mode: str = "walkrun"
        self.passive = False
        self.lock = threading.Lock()

    def set_bridge_mode(self, mode: str):
        if hasattr(self.bridge, "set_command_mode"):
            self.bridge.set_command_mode(mode)

    def walkrun_running(self) -> bool:
        return self.walkrun_proc is not None and self.walkrun_proc.poll() is None

    def action_running(self) -> bool:
        return self.active_proc is not None and self.active_proc.poll() is None

    def any_motion_running(self) -> bool:
        return self.action_running() or self.walkrun_running()

    def start_walkrun(self):
        with self.lock:
            if self.action_running():
                return
            if self.walkrun_running():
                self.passive = False
                self.set_bridge_mode("lowcmd")
                return
            model = str(Path(self.args.walkrun_model).expanduser())
            params = str(Path(self.args.walkrun_params).expanduser()) if self.args.walkrun_params else ""
            argv = [
                str(ACTION_GO_ROOT / "build" / "generic_policy_player"),
                "--spec", str(ACTION_GO_ROOT / "actions/unitree_rl_mjlab/config/generic_unitree_rl_mjlab.yaml"),
                "--net", self.args.interface,
                "--domain-id", str(int(self.args.domain_id)),
                "--cmd-topic", "rt/lowcmd",
                "--state-topic", "rt/lowstate",
                "--model", model,
                "--hz", str(int(self.args.walkrun_hz)),
                "--publish-hz", str(int(self.walkrun_publish_hz)),
                "--no-damping-on-stop",
            ]
            if params:
                argv += ["--params", params]
            if self.args.walkrun_zero_command:
                print("[action_go_sim] WARNING: --walkrun-zero-command is ignored by generic_policy_player")
            self.passive = False
            self.set_bridge_mode("lowcmd")
            self.walkrun_proc = start_argv(argv)

    def stop_walkrun(self):
        with self.lock:
            stop_process(self.walkrun_proc, "walkrun")
            self.walkrun_proc = None

    def enter_passive(self):
        print("[action_go_sim] enter simulated PASSIVE damping", flush=True)
        self.stop_active(switch_mode="none")
        self.stop_walkrun()
        self.set_bridge_mode("none")
        self.passive = True

    def apply_passive(self, data):
        if not self.passive:
            return
        n = min(29, data.ctrl.size, max(0, data.qvel.size - 6))
        if n > 0:
            data.ctrl[:n] = -float(self.args.passive_kd) * data.qvel[6:6 + n]

    def _prepare_prestart_files(self, action: ActionConfig):
        if action.ready_file:
            try:
                Path(action.ready_file).unlink(missing_ok=True)
            except Exception:
                pass
        if action.start_file:
            try:
                Path(action.start_file).unlink(missing_ok=True)
            except Exception:
                pass

    def _wait_ready(self, action: ActionConfig, proc: subprocess.Popen) -> bool:
        if not action.prestart_before_userctrl or not action.ready_file:
            return True
        deadline = time.monotonic() + float(action.prestart_ready_timeout_sec)
        while time.monotonic() < deadline and not STOP.is_set():
            # With the native bridge LowState is published from the MuJoCo main
            # loop, but start_action() blocks that loop while waiting for a
            # prestarted player.  Pump LowState here so the child can connect,
            # publish its hold command, and touch ready_file.
            if hasattr(self.bridge, "publish_lowstate"):
                self.bridge.publish_lowstate()
            if proc.poll() is not None:
                print(f"[action_go_sim] action exited before ready: {action.name} status={proc.returncode}", flush=True)
                return False
            if Path(action.ready_file).exists():
                return True
            time.sleep(0.01)
        print(f"[action_go_sim] ready timeout for action {action.name}: {action.ready_file}", flush=True)
        return False

    def _signal_action_start(self, action: ActionConfig):
        if action.prestart_before_userctrl and action.start_file:
            Path(action.start_file).write_text("start\n", encoding="utf-8")

    def start_action(self, action_name: str, source: str = "remote") -> bool:
        if action_name not in self.actions:
            print(f"[action_go_sim] unknown action from {source}: {action_name}", flush=True)
            return False
        if self.action_running():
            print(f"[action_go_sim] ignore action while running: {action_name} (active={self.active_action})", flush=True)
            return False
        action = self.actions[action_name]
        if action.type in SIM_DISABLED_ACTION_TYPES:
            print(
                f"[action_go_sim] sim action disabled: {action_name} type={action.type}; "
                "skip in unified simulation",
                flush=True,
            )
            return False
        command = adapt_command_for_sim(
            expand_command(action),
            action_type=action.type,
            publish_hz=int(self.sim_publish_hz),
            lowstate_timeout_sec=float(self.args.sim_lowstate_timeout_sec),
            normalize_publish_hz=not bool(getattr(self.args, "no_normalize_action_publish_hz", False)),
            map_user_lowcmd=not self.native_bridge,
        )
        if not command:
            print(f"[action_go_sim] action has no command: {action_name}", flush=True)
            return False
        self._prepare_prestart_files(action)
        if not self.native_bridge:
            self.stop_walkrun()
            self.passive = False
        proc = start_shell_command(command)
        if not self._wait_ready(action, proc):
            stop_process(proc, action_name)
            if not self.walkrun_running() and not self.passive:
                self.start_walkrun()
            return False
        if self.native_bridge and action.prestart_before_userctrl:
            # Real handoff order: action player is already publishing a hold on
            # rt/user_lowcmd; switch the simulated UserCtrl gate to that topic,
            # stop walkrun rt/lowcmd, then release the player from hold.
            self.set_bridge_mode("user")
            self.stop_walkrun()
            self.passive = False
            self._signal_action_start(action)
        else:
            self.stop_walkrun()
            self.set_bridge_mode("user")
            self.passive = False
            self._signal_action_start(action)
        self.active_proc = proc
        self.active_action = action_name
        self.active_switch_mode = normalize_mode(action.switch_back_mode)
        print(f"[action_go_sim] active action={action_name} source={source} switch_back={self.active_switch_mode}", flush=True)
        return True

    def stop_active(self, switch_mode: str = "walkrun"):
        if self.active_proc is not None:
            stop_process(self.active_proc, self.active_action or "action")
        self.active_proc = None
        self.active_action = ""
        self.active_switch_mode = normalize_mode(switch_mode) if switch_mode != "none" else "none"
        if switch_mode == "walkrun":
            self.start_walkrun()
        elif switch_mode == "passive":
            self.stop_walkrun()
            self.set_bridge_mode("none")
            self.passive = True

    def poll_action_done(self) -> tuple[str, int, str] | None:
        if self.active_proc is None:
            return None
        code = self.active_proc.poll()
        if code is None:
            return None
        name = self.active_action
        mode = self.active_switch_mode
        print(f"[action_go_sim] action exited: {name} status={code} switch_back={mode}", flush=True)
        self.active_proc = None
        self.active_action = ""
        if mode == "passive":
            self.stop_walkrun()
            self.set_bridge_mode("none")
            self.passive = True
        elif mode in {"walkrun", "last"}:
            self.start_walkrun()
        return name, int(code), mode

    def shutdown(self, passive: bool = False):
        self.stop_active(switch_mode="none")
        self.stop_walkrun()
        self.set_bridge_mode("none")
        self.passive = passive


class SafetyState:
    NORMAL = "normal"
    RECOVERING = "recovering"


def resolve_default_path(raw: str) -> str:
    p = Path(raw)
    if p.exists():
        return str(p)
    p2 = ACTION_GO_ROOT / raw
    return str(p2)


def parse_args(argv=None) -> argparse.Namespace:
    unitree_mujoco = os.environ.get("UNITREE_MUJOCO_ROOT", str(ACTION_GO_ROOT.parent / "unitree_mujoco"))
    unitree_sdk2py = os.environ.get("UNITREE_SDK2PY_ROOT", str(ACTION_GO_ROOT.parent / "unitree_sdk2_python"))
    p = argparse.ArgumentParser(description="action_go unified remote interface for unitree_mujoco")
    p.add_argument("--config", default=str(ACTION_GO_ROOT / "config/motion_actions.yaml"))
    p.add_argument("--remote-map", default=str(ACTION_GO_ROOT / "config/remote_actions.yaml"))
    p.add_argument("--unitree-mujoco-root", default=unitree_mujoco)
    p.add_argument("--unitree-sdk2py-root", default=unitree_sdk2py)
    p.add_argument("--xml", default="", help="default: <unitree_mujoco_root>/unitree_robots/g1/scene_29dof.xml")
    p.add_argument("--interface", default="lo")
    p.add_argument("--domain-id", type=int, default=0, help="use 0 for compatibility with all action_go C++ players")
    p.add_argument("--sim-dt", type=float, default=0.002, help="default matches a 500 Hz low-level robot/sim control loop")
    p.add_argument("--viewer-hz", type=float, default=50.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--paused", action="store_true")
    p.add_argument("--no-realtime", action="store_true")
    p.add_argument("--bridge", default="native", choices=["native", "unitree"], help="native keeps rt/lowcmd and rt/user_lowcmd separated like real UserCtrl")
    p.add_argument("--use-joystick", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--joystick-type", default="xbox", choices=["xbox", "switch"])
    p.add_argument("--joystick-device", type=int, default=0)
    p.add_argument("--require-joystick", action="store_true", help="fail if joystick is requested but missing")
    p.add_argument("--walkrun-model", default=str(DEFAULT_WALKRUN_MODEL))
    p.add_argument("--walkrun-params", default="", help="default inferred from model path")
    p.add_argument("--walkrun-hz", type=int, default=50)
    p.add_argument("--walkrun-publish-hz", type=int, default=0, help="0 means match 1/--sim-dt lowstate/control cadence")
    p.add_argument("--sim-action-publish-hz", type=int, default=0, help="0 means rewrite action player --publish-hz to 1/--sim-dt")
    p.add_argument("--sim-lowstate-timeout-sec", type=float, default=0.2, help="LowState safety timeout passed only to simulated action players")
    p.add_argument("--no-normalize-action-publish-hz", action="store_true", help="keep action YAML/player --publish-hz values instead of matching --sim-dt")
    p.add_argument("--walkrun-startup-sec", type=float, default=0.2)
    p.add_argument("--walkrun-target-interp-sec", type=float, default=0.02)
    p.add_argument("--walkrun-stats-interval", type=int, default=100)
    p.add_argument("--walkrun-zero-command", action="store_true", help="ignore joystick axes in simulated walkrun")
    p.add_argument("--passive-kd", type=float, default=3.0)
    p.add_argument("--remote-period-sec", type=float, default=0.02)
    p.add_argument("--auto-action", default="", help="start this action automatically after lowstate connects; useful for headless demos")
    p.add_argument("--auto-arm-sec", type=float, default=-1.0, help="arm the shell after this many seconds; useful for headless safety tests")
    p.add_argument("--auto-arm-alt-sec", type=float, default=-1.0, help="arm with the alternate recovery backend after this many seconds")
    p.add_argument("--auto-exit-sec", type=float, default=0.0, help="exit automatically after this many seconds; 0 disables")
    p.add_argument("--record-video", default="", help="write an mp4 from the MuJoCo camera using ffmpeg")
    p.add_argument("--record-fps", type=float, default=30.0)
    p.add_argument("--record-width", type=int, default=1280)
    p.add_argument("--record-height", type=int, default=720)
    return p.parse_args(argv)


def setup_unitree_mujoco(args: argparse.Namespace):
    unitree_root = Path(args.unitree_mujoco_root).expanduser().resolve()
    sim_py = unitree_root / "simulate_python"
    if args.unitree_sdk2py_root:
        sys.path.insert(0, args.unitree_sdk2py_root)
    sys.path.insert(0, str(sim_py))

    cfg = types.ModuleType("config")
    cfg.ROBOT = "g1"
    cfg.ROBOT_SCENE = str(Path(args.xml).expanduser().resolve() if args.xml else unitree_root / "unitree_robots/g1/scene_29dof.xml")
    cfg.DOMAIN_ID = int(args.domain_id)
    cfg.INTERFACE = args.interface
    cfg.USE_JOYSTICK = 1 if args.use_joystick else 0
    cfg.JOYSTICK_TYPE = args.joystick_type
    cfg.JOYSTICK_DEVICE = int(args.joystick_device)
    cfg.PRINT_SCENE_INFORMATION = False
    cfg.ENABLE_ELASTIC_BAND = False
    cfg.SIMULATE_DT = float(args.sim_dt)
    cfg.VIEWER_DT = 1.0 / max(1.0, float(args.viewer_hz))
    sys.modules["config"] = cfg

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        if args.bridge == "unitree":
            from unitree_sdk2py_bridge import UnitreeSdk2Bridge
        else:
            UnitreeSdk2Bridge = None
    except Exception as e:
        raise RuntimeError(
            "failed to import unitree_mujoco/unitree_sdk2py modules. "
            f"Current Python is {sys.executable}; try /usr/bin/python3 or set --unitree-sdk2py-root."
        ) from e

    model = mujoco.MjModel.from_xml_path(cfg.ROBOT_SCENE)
    model.opt.timestep = float(args.sim_dt)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ChannelFactoryInitialize(int(args.domain_id), args.interface)
    if args.bridge == "unitree":
        bridge = UnitreeSdk2Bridge(model, data)
    else:
        bridge = NativeActionGoBridge(model, data, joystick_type=args.joystick_type, joystick_device=int(args.joystick_device))
    joystick_ok = False
    if args.use_joystick:
        try:
            bridge.SetupJoystick(device_id=int(args.joystick_device), js_type=args.joystick_type)
            joystick_ok = True
        except SystemExit:
            if args.require_joystick:
                raise
            print("[action_go_sim] WARNING: no joystick detected; remote buttons will stay idle. Use --no-use-joystick or connect a gamepad.", flush=True)
        except Exception as e:
            if args.require_joystick:
                raise
            print(f"[action_go_sim] WARNING: joystick setup failed: {e}", flush=True)
    return model, data, bridge, joystick_ok, cfg.ROBOT_SCENE


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        sim_lowlevel_hz = sim_cadence_hz(args.sim_dt)
        model, data, _bridge, joystick_ok, xml_path = setup_unitree_mujoco(args)
        actions = load_action_registry(Path(args.config), args.interface)
        remote_cfg = load_remote_config(Path(args.remote_map) if args.remote_map else None)
        remote = RemoteMonitor("rt/lowstate")
        manager = SimActionManager(args, actions, _bridge)
        disabled_bindings: list[Binding] = []
        enabled_bindings: list[Binding] = []
        for binding in remote_cfg.bindings:
            action = actions.get(binding.action)
            if action is not None and action.type in SIM_DISABLED_ACTION_TYPES:
                disabled_bindings.append(binding)
            else:
                enabled_bindings.append(binding)
        remote_cfg.bindings = enabled_bindings

        print(
            f"[action_go_sim] xml={xml_path} domain={args.domain_id} interface={args.interface} "
            f"bridge={args.bridge} sim_dt={args.sim_dt} sim_lowlevel_hz={sim_lowlevel_hz} joystick={joystick_ok}",
            flush=True,
        )
        action_pub_text = (
            "config/default"
            if args.no_normalize_action_publish_hz
            else str(int(manager.sim_publish_hz))
        )
        print(
            f"[action_go_sim] walkrun model={args.walkrun_model} "
            f"walkrun_hz={args.walkrun_hz} walkrun_publish_hz={manager.walkrun_publish_hz} "
            f"action_publish_hz={action_pub_text}",
            flush=True,
        )
        print("[action_go_sim] Bindings:", flush=True)
        for b in remote_cfg.bindings:
            print(f"  {combo_name(b.combo)} -> {b.action}", flush=True)
        if disabled_bindings:
            print("[action_go_sim] Disabled in unified sim:", flush=True)
            for b in disabled_bindings:
                action = actions.get(b.action)
                typ = action.type if action is not None else "unknown"
                print(f"  {combo_name(b.combo)} -> {b.action} ({typ})", flush=True)
        def special_combo_text(combo: int, desc: str) -> str:
            if combo == 0:
                return f"  disabled -> {desc}"
            return f"  {combo_name(combo)} -> {desc}"
        print(
            "\n".join([
                special_combo_text(remote_cfg.shell_start_combo, "ARM shell"),
                special_combo_text(remote_cfg.shell_start_alt_combo, "ARM shell (alternate)"),
                f"  primary ARM recovery -> {remote_cfg.shell_start_action}",
                f"  alternate ARM recovery -> {remote_cfg.shell_start_alt_action}",
                special_combo_text(remote_cfg.shell_walkrun_combo, "simulated WALKRUN shell"),
                special_combo_text(remote_cfg.shell_damping_combo, "simulated PASSIVE shell"),
                special_combo_text(remote_cfg.emergency_combo, "emergency stop"),
                special_combo_text(remote_cfg.walkrun_exit_combo, "walkrun exit"),
                special_combo_text(remote_cfg.damping_exit_combo, "damping exit"),
                special_combo_text(remote_cfg.lie_down_combo, f"{remote_cfg.lie_action}, then passive shell"),
            ]),
            flush=True,
        )

        if args.auto_action:
            manager.set_bridge_mode("none")
            manager.passive = False
            print(
                f"[action_go_sim][auto] will start action '{args.auto_action}' after lowstate connects",
                flush=True,
            )
        else:
            manager.enter_passive()
            print(
                "[action_go_sim][shell] initial state=SHELL_IDLE + simulated PASSIVE: "
                "robot is damped and will fall. Use Start/S for BFM stand or "
                "Left+B/B for UFO stand; the selected backend remains active for safety.",
                flush=True,
            )

        # Wait briefly for lowstate; do not block forever if running headless/no joystick.
        wait_deadline = time.monotonic() + 5.0
        while not remote.connected and time.monotonic() < wait_deadline and not STOP.is_set():
            if hasattr(_bridge, "publish_lowstate"):
                _bridge.publish_lowstate()
            time.sleep(0.02)
        if not remote.connected:
            print("[action_go_sim] WARNING: no rt/lowstate seen yet", flush=True)

        run_started_at = time.monotonic()
        pending_auto_action = bool(args.auto_action)

        paused = bool(args.paused)
        keyboard_events: list[str] = []
        keyboard_lock = threading.Lock()

        def key_callback(keycode):
            nonlocal paused
            try:
                c = chr(keycode).lower()
            except Exception:
                c = ""
            if keycode == ord(" "):
                paused = not paused
                print(f"[action_go_sim] paused={paused}", flush=True)
            elif c == "r":
                with keyboard_lock:
                    keyboard_events.append("reset")
            elif c == "s":
                with keyboard_lock:
                    keyboard_events.append("start")
            elif c == "b":
                with keyboard_lock:
                    keyboard_events.append("start_alt")
            elif c == "w":
                with keyboard_lock:
                    keyboard_events.append("walkrun")
            elif c == "p":
                with keyboard_lock:
                    keyboard_events.append("passive")
            elif c == "t":
                with keyboard_lock:
                    keyboard_events.append("ufo_tracking")
            elif c == "g":
                with keyboard_lock:
                    keyboard_events.append("ufo_stand")

        prev_mask = remote.mask()
        shell_armed = False
        lie_action_active = False
        selected_safety_action = remote_cfg.safety.action
        safety_state = SafetyState.NORMAL
        fall_timer_active = False
        recover_timer_active = False
        fall_since = time.monotonic()
        recover_since = time.monotonic()
        last_safety_finish = time.monotonic() - max(0, remote_cfg.safety.cooldown_ms) / 1000.0
        next_remote_tick = 0.0
        next_viewer_sync = 0.0

        def reset_safety_state():
            nonlocal safety_state, fall_timer_active, recover_timer_active, last_safety_finish
            safety_state = SafetyState.NORMAL
            fall_timer_active = False
            recover_timer_active = False
            last_safety_finish = time.monotonic() - max(0, remote_cfg.safety.cooldown_ms) / 1000.0

        def arm_shell(source: str, recovery_action: str):
            nonlocal shell_armed, selected_safety_action, safety_state, fall_timer_active, recover_timer_active
            if manager.action_running():
                print(f"[action_go_sim][shell] cannot arm while action is running: {manager.active_action}", flush=True)
                return
            shell_armed = True
            selected_safety_action = recovery_action or remote_cfg.safety.action
            reset_safety_state()
            # ARM immediately starts the selected stand backend.  Once upright,
            # safetyTick() switches to walkrun and remembers that backend for later falls.
            print(
                f"[action_go_sim][shell] ARMED by {source}: actions/safety enabled; "
                f"start recovery={selected_safety_action}",
                flush=True,
            )
            safety_state = SafetyState.RECOVERING
            fall_timer_active = False
            recover_timer_active = False
            manager.stop_walkrun()
            manager.start_action(selected_safety_action, source="arm")

        def start_action(action_name: str, source: str):
            nonlocal lie_action_active
            if safety_state == SafetyState.RECOVERING and action_name != selected_safety_action:
                print(f"[action_go_sim][safety] ignore action while recovering: {action_name}", flush=True)
                return
            if manager.start_action(action_name, source=source):
                lie_action_active = (action_name == remote_cfg.lie_action)

        def start_safety_recovery(tilt: float):
            nonlocal safety_state, fall_timer_active, recover_timer_active
            safety_state = SafetyState.RECOVERING
            fall_timer_active = False
            recover_timer_active = False
            print(
                f"[action_go_sim][safety] fallen tilt={tilt:.1f} >= {remote_cfg.safety.fall_angle_deg}; "
                f"start {selected_safety_action}",
                flush=True,
            )
            manager.stop_active(switch_mode="none")
            manager.stop_walkrun()
            manager.start_action(selected_safety_action, source="safety")

        def safety_tick():
            nonlocal fall_timer_active, recover_timer_active, fall_since, recover_since, safety_state, last_safety_finish
            if not remote_cfg.safety.enable or not shell_armed:
                return
            imu = remote.imu()
            now = time.monotonic()
            if not imu.valid or now - imu.stamp > 0.5:
                return
            fallen = imu.tilt_deg >= remote_cfg.safety.fall_angle_deg
            upright = imu.tilt_deg <= remote_cfg.safety.recover_angle_deg
            if safety_state == SafetyState.NORMAL:
                recover_timer_active = False
                if now - last_safety_finish < max(0, remote_cfg.safety.cooldown_ms) / 1000.0:
                    fall_timer_active = False
                    return
                if not fallen:
                    fall_timer_active = False
                    return
                if not fall_timer_active:
                    fall_since = now
                    fall_timer_active = True
                if now - fall_since >= max(0, remote_cfg.safety.fall_hold_ms) / 1000.0:
                    start_safety_recovery(imu.tilt_deg)
                return

            # Recovering: keep recovery action alive until upright is stable.
            fall_timer_active = False
            if not manager.action_running():
                manager.stop_walkrun()
                manager.start_action(selected_safety_action, source="safety_watchdog")
            if not upright:
                recover_timer_active = False
                return
            if not recover_timer_active:
                recover_since = now
                recover_timer_active = True
            if now - recover_since < max(0, remote_cfg.safety.recover_hold_ms) / 1000.0:
                return
            print(
                f"[action_go_sim][safety] recovered tilt={imu.tilt_deg:.1f}; return {remote_cfg.safety.return_mode}",
                flush=True,
            )
            manager.stop_active(switch_mode="none")
            if normalize_mode(remote_cfg.safety.return_mode) == "passive":
                manager.enter_passive()
            else:
                manager.start_walkrun()
            safety_state = SafetyState.NORMAL
            recover_timer_active = False
            last_safety_finish = now

        def remote_tick():
            nonlocal prev_mask, shell_armed, lie_action_active, safety_state
            with keyboard_lock:
                events = list(keyboard_events)
                keyboard_events.clear()
            for ev in events:
                if ev == "reset":
                    print("[action_go_sim] reset requested", flush=True)
                    shell_armed = False
                    lie_action_active = False
                    reset_safety_state()
                    manager.shutdown(passive=True)
                    mujoco.mj_resetData(model, data)
                    mujoco.mj_forward(model, data)
                    manager.enter_passive()
                elif ev == "start":
                    arm_shell("keyboard S", remote_cfg.shell_start_action)
                elif ev == "start_alt":
                    arm_shell("keyboard B", remote_cfg.shell_start_alt_action)
                elif ev == "walkrun":
                    shell_armed = False
                    reset_safety_state()
                    manager.stop_active(switch_mode="none")
                    manager.start_walkrun()
                    print("[action_go_sim][shell] keyboard W -> WALKRUN shell", flush=True)
                elif ev == "passive":
                    shell_armed = False
                    reset_safety_state()
                    manager.enter_passive()
                    print("[action_go_sim][shell] keyboard P -> PASSIVE shell", flush=True)
                elif ev in {"ufo_tracking", "ufo_stand"}:
                    action_name = "ufo_tracking_zs7" if ev == "ufo_tracking" else "ufo_stand"
                    if not shell_armed:
                        print(
                            f"[action_go_sim][keyboard] press S to ARM before starting {action_name}",
                            flush=True,
                        )
                    elif manager.action_running():
                        print(
                            f"[action_go_sim][keyboard] stop the current action with W or P before starting {action_name}",
                            flush=True,
                        )
                    else:
                        key_name = "T" if ev == "ufo_tracking" else "G"
                        start_action(action_name, f"keyboard {key_name}")

            now_mask = remote.mask()
            if combo_rising(now_mask, prev_mask, remote_cfg.damping_exit_combo):
                print("[action_go_sim] damping exit requested", flush=True)
                manager.shutdown(passive=True)
                STOP.set()
            elif combo_rising(now_mask, prev_mask, remote_cfg.walkrun_exit_combo):
                print("[action_go_sim] walkrun exit requested", flush=True)
                manager.shutdown(passive=False)
                STOP.set()
            elif combo_rising(now_mask, prev_mask, remote_cfg.shell_walkrun_combo):
                shell_armed = False
                lie_action_active = False
                reset_safety_state()
                manager.stop_active(switch_mode="none")
                manager.start_walkrun()
                print("[action_go_sim][shell] return to SHELL_IDLE + simulated WALKRUN", flush=True)
            elif combo_rising(now_mask, prev_mask, remote_cfg.shell_damping_combo):
                shell_armed = False
                lie_action_active = False
                reset_safety_state()
                manager.enter_passive()
                print("[action_go_sim][shell] return to SHELL_IDLE + simulated PASSIVE", flush=True)
            elif combo_rising(now_mask, prev_mask, remote_cfg.shell_start_combo):
                arm_shell(
                    "remote " + combo_name(remote_cfg.shell_start_combo),
                    remote_cfg.shell_start_action,
                )
            elif combo_rising(now_mask, prev_mask, remote_cfg.shell_start_alt_combo):
                arm_shell(
                    "remote " + combo_name(remote_cfg.shell_start_alt_combo),
                    remote_cfg.shell_start_alt_action,
                )
            elif combo_rising(now_mask, prev_mask, remote_cfg.lie_down_combo):
                shell_armed = False
                reset_safety_state()
                lie_action_active = True
                manager.stop_active(switch_mode="none")
                manager.stop_walkrun()
                start_action(remote_cfg.lie_action, "remote lie-down")
            elif combo_rising(now_mask, prev_mask, remote_cfg.emergency_combo):
                print("[action_go_sim] emergency stop requested", flush=True)
                manager.stop_active(switch_mode="walkrun")
                lie_action_active = False

            if shell_armed:
                safety_tick()
                if not manager.action_running():
                    for binding in remote_cfg.bindings:
                        if combo_rising(now_mask, prev_mask, binding.combo):
                            start_action(binding.action, "remote")
                            break

            done = manager.poll_action_done()
            if done is not None:
                name, _code, mode = done
                if lie_action_active or name == remote_cfg.lie_action:
                    lie_action_active = False
                    shell_armed = False
                    reset_safety_state()
                    manager.enter_passive()
                    print("[action_go_sim][shell] lie-down done -> PASSIVE SHELL_IDLE", flush=True)
                elif safety_state == SafetyState.RECOVERING:
                    # safety_tick will restart/stop recovery based on IMU state.
                    pass
                elif mode == "passive":
                    shell_armed = False
            prev_mask = now_mask

        record_proc = None
        record_renderer = None
        record_camera = None
        record_body_id = -1
        next_record_frame = 0.0
        if args.record_video:
            out_path = Path(args.record_video).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            record_renderer = mujoco.Renderer(model, height=int(args.record_height), width=int(args.record_width))
            record_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(record_camera)
            record_camera.distance = 3.0
            record_camera.elevation = -10.0
            record_camera.azimuth = 180.0
            record_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            record_proc = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{int(args.record_width)}x{int(args.record_height)}",
                "-r", str(float(args.record_fps)),
                "-i", "-",
                "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                str(out_path),
            ], stdin=subprocess.PIPE)
            print(f"[action_go_sim] recording video -> {out_path}", flush=True)

        def record_frame_if_needed(now: float):
            nonlocal next_record_frame
            if record_proc is None or record_renderer is None or record_camera is None:
                return
            if now < next_record_frame:
                return
            next_record_frame = now + 1.0 / max(1.0, float(args.record_fps))
            if record_body_id >= 0:
                record_camera.lookat[:] = data.xpos[record_body_id]
            record_renderer.update_scene(data, camera=record_camera)
            frame = record_renderer.render()
            try:
                record_proc.stdin.write(frame.tobytes())
            except Exception as e:
                print(f"[action_go_sim] record write failed: {e}", flush=True)

        def close_recorder():
            if record_proc is not None:
                try:
                    if record_proc.stdin:
                        record_proc.stdin.close()
                    record_proc.wait(timeout=5.0)
                except Exception:
                    try:
                        record_proc.kill()
                    except Exception:
                        pass
            if record_renderer is not None:
                try:
                    record_renderer.close()
                except Exception:
                    pass

        def run_loop(viewer=None):
            nonlocal next_remote_tick, next_viewer_sync, pending_auto_action
            auto_arm_done = False
            auto_arm_alt_done = False
            # Releasing a prestarted action before the viewer/main loop is ready
            # leaves LowState unpublished during viewer initialization.  Start it
            # here so _wait_ready() hands directly back to the simulation loop.
            if pending_auto_action and not STOP.is_set():
                pending_auto_action = False
                if not manager.start_action(args.auto_action, source="auto"):
                    print(f"[action_go_sim][auto] failed to start action: {args.auto_action}", flush=True)
                    STOP.set()
            while not STOP.is_set():
                elapsed = time.monotonic() - run_started_at
                if args.auto_arm_sec >= 0.0 and not auto_arm_done and elapsed >= float(args.auto_arm_sec):
                    auto_arm_done = True
                    arm_shell("--auto-arm-sec", remote_cfg.shell_start_action)
                if args.auto_arm_alt_sec >= 0.0 and not auto_arm_alt_done and elapsed >= float(args.auto_arm_alt_sec):
                    auto_arm_alt_done = True
                    arm_shell("--auto-arm-alt-sec", remote_cfg.shell_start_alt_action)
                if args.auto_exit_sec > 0.0 and elapsed >= float(args.auto_exit_sec):
                    print(f"[action_go_sim][auto] exit after {args.auto_exit_sec:.2f}s", flush=True)
                    STOP.set()
                    break
                if viewer is not None and not viewer.is_running():
                    STOP.set()
                    break
                step_start = time.perf_counter()
                now = time.monotonic()
                if hasattr(_bridge, "publish_lowstate"):
                    _bridge.publish_lowstate()
                if now >= next_remote_tick:
                    next_remote_tick = now + max(0.005, float(args.remote_period_sec))
                    remote_tick()
                if not paused:
                    if manager.passive:
                        manager.apply_passive(data)
                    elif hasattr(_bridge, "apply_command"):
                        _bridge.apply_command()
                    mujoco.mj_step(model, data)
                record_frame_if_needed(now)
                if viewer is not None and now >= next_viewer_sync:
                    next_viewer_sync = now + 1.0 / max(1.0, float(args.viewer_hz))
                    viewer.sync()
                if not args.no_realtime:
                    sleep_time = float(args.sim_dt) - (time.perf_counter() - step_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        print(
            "[action_go_sim] viewer keys: SPACE pause, R reset, S ARM, "
            "B alternate ARM, T UFO tracking, G UFO stand, W WALKRUN, P PASSIVE",
            flush=True,
        )
        try:
            if args.headless:
                run_loop(None)
            else:
                if mujoco_viewer is None:
                    raise RuntimeError("mujoco.viewer is unavailable; use --headless")
                with mujoco_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                    viewer.cam.distance = 3.0
                    viewer.cam.elevation = -10.0
                    viewer.cam.azimuth = 180.0
                    viewer.cam.trackbodyid = 1
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    run_loop(viewer)
        finally:
            close_recorder()
            manager.shutdown(passive=True)
        return 0
    except KeyboardInterrupt:
        STOP.set()
        return 130
    except Exception as e:
        print(f"[action_go_sim] ERROR: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
