#!/usr/bin/env python3
"""ROS-free Instinct G1 parkour ONNX player for action_go.

This player intentionally does not import ROS.  Robot state/command use Unitree
DDS directly (unitree_sdk2py) and depth can be either RealSense SDK or a zero
image for dry checks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import struct
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import yaml

# Python on the robot can be 3.8, where argparse.BooleanOptionalAction does not
# exist yet.  Keep the same CLI syntax (--foo / --no-foo) without requiring a
# newer system Python.
if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, type=None, choices=None, required=False, help=None, metavar=None):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith("--"):
                    _option_strings.append("--no-" + option_string[2:])
            super().__init__(
                option_strings=_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not (option_string or "").startswith("--no-"))

        def format_usage(self):
            return " | ".join(self.option_strings)

    argparse.BooleanOptionalAction = _BooleanOptionalAction

# -----------------------------------------------------------------------------
# Robot constants: copied from upstream instinct_onboard.robot_cfgs.G1_29Dof_TorsoBase.
# The bundled actions/instinct checkpoints are InstinctLab onboard exports and
# use the older shoulder-first simulation/policy order.  DDS motor_state/cmd is
# Unitree real motor order, therefore JOINT_MAP/SIGNS must always be applied.
# -----------------------------------------------------------------------------
NUM_JOINTS = 29
SIM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "waist_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "waist_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "waist_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]
JOINT_MAP = np.asarray([
    15, 22, 14, 16, 23, 13, 17, 24, 12, 18, 25,
    0, 6, 19, 26, 1, 7, 20, 27, 2, 8, 21, 28, 3, 9, 4, 10, 5, 11,
], dtype=np.int64)
JOINT_SIGNS = np.asarray([
    1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
], dtype=np.float32)
JOINT_LIMITS_HIGH = np.asarray([
    2.6704, 2.6704, 0.5200, 2.2515, 1.5882, 0.5200, 2.6180, 2.6180, 2.6180,
    2.0944, 2.0944, 2.8798, 2.8798, 1.9722, 1.9722, 2.9671, 0.5236, 1.6144,
    1.6144, 2.7576, 2.7576, 1.6144, 1.6144, 2.8798, 2.8798, 0.5236, 0.5236,
    0.2618, 0.2618,
], dtype=np.float32)
JOINT_LIMITS_LOW = np.asarray([
    -3.0892, -3.0892, -0.5200, -1.5882, -2.2515, -0.5200, -2.6180, -2.6180,
    -2.6180, -1.0472, -1.0472, -2.5307, -2.5307, -1.9722, -1.9722, -0.5236,
    -2.9671, -1.6144, -1.6144, -2.7576, -2.7576, -1.6144, -1.6144, -0.0873,
    -0.0873, -0.8727, -0.8727, -0.2618, -0.2618,
], dtype=np.float32)
TORQUE_LIMITS = np.asarray([
    25, 25, 50, 25, 25, 50, 25, 25, 88, 25, 25, 88, 88, 25, 25,
    88, 88, 5, 5, 88, 88, 5, 5, 139, 139, 50, 50, 50, 50,
], dtype=np.float32)

STOP = threading.Event()


def _signal_handler(_signum, _frame):
    STOP.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# -----------------------------------------------------------------------------
# YAML helpers. env.yaml contains python tags pointing to training modules that
# are not needed on the robot.  This loader keeps normal YAML data and turns all
# unknown tagged nodes into plain scalar/list/dict objects.
# -----------------------------------------------------------------------------
class LenientSafeLoader(yaml.SafeLoader):
    pass


def _construct_unknown(loader: LenientSafeLoader, _tag_suffix: str, node: yaml.Node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


LenientSafeLoader.add_multi_constructor("", _construct_unknown)


def load_yaml_lenient(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=LenientSafeLoader)


def apply_regex_values(pattern_values: dict[str, Any], default: float = 0.0) -> np.ndarray:
    out = np.full(NUM_JOINTS, default, dtype=np.float32)
    for pattern, value in (pattern_values or {}).items():
        for i, name in enumerate(SIM_JOINT_NAMES):
            if re.search(str(pattern), name):
                out[i] = float(value)
    return out


def apply_regex_scale(scale_cfg: Any) -> np.ndarray:
    out = np.zeros(NUM_JOINTS, dtype=np.float32)
    if isinstance(scale_cfg, dict):
        for pattern, value in scale_cfg.items():
            for i, name in enumerate(SIM_JOINT_NAMES):
                if re.search(str(pattern), name):
                    out[i] = float(value)
    else:
        out[:] = float(scale_cfg)
    return out


def get_nested(d: dict[str, Any], path: list[str], default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-6:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / n


def projected_gravity_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    # Same as quaternion.rotate_vectors(q.conjugate(), [0,0,-1]).
    w, x, y, z = quat_normalize(q)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.asarray([-r20, -r21, -r22], dtype=np.float32)


def tilt_deg_from_quat_wxyz(q: np.ndarray) -> float:
    g = projected_gravity_from_quat_wxyz(q)
    return float(np.degrees(np.arccos(np.clip(-float(g[2]), -1.0, 1.0))))


class HistoryBuffer:
    def __init__(self, length: int):
        self.length = int(length)
        self.buffer: np.ndarray | None = None
        self.num_pushes = 0

    def append(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if self.buffer is None:
            self.buffer = np.zeros((self.length,) + value.shape, dtype=np.float32)
        if self.num_pushes == 0:
            self.buffer[:] = value
        else:
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = value
        self.num_pushes += 1
        return self.buffer

    def reset(self):
        if self.buffer is not None:
            self.buffer[:] = 0.0
        self.num_pushes = 0


@dataclass
class RobotState:
    connected: bool = False
    mode_machine: int = 0
    quat: np.ndarray = field(default_factory=lambda: np.asarray([1, 0, 0, 0], dtype=np.float32))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    q_sim: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, dtype=np.float32))
    dq_sim: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, dtype=np.float32))
    wireless_remote: bytes = bytes(40)


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray([self.vx, self.vy, self.yaw], dtype=np.float32)


def unpack_float_le(data: bytes, begin: int) -> float:
    if len(data) < begin + 4:
        return 0.0
    try:
        return float(struct.unpack("<f", data[begin:begin + 4])[0])
    except Exception:
        return 0.0


def sanitize_axis(v: float) -> float:
    # Do not let stale/corrupt wireless bytes become a huge velocity command.
    if not math.isfinite(v):
        return 0.0
    return float(np.clip(v, -1.0, 1.0))


def joystick_command(wireless_remote: bytes, args: argparse.Namespace) -> Command:
    if getattr(args, "command_source", "remote") == "zero":
        return Command(0.0, 0.0, 0.0)
    # Unitree wireless_remote layout follows unitree_sdk2py.utils.joystick.Joystick.extract().
    lx_raw = sanitize_axis(unpack_float_le(wireless_remote, 4))
    rx_raw = sanitize_axis(unpack_float_le(wireless_remote, 8))
    ly_raw = sanitize_axis(unpack_float_le(wireless_remote, 20))

    ly = ly_raw
    if ly > args.lin_vel_deadband:
        vx = (ly - args.lin_vel_deadband) / max(1e-6, 1.0 - args.lin_vel_deadband)
        vx = vx * (args.lin_vel_range[1] - args.lin_vel_range[0]) + args.lin_vel_range[0]
    elif ly < -args.lin_vel_deadband:
        vx = (ly + args.lin_vel_deadband) / max(1e-6, 1.0 - args.lin_vel_deadband)
        vx = vx * (args.lin_vel_negative_range[1] - args.lin_vel_negative_range[0]) - args.lin_vel_negative_range[0]
    else:
        vx = 0.0

    lx = -lx_raw
    if lx > args.lin_vel_deadband:
        vy = (lx - args.lin_vel_deadband) / max(1e-6, 1.0 - args.lin_vel_deadband)
        vy = vy * (args.lat_vel_range[1] - args.lat_vel_range[0]) + args.lat_vel_range[0]
    elif lx < -args.lin_vel_deadband:
        vy = (lx + args.lin_vel_deadband) / max(1e-6, 1.0 - args.lin_vel_deadband)
        vy = vy * (args.lat_vel_negative_range[1] - args.lat_vel_negative_range[0]) - args.lat_vel_negative_range[0]
    else:
        vy = 0.0

    rx = -rx_raw
    if rx > args.ang_vel_deadband:
        yaw = (rx - args.ang_vel_deadband) / max(1e-6, 1.0 - args.ang_vel_deadband)
        yaw = yaw * (args.ang_vel_range[1] - args.ang_vel_range[0]) + args.ang_vel_range[0]
    elif rx < -args.ang_vel_deadband:
        yaw = (rx + args.ang_vel_deadband) / max(1e-6, 1.0 - args.ang_vel_deadband)
        yaw = yaw * (args.ang_vel_negative_range[1] - args.ang_vel_negative_range[0]) - args.ang_vel_negative_range[0]
    else:
        yaw = 0.0
    return Command(vx=float(vx), vy=float(vy), yaw=float(yaw))


class DepthSource:
    def get_observation(self) -> np.ndarray:
        raise NotImplementedError

    def stats_text(self) -> str:
        return ""

    def close(self):
        pass


class ZeroDepthSource(DepthSource):
    def __init__(self, frames: int, height: int, width: int):
        self.obs = np.zeros((frames, height, width), dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        return self.obs

    def stats_text(self) -> str:
        return "source=zero"


class RealSenseDepthSource(DepthSource):
    def __init__(self, cfg: dict[str, Any], args: argparse.Namespace):
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise RuntimeError("pyrealsense2 is required for --depth-source realsense") from e
        self.rs = rs
        self.args = args
        self.resolution = tuple(args.rs_resolution)
        self.fps = int(args.rs_fps)
        self.output_resolution = tuple(cfg["output_resolution"])
        self.crop_region = cfg.get("crop_region")
        self.depth_range = cfg["depth_range"]
        self.depth_output_range = cfg["depth_output_range"]
        self.gaussian_kernel_size = cfg.get("gaussian_kernel_size")
        self.gaussian_sigma = cfg.get("gaussian_sigma", 1.0)
        self.num_frames = int(cfg["num_output_frames"])
        self.depth_h = int(cfg["depth_height"])
        self.depth_w = int(cfg["depth_width"])
        self.sim_history_skip_frames = max(1, int(cfg.get("history_skip_frames", 5)))
        sim_update_period = float(cfg.get("sim_update_period", 0.02) or 0.02)
        sim_frequency = max(1, int(1.0 / sim_update_period))
        # Match instinct_onboard ParkourAgent: RealSense runs at 60Hz while the
        # sim camera in the export runs at camera.update_period (normally 50Hz).
        # Therefore a history_skip_frames=5 policy frame skip becomes 6 real
        # RealSense frames at 60Hz.
        self.real_history_skip_frames = max(1, int(self.fps / sim_frequency * self.sim_history_skip_frames))
        self.indices = np.linspace(-1 - self.real_history_skip_frames * (self.num_frames - 1), -1, self.num_frames).astype(int)
        self.buffer_len = max(self.fps, abs(int(self.indices[0])) + 2)
        self.buffer = deque(maxlen=self.buffer_len)
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.last_raw_stats = (0.0, 0.0, 0.0, 1.0)  # min, max, mean, zero fraction
        self.last_proc_stats = (0.0, 0.0, 0.0)      # min, max, mean after policy preprocessing
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
        self.profile = self.pipeline.start(rs_cfg)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        # Prime camera and history.
        _ = self.pipeline.wait_for_frames(1000)
        self.thread = threading.Thread(target=self._thread_main, name="realsense-depth", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with self.lock:
                if self.buffer:
                    break
            time.sleep(0.01)
        print(
            f"[instinct_parkour] RealSense started resolution={self.resolution} fps={self.fps} "
            f"sim_skip={self.sim_history_skip_frames} real_skip={self.real_history_skip_frames} "
            f"depth_indices={self.indices.tolist()}",
            flush=True,
        )

    def _preprocess(self, depth_m: np.ndarray) -> np.ndarray:
        depth = cv2.resize(depth_m.astype(np.float32), self.output_resolution, interpolation=cv2.INTER_NEAREST)
        if self.crop_region is not None:
            x1, x2, y1, y2 = [int(x) for x in self.crop_region]
            h, w = depth.shape
            depth = depth[x1:h - x2, y1:w - y2]
        mask = (depth < 0.2).astype(np.uint8)
        if mask.any():
            depth = cv2.inpaint(depth.astype(np.float32), mask, 3, cv2.INPAINT_NS)
        if self.gaussian_kernel_size is not None:
            depth = cv2.GaussianBlur(depth, self.gaussian_kernel_size, self.gaussian_sigma, self.gaussian_sigma)
        lo, hi = self.depth_range
        out_lo, out_hi = self.depth_output_range
        filt = np.clip(depth, lo, hi)
        norm = (filt - lo) / max(1e-6, hi - lo)
        out = norm * (out_hi - out_lo) + out_lo
        return out.astype(np.float32)

    def _thread_main(self):
        while not self.stop.is_set() and not STOP.is_set():
            try:
                frames = self.pipeline.wait_for_frames(max(1, int(2000 / max(1, self.fps))))
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_m = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * float(self.depth_scale)
                valid = depth_m > 0.0
                if valid.any():
                    raw_valid = depth_m[valid]
                    raw_stats = (
                        float(raw_valid.min()),
                        float(raw_valid.max()),
                        float(raw_valid.mean()),
                        float(1.0 - valid.mean()),
                    )
                else:
                    raw_stats = (0.0, 0.0, 0.0, 1.0)
                frame = self._preprocess(depth_m)
                if frame.shape != (self.depth_h, self.depth_w):
                    raise RuntimeError(f"depth frame shape {frame.shape} != {(self.depth_h, self.depth_w)}")
                proc_stats = (float(frame.min()), float(frame.max()), float(frame.mean()))
                with self.lock:
                    self.frame_count += 1
                    self.last_frame_time = time.monotonic()
                    self.last_raw_stats = raw_stats
                    self.last_proc_stats = proc_stats
                    if not self.buffer:
                        for _ in range(self.buffer_len):
                            self.buffer.append(frame.copy())
                    else:
                        self.buffer.append(frame)
            except Exception as e:
                print(f"[instinct_parkour] RealSense warning: {e}", file=sys.stderr, flush=True)
                time.sleep(0.05)

    def stats_text(self) -> str:
        with self.lock:
            n = self.frame_count
            t = self.last_frame_time
            raw_min, raw_max, raw_mean, zero_frac = self.last_raw_stats
            proc_min, proc_max, proc_mean = self.last_proc_stats
            buffered = len(self.buffer)
        age_ms = (time.monotonic() - t) * 1000.0 if t > 0.0 else float("inf")
        return (
            f"rs_frames={n} buffered={buffered} age={age_ms:.0f}ms "
            f"raw_m=[{raw_min:.2f},{raw_max:.2f}] raw_mean={raw_mean:.2f} zero={zero_frac:.2f} "
            f"proc=[{proc_min:.2f},{proc_max:.2f}] proc_mean={proc_mean:.2f}"
        )

    def get_observation(self) -> np.ndarray:
        with self.lock:
            if not self.buffer:
                return np.zeros((self.num_frames, self.depth_h, self.depth_w), dtype=np.float32)
            frames = list(self.buffer)
        out = []
        for idx in self.indices:
            pos = len(frames) + int(idx)
            pos = max(0, min(len(frames) - 1, pos))
            out.append(frames[pos])
        return np.stack(out, axis=0).astype(np.float32)

    def close(self):
        self.stop.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass


class InstinctParkourPolicy:
    def __init__(self, logdir: Path, args: argparse.Namespace):
        self.logdir = logdir
        self.args = args
        # action_go layout uses model/ + config/.  Keep compatibility with
        # older exported training directories that used exported/ + params/.
        self.exported = logdir / "model" if (logdir / "model").exists() else logdir / "exported"
        self.params = logdir / "config" if (logdir / "config").exists() else logdir / "params"
        metadata_path = self.exported / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.env_cfg = load_yaml_lenient(self.params / "env.yaml")
        self.agent_cfg = load_yaml_lenient(self.params / "agent.yaml") if (self.params / "agent.yaml").exists() else {}
        if self.agent_cfg is None:
            self.agent_cfg = {}
        self.policy_format = OrderedDict(self.metadata.get("obs_format", {}).get("policy", {}))

        self._parse_config()
        if not self.policy_format:
            self.policy_format = self._infer_policy_format_from_env()
        self._load_normalizer()
        self._load_models()
        self._init_histories()
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.step_count = 0

    def _parse_config(self):
        robot = get_nested(self.env_cfg, ["scene", "entities", "robot"], None)
        if robot is None:
            robot = get_nested(self.env_cfg, ["scene", "robot"], {})
        init_state = robot.get("init_state", {})
        self.default_joint_pos = apply_regex_values(init_state.get("joint_pos", {}), default=0.0)
        self.default_joint_vel = apply_regex_values(init_state.get("joint_vel", {".*": 0.0}), default=0.0)

        action_cfg = get_nested(self.env_cfg, ["actions", "joint_pos"], {})
        self.action_scale = apply_regex_scale(action_cfg.get("scale", 1.0))
        use_default_offset = bool(action_cfg.get("use_default_offset", True))
        if use_default_offset:
            self.action_offset = self.default_joint_pos.copy()
        else:
            self.action_offset = apply_regex_values(action_cfg.get("offset", {}), default=float(action_cfg.get("offset", 0.0)))

        self.kp = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.kd = np.zeros(NUM_JOINTS, dtype=np.float32)
        actuators = get_nested(robot, ["articulation", "actuators"], None)
        if actuators is None:
            actuators = robot.get("actuators", [])
        actuator_iter = actuators.values() if isinstance(actuators, dict) else (actuators or [])

        def _patterns(x: Any) -> list[str]:
            if x is None:
                return []
            if isinstance(x, (list, tuple)):
                return [str(v) for v in x]
            return [str(x)]

        def _regex_value(value_cfg: Any, joint_name: str, default: float) -> float:
            if isinstance(value_cfg, dict):
                out = default
                for pattern, value in value_cfg.items():
                    if re.search(str(pattern), joint_name):
                        out = float(value)
                return out
            if value_cfg is None:
                return default
            return float(value_cfg)

        for actuator in actuator_iter:
            cfg = actuator.get("base_cfg", actuator)
            patterns = _patterns(
                cfg.get(
                    "target_names_expr",
                    actuator.get("target_names_expr", cfg.get("joint_names_expr", actuator.get("joint_names_expr", []))),
                )
            )
            stiffness = cfg.get("stiffness", actuator.get("stiffness", 0.0))
            damping = cfg.get("damping", actuator.get("damping", 0.0))
            for i, name in enumerate(SIM_JOINT_NAMES):
                if any(re.search(str(p), name) for p in patterns):
                    self.kp[i] = _regex_value(stiffness, name, self.kp[i])
                    self.kd[i] = _regex_value(damping, name, self.kd[i])
        # Match instinct_onboard defaults if the config format changes unexpectedly.
        if not np.any(self.kp):
            self.kp[:] = 40.0
            self.kd[:] = 1.0
        self.kp = np.clip(self.kp * float(self.args.kp_factor), 0.0, 500.0).astype(np.float32)
        self.kd = np.clip(self.kd * float(self.args.kd_factor), 0.0, 5.0).astype(np.float32)

        # Observation term config.
        terms = get_nested(self.env_cfg, ["observations", "policy", "terms"], None)
        if terms is None:
            terms = get_nested(self.env_cfg, ["observations", "policy"], {})
        self.obs_scales: dict[str, float] = {}
        self.obs_history_len: dict[str, int] = {}
        for name, cfg in terms.items():
            if not isinstance(cfg, dict):
                continue
            if cfg.get("scale") is not None:
                self.obs_scales[name] = float(cfg["scale"])
            if int(cfg.get("history_length", 0) or 0) > 0:
                self.obs_history_len[name] = int(cfg["history_length"])
        depth_term = terms.get("depth_image", {}) if isinstance(terms, dict) else {}
        depth_params = depth_term.get("params", {}) if isinstance(depth_term, dict) else {}

        # Camera config may be stored as scene.camera in newer exported models,
        # or as a named sensor in scene.sensors in older exported models.
        camera = get_nested(self.env_cfg, ["scene", "camera"], None)
        if camera is None:
            for sensor in get_nested(self.env_cfg, ["scene", "sensors"], []) or []:
                if isinstance(sensor, dict) and sensor.get("name") == "camera":
                    camera = sensor
                    break
        if camera is None:
            raise RuntimeError("camera config not found in env.yaml (expected scene.camera or scene.sensors[name=camera])")
        pattern = camera.get("pattern", camera.get("pattern_cfg", {}))
        noise_pipeline = camera.get("noise_pipeline", {})
        crop_region = noise_pipeline.get("crop_and_resize", {}).get("crop_region")
        depth_norm = noise_pipeline.get("depth_normalization", {})
        camera_update_period = float(camera.get("update_period", 0.02) or 0.02)
        data_histories = camera.get("data_histories", {}) if isinstance(camera, dict) else {}
        camera_history_length = data_histories.get("distance_to_image_plane_noised")
        depth_range = [float(x) for x in depth_norm.get("depth_range", [0.0, 2.5])]
        output_range = [float(x) for x in depth_norm.get("output_range", [0.0, 1.0])]
        if not bool(depth_norm.get("normalize", True)):
            output_range = depth_range
        out_w, out_h = int(pattern.get("width", 64)), int(pattern.get("height", 36))
        if crop_region is not None:
            x1, x2, y1, y2 = [int(x) for x in crop_region]
            depth_h = out_h - x1 - x2
            depth_w = out_w - y1 - y2
        else:
            depth_h, depth_w = out_h, out_w
        blur_cfg = noise_pipeline.get("gaussian_blur", {})
        kernel_size = blur_cfg.get("kernel_size")
        history_skip_frames = int(depth_params.get("history_skip_frames", depth_params.get("time_downsample_factor", 5)))
        if camera_history_length is not None:
            num_output_frames = int((int(camera_history_length) - 1) / max(1, history_skip_frames) + 1)
        else:
            num_output_frames = int(depth_params.get("num_output_frames", 8))
        self.depth_cfg = {
            "output_resolution": (out_w, out_h),
            "crop_region": crop_region,
            "depth_range": depth_range,
            "depth_output_range": output_range,
            "gaussian_kernel_size": (int(kernel_size), int(kernel_size)) if kernel_size else None,
            "gaussian_sigma": float(blur_cfg.get("sigma", 1.0)),
            "history_skip_frames": int(history_skip_frames),
            "num_output_frames": int(num_output_frames),
            "depth_height": int(depth_h),
            "depth_width": int(depth_w),
            "sim_update_period": float(camera_update_period),
            "camera_history_length": int(camera_history_length) if camera_history_length is not None else None,
        }

    def _infer_policy_format_from_env(self) -> OrderedDict[str, list[int]]:
        terms = get_nested(self.env_cfg, ["observations", "policy", "terms"], None)
        if terms is None:
            terms = get_nested(self.env_cfg, ["observations", "policy"], {})
        fmt: OrderedDict[str, list[int]] = OrderedDict()
        for name, cfg in terms.items():
            if not isinstance(cfg, dict) or "func" not in cfg:
                continue
            hist = int(cfg.get("history_length", 1) or 1)
            if name in ("base_ang_vel", "projected_gravity", "velocity_commands"):
                fmt[name] = [3 * hist]
            elif name in ("joint_pos", "joint_pos_rel", "joint_vel", "actions"):
                fmt[name] = [NUM_JOINTS * hist]
            elif name == "depth_image":
                fmt[name] = [
                    int(self.depth_cfg["num_output_frames"]),
                    int(self.depth_cfg["depth_height"]),
                    int(self.depth_cfg["depth_width"]),
                ]
        required = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions", "depth_image")
        missing = [name for name in required if name not in fmt]
        if missing:
            raise RuntimeError(f"cannot infer policy obs format from env.yaml; missing terms: {missing}")
        return fmt

    @staticmethod
    def _resolve_onnx_shape(shape: list[Any], defaults: list[int]) -> list[int]:
        out: list[int] = []
        for i, dim in enumerate(shape):
            default = int(defaults[i]) if i < len(defaults) else 1
            if isinstance(dim, int) and dim > 0:
                out.append(int(dim))
            else:
                out.append(default)
        return out

    def _load_normalizer(self):
        path = self.exported / "policy_normalizer.npz"
        raw_dim = sum(int(np.prod(shape)) for shape in self.policy_format.values())
        if not path.exists():
            if bool(self.args.use_normalizer):
                raise RuntimeError(f"--use-normalizer requested but {path} does not exist")
            self.norm_mean = np.zeros(raw_dim, dtype=np.float32)
            self.norm_std = np.ones(raw_dim, dtype=np.float32)
            self.norm_eps = 1e-8
            self.normalizer_source = "none"
            return
        data = np.load(path, allow_pickle=True)
        self.norm_mean = data["mean"].astype(np.float32).reshape(-1)
        self.norm_std = data["std"].astype(np.float32).reshape(-1)
        self.norm_eps = float(np.asarray(data["eps"]).reshape(()))
        self.normalizer_source = str(path)
        if self.norm_mean.size != raw_dim or self.norm_std.size != raw_dim:
            raise RuntimeError(f"normalizer dim mismatch mean={self.norm_mean.size} std={self.norm_std.size} raw_obs={raw_dim}")

    def _load_models(self):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.intra_op_num_threads = max(1, int(self.args.onnx_threads))
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.depth_sess = ort.InferenceSession(str(self.exported / "0-depth_encoder.onnx"), sess_options=options, providers=providers)
        self.actor_sess = ort.InferenceSession(str(self.exported / "actor.onnx"), sess_options=options, providers=providers)
        self.depth_input_name = self.depth_sess.get_inputs()[0].name
        self.actor_input_name = self.actor_sess.get_inputs()[0].name
        raw_dim = sum(int(np.prod(shape)) for shape in self.policy_format.values())
        depth_size = int(np.prod(self.policy_format["depth_image"]))
        self.depth_shape = self._resolve_onnx_shape(
            list(self.depth_sess.get_inputs()[0].shape),
            [1, 8, int(self.depth_cfg["depth_height"]), int(self.depth_cfg["depth_width"])],
        )
        self.depth_output_shape = self._resolve_onnx_shape(list(self.depth_sess.get_outputs()[0].shape), [self.depth_shape[0], 128])
        expected_actor_input = raw_dim - depth_size + int(self.depth_output_shape[-1])
        self.actor_shape = self._resolve_onnx_shape(list(self.actor_sess.get_inputs()[0].shape), [self.depth_shape[0], expected_actor_input])
        self.actor_output_shape = self._resolve_onnx_shape(list(self.actor_sess.get_outputs()[0].shape), [self.actor_shape[0], NUM_JOINTS])
        if self.depth_shape[1:] != [8, self.depth_cfg["depth_height"], self.depth_cfg["depth_width"]]:
            raise RuntimeError(f"depth encoder shape {self.depth_shape} does not match parsed depth cfg {self.depth_cfg}")
        if self.actor_output_shape[-1] != NUM_JOINTS:
            raise RuntimeError(f"actor output dim {self.actor_output_shape} != {NUM_JOINTS}")
        self.batch = int(self.actor_shape[0])
        if self.batch != int(self.depth_shape[0]):
            raise RuntimeError(f"actor batch {self.actor_shape[0]} != depth batch {self.depth_shape[0]}")
        if self.actor_shape[-1] != expected_actor_input:
            raise RuntimeError(f"actor input dim {self.actor_shape[-1]} != inferred proprio+depth_latent {expected_actor_input}")

    def _init_histories(self):
        self.histories = {name: HistoryBuffer(length) for name, length in self.obs_history_len.items()}

    def check(self):
        raw_dim = sum(int(np.prod(shape)) for shape in self.policy_format.values())
        proprio_dim = raw_dim - int(np.prod(self.policy_format["depth_image"]))
        print("[instinct_parkour] check-only OK")
        print(f"  logdir: {self.logdir}")
        print(
            f"  raw_obs_dim={raw_dim} proprio_dim={proprio_dim} "
            f"normalizer_dim={self.norm_mean.size} normalizer_source={self.normalizer_source} "
            f"use_normalizer={self.args.use_normalizer}"
        )
        print(f"  depth_encoder input={self.depth_shape} output={self.depth_output_shape}")
        print(f"  actor input={self.actor_shape} output={self.actor_output_shape}")
        print(f"  joint_order=InstinctLab shoulder-first first_joints={SIM_JOINT_NAMES[:3]}")
        print(f"  default_joint_pos: min={self.default_joint_pos.min():.3f} max={self.default_joint_pos.max():.3f}")
        print(f"  action_scale: min={self.action_scale.min():.3f} max={self.action_scale.max():.3f}")
        print(f"  kp: min={self.kp.min():.3f} max={self.kp.max():.3f}; kd: min={self.kd.min():.3f} max={self.kd.max():.3f}")
        print(f"  depth_cfg: {self.depth_cfg}")

    def _history_term(self, name: str, value: np.ndarray) -> np.ndarray:
        if name not in self.histories:
            return value.astype(np.float32).reshape(-1)
        return self.histories[name].append(value).astype(np.float32).reshape(-1)

    def build_raw_obs(self, state: RobotState, command: Command, depth_obs: np.ndarray) -> np.ndarray:
        base_ang_vel = state.gyro.astype(np.float32) * float(self.obs_scales.get("base_ang_vel", 1.0))
        projected_gravity = projected_gravity_from_quat_wxyz(state.quat)
        velocity_commands = command.as_array()
        joint_pos = state.q_sim.astype(np.float32) - self.default_joint_pos
        joint_vel = (state.dq_sim.astype(np.float32) - self.default_joint_vel) * float(self.obs_scales.get("joint_vel", 1.0))
        actions = self.last_action.astype(np.float32)
        terms = {
            "base_ang_vel": self._history_term("base_ang_vel", base_ang_vel),
            "projected_gravity": self._history_term("projected_gravity", projected_gravity),
            "velocity_commands": self._history_term("velocity_commands", velocity_commands),
            "joint_pos": self._history_term("joint_pos", joint_pos),
            "joint_vel": self._history_term("joint_vel", joint_vel),
            "actions": self._history_term("actions", actions),
            "joint_pos_rel": self._history_term("joint_pos_rel", joint_pos),
            "joint_vel_rel": self._history_term("joint_vel_rel", joint_vel),
            "last_action": self._history_term("last_action", actions),
            "depth_image": depth_obs.astype(np.float32).reshape(-1),
        }
        obs = np.concatenate([terms[name] for name in self.policy_format.keys()], axis=0).astype(np.float32)
        if obs.size != self.norm_mean.size:
            raise RuntimeError(f"obs dim mismatch {obs.size} != normalizer {self.norm_mean.size}")
        return obs

    def infer_action(self, state: RobotState, command: Command, depth_obs: np.ndarray) -> np.ndarray:
        raw = self.build_raw_obs(state, command, depth_obs)
        depth_size = int(np.prod(self.policy_format["depth_image"]))
        # Match instinct_onboard ParkourAgent: depth_encoder receives the
        # already camera-normalized depth image, and actor receives raw
        # proprioception + depth latent.  The exported policy_normalizer.npz is
        # kept only as an optional compatibility/debug path.  Using it by
        # default makes the real robot behave like the observation dimensions
        # are wrong.
        if bool(self.args.use_normalizer):
            obs = ((raw - self.norm_mean) / (self.norm_std + self.norm_eps)).astype(np.float32)
        else:
            obs = raw.astype(np.float32)
        proprio = obs[:-depth_size]
        depth = obs[-depth_size:].reshape(1, *self.policy_format["depth_image"]).astype(np.float32)
        proprio_batch = np.repeat(proprio.reshape(1, -1), self.batch, axis=0).astype(np.float32)
        depth_batch = np.repeat(depth, self.batch, axis=0).astype(np.float32)
        depth_latent = self.depth_sess.run(None, {self.depth_input_name: depth_batch})[0]
        actor_input = np.concatenate([proprio_batch, depth_latent.astype(np.float32)], axis=1)
        if actor_input.shape != tuple(self.actor_shape):
            raise RuntimeError(f"actor_input shape {actor_input.shape} != {self.actor_shape}")
        action_batch = self.actor_sess.run(None, {self.actor_input_name: actor_input})[0]
        action = action_batch[0].astype(np.float32).reshape(NUM_JOINTS)
        self.last_action = action
        self.step_count += 1
        return action

    def action_to_target(self, action: np.ndarray, state: RobotState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(NUM_JOINTS)
        target = self.action_offset + action * self.action_scale
        target = np.clip(target, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        if bool(self.args.clip_by_torque):
            p = np.maximum(self.kp, 1e-6)
            low = (-TORQUE_LIMITS + self.kd * state.dq_sim) / p + state.q_sim
            high = (TORQUE_LIMITS + self.kd * state.dq_sim) / p + state.q_sim
            target = np.clip(target, low, high)
        return target.astype(np.float32), self.kp.copy(), self.kd.copy()


class UnitreeDDS:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.state = RobotState()
        self._import_unitree(args.unitree_sdk2py_root)
        self.ChannelFactoryInitialize(0, args.net if args.net else None)
        self.publish_enabled = not bool(getattr(args, "no_publish", False))
        self.pub = None
        if self.publish_enabled:
            self.pub = self.ChannelPublisher(args.cmd_topic, self.LowCmd_)
            self.pub.Init()
        self.sub = self.ChannelSubscriber(args.state_topic, self.LowState_)
        self.sub.Init(self._lowstate_handler, 1)
        self.crc = self.CRC()
        self.low_cmd = self.unitree_hg_msg_dds__LowCmd_()

    def _import_unitree(self, extra_root: str | None):
        candidates = []
        if extra_root:
            candidates.append(extra_root)
        if os.environ.get("UNITREE_SDK2PY_ROOT"):
            candidates.append(os.environ["UNITREE_SDK2PY_ROOT"])
        here = Path(__file__).resolve()
        candidates.extend([
            str(here.parents[2] / "thirdparty" / "unitree_sdk2_python"),
            str(Path.home() / "unitree_sdk2_python"),
        ])
        for c in candidates:
            p = Path(c)
            if (p / "unitree_sdk2py").exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC
        except Exception as e:
            raise RuntimeError(
                "failed to import unitree_sdk2py. Install it or set UNITREE_SDK2PY_ROOT / --unitree-sdk2py-root. "
                f"Tried: {candidates}"
            ) from e
        self.ChannelFactoryInitialize = ChannelFactoryInitialize
        self.ChannelPublisher = ChannelPublisher
        self.ChannelSubscriber = ChannelSubscriber
        self.unitree_hg_msg_dds__LowCmd_ = unitree_hg_msg_dds__LowCmd_
        self.LowCmd_ = LowCmd_
        self.LowState_ = LowState_
        self.CRC = CRC

    def _lowstate_handler(self, msg):
        q = np.zeros(NUM_JOINTS, dtype=np.float32)
        dq = np.zeros(NUM_JOINTS, dtype=np.float32)
        for sim_idx, real_idx in enumerate(JOINT_MAP):
            q[sim_idx] = float(msg.motor_state[int(real_idx)].q) * float(JOINT_SIGNS[sim_idx])
            dq[sim_idx] = float(msg.motor_state[int(real_idx)].dq) * float(JOINT_SIGNS[sim_idx])
        quat = np.asarray(list(msg.imu_state.quaternion), dtype=np.float32)
        gyro = np.asarray(list(msg.imu_state.gyroscope), dtype=np.float32)
        remote = bytes(int(x) & 0xFF for x in list(msg.wireless_remote))
        with self.lock:
            self.state = RobotState(
                connected=True,
                mode_machine=int(msg.mode_machine),
                quat=quat,
                gyro=gyro,
                q_sim=q,
                dq_sim=dq,
                wireless_remote=remote,
            )

    def wait_for_state(self, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not STOP.is_set():
            with self.lock:
                if self.state.connected:
                    print("[instinct_parkour] lowstate connected", flush=True)
                    return
            time.sleep(0.01)
        raise RuntimeError("timeout waiting for rt/lowstate")

    def snapshot(self) -> RobotState:
        with self.lock:
            s = self.state
            return RobotState(
                connected=s.connected,
                mode_machine=s.mode_machine,
                quat=s.quat.copy(), gyro=s.gyro.copy(),
                q_sim=s.q_sim.copy(), dq_sim=s.dq_sim.copy(), wireless_remote=s.wireless_remote,
            )

    def publish(self, target_sim: np.ndarray, kp_sim: np.ndarray, kd_sim: np.ndarray):
        if not self.publish_enabled or self.pub is None:
            return
        cmd = self.low_cmd
        cmd.mode_pr = 0
        state = self.snapshot()
        cmd.mode_machine = int(state.mode_machine if state.mode_machine > 0 else self.args.mode_machine)
        for sim_idx, real_idx in enumerate(JOINT_MAP):
            mc = cmd.motor_cmd[int(real_idx)]
            mc.mode = 1
            mc.q = float(target_sim[sim_idx] * JOINT_SIGNS[sim_idx])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(kp_sim[sim_idx])
            mc.kd = float(kd_sim[sim_idx])
        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)

    def publish_damping_for_stop(self, seconds: float = 0.12):
        if not self.publish_enabled or self.pub is None:
            return
        state = self.snapshot()
        if not state.connected:
            return
        end = time.monotonic() + seconds
        cmd = self.low_cmd
        cmd.mode_pr = 0
        cmd.mode_machine = int(state.mode_machine if state.mode_machine > 0 else self.args.mode_machine)
        while time.monotonic() < end:
            for sim_idx, real_idx in enumerate(JOINT_MAP):
                mc = cmd.motor_cmd[int(real_idx)]
                mc.mode = 1
                mc.q = float(state.q_sim[sim_idx] * JOINT_SIGNS[sim_idx])
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 0.0
                mc.kd = 3.0
            cmd.crc = self.crc.Crc(cmd)
            self.pub.Write(cmd)
            time.sleep(0.005)


class ParkourRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.policy = InstinctParkourPolicy(Path(args.logdir).resolve(), args)
        # Do not initialize RealSense in __init__: action_go has already switched
        # to UserCtrl before launching this process.  In real_run() we first
        # create DDS and publish a hold-current-pose command, then start the
        # camera while the publish loop keeps the robot alive.
        self.depth: DepthSource | None = None
        self.dds: UnitreeDDS | None = None
        self.target_lock = threading.Lock()
        self.latest_target: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.target_seq = 0
        self.filtered_target: np.ndarray | None = None
        self.filtered_target_time = 0.0
        self.last_target_raw_delta_max = 0.0
        self.last_target_limited_delta_max = 0.0
        self.run_stats: dict[str, float | int | bool] = {}
        self.inference_thread: threading.Thread | None = None
        self.publish_thread: threading.Thread | None = None

    def init_depth(self):
        if self.depth is not None:
            return
        # check-only validates files/shapes and must not touch hardware.
        if self.args.check_only or self.args.depth_source == "zero":
            self.depth = ZeroDepthSource(
                self.policy.depth_cfg["num_output_frames"],
                self.policy.depth_cfg["depth_height"],
                self.policy.depth_cfg["depth_width"],
            )
        elif self.args.depth_source == "realsense":
            self.depth = RealSenseDepthSource(self.policy.depth_cfg, self.args)
        else:
            raise RuntimeError(f"unsupported depth source: {self.args.depth_source}")

    def close(self):
        if self.depth is not None:
            self.depth.close()

    def fake_state(self) -> RobotState:
        return RobotState(
            connected=True,
            mode_machine=self.args.mode_machine,
            quat=np.asarray([1, 0, 0, 0], dtype=np.float32),
            gyro=np.zeros(3, dtype=np.float32),
            q_sim=self.policy.default_joint_pos.copy(),
            dq_sim=np.zeros(NUM_JOINTS, dtype=np.float32),
            wireless_remote=bytes(40),
        )

    def _check_joint_protection(self, state: RobotState):
        mid = (JOINT_LIMITS_HIGH + JOINT_LIMITS_LOW) * 0.5
        half = (JOINT_LIMITS_HIGH - JOINT_LIMITS_LOW) * 0.5
        hi = mid + half * float(self.args.joint_pos_protect_ratio)
        lo = mid - half * float(self.args.joint_pos_protect_ratio)
        bad = np.where((state.q_sim > hi) | (state.q_sim < lo))[0]
        if bad.size:
            names = ", ".join(f"{int(i)}:{SIM_JOINT_NAMES[int(i)]}={state.q_sim[int(i)]:.3f}" for i in bad[:6])
            raise RuntimeError(f"joint position protection triggered: {names}")

    def set_latest(self, target: np.ndarray, kp: np.ndarray, kd: np.ndarray):
        with self.target_lock:
            self.latest_target = (target.copy(), kp.copy(), kd.copy())
            self.target_seq += 1

    def get_latest(self):
        with self.target_lock:
            if self.latest_target is None:
                return None, self.target_seq
            t, kp, kd = self.latest_target
            return (t.copy(), kp.copy(), kd.copy()), self.target_seq

    def reset_target_filter(self, target: np.ndarray):
        self.filtered_target = np.asarray(target, dtype=np.float32).reshape(NUM_JOINTS).copy()
        self.filtered_target_time = time.monotonic()
        self.last_target_raw_delta_max = 0.0
        self.last_target_limited_delta_max = 0.0

    def limit_target_rate(self, target: np.ndarray, initial_target: np.ndarray | None = None) -> np.ndarray:
        """Optionally limit policy target jumps before they reach the 500 Hz publisher.

        The official ROS pipeline transitions through a stand agent before parkour.  In
        action_go we may start parkour directly after a pose ramp, so the first policy
        target can otherwise jump by several tenths of a radian in one 20 ms control
        frame.  This limiter is disabled by default and enabled from motion_actions.yaml
        for safer real-robot bring-up.
        """
        target = np.asarray(target, dtype=np.float32).reshape(NUM_JOINTS)
        rate = float(getattr(self.args, "target_rate_limit", 0.0))
        if rate <= 1e-6:
            self.last_target_raw_delta_max = 0.0
            self.last_target_limited_delta_max = 0.0
            return target
        now = time.monotonic()
        if self.filtered_target is None:
            self.reset_target_filter(target if initial_target is None else initial_target)

        # Use a bounded wall-clock dt.  The lower bound avoids a near-zero
        # allowance on the first inference tick after startup; the upper bound
        # avoids one scheduler hiccup creating a very large step.
        nominal = 1.0 / max(1, int(self.args.hz))
        dt = float(np.clip(now - self.filtered_target_time, 0.5 * nominal, 0.1))
        max_step = rate * dt
        raw_delta = target - self.filtered_target
        limited_delta = np.clip(raw_delta, -max_step, max_step).astype(np.float32)
        self.last_target_raw_delta_max = float(np.max(np.abs(raw_delta)))
        self.last_target_limited_delta_max = float(np.max(np.abs(limited_delta)))
        self.filtered_target = (self.filtered_target + limited_delta).astype(np.float32)
        self.filtered_target_time = now
        return self.filtered_target.copy()

    def reset_run_stats(self):
        self.run_stats = {
            "count": 0,
            "nonfinite": False,
            "max_action_abs": 0.0,
            "max_target_err": 0.0,
            "max_dq": 0.0,
            "max_tilt": 0.0,
        }

    def update_run_stats(self, state: RobotState, action: np.ndarray, target: np.ndarray, depth_obs: np.ndarray):
        if not self.run_stats:
            self.reset_run_stats()
        self.run_stats["count"] = int(self.run_stats.get("count", 0)) + 1
        nonfinite = (
            not np.all(np.isfinite(action))
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(depth_obs))
            or not np.all(np.isfinite(state.q_sim))
            or not np.all(np.isfinite(state.dq_sim))
        )
        self.run_stats["nonfinite"] = bool(self.run_stats.get("nonfinite", False) or nonfinite)
        self.run_stats["max_action_abs"] = max(
            float(self.run_stats.get("max_action_abs", 0.0)),
            float(np.max(np.abs(action))),
        )
        self.run_stats["max_target_err"] = max(
            float(self.run_stats.get("max_target_err", 0.0)),
            float(np.max(np.abs(target - state.q_sim))),
        )
        self.run_stats["max_dq"] = max(
            float(self.run_stats.get("max_dq", 0.0)),
            float(np.max(np.abs(state.dq_sim))),
        )
        self.run_stats["max_tilt"] = max(
            float(self.run_stats.get("max_tilt", 0.0)),
            tilt_deg_from_quat_wxyz(state.quat),
        )

    def check_preflight_sanity(self):
        stats = self.run_stats
        failures = []
        count = int(stats.get("count", 0))
        if count <= 0:
            failures.append("no inference samples")
        if bool(stats.get("nonfinite", False)):
            failures.append("non-finite value in state/depth/action/target")
        if float(stats.get("max_tilt", 0.0)) > float(self.args.max_preflight_tilt_deg):
            failures.append(
                f"tilt {float(stats.get('max_tilt', 0.0)):.1f}deg > {self.args.max_preflight_tilt_deg:.1f}deg"
            )
        if float(stats.get("max_dq", 0.0)) > float(self.args.max_preflight_dq):
            failures.append(
                f"dq {float(stats.get('max_dq', 0.0)):.2f} > {self.args.max_preflight_dq:.2f}"
            )
        if float(stats.get("max_action_abs", 0.0)) > float(self.args.max_preflight_action_abs):
            failures.append(
                f"action_abs {float(stats.get('max_action_abs', 0.0)):.2f} > {self.args.max_preflight_action_abs:.2f}"
            )
        if float(stats.get("max_target_err", 0.0)) > float(self.args.max_preflight_target_err):
            failures.append(
                f"target_err {float(stats.get('max_target_err', 0.0)):.2f} > {self.args.max_preflight_target_err:.2f}"
            )
        if self.args.depth_source == "realsense":
            frame_count = int(getattr(self.depth, "frame_count", 0)) if self.depth is not None else 0
            if frame_count < int(self.args.min_preflight_rs_frames):
                failures.append(f"RealSense frames {frame_count} < {self.args.min_preflight_rs_frames}")
        print(
            "[instinct_parkour] preflight summary "
            f"samples={count} max_tilt={float(stats.get('max_tilt', 0.0)):.1f}deg "
            f"max_dq={float(stats.get('max_dq', 0.0)):.3f} "
            f"max_action_abs={float(stats.get('max_action_abs', 0.0)):.3f} "
            f"max_target_err={float(stats.get('max_target_err', 0.0)):.3f}",
            flush=True,
        )
        if failures:
            raise RuntimeError("preflight sanity failed: " + "; ".join(failures))

    def inference_loop(self):
        if self.depth is None:
            raise RuntimeError("depth source is not initialized")
        self.reset_run_stats()
        period = 1.0 / max(1, int(self.args.hz))
        next_t = time.monotonic()
        loop_start = next_t
        count = 0
        avg_ms = 0.0
        max_ms = 0.0
        while not STOP.is_set():
            t0 = time.perf_counter()
            state = self.dds.snapshot() if self.dds is not None else self.fake_state()
            self._check_joint_protection(state)
            command = joystick_command(state.wireless_remote, self.args)
            depth_obs = self.depth.get_observation()
            action = self.policy.infer_action(state, command, depth_obs)
            target, kp, kd = self.policy.action_to_target(action, state)
            target = self.limit_target_rate(target, state.q_sim)
            self.update_run_stats(state, action, target, depth_obs)
            self.set_latest(target, kp, kd)
            ms = (time.perf_counter() - t0) * 1000.0
            count += 1
            avg_ms += (ms - avg_ms) / count
            max_ms = max(max_ms, ms)
            if count % int(self.args.stats_interval) == 0:
                depth_min = float(depth_obs.min())
                depth_max = float(depth_obs.max())
                depth_mean = float(depth_obs.mean())
                src_stats = self.depth.stats_text() if self.depth is not None else ""
                q_err = state.q_sim - self.policy.default_joint_pos
                target_err = target - state.q_sim
                rate_text = ""
                if float(getattr(self.args, "target_rate_limit", 0.0)) > 1e-6:
                    rate_text = (
                        f" target_delta_raw={self.last_target_raw_delta_max:.3f}"
                        f" target_delta_limited={self.last_target_limited_delta_max:.3f}"
                    )
                print(
                    f"[instinct_parkour] infer avg={avg_ms:.3f}ms max={max_ms:.3f}ms "
                    f"action=[{action.min():.3f},{action.max():.3f}] "
                    f"cmd=({command.vx:.2f},{command.vy:.2f},{command.yaw:.2f}) "
                    f"tilt={tilt_deg_from_quat_wxyz(state.quat):.1f}deg "
                    f"qerr_max={float(np.max(np.abs(q_err))):.3f} dq_max={float(np.max(np.abs(state.dq_sim))):.3f} "
                    f"target_err_max={float(np.max(np.abs(target_err))):.3f}{rate_text} "
                    f"depth_obs=[{depth_min:.2f},{depth_max:.2f}] depth_mean={depth_mean:.2f} {src_stats}",
                    flush=True,
                )
                max_ms = 0.0
            if (
                self.args.duration_sec > 0
                and (self.dds is None or bool(getattr(self.args, "dds_dry_run", False)))
                and time.monotonic() - loop_start >= self.args.duration_sec
            ):
                STOP.set()
                break
            next_t += period
            now = time.monotonic()
            if next_t < now:
                next_t = now
            time.sleep(max(0.0, next_t - now))

    def dry_run(self):
        self.init_depth()
        print(f"[instinct_parkour] dry-run depth_source={self.args.depth_source} hz={self.args.hz} duration={self.args.duration_sec} (no DDS, no LowCmd publish)", flush=True)
        if self.args.duration_sec <= 0:
            self.args.duration_sec = 2.0
        self.inference_loop()
        print("[instinct_parkour] dry-run completed", flush=True)

    def publish_loop(self):
        if self.dds is None:
            raise RuntimeError("DDS is not initialized")
        publish_hz = max(int(self.args.publish_hz), int(self.args.hz))
        period = 1.0 / publish_hz
        next_t = time.monotonic()
        latest, seq = self.get_latest()
        if latest is not None:
            q, kp, kd = latest
        else:
            state = self.dds.snapshot()
            q = state.q_sim.copy()
            kp = np.ones(NUM_JOINTS, dtype=np.float32) * 20.0
            kd = np.ones(NUM_JOINTS, dtype=np.float32) * 1.0
        last_seq = seq
        interp_from = q.copy()
        interp_to = q.copy()
        interp_start = time.monotonic()
        print(f"[instinct_parkour] publish loop cmd_topic={self.args.cmd_topic} state_topic={self.args.state_topic} publish_hz={publish_hz}", flush=True)
        while not STOP.is_set():
            latest, seq = self.get_latest()
            if latest is not None and seq != last_seq:
                target, kp, kd = latest
                interp_from = q.copy()
                interp_to = target.copy()
                interp_start = time.monotonic()
                last_seq = seq
            if self.args.target_interp_sec > 1e-6:
                a = np.clip((time.monotonic() - interp_start) / self.args.target_interp_sec, 0.0, 1.0)
                q = (1.0 - a) * interp_from + a * interp_to
            else:
                q = interp_to
            self.dds.publish(q, kp, kd)
            next_t += period
            now = time.monotonic()
            if next_t < now:
                next_t = now
            time.sleep(max(0.0, next_t - now))

    def ramp_to_pose(self, target: np.ndarray, seconds: float, hold_sec: float = 0.0):
        if self.dds is None:
            raise RuntimeError("DDS is not initialized")
        target = np.asarray(target, dtype=np.float32).reshape(NUM_JOINTS)
        kp = self.policy.kp.copy()
        kd = self.policy.kd.copy()
        start = self.dds.snapshot().q_sim.copy()
        seconds = max(0.0, float(seconds))
        if seconds > 1e-6:
            steps = max(2, int(seconds * 100))
            print(f"[instinct_parkour] startup ramp to default pose over {seconds:.2f}s", flush=True)
            for i in range(steps):
                if STOP.is_set():
                    return
                a = (i + 1) / steps
                q = ((1.0 - a) * start + a * target).astype(np.float32)
                self.set_latest(q, kp, kd)
                time.sleep(seconds / steps)
        self.set_latest(target, kp, kd)
        if hold_sec > 1e-6:
            print(f"[instinct_parkour] startup hold default pose for {hold_sec:.2f}s", flush=True)
            end = time.monotonic() + hold_sec
            while time.monotonic() < end and not STOP.is_set():
                self.set_latest(target, kp, kd)
                time.sleep(0.02)
        self.reset_target_filter(target)

    def dds_dry_run(self):
        self.args.no_publish = True
        self.dds = UnitreeDDS(self.args)
        self.dds.wait_for_state()
        self.init_depth()
        if self.args.duration_sec <= 0:
            self.args.duration_sec = 5.0
        print(
            f"[instinct_parkour] DDS dry-run depth_source={self.args.depth_source} "
            f"hz={self.args.hz} duration={self.args.duration_sec} command_source={self.args.command_source} (subscribes lowstate/camera; no LowCmd publish)",
            flush=True,
        )
        self.inference_loop()
        if bool(getattr(self.args, "fail_on_bad_input", False)):
            self.check_preflight_sanity()
        print("[instinct_parkour] DDS dry-run completed", flush=True)

    def real_run(self):
        self.args.no_publish = False
        self.dds = UnitreeDDS(self.args)
        self.dds.wait_for_state()
        state = self.dds.snapshot()
        hold = state.q_sim.copy()
        self.set_latest(
            hold,
            np.ones(NUM_JOINTS, dtype=np.float32) * 20.0,
            np.ones(NUM_JOINTS, dtype=np.float32) * 1.0,
        )
        self.reset_target_filter(hold)
        self.publish_thread = threading.Thread(target=self.publish_loop, name="parkour-publish", daemon=True)
        self.publish_thread.start()
        try:
            print("[instinct_parkour] holding current pose while initializing depth source", flush=True)
            self.init_depth()
            self.ramp_to_pose(
                self.policy.default_joint_pos,
                float(self.args.startup_sec),
                float(self.args.startup_hold_sec),
            )
            self.inference_thread = threading.Thread(target=self.inference_loop, name="parkour-inference", daemon=True)
            self.inference_thread.start()
            print(
                f"[instinct_parkour] start real DDS depth_source={self.args.depth_source} "
                f"hz={self.args.hz} duration={self.args.duration_sec} command_source={self.args.command_source} "
                f"use_normalizer={self.args.use_normalizer} target_rate_limit={self.args.target_rate_limit}",
                flush=True,
            )
            start = time.monotonic()
            while not STOP.is_set():
                if self.args.duration_sec > 0 and time.monotonic() - start >= self.args.duration_sec:
                    break
                time.sleep(0.05)
        finally:
            STOP.set()
            if self.inference_thread is not None:
                self.inference_thread.join(timeout=1.0)
            if self.publish_thread is not None:
                self.publish_thread.join(timeout=1.0)
            if not self.args.no_damping_on_stop and self.dds is not None:
                print("[instinct_parkour] stop requested, publishing damping packet", flush=True)
                self.dds.publish_damping_for_stop()
            print("[instinct_parkour] completed", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ROS-free action_go player for Instinct G1 parkour ONNX")
    p.add_argument("--logdir", required=True, help="Run directory containing model/ and config/ (or legacy exported/ and params/)")
    p.add_argument("--net", default="", help="Unitree DDS network interface, e.g. enP8p1s0")
    p.add_argument("--cmd-topic", default="rt/user_lowcmd")
    p.add_argument("--state-topic", default="rt/lowstate")
    p.add_argument("--unitree-sdk2py-root", default=None)
    p.add_argument("--depth-source", choices=["zero", "realsense"], default="realsense")
    p.add_argument("--rs-resolution", nargs=2, type=int, default=[480, 270], metavar=("W", "H"))
    p.add_argument("--rs-fps", type=int, default=60)
    p.add_argument("--hz", type=int, default=50, help="ONNX inference/control update frequency")
    p.add_argument("--publish-hz", type=int, default=50, help="LowCmd publish frequency")
    p.add_argument("--onnx-threads", type=int, default=2)
    p.add_argument("--target-interp-sec", type=float, default=0.02)
    p.add_argument("--target-rate-limit", type=float, default=0.0, help="rad/s per-joint target limiter after inference; 0 disables it")
    p.add_argument("--startup-sec", type=float, default=2.0, help="real run only: ramp from current pose to policy default pose before inference")
    p.add_argument("--startup-hold-sec", type=float, default=0.5, help="real run only: hold policy default pose before starting inference")
    p.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until action_go terminates the process")
    p.add_argument("--mode-machine", type=int, default=5)
    p.add_argument("--command-source", choices=["remote", "zero"], default="remote", help="velocity command source; zero ignores wireless stick axes")
    p.add_argument("--kp-factor", type=float, default=1.0)
    p.add_argument("--kd-factor", type=float, default=1.0)
    p.add_argument("--joint-pos-protect-ratio", type=float, default=2.0)
    p.add_argument("--clip-by-torque", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-normalizer", action=argparse.BooleanOptionalAction, default=False, help="Compatibility/debug only; official instinct_onboard parkour uses raw proprio/depth, so default is false")
    p.add_argument("--no-damping-on-stop", action="store_true", help="Skip soft damping packet on SIGTERM; action_go will switch back to official mode")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Run ONNX with fake robot state and no DDS publish")
    p.add_argument("--dds-dry-run", action="store_true", help="Subscribe real lowstate + camera and run ONNX, but do not publish LowCmd or switch UserCtrl")
    p.add_argument("--fail-on-bad-input", action="store_true", help="For --dds-dry-run preflight: exit non-zero if state/camera/action sanity checks fail")
    p.add_argument("--max-preflight-tilt-deg", type=float, default=20.0)
    p.add_argument("--max-preflight-dq", type=float, default=2.0)
    p.add_argument("--max-preflight-action-abs", type=float, default=5.0)
    p.add_argument("--max-preflight-target-err", type=float, default=2.0)
    p.add_argument("--min-preflight-rs-frames", type=int, default=10)
    p.add_argument("--camera-check", action="store_true", help="Alias for --dry-run --depth-source realsense; opens camera and prints depth stats without DDS/LowCmd")
    p.add_argument("--stats-interval", type=int, default=100)
    p.add_argument("--lin-vel-deadband", type=float, default=0.5)
    p.add_argument("--lin-vel-range", nargs=2, type=float, default=[0.5, 0.5])
    p.add_argument("--lin-vel-negative-range", nargs=2, type=float, default=[0.0, 0.0])
    p.add_argument("--lat-vel-range", nargs=2, type=float, default=[0.0, 0.0])
    p.add_argument("--lat-vel-negative-range", nargs=2, type=float, default=[0.0, 0.0])
    p.add_argument("--ang-vel-deadband", type=float, default=0.5)
    p.add_argument("--ang-vel-range", nargs=2, type=float, default=[0.0, 1.0])
    p.add_argument("--ang-vel-negative-range", nargs=2, type=float, default=[0.0, 1.0])
    args = p.parse_args(argv)
    if args.camera_check:
        args.dry_run = True
        args.depth_source = "realsense"
        if args.duration_sec <= 0.0:
            args.duration_sec = 5.0
        if args.stats_interval == 100:
            args.stats_interval = 25
    if args.dds_dry_run:
        args.no_publish = True
        if args.duration_sec <= 0.0:
            args.duration_sec = 5.0
        if args.stats_interval == 100:
            args.stats_interval = 25
    else:
        args.no_publish = False
    return args


def main(argv=None):
    args = parse_args(argv)
    runtime = ParkourRuntime(args)
    try:
        if args.check_only:
            runtime.policy.check()
            return 0
        if args.dry_run:
            runtime.policy.check()
            runtime.dry_run()
            return 0
        if args.dds_dry_run:
            runtime.policy.check()
            runtime.dds_dry_run()
            return 0
        runtime.real_run()
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[instinct_parkour] ERROR: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)
