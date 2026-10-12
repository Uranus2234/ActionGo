#!/usr/bin/env python3
"""Visual Unitree MuJoCo preview for the action_go OmniXtreme composite player.

The viewer keeps Unitree's MuJoCo DDS bridge in-process, then launches the same
``build/generic_composite_policy_player`` binary used by action_go.  Python is only the visual
simulator wrapper; OmniXtreme ONNX inference and LowCmd publishing are handled
by the C++ player.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
import types
import threading
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")

import mujoco
import mujoco.viewer
import numpy as np
import yaml


ACTION_GO_ROOT = Path(__file__).resolve().parents[2]
PERM = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28], dtype=np.int64)
OMNI_KP = np.array([99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012, 99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012, 40.1792, 28.5012, 28.5012, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783], dtype=np.float64)
OMNI_KD = np.array([6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144, 6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144, 2.5579, 1.8144, 1.8144, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681], dtype=np.float64)


def quat_wxyz_to_rpy(q):
    w, x, y, z = [float(v) for v in q]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


def parse_args(argv=None):
    unitree_mujoco = os.environ.get("UNITREE_MUJOCO_ROOT", str(ACTION_GO_ROOT.parent / "unitree_mujoco"))
    unitree_sdk2py = os.environ.get("UNITREE_SDK2PY_ROOT", str(ACTION_GO_ROOT.parent / "unitree_sdk2_python"))
    omnixtreme_root = Path(os.environ.get("OMNIXTREME_ROOT", str(ACTION_GO_ROOT.parent / "OmniXtreme")))
    p = argparse.ArgumentParser(description="Open MuJoCo viewer for action_go OmniXtreme composite player")
    p.add_argument("--bridge", choices=["native", "unitree"], default="native",
                   help="native uses OmniXtreme/no_hand.xml with a minimal DDS bridge; unitree uses unitree_mujoco's bridge")
    p.add_argument("--unitree-mujoco-root", default=unitree_mujoco)
    p.add_argument("--unitree-sdk2py-root", default=unitree_sdk2py)
    p.add_argument("--xml", default=str(omnixtreme_root / "robots/g1/no_hand.xml"),
                   help="MuJoCo XML for --bridge native")
    p.add_argument("--player", default=str(ACTION_GO_ROOT / "build/generic_composite_policy_player"))
    p.add_argument("--action", default="",
                   help="OmniXtreme action name from actions/omnixtreme/*.yaml, e.g. omni_motion_default or omni_seg04_wbst_stand")
    p.add_argument("--action-yaml", default=str(ACTION_GO_ROOT / "actions/omnixtreme/omnixtreme.yaml"),
                   help="Root OmniXtreme action YAML used by --action")
    p.add_argument("--base-model", "--model", dest="base_model", default=str(ACTION_GO_ROOT / "actions/omnixtreme/model/base_policy_trt.onnx"))
    p.add_argument("--residual-model", default=str(ACTION_GO_ROOT / "actions/omnixtreme/model/residual_policy.onnx"))
    p.add_argument("--fk-model", default=str(ACTION_GO_ROOT / "actions/omnixtreme/model/fk_trt.onnx"))
    p.add_argument("--spec", default=str(ACTION_GO_ROOT / "actions/omnixtreme/config/generic_omnixtreme.yaml"))
    p.add_argument("--params", default=str(ACTION_GO_ROOT / "actions/omnixtreme/config/OmniXtreme.yaml"))
    p.add_argument("--motion", default=str(ACTION_GO_ROOT / "actions/omnixtreme/motions/motion.npz"))
    p.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until viewer is closed/player exits/motion ends")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop C++ player after N policy frames; default 500, or action max_frames when --action is used; 0 means full motion")
    p.add_argument("--motion-start-frame", type=int, default=0)
    p.add_argument("--sim-dt", type=float, default=0.004)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--publish-hz", type=float, default=0.0, help="0 means 1/sim-dt")
    p.add_argument("--viewer-hz", type=float, default=50.0)
    p.add_argument("--print-interval", type=float, default=0.5)
    p.add_argument("--domain-id", type=int, default=1)
    p.add_argument("--interface", default="lo")
    p.add_argument("--onnx-threads", type=int, default=2)
    p.add_argument("--startup-interp-sec", type=float, default=0.0)
    p.add_argument("--target-interp-sec", type=float, default=0.0)
    p.add_argument("--deterministic", action="store_true", help="Use zero initial_noise in base policy")
    p.add_argument("--initial-noise-scale", type=float, default=None)
    p.add_argument("--keep-motion-xy", action="store_true", help="Reset root x/y to motion x/y instead of origin")
    p.add_argument("--root-z", type=float, default=0.79, help="Root z at reset; <=0 uses motion root z")
    p.add_argument("--bootstrap-hold-sec", type=float, default=0.5, help="Python-side PD hold before the C++ DDS player finishes loading")
    p.add_argument("--paused", action="store_true")
    p.add_argument("--headless", action="store_true", help="Run bridge/player without opening viewer")
    p.add_argument("--no-realtime", action="store_true")
    return p.parse_args(argv)


def _iter_action_includes(value):
    if value is None:
        return []
    if isinstance(value, dict):
        return [v for v in value.values() if v]
    if isinstance(value, (list, tuple)):
        return [v for v in value if v]
    return [value]


def _abs_no_symlink(path: Path) -> str:
    # Keep symlink paths stable in logs/commands; do not resolve to the local
    # Keep an external OmniXtreme checkout target out of generated commands.
    return str(Path(os.path.abspath(os.path.expanduser(str(path)))))


def _resolve_action_path(base_dir: Path, value: str) -> str:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return _abs_no_symlink(p)
    from_base = base_dir / p
    if from_base.exists():
        return _abs_no_symlink(from_base)
    from_root = ACTION_GO_ROOT / p
    if from_root.exists() or str(p).startswith("actions/"):
        return _abs_no_symlink(from_root)
    return _abs_no_symlink(from_base)


def load_omni_action(action_name: str, root_yaml: str):
    """Return (action_dict, action_yaml_dir) for an OmniXtreme action name."""
    root = Path(root_yaml).expanduser().resolve()
    visited = set()

    def visit(path: Path):
        path = path.resolve()
        if path in visited:
            return None
        visited.add(path)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}

        action = (((data.get("actions") or {}).get("omnixtreme") or {}).get(action_name))
        if action is not None:
            return action, path.parent

        for inc in _iter_action_includes(data.get("action_files") or data.get("actions_files")
                                         or data.get("include_actions") or data.get("includes")):
            inc_path = Path(str(inc)).expanduser()
            if not inc_path.is_absolute():
                inc_path = path.parent / inc_path
            found = visit(inc_path)
            if found is not None:
                return found
        return None

    found = visit(root)
    if found is None:
        raise KeyError(f"action '{action_name}' not found from {root}")
    return found


def apply_action_args(args):
    if not args.action:
        if args.max_frames is None:
            args.max_frames = 500
        return

    action, base_dir = load_omni_action(args.action, args.action_yaml)
    if action.get("model"):
        args.base_model = _resolve_action_path(base_dir, action["model"])
    if action.get("params"):
        args.params = _resolve_action_path(base_dir, action["params"])
    if action.get("trajectory"):
        args.motion = _resolve_action_path(base_dir, action["trajectory"])
    if args.max_frames is None and action.get("max_frames") is not None:
        args.max_frames = int(action["max_frames"])
    if args.max_frames is None:
        args.max_frames = 500


def reset_to_motion_start(model, data, motion, frame: int, keep_motion_xy: bool, root_z: float):
    frame = int(np.clip(frame, 0, motion["joint_pos"].shape[0] - 1))
    root_pos = np.asarray(motion["body_pos_w"][frame, 0], dtype=np.float64).copy()
    if not keep_motion_xy:
        root_pos[0:2] = 0.0
    if root_z > 0.0:
        root_pos[2] = float(root_z)
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = np.asarray(motion["body_quat_w"][frame, 0], dtype=np.float64)
    data.qpos[7:7 + 29] = np.asarray(motion["joint_pos"][frame, PERM], dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


class NativeOmniDDSBridge:
    """Minimal DDS bridge for OmniXtreme's original MuJoCo XML.

    unitree_mujoco's stock bridge requires its own XML/sensor layout.  For
    OmniXtreme parity we publish LowState directly from qpos/qvel and apply
    rt/lowcmd to data.ctrl with the same PD equation as Unitree's bridge.
    """

    def __init__(self, model, data):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default

        self.model = model
        self.data = data
        self.LowState_default = LowState_default
        self.low_state = LowState_default()
        self.low_state.mode_machine = 5
        self.lock = threading.Lock()
        self.cmd_q = np.asarray(data.qpos[7:36], dtype=np.float64).copy()
        self.cmd_dq = np.zeros(29, dtype=np.float64)
        self.cmd_kp = np.zeros(29, dtype=np.float64)
        self.cmd_kd = np.zeros(29, dtype=np.float64)
        self.cmd_tau = np.zeros(29, dtype=np.float64)
        self.has_cmd = False
        self.cmd_sequence = 0
        self.low_state_pub = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_pub.Init()
        self.low_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_sub.Init(self.low_cmd_handler, 10)

    def low_cmd_handler(self, msg):
        with self.lock:
            for i in range(29):
                self.cmd_q[i] = float(msg.motor_cmd[i].q)
                self.cmd_dq[i] = float(msg.motor_cmd[i].dq)
                self.cmd_kp[i] = float(msg.motor_cmd[i].kp)
                self.cmd_kd[i] = float(msg.motor_cmd[i].kd)
                self.cmd_tau[i] = float(msg.motor_cmd[i].tau)
            self.has_cmd = True
            self.cmd_sequence += 1

    def command_sequence(self) -> int:
        with self.lock:
            return int(self.cmd_sequence)

    def wait_for_command_after(self, seq: int, timeout_sec: float = 0.2) -> bool:
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            with self.lock:
                if self.cmd_sequence != seq:
                    return True
            time.sleep(0.0001)
        return False

    def apply_cmd(self):
        with self.lock:
            if not self.has_cmd:
                return
            q = np.asarray(self.data.qpos[7:36], dtype=np.float64)
            dq = np.asarray(self.data.qvel[6:35], dtype=np.float64)
            self.data.ctrl[:29] = self.cmd_tau + self.cmd_kp * (self.cmd_q - q) + self.cmd_kd * (self.cmd_dq - dq)

    def publish_lowstate(self):
        s = self.low_state
        s.mode_machine = 5
        for i in range(29):
            s.motor_state[i].q = float(self.data.qpos[7 + i])
            s.motor_state[i].dq = float(self.data.qvel[6 + i])
            s.motor_state[i].tau_est = float(self.data.ctrl[i]) if i < self.data.ctrl.size else 0.0
        # MuJoCo stores free-joint quaternion as wxyz.  deploy_mujoco.py reads
        # qvel[3:6] for angular velocity, so publish the same signal here.
        for i in range(4):
            s.imu_state.quaternion[i] = float(self.data.qpos[3 + i])
        for i in range(3):
            s.imu_state.gyroscope[i] = float(self.data.qvel[3 + i])
        self.low_state_pub.Write(s)


def stop_player(proc: subprocess.Popen | None):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)


def main(argv=None):
    args = parse_args(argv)
    apply_action_args(args)

    if args.unitree_sdk2py_root:
        sys.path.insert(0, args.unitree_sdk2py_root)

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except Exception as e:
        raise RuntimeError(
            "failed to import unitree_sdk2py modules. "
            f"Current Python is {sys.executable}; on this workstation try /usr/bin/python3."
        ) from e

    unitree_root = Path(args.unitree_mujoco_root).resolve()
    if args.bridge == "unitree":
        sim_py = unitree_root / "simulate_python"
        sys.path.insert(0, str(sim_py))
        cfg = types.ModuleType("config")
        cfg.ROBOT = "g1"
        cfg.ROBOT_SCENE = str(unitree_root / "unitree_robots/g1/scene_29dof.xml")
        cfg.DOMAIN_ID = int(args.domain_id)
        cfg.INTERFACE = args.interface
        cfg.USE_JOYSTICK = 0
        cfg.JOYSTICK_TYPE = "xbox"
        cfg.JOYSTICK_DEVICE = 0
        cfg.PRINT_SCENE_INFORMATION = False
        cfg.ENABLE_ELASTIC_BAND = False
        cfg.SIMULATE_DT = float(args.sim_dt)
        cfg.VIEWER_DT = 1.0 / max(1.0, float(args.viewer_hz))
        sys.modules["config"] = cfg
        try:
            from unitree_sdk2py_bridge import UnitreeSdk2Bridge
        except Exception as e:
            raise RuntimeError(
                "failed to import unitree_mujoco bridge. "
                f"Current Python is {sys.executable}; on this workstation try /usr/bin/python3."
            ) from e
        xml_path = cfg.ROBOT_SCENE
    else:
        UnitreeSdk2Bridge = None
        xml_path = str(Path(args.xml).resolve())

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = float(args.sim_dt)
    data = mujoco.MjData(model)
    motion = np.load(args.motion, allow_pickle=False)
    reset_to_motion_start(model, data, motion, args.motion_start_frame, args.keep_motion_xy, args.root_z)

    ChannelFactoryInitialize(int(args.domain_id), args.interface)
    native_bridge = None
    if args.bridge == "unitree":
        UnitreeSdk2Bridge(model, data)
    else:
        native_bridge = NativeOmniDDSBridge(model, data)

    player_path = Path(args.player).resolve()
    if not player_path.exists():
        raise FileNotFoundError(f"C++ player not found: {player_path}; run cmake --build build -j first")

    publish_hz = int(round(args.publish_hz if args.publish_hz > 0 else 1.0 / float(args.sim_dt)))
    player_cmd = [
        str(player_path),
        "--net", args.interface,
        "--domain-id", str(int(args.domain_id)),
        "--cmd-topic", "rt/lowcmd",
        "--state-topic", "rt/lowstate",
        "--spec", str(Path(args.spec).resolve()),
        "--model", str(Path(args.base_model).resolve()),
        "--residual-model", str(Path(args.residual_model).resolve()),
        "--fk-model", str(Path(args.fk_model).resolve()),
        "--params", str(Path(args.params).resolve()),
        "--trajectory", str(Path(args.motion).resolve()),
        "--hz", str(int(round(args.control_hz))),
        "--publish-hz", str(publish_hz),
        "--onnx-threads", str(int(args.onnx_threads)),
        "--motion-start-frame", str(int(args.motion_start_frame)),
        "--startup-interp-sec", str(float(args.startup_interp_sec)),
        "--target-interp-sec", str(float(args.target_interp_sec)),
        "--sync-lowstate",
        # Match official deploy_mujoco.py's actuator loss after the dynamic
        # torque-speed envelope. This is simulation modeling, not a hardware
        # recommendation to compensate physical motor friction via tau_ff.
        "--friction-feedforward",
    ]
    if args.max_frames > 0:
        player_cmd += ["--max-frames", str(int(args.max_frames))]
    if args.duration_sec > 0:
        player_cmd += ["--duration-sec", str(float(args.duration_sec))]
    if args.deterministic:
        player_cmd += ["--deterministic"]
    if args.initial_noise_scale is not None:
        player_cmd += ["--initial-noise-scale", str(float(args.initial_noise_scale))]

    player_proc: subprocess.Popen | None = None
    paused = bool(args.paused)

    def start_player():
        print("[omnixtreme_mujoco_viewer] launching C++ player:\n  " + " ".join(player_cmd), flush=True)
        return subprocess.Popen(player_cmd)

    def key_callback(keycode):
        nonlocal paused, player_proc, sim_t0, min_z, max_abs_rp, max_ctrl
        if keycode == ord(" "):
            paused = not paused
            print(f"[omnixtreme_mujoco_viewer] paused={paused}", flush=True)
        elif keycode in (ord("R"), ord("r")):
            stop_player(player_proc)
            reset_to_motion_start(model, data, motion, args.motion_start_frame, args.keep_motion_xy, args.root_z)
            sim_t0 = float(data.time)
            min_z = float(data.qpos[2])
            max_abs_rp = 0.0
            max_ctrl = 0.0
            player_proc = start_player()
            print("[omnixtreme_mujoco_viewer] reset and restarted C++ player", flush=True)

    viewer_period = 1.0 / max(1.0, float(args.viewer_hz))
    next_viewer_sync_wall = 0.0
    start_wall = time.perf_counter()
    sim_t0 = float(data.time)
    min_z = float(data.qpos[2])
    max_abs_rp = 0.0
    max_ctrl = 0.0
    next_print = 0.0
    bootstrap_q = data.qpos[7:7 + 29].copy()

    print(
        f"[omnixtreme_mujoco_viewer] bridge={args.bridge} xml={xml_path}\n"
        f"  sim_dt={model.opt.timestep} control_hz={args.control_hz} publish_hz={publish_hz}\n"
        f"  base={args.base_model}\n"
        f"  residual={args.residual_model}\n"
        f"  fk={args.fk_model}\n"
        f"  action={args.action or '(manual paths)'}\n"
        f"  motion={args.motion}\n"
        f"  max_frames={args.max_frames} bootstrap_hold_sec={args.bootstrap_hold_sec}\n"
        f"  C++ player={player_path}\n"
        f"Controls: SPACE pause/resume, R reset/restart player, close window to stop.",
        flush=True,
    )

    def sim_step_and_log(viewer=None):
        nonlocal next_print, min_z, max_abs_rp, max_ctrl
        sim_elapsed_pre = float(data.time) - sim_t0
        cmd_seq_before = None
        if native_bridge is not None:
            cmd_seq_before = native_bridge.command_sequence()
            native_bridge.publish_lowstate()
            if not native_bridge.has_cmd:
                # Do not advance the physics while the C++ player is loading
                # ONNX.  The first OmniXtreme pose is a motion-tracking state,
                # not a passive statically balanced hold pose.
                native_bridge.wait_for_command_after(cmd_seq_before, timeout_sec=0.2)
                if not native_bridge.has_cmd:
                    return
        need_bootstrap = (
            args.bootstrap_hold_sec > 1e-9
            and native_bridge is None
            and sim_elapsed_pre < float(args.bootstrap_hold_sec)
        )
        if need_bootstrap:
            # The C++ player loads a ~100 MB base ONNX before it can publish the
            # prestart/current-pose hold.  Keep the robot upright during that
            # startup window; after this, all commands come from the composite player.
            q = np.asarray(data.sensordata[:29], dtype=np.float64)
            dq = np.asarray(data.sensordata[29:58], dtype=np.float64)
            if q.shape[0] < 29 or dq.shape[0] < 29:
                q = np.asarray(data.qpos[7:36], dtype=np.float64)
                dq = np.asarray(data.qvel[6:35], dtype=np.float64)
            data.ctrl[:29] = OMNI_KP * (bootstrap_q - q) - OMNI_KD * dq
        elif native_bridge is not None:
            # Lock-step simulation: pause MuJoCo until the composite player has
            # consumed the freshly published LowState and returned a LowCmd.
            # This removes the 1-2 physics-step command latency that otherwise
            # destabilizes the high-dynamic segment after ~500 frames.
            native_bridge.wait_for_command_after(cmd_seq_before, timeout_sec=0.2)
            native_bridge.apply_cmd()
        mujoco.mj_step(model, data)
        sim_elapsed = float(data.time) - sim_t0
        z = float(data.qpos[2])
        min_z = min(min_z, z)
        r, p, _ = quat_wxyz_to_rpy(data.qpos[3:7])
        max_abs_rp = max(max_abs_rp, abs(r), abs(p))
        max_ctrl = max(max_ctrl, float(np.max(np.abs(data.ctrl))) if data.ctrl.size else 0.0)
        if sim_elapsed + 1e-9 >= next_print:
            print(
                f"[omnixtreme_mujoco_viewer] t={sim_elapsed:.2f}s z={z:.3f} "
                f"roll={math.degrees(r):.1f} pitch={math.degrees(p):.1f} "
                f"q=[{data.qpos[7:].min():.3f},{data.qpos[7:].max():.3f}] "
                f"dqmax={np.max(np.abs(data.qvel[6:])):.2f} ctrlmax={np.max(np.abs(data.ctrl)):.1f}",
                flush=True,
            )
            next_print += float(args.print_interval)

    try:
        player_proc = start_player()
        if args.headless:
            while True:
                sim_elapsed = float(data.time) - sim_t0
                if args.duration_sec > 0 and sim_elapsed >= float(args.duration_sec):
                    break
                if player_proc is not None and player_proc.poll() is not None:
                    print(f"[omnixtreme_mujoco_viewer] C++ player exited status={player_proc.returncode}", flush=True)
                    break
                if not paused:
                    sim_step_and_log()
                if not args.no_realtime:
                    elapsed_wall = time.perf_counter() - start_wall
                    desired_wall = float(data.time) - sim_t0
                    sleep = desired_wall - elapsed_wall
                    if sleep > 0:
                        time.sleep(min(sleep, model.opt.timestep))
                else:
                    time.sleep(0.0005)
        else:
            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                viewer.cam.distance = 3.0
                viewer.cam.elevation = -10.0
                viewer.cam.azimuth = 180.0
                viewer.cam.trackbodyid = 1
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

                while viewer.is_running():
                    sim_elapsed = float(data.time) - sim_t0
                    if args.duration_sec > 0 and sim_elapsed >= float(args.duration_sec):
                        break
                    if player_proc is not None and player_proc.poll() is not None:
                        print(f"[omnixtreme_mujoco_viewer] C++ player exited status={player_proc.returncode}", flush=True)
                        break
                    if not paused:
                        sim_step_and_log(viewer)

                    wall_now = time.perf_counter()
                    if wall_now >= next_viewer_sync_wall:
                        viewer.sync()
                        next_viewer_sync_wall = wall_now + viewer_period

                    if not args.no_realtime:
                        elapsed_wall = time.perf_counter() - start_wall
                        desired_wall = float(data.time) - sim_t0
                        sleep = desired_wall - elapsed_wall
                        if sleep > 0:
                            time.sleep(min(sleep, model.opt.timestep))
                    else:
                        time.sleep(0.0005)
    finally:
        stop_player(player_proc)

    print(
        f"[omnixtreme_mujoco_viewer] done sim_t={float(data.time)-sim_t0:.2f}s "
        f"min_z={min_z:.3f} max_abs_rp_deg={math.degrees(max_abs_rp):.1f} max_ctrl={max_ctrl:.1f}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
