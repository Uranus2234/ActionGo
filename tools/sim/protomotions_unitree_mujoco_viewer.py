#!/usr/bin/env python3
"""Visual unitree_mujoco preview for the action_go ProtoMotions generic runtime.

This viewer keeps Unitree's MuJoCo bridge in-process, then launches the same
C++ ``build/generic_motion_tracking_player`` binary used by action_go.  Python is
only the visual simulator wrapper; policy inference and LowCmd publishing are
handled by the C++ player.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


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
    root = Path(__file__).resolve().parents[2]
    unitree_mujoco = os.environ.get("UNITREE_MUJOCO_ROOT", str(root.parent / "unitree_mujoco"))
    unitree_sdk2py = os.environ.get("UNITREE_SDK2PY_ROOT", str(root.parent / "unitree_sdk2_python"))
    p = argparse.ArgumentParser(description="Open MuJoCo viewer for action_go ProtoMotions C++ player")
    p.add_argument("--unitree-mujoco-root", default=unitree_mujoco)
    p.add_argument("--unitree-sdk2py-root", default=unitree_sdk2py)
    p.add_argument("--player", default=str(root / "build/generic_motion_tracking_player"))
    p.add_argument("--spec", default=str(root / "actions/tracking/protomotions/config/generic_protomotions.yaml"))
    p.add_argument("--onnx", default=str(root / "actions/tracking/protomotions/model/unified_pipeline.onnx"))
    p.add_argument("--yaml", default=str(root / "actions/tracking/protomotions/config/unified_pipeline.yaml"))
    p.add_argument("--motion", default=str(root / "actions/tracking/protomotions/motions/output_walk.50fps.npz"))
    p.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until viewer is closed or motion ends")
    p.add_argument("--max-frames", type=int, default=1000, help="Stop C++ player after this many policy/motion frames; 0 means full motion")
    p.add_argument("--sim-dt", type=float, default=0.001, help="Use 0.001 for stable G1 ProtoMotions tracking")
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--viewer-hz", type=float, default=50.0)
    p.add_argument("--print-interval", type=float, default=0.5)
    p.add_argument("--domain-id", type=int, default=1)
    p.add_argument("--interface", default="lo")
    p.add_argument("--anchor-source", choices=["fk", "imu"], default="fk")
    p.add_argument("--onnx-threads", type=int, default=2)
    p.add_argument("--startup-interp-sec", type=float, default=0.5)
    p.add_argument("--paused", action="store_true", help="Start MuJoCo viewer paused")
    p.add_argument("--no-realtime", action="store_true", help="Run simulation as fast as possible while still syncing viewer")
    return p.parse_args(argv)


def reset_to_motion_start(model, data, motion):
    data.qpos[0:3] = np.asarray(motion["body_pos"][0, 0], dtype=np.float64)
    root_q_xyzw = np.asarray(motion["body_rot"][0, 0], dtype=np.float64)
    data.qpos[3:7] = root_q_xyzw[[3, 0, 1, 2]]
    data.qpos[7:] = np.asarray(motion["dof_pos"][0], dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


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

    unitree_root = Path(args.unitree_mujoco_root).resolve()
    sim_py = unitree_root / "simulate_python"
    if args.unitree_sdk2py_root:
        sys.path.insert(0, args.unitree_sdk2py_root)
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
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py_bridge import UnitreeSdk2Bridge
    except Exception as e:
        this_python = sys.executable
        system_python = "/usr/bin/python3"
        raise RuntimeError(
            "failed to import unitree_mujoco/unitree_sdk2py modules. "
            f"Current Python is {this_python}. "
            f"On this workstation the Unitree CycloneDDS Python dependency is usually installed for {system_python}; "
            f"try: {system_python} tools/sim/protomotions_unitree_mujoco_viewer.py --motion {args.motion}"
        ) from e

    model = mujoco.MjModel.from_xml_path(cfg.ROBOT_SCENE)
    model.opt.timestep = float(args.sim_dt)
    data = mujoco.MjData(model)
    motion = np.load(args.motion, allow_pickle=False)
    reset_to_motion_start(model, data, motion)

    ChannelFactoryInitialize(int(args.domain_id), args.interface)
    UnitreeSdk2Bridge(model, data)

    player_path = Path(args.player).resolve()
    if not player_path.exists():
        raise FileNotFoundError(f"C++ player not found: {player_path}; run cmake --build build -j first")

    player_cmd = [
        str(player_path),
        "--net", args.interface,
        "--domain-id", str(int(args.domain_id)),
        "--cmd-topic", "rt/lowcmd",
        "--state-topic", "rt/lowstate",
        "--spec", str(Path(args.spec).resolve()),
        "--model", str(Path(args.onnx).resolve()),
        "--params", str(Path(args.yaml).resolve()),
        "--trajectory", str(Path(args.motion).resolve()),
        "--hz", str(int(round(args.control_hz))),
        "--publish-hz", str(int(round(1.0 / float(args.sim_dt)))),
        "--onnx-threads", str(int(args.onnx_threads)),
    ]
    if args.max_frames > 0:
        player_cmd += ["--max-frames", str(int(args.max_frames))]
    if args.duration_sec > 0:
        player_cmd += ["--duration-sec", str(float(args.duration_sec))]

    player_proc: subprocess.Popen | None = None
    paused = bool(args.paused)

    def start_player():
        print("[proto_unitree_mujoco_viewer] launching C++ player:\n  " + " ".join(player_cmd), flush=True)
        return subprocess.Popen(player_cmd)

    def key_callback(keycode):
        nonlocal paused, player_proc, sim_t0, min_z, max_abs_rp, max_ctrl
        if keycode == ord(" "):
            paused = not paused
            print(f"[proto_unitree_mujoco_viewer] paused={paused}", flush=True)
        elif keycode in (ord("R"), ord("r")):
            stop_player(player_proc)
            reset_to_motion_start(model, data, motion)
            sim_t0 = float(data.time)
            min_z = float(data.qpos[2])
            max_abs_rp = 0.0
            max_ctrl = 0.0
            player_proc = start_player()
            print("[proto_unitree_mujoco_viewer] reset and restarted C++ player", flush=True)

    viewer_period = 1.0 / max(1.0, float(args.viewer_hz))
    next_viewer_sync_wall = 0.0
    start_wall = time.perf_counter()
    sim_t0 = float(data.time)
    min_z = float(data.qpos[2])
    max_abs_rp = 0.0
    max_ctrl = 0.0
    next_print = 0.0

    print(
        f"[proto_unitree_mujoco_viewer] opening viewer\n"
        f"  xml={cfg.ROBOT_SCENE}\n"
        f"  sim_dt={model.opt.timestep} control_hz={args.control_hz}\n"
        f"  onnx={args.onnx}\n"
        f"  motion={args.motion}\n"
        f"  max_frames={args.max_frames}\n"
        f"  C++ player={player_path}\n"
        f"Controls: SPACE pause/resume, R reset/restart player, close window to stop.",
        flush=True,
    )

    try:
        player_proc = start_player()
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
                    print(f"[proto_unitree_mujoco_viewer] C++ player exited status={player_proc.returncode}", flush=True)
                    break

                if not paused:
                    mujoco.mj_step(model, data)
                    z = float(data.qpos[2])
                    min_z = min(min_z, z)
                    r, p, _ = quat_wxyz_to_rpy(data.qpos[3:7])
                    max_abs_rp = max(max_abs_rp, abs(r), abs(p))
                    max_ctrl = max(max_ctrl, float(np.max(np.abs(data.ctrl))) if data.ctrl.size else 0.0)

                    if sim_elapsed + 1e-9 >= next_print:
                        print(
                            f"[proto_unitree_mujoco_viewer] t={sim_elapsed:.2f}s z={z:.3f} "
                            f"roll={math.degrees(r):.1f} pitch={math.degrees(p):.1f} "
                            f"q=[{data.qpos[7:].min():.3f},{data.qpos[7:].max():.3f}] "
                            f"dqmax={np.max(np.abs(data.qvel[6:])):.2f} ctrlmax={np.max(np.abs(data.ctrl)):.1f}",
                            flush=True,
                        )
                        next_print += float(args.print_interval)

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
        f"[proto_unitree_mujoco_viewer] done sim_t={float(data.time)-sim_t0:.2f}s "
        f"min_z={min_z:.3f} max_abs_rp_deg={math.degrees(max_abs_rp):.1f} "
        f"max_ctrl={max_ctrl:.1f}",
        flush=True,
    )
    # unitree_sdk2py RecurrentThread/CycloneDDS and MuJoCo viewer destructors can
    # occasionally segfault on interpreter shutdown after the viewer closes.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
