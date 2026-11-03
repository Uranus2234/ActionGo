#!/usr/bin/env python3
"""Run the ActionGo C++ SuperMimic player against native MuJoCo over loopback DDS."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import secrets
import signal
import subprocess
import sys
import threading
import time


ACTION_GO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = ACTION_GO_ROOT.parent / "ActionGo-local"
LOCKSTEP_MAGIC = 0x41474C53


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=ACTION_GO_ROOT.parent / "SuperMimic-MuJoCo")
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--clip")
    parser.add_argument("--package", type=Path, default=LOCAL_ROOT / "supermimic")
    parser.add_argument("--player", type=Path, default=LOCAL_ROOT / "build/generic_motion_tracking_player")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=0.0, help="Simulated seconds; zero runs until stopped")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--domain-id", type=int, help="Dedicated DDS domain, 1 through 200; random by default")
    parser.add_argument("--sdk-root", type=Path, help="Directory containing unitree_sdk2py")
    parser.add_argument("--python-deps", type=Path, default=LOCAL_ROOT / "python")
    parser.add_argument("--reference", choices=("beside", "ghost", "off"), default="beside")
    parser.add_argument("--gpu", action="store_true", help="Render native viewer using VirtualGL/NVIDIA EGL")
    parser.add_argument("--vglrun", type=Path)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--no-realtime", action="store_true", help="Headless lockstep as fast as the player permits")
    parser.add_argument("--trace", action="store_true", help="Record full C++ actor inputs/outputs and DDS trace for offline parity validation")
    parser.add_argument("--command-timeout-sec", type=float, default=10.0)
    parser.add_argument("--control-file", type=Path, help="Optional append-only JSONL commands: pause, resume, reset, set_root_pose")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_sec) or args.duration_sec < 0:
        parser.error("--duration-sec must be finite and nonnegative")
    for key in ("fps", "command_timeout_sec"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.domain_id is not None and not 1 <= args.domain_id <= 200:
        parser.error("--domain-id must be a dedicated nonzero domain in [1, 200]")
    return args


def configure_graphics(args):
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
        return
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ["PYOPENGL_PLATFORM"] = "glx"
    if not args.gpu or os.environ.get("ACTION_GO_SUPERMIMIC_VGL") == "1":
        return
    dependencies = args.source_repo.resolve().parent / "SuperMimic-MuJoCo-local/dependencies/virtualgl"
    launcher = args.vglrun or dependencies / "opt/VirtualGL/bin/vglrun"
    if not launcher.is_file():
        raise FileNotFoundError(f"VirtualGL not found: {launcher}; specify --vglrun")
    command = [str(launcher)]
    library = dependencies / "usr/lib"
    if library.is_dir():
        command += ["-ld", str(library)]
    command += ["-d", "egl0", "-c", "proxy", "-fps", str(args.fps),
                sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    os.environ["ACTION_GO_SUPERMIMIC_VGL"] = "1"
    os.execv(command[0], command)


def load_sdk(args):
    if args.python_deps.is_dir():
        sys.path.insert(0, str(args.python_deps.resolve()))
    program_root = ACTION_GO_ROOT.parents[1]
    candidates = [args.sdk_root, os.environ.get("UNITREE_SDK2PY_ROOT"),
                  program_root / "projects/vendor/unitree_sdk2_python_tar",
                  program_root / "projects/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "unitree_sdk2py").is_dir():
            sys.path.insert(0, str(Path(candidate).resolve()))
            break
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ImportError as error:
        raise RuntimeError("DDS dependencies missing: supply --sdk-root and --python-deps (cyclonedds==0.10.2)") from error
    return ChannelFactoryInitialize


class DDSBridge:
    def __init__(self, model, data, indices):
        import numpy as np
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

        self.model, self.data, self.indices = model, data, indices
        self.state = unitree_hg_msg_dds__LowState_()
        self.state.mode_machine = 5
        self.condition = threading.Condition()
        self.pending = deque()
        self.latest_sequence = 0
        self.messages = 0
        self.duplicates = 0
        self.hold_messages = 0
        self.error = None
        self.publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/user_lowcmd", LowCmd_)
        self.subscriber.Init(self.receive, 0)
        actuators = indices["actuators"]
        if not np.allclose(model.actuator_gear[actuators, 0], 1.0):
            raise ValueError("SuperMimic implicit PD bridge requires unit transmission gear")
        self.effort = model.actuator_forcerange[actuators].copy()
        if not np.all(model.actuator_forcelimited[actuators]):
            raise ValueError("Training actuators must have finite effort limits")

    def receive(self, message):
        import numpy as np

        motors = message.motor_cmd[:29]
        command = {key: np.asarray([getattr(motor, key) for motor in motors], dtype=np.float64)
                   for key in ("q", "dq", "tau", "kp", "kd")}
        with self.condition:
            self.messages += 1
            if not all(values.shape == (29,) and np.isfinite(values).all() for values in command.values()):
                self.error = "Received nonfinite or malformed LowCmd"
            elif np.any(command["kp"] < 0) or np.any(command["kd"] < 0):
                self.error = "Received negative PD gains"
            elif int(message.reserve[1]) != LOCKSTEP_MAGIC:
                self.hold_messages += 1
            else:
                sequence = int(message.reserve[0])
                if sequence <= self.latest_sequence:
                    self.duplicates += 1
                elif sequence != self.latest_sequence + 1:
                    self.error = f"DDS policy sequence gap: {self.latest_sequence} -> {sequence}"
                else:
                    command["sequence"] = sequence
                    self.pending.append(command)
                    self.latest_sequence = sequence
            self.condition.notify_all()

    def publish(self):
        state, data, indices = self.state, self.data, self.indices
        state.tick = int(round(data.time * 1000)) & 0xFFFFFFFF
        for motor, q_address, v_address, actuator in zip(state.motor_state, indices["q"], indices["v"], indices["actuators"]):
            motor.q = float(data.qpos[q_address])
            motor.dq = float(data.qvel[v_address])
            motor.tau_est = float(data.actuator_force[actuator])
        root_q, root_v = indices["root_q"], indices["root_v"]
        for index in range(4):
            state.imu_state.quaternion[index] = float(data.qpos[root_q + 3 + index])
        for index in range(3):
            state.imu_state.gyroscope[index] = float(data.qvel[root_v + 3 + index])
        self.publisher.Write(state)

    def pop(self):
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            return self.pending.popleft() if self.pending else None

    def reset_queue(self):
        with self.condition:
            self.pending.clear()
            self.latest_sequence = 0
            self.error = None

    def apply(self, command):
        import mujoco

        actuators = self.indices["actuators"]
        self.model.actuator_gaintype[actuators] = mujoco.mjtGain.mjGAIN_FIXED
        self.model.actuator_biastype[actuators] = mujoco.mjtBias.mjBIAS_AFFINE
        self.model.actuator_gainprm[actuators, 0] = command["kp"]
        self.model.actuator_biasprm[actuators, 0] = command["tau"] + command["kd"] * command["dq"]
        self.model.actuator_biasprm[actuators, 1] = -command["kp"]
        self.model.actuator_biasprm[actuators, 2] = -command["kd"]
        self.model.actuator_forcerange[actuators] = self.effort
        self.model.actuator_forcelimited[actuators] = True
        self.model.actuator_ctrllimited[actuators] = False
        self.data.ctrl[actuators] = command["q"]


def stop_player(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def runtime_spec(package, log_dir, trace=False):
    import yaml

    source = package / "generic_supermimic.yaml"
    if not source.is_file():
        source = ACTION_GO_ROOT / "actions/tracking/supermimic/config/generic_supermimic.yaml"
    spec = yaml.safe_load(source.read_text())
    spec.setdefault("runtime", {}).update(duration_sec=0, publish_hz=500)
    spec.setdefault("robot", {})["params"] = str(package / "SuperMimic.yaml")
    spec.setdefault("model", {})["path"] = str(package / "policy.onnx")
    spec.setdefault("reference", {}).update(policy_step_source="lowstate_tick", sim_lockstep_sequence=True,
                                            path=str(package / "reference.csv"), end_behavior="hold", trace=trace)
    spec.setdefault("handoff", {}).update(activation_blend_sec=0.0, damping_on_stop=False,
                                          cmd_topic="rt/user_lowcmd", state_topic="rt/lowstate")
    spec.setdefault("safety", {}).update(profile="simulation_parity", enabled=True, finite_check=True,
                                         joint_limit_enabled=False, max_position_step_enabled=False)
    path = log_dir / "simulation_spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def main(argv=None):
    args = arguments(argv)
    configure_graphics(args)
    sys.path.insert(0, str(args.source_repo.resolve()))
    import mujoco
    import numpy as np
    import torch
    from supermimic.env import SimCfg, TrackingSim, _model_indices
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.motion import MotionLib
    from supermimic.robot_contract import load_contract

    package = args.package.expanduser().resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    package_clip = manifest.get("clip") or manifest.get("reference", {}).get("clip")
    clip = args.clip or package_clip
    motion_path = args.motion or manifest.get("motion") or manifest.get("motion_source")
    if not motion_path or not clip:
        raise ValueError("Specify --motion and --clip, or provide motion/clip in manifest.json")
    if args.clip and package_clip and args.clip != package_clip:
        raise ValueError("Requested clip does not match the exported reference package")
    player = args.player.expanduser().resolve()
    if not player.is_file():
        raise FileNotFoundError(f"C++ player not built: {player}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_dir = (args.log_dir or LOCAL_ROOT / "runs" / f"supermimic-{timestamp}-{os.getpid()}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    if (log_dir / "metrics.json").exists():
        raise FileExistsError(f"Choose a fresh log directory: {log_dir}")
    domain = args.domain_id if args.domain_id is not None else secrets.randbelow(151) + 40
    torch.set_num_threads(2)
    contract = load_contract("upstream", robot="g1_29dof")
    kin = G1Kinematics("cpu", xml_path=contract.xml_path)
    source_fps = float(manifest.get("provenance", {}).get("motion", {}).get("source_fps", 30.0))
    motion = MotionLib.from_path(str(Path(motion_path).expanduser().resolve()), kin, "cpu", src_fps=source_fps)
    if clip not in motion.names:
        raise ValueError(f"Unknown clip {clip}; choose from {motion.names}")
    clip_id = motion.names.index(clip)
    first_frame = int(motion.clip_start[clip_id])
    last_frame = int(motion.clip_end[clip_id])
    clip_frames = last_frame - first_frame + 1
    reference = np.loadtxt(package / "reference.csv", delimiter=",", ndmin=2)
    if reference.shape != (clip_frames, 739) or not np.isfinite(reference).all():
        raise ValueError(f"Reference cache must have shape {(clip_frames, 739)}, got {reference.shape}")
    expected_rotations = motion.root_R[first_frame:last_frame + 1].numpy().reshape(-1, 9)
    if not np.allclose(reference[:, 730:], expected_rotations, atol=2e-5):
        raise ValueError("Reference cache orientation does not match the requested motion")
    mdp = MDP(kin, motion, torch.tensor(contract.default_q), observation_mode="legacy99")
    environment = TrackingSim(SimCfg("cpu", "mujoco", contract), 1, motion, mdp, eval_mode=True)
    model, data = environment._physics.models[0], environment._physics.data[0]
    indices = _model_indices(model, contract.joint_names)
    if abs(model.opt.timestep * 4 - 0.02) > 1e-9:
        raise ValueError("Lockstep protocol requires four 5 ms physics steps per inference")
    for geom in indices["robot_geoms"]:
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            model.geom_group[geom] = 3
    initializer = load_sdk(args)
    initializer(domain, "lo")
    bridge = DDSBridge(model, data, indices)
    spec_path = runtime_spec(package, log_dir, args.trace)
    ready_path, start_path = log_dir / "ready", log_dir / "start"
    command = [str(player), "--net", "lo", "--domain-id", str(domain), "--spec", str(spec_path),
               "--params", str(package / "SuperMimic.yaml"), "--model", str(package / "policy.onnx"),
               "--trajectory", str(package / "reference.csv"), "--hz", "50", "--publish-hz", "500",
               "--activation-blend-sec", "0", "--ready-file", str(ready_path), "--start-file", str(start_path),
               "--no-damping-on-stop"]
    events = queue.SimpleQueue()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    process = None
    player_log = None
    traces = []
    stats = {"interface": "lo", "domain_id": domain, "clip": clip, "clip_frames": clip_frames,
             "control_dt": 0.02, "physics_dt": float(model.opt.timestep), "physics_steps": 0,
             "policy_steps": 0, "manual_resets": 0, "manual_pauses": 0, "manual_state_edits": 0, "finite": True,
             "automatic_reset": False, "automatic_pause": False, "reference_end_behavior": "clamp",
             "inference_backend": "ActionGo C++ ONNX Runtime via Unitree DDS", "player_command": command,
             "trace_enabled": args.trace, "policy_steps_after_reference_end": 0}
    error_sum = np.zeros(3)
    error_count = 0
    force_max = 0.0
    saturation_max = 0.0
    episode_steps = 0
    substeps_left = 0
    paused = False
    handed_off = False
    active_command = None
    reset_index = -1
    waiting_since = time.monotonic()
    previous_sequence = 0
    reference_mode = args.reference
    next_sync = 0.0
    next_state = 0.0
    next_step = 0.0
    next_status = 0.0
    next_control = 0.0
    control_offset = 0
    control_events = []
    last_state_edit_step = None
    wall_started = time.monotonic()
    last_loop_time = wall_started
    playback_wall_seconds = 0.0

    def restart():
        nonlocal process, player_log, handed_off, episode_steps, substeps_left
        nonlocal active_command, waiting_since, previous_sequence, next_step, reset_index
        stop_player(process)
        if player_log:
            player_log.close()
        reset_index += 1
        ready_path.unlink(missing_ok=True)
        start_path.unlink(missing_ok=True)
        environment.reset(torch.tensor([0]), torch.tensor([first_frame]))
        environment.refresh()
        data.xfrc_applied[:] = 0
        data.qfrc_applied[:] = 0
        bridge.reset_queue()
        episode_steps = 0
        previous_sequence = 0
        substeps_left = 0
        active_command = None
        handed_off = False
        waiting_since = time.monotonic()
        next_step = waiting_since
        player_log = (log_dir / f"player-{reset_index}.log").open("w", buffering=1)
        event_path = log_dir / f"player-{reset_index}.jsonl"
        process = subprocess.Popen(command + ["--event-log", str(event_path)], stdout=player_log, stderr=subprocess.STDOUT)
        print(f"C++ player pid={process.pid}, DDS=lo/domain {domain}, clip={clip}", flush=True)

    reference_data = mujoco.MjData(model)
    reference_scene = None
    reference_options = None
    if not args.headless:
        import mujoco.viewer
        from tools.render_rollout import append_reference_geoms

        reference_scene = mujoco.MjvScene(model, maxgeom=max(1000, 2 * model.ngeom))
        reference_options = mujoco.MjvOption()
        reference_options.geomgroup[3:] = 0

    def update_reference(viewer):
        destination = viewer.user_scn
        destination.ngeom = 0
        frame = first_frame + min(episode_steps, clip_frames - 1)
        position = motion.root_p[frame].numpy().astype(np.float64).copy()
        if reference_mode == "beside":
            position[1] += 1.6
        if reference_mode != "off":
            root = indices["root_q"]
            reference_data.qpos[:] = model.qpos0
            reference_data.qpos[root:root + 3] = position
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, motion.root_R[frame].numpy().astype(np.float64).reshape(9))
            reference_data.qpos[root + 3:root + 7] = quaternion
            reference_data.qpos[indices["q"]] = motion.q[frame].numpy()
            mujoco.mj_forward(model, reference_data)
            mujoco.mjv_updateScene(model, reference_data, reference_options, None, viewer.cam,
                                  mujoco.mjtCatBit.mjCAT_ALL, reference_scene)
            append_reference_geoms(destination, reference_scene, indices["robot_geoms"],
                                   np.asarray([0.0, 0.8, 1.0, 0.45], dtype=np.float32))
            for geom_index in range(destination.ngeom):
                destination.geoms[geom_index].objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                destination.geoms[geom_index].objid = -1
        root_position = data.qpos[indices["root_q"]:indices["root_q"] + 3]
        midpoint = root_position if reference_mode == "off" else (position + root_position) * 0.5
        viewer.cam.lookat[:] = [midpoint[0], midpoint[1], max(0.7, midpoint[2])]

    try:
        restart()
        viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data, key_callback=events.put)
        with viewer_context as viewer:
            if viewer:
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.5, 135, -15
                viewer.opt.geomgroup[3:] = 0
            print("SPACE: manual pause | BACKSPACE: reset robot/reference/history | G: reference beside/ghost/off", flush=True)
            while not stop.is_set() and (viewer is None or viewer.is_running()):
                now = time.monotonic()
                if handed_off and not paused:
                    playback_wall_seconds += now - last_loop_time
                last_loop_time = now
                if args.duration_sec and stats["physics_steps"] * model.opt.timestep >= args.duration_sec - 1e-9:
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"C++ player exited with status {process.returncode}; see {player_log.name}")
                if now >= next_state:
                    bridge.publish()
                    next_state = now + 0.002
                if args.control_file and now >= next_control:
                    if args.control_file.is_file():
                        with args.control_file.open() as control_stream:
                            control_stream.seek(control_offset)
                            while True:
                                offset = control_stream.tell()
                                line = control_stream.readline()
                                if not line.endswith("\n"):
                                    control_stream.seek(offset)
                                    break
                                if line.strip():
                                    events.put(json.loads(line))
                            control_offset = control_stream.tell()
                    next_control = now + 0.02
                render_due = viewer is not None and now >= next_sync
                with viewer.lock() if viewer else nullcontext():
                    while not events.empty():
                        key = events.get()
                        if isinstance(key, dict):
                            control = key
                            action = control.get("command")
                            control_events.append(dict(command=action, tick_ms=int(round(data.time * 1000)),
                                                       policy_steps=stats["policy_steps"], reset_index=reset_index))
                            if action in ("pause", "resume"):
                                key = 32 if paused != (action == "pause") else None
                            elif action == "reset":
                                key = 259
                            elif action == "set_root_pose":
                                quaternion = np.asarray(control["quaternion"], dtype=np.float64)
                                if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-9:
                                    raise ValueError("set_root_pose requires a finite nonzero wxyz quaternion")
                                root = indices["root_q"]
                                data.qpos[root + 3:root + 7] = quaternion / np.linalg.norm(quaternion)
                                if "position" in control:
                                    position = np.asarray(control["position"], dtype=np.float64)
                                    if position.shape != (3,) or not np.isfinite(position).all():
                                        raise ValueError("set_root_pose requires a finite position vector")
                                    data.qpos[root:root + 3] = position
                                data.qvel[indices["root_v"]:indices["root_v"] + 6] = 0
                                mujoco.mj_forward(model, data)
                                stats["manual_state_edits"] += 1
                                last_state_edit_step = stats["policy_steps"]
                                key = None
                            else:
                                raise ValueError(f"Unknown simulation control command: {action}")
                        if key == 32:
                            paused = not paused
                            stats["manual_pauses"] += int(paused)
                            next_step = now
                            waiting_since = now
                            print("Paused" if paused else "Playing", flush=True)
                        elif key == 259:
                            restart()
                            stats["manual_resets"] += 1
                            if viewer:
                                viewer.perturb.active = 0
                                viewer.perturb.active2 = 0
                        elif key == ord("G"):
                            modes = ("beside", "ghost", "off")
                            reference_mode = modes[(modes.index(reference_mode) + 1) % len(modes)]
                    if not handed_off and ready_path.is_file() and bridge.hold_messages:
                        start_path.touch()
                        handed_off = True
                        waiting_since = now
                        next_step = now
                    if handed_off and not paused:
                        if substeps_left == 0:
                            active_command = bridge.pop()
                            if active_command is not None:
                                sequence = active_command["sequence"]
                                if sequence != previous_sequence + 1:
                                    raise RuntimeError("Nonconsecutive command consumed by simulator")
                                previous_sequence = sequence
                                root_q, root_v = indices["root_q"], indices["root_v"]
                                frame = first_frame + min(sequence - 1, clip_frames - 1)
                                if args.trace:
                                    traces.append(dict(reset_index=reset_index, sequence=sequence,
                                                       reference_frame=frame - first_frame, tick_ms=int(round(data.time * 1000)),
                                                       q=data.qpos[indices["q"]].copy(), qd=data.qvel[indices["v"]].copy(),
                                                       root_p=data.qpos[root_q:root_q + 3].copy(),
                                                       root_quat=data.qpos[root_q + 3:root_q + 7].copy(),
                                                       root_w_local=data.qvel[root_v + 3:root_v + 6].copy(),
                                                       target_q=active_command["q"].copy(), kp=active_command["kp"].copy(),
                                                       kd=active_command["kd"].copy(), tau=active_command["tau"].copy(),
                                                       target_qd=active_command["dq"].copy()))
                                stats["policy_steps_after_reference_end"] += int(frame == last_frame)
                                bridge.apply(active_command)
                                substeps_left = 4
                                waiting_since = now
                                stats["policy_steps"] += 1
                        if substeps_left and (args.no_realtime or now >= next_step):
                            mujoco.mj_step(model, data)
                            substeps_left -= 1
                            stats["physics_steps"] += 1
                            next_step += model.opt.timestep
                            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                                stats["finite"] = False
                                raise RuntimeError("Nonfinite MuJoCo state")
                            torques = data.actuator_force[indices["actuators"]]
                            force_max = max(force_max, float(np.max(np.abs(torques))))
                            saturation_max = max(saturation_max, float(np.max(np.maximum(
                                torques - bridge.effort[:, 1], bridge.effort[:, 0] - torques))))
                            if substeps_left == 0:
                                mujoco.mj_forward(model, data)
                                episode_steps += 1
                                frame = first_frame + min(episode_steps, clip_frames - 1)
                                rotation = np.empty(9)
                                root = indices["root_q"]
                                mujoco.mju_quat2Mat(rotation, data.qpos[root + 3:root + 7])
                                rotation_delta = rotation.reshape(3, 3).T @ motion.root_R[frame].numpy()
                                error_sum += (float(np.mean((data.qpos[indices["q"]] - motion.q[frame].numpy()) ** 2)),
                                              float(np.linalg.norm(data.qpos[root:root + 3] - motion.root_p[frame].numpy())),
                                              float(np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1, 1))))
                                error_count += 1
                                bridge.publish()
                                waiting_since = now
                    if not paused and now - waiting_since > args.command_timeout_sec:
                        raise TimeoutError(f"Timed out waiting for C++ DDS command; see {player_log.name}")
                    if render_due:
                        if paused:
                            mujoco.mjv_applyPerturbPose(model, data, viewer.perturb, 1)
                            mujoco.mj_forward(model, data)
                        update_reference(viewer)
                        next_sync = now + 1.0 / args.fps
                if render_due:
                    viewer.sync()
                if now >= next_status:
                    status = dict(tick_ms=int(round(data.time * 1000)), policy_steps=stats["policy_steps"],
                                  sequence=previous_sequence, paused=paused, reset_index=reset_index,
                                  playback_wall_seconds=playback_wall_seconds,
                                  realtime_factor=(stats["physics_steps"] * model.opt.timestep / playback_wall_seconds
                                                   if playback_wall_seconds > 0 else None),
                                  manual_state_edits=stats["manual_state_edits"],
                                  root_position=data.qpos[indices["root_q"]:indices["root_q"] + 3].tolist())
                    status_temp = log_dir / "status.tmp"
                    status_temp.write_text(json.dumps(status))
                    status_temp.replace(log_dir / "status.json")
                    next_status = now + 0.2
                time.sleep(0.0001 if args.no_realtime else 0.0005)
    except Exception as error:
        stats["error"] = str(error)
        raise
    finally:
        stop_player(process)
        if player_log:
            player_log.close()
        elapsed = time.monotonic() - wall_started
        stats.update(simulated_seconds=stats["physics_steps"] * float(model.opt.timestep), wall_seconds=elapsed,
                     playback_wall_seconds=playback_wall_seconds,
                     realtime_factor=(stats["physics_steps"] * model.opt.timestep / playback_wall_seconds
                                      if playback_wall_seconds > 0 else None),
                     last_policy_sequence=previous_sequence, reference_frame=min(episode_steps, clip_frames - 1),
                     dds_messages=bridge.messages, dds_duplicate_messages=bridge.duplicates,
                     max_abs_actuator_force_nm=force_max, torque_limit_violation_nm=max(0.0, saturation_max),
                     player_exit_code=process.returncode if process else None)
        stats["control_events"] = control_events
        stats["policy_steps_after_last_state_edit"] = (stats["policy_steps"] - last_state_edit_step
                                                        if last_state_edit_step is not None else 0)
        if error_count:
            stats.update(joint_rmse_rad=float(np.sqrt(error_sum[0] / error_count)),
                         mean_root_position_error_m=float(error_sum[1] / error_count),
                         mean_root_orientation_error_rad=float(error_sum[2] / error_count))
        stats["passed"] = ("error" not in stats and stats["finite"] and stats["policy_steps"] > 0
                           and stats["physics_steps"] > 0 and stats["torque_limit_violation_nm"] < 1e-6)
        if traces:
            np.savez_compressed(log_dir / "trace.npz", **{key: np.asarray([row[key] for row in traces]) for key in traces[0]})
        (log_dir / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2), flush=True)
        print(f"Metrics and DDS trace: {log_dir}", flush=True)
        environment.close()
    return 0 if stats["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
