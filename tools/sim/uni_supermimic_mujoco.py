#!/usr/bin/env python3
"""Run a Uni-SuperMimic or AllMimic ActionGo C++ actor in its native MuJoCo scene."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.util
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
from types import SimpleNamespace

import numpy as np


ACTION_GO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = ACTION_GO_ROOT.parent / "ActionGo-local"
SIM_HELPER_PATH = ACTION_GO_ROOT / "tools/sim/supermimic_mujoco.py"
HELPER_SPEC = importlib.util.spec_from_file_location("actiongo_supermimic_helpers", SIM_HELPER_PATH)
HELPERS = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPERS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("uni_supermimic", "allmimic"), default="uni_supermimic")
    parser.add_argument("--clip", default="paoxiao")
    parser.add_argument("--training-repo", type=Path, default=ACTION_GO_ROOT.parent / "Uni-SuperMimic")
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--player", type=Path, default=LOCAL_ROOT / "build/generic_motion_tracking_player")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--domain-id", type=int)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--python-deps", type=Path, default=LOCAL_ROOT / "python")
    parser.add_argument("--reference", choices=("beside", "ghost", "off"), default="beside")
    parser.add_argument("--safety-profile", choices=("simulation_parity", "hardware"), default="simulation_parity")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--vglrun", type=Path)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--control-file", type=Path)
    args = parser.parse_args(argv)
    if args.variant == "uni_supermimic" and args.motion is None:
        parser.error("--motion is required for Uni-SuperMimic")
    if not math.isfinite(args.duration_sec) or args.duration_sec < 0:
        parser.error("--duration-sec must be finite and nonnegative")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if args.domain_id is not None and not 1 <= args.domain_id <= 200:
        parser.error("--domain-id must be in [1, 200]")
    if args.no_realtime and not args.headless:
        parser.error("--no-realtime is only supported with --headless")
    return args


def configure_graphics(args):
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
        return
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ["PYOPENGL_PLATFORM"] = "glx"
    if not args.gpu or os.environ.get("ACTION_GO_UNI_SUPERMIMIC_VGL") == "1":
        return
    vgl_root = args.training_repo.resolve().parent / "SuperMimic-MuJoCo-local/dependencies/virtualgl"
    launcher = args.vglrun or vgl_root / "opt/VirtualGL/bin/vglrun"
    if not launcher.is_file():
        raise FileNotFoundError(f"VirtualGL not found: {launcher}; specify --vglrun")
    command = [str(launcher)]
    library = vgl_root / "usr/lib"
    if library.is_dir():
        command += ["-ld", str(library)]
    command += ["-d", "egl0", "-c", "proxy", "-fps", str(args.fps), sys.executable,
                str(Path(__file__).resolve()), *sys.argv[1:]]
    os.environ["ACTION_GO_UNI_SUPERMIMIC_VGL"] = "1"
    os.execv(command[0], command)


def load_training(training_repo, motion_path, source_fps):
    sys.path.insert(0, str(training_repo / "src"))
    import mujoco
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
    from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
    from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import MotionLib

    xml = training_repo / "src/unisupermimic/assets/robots/g1/scene_supermimic.xml"
    mjcf = training_repo / "src/unisupermimic/assets/robots/g1/supermimic.xml"
    kin = G1Kinematics(str(mjcf), "cpu")
    motion = MotionLib([str(motion_path)], kin, "cpu", src_fps=source_fps)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_joint < 0 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("Training MJCF must have a floating_base_joint free joint")
    qpos_addresses = []
    dof_addresses = []
    actuator_addresses = []
    for name in __import__("unisupermimic.tasks.motion_tracking.g1.supermimic.robot", fromlist=["G1_JOINT_NAMES"]).G1_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"Training MJCF is missing joint/actuator {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        actuator_addresses.append(actuator_id)
    indices = {
        "q": np.asarray(qpos_addresses, dtype=np.int32),
        "v": np.asarray(dof_addresses, dtype=np.int32),
        "actuators": np.asarray(actuator_addresses, dtype=np.int32),
        "root_q": int(model.jnt_qposadr[root_joint]),
        "root_v": int(model.jnt_dofadr[root_joint]),
        "robot_geoms": np.flatnonzero(model.geom_bodyid > 0),
    }
    if not np.allclose(model.actuator_gear[indices["actuators"], 0], 1.0):
        raise ValueError("The training scene uses an unsupported actuator gear")
    return mujoco, torch, mat_to_quat, motion, model, data, indices


def set_motion_state(mujoco, torch, mat_to_quat, model, data, indices, motion, frame):
    root_q, root_v = indices["root_q"], indices["root_v"]
    rotation = motion.root_R[frame]
    data.qpos[:] = model.qpos0
    data.qpos[root_q:root_q + 3] = motion.root_p[frame].numpy()
    data.qpos[root_q + 3:root_q + 7] = mat_to_quat(rotation).numpy()
    data.qpos[indices["q"]] = motion.q[frame].numpy()
    data.qvel[:] = 0
    data.qvel[root_v:root_v + 3] = motion.root_v[frame].numpy()
    data.qvel[root_v + 3:root_v + 6] = (rotation.T @ motion.root_w[frame]).numpy()
    data.qvel[indices["v"]] = motion.qd[frame].numpy()
    data.ctrl[indices["actuators"]] = motion.q[frame].numpy()
    data.xfrc_applied[:] = 0
    data.qfrc_applied[:] = 0
    mujoco.mj_forward(model, data)


def load_allmimic(package, clip):
    import hashlib
    import mujoco
    import torch

    clip_dir = package / clip
    manifest = json.loads((clip_dir / "manifest.json").read_text())
    for path, expected in ((package / "policy.onnx", manifest["policy"]["sha256"]),
                           (clip_dir / "reference.csv", manifest["reference"]["csv_sha256"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"AllMimic asset hash mismatch: {path}")
    model = mujoco.MjModel.from_binary_path(str(package / "scene.mjb"))
    data = mujoco.MjData(model)
    with np.load(clip_dir / "motion.npz") as archive:
        qpos, qvel = archive["qpos"].copy(), archive["qvel"].copy()
    if qpos.shape != (manifest["reference"]["num_frames"], 36) or qvel.shape != (len(qpos), 35):
        raise ValueError("AllMimic motion has unexpected dimensions")
    rotations = np.empty((len(qpos), 9))
    for frame, quat in enumerate(qpos[:, 3:7]):
        mujoco.mju_quat2Mat(rotations[frame], quat.astype(np.float64))
    motion = SimpleNamespace(names=[manifest["reference"]["clip"]], num_frames=len(qpos),
        root_p=torch.from_numpy(qpos[:, :3]), root_R=torch.from_numpy(rotations.reshape(-1, 3, 3)).float(),
        q=torch.from_numpy(qpos[:, 7:]), root_v=torch.from_numpy(qvel[:, :3]),
        root_w=torch.from_numpy(qvel[:, 3:6]), qd=torch.from_numpy(qvel[:, 6:]))

    def mat_to_quat(rotation):
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.numpy().astype(np.float64).reshape(9))
        return torch.from_numpy(quat)

    indices = {"q": np.arange(7, 36), "v": np.arange(6, 35), "root_q": 0, "root_v": 0,
               "actuators": np.array([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]
                                       for joint in range(1, 30)]),
               "robot_geoms": np.flatnonzero(model.body_rootid[model.geom_bodyid] == model.jnt_bodyid[0])}
    return mujoco, torch, mat_to_quat, motion, model, data, indices, manifest


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def append_reference_geoms(mujoco, destination, source_scene, robot_geoms, rgba):
    robot_geoms = set(robot_geoms)
    for index in range(source_scene.ngeom):
        source = source_scene.geoms[index]
        if source.objtype != mujoco.mjtObj.mjOBJ_GEOM or source.objid not in robot_geoms:
            continue
        if destination.ngeom >= destination.maxgeom:
            raise ValueError("Reference visuals exceed the viewer scene capacity")
        target = destination.geoms[destination.ngeom]
        mujoco.mjv_initGeom(target, source.type, source.size, source.pos, source.mat.reshape(9), rgba)
        target.dataid = source.dataid
        target.modelrbound = source.modelrbound
        target.objtype, target.objid = mujoco.mjtObj.mjOBJ_UNKNOWN, -1
        target.category = mujoco.mjtCatBit.mjCAT_DYNAMIC
        target.matid, target.texcoord, target.segid = -1, 0, -1
        target.specular, target.shininess, target.reflectance = 0.1, 0.1, 0.0
        target.emission, target.transparent = 0.2, 1
        destination.ngeom += 1


def main(argv=None):
    args = parse_args(argv)
    configure_graphics(args)
    import yaml

    training_repo = args.training_repo.expanduser().resolve()
    package = args.package.expanduser().resolve()
    if args.variant == "allmimic":
        mujoco, torch, mat_to_quat, motion, model, data, indices, manifest = load_allmimic(package, args.clip)
        reference_dir = package / args.clip
        params_path = package / "AllMimic.yaml"
        checkpoint = manifest["checkpoint"]
    else:
        motion_path = args.motion.expanduser().resolve()
        manifest = json.loads((package / "manifest.json").read_text())
        source_fps = float(manifest["provenance"]["motion"]["source_fps"])
        if not motion_path.is_file() or motion_path.name != manifest["provenance"]["motion"]["name"]:
            raise ValueError("The motion must match the clip packaged with the policy")
        mujoco, torch, mat_to_quat, motion, model, data, indices = load_training(training_repo, motion_path, source_fps)
        reference_dir = package
        params_path = package / "SuperMimic.yaml"
        checkpoint = manifest["provenance"]["checkpoint"]
    clip = str(manifest["reference"]["clip"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_dir = (args.log_dir or LOCAL_ROOT / "runs" / f"{args.variant}-{timestamp}-{os.getpid()}").resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    player = args.player.expanduser().resolve()
    if not player.is_file():
        raise FileNotFoundError(f"ActionGo C++ player not found: {player}")
    if motion.names != [clip] or motion.num_frames != int(manifest["reference"]["num_frames"]):
        raise ValueError("Training MotionLib output does not match the exported reference package")
    reference_cache = np.loadtxt(reference_dir / "reference.csv", delimiter=",", dtype=np.float32, ndmin=2)
    if reference_cache.shape != (motion.num_frames, 739):
        raise ValueError("Packed reference cache has an unexpected shape")
    if not np.allclose(reference_cache[:, 730:], motion.root_R.numpy().reshape(-1, 9), atol=2e-5):
        raise ValueError("Cached reference rotations differ from the training MotionLib")
    if abs(float(model.opt.timestep) * 4 - 0.02) > 1e-9:
        raise ValueError("ActionGo lockstep expects four 5 ms simulation steps per inference")
    for geom in indices["robot_geoms"]:
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            model.geom_group[geom] = 3

    # Resolve the SDK only after all simulation inputs have been validated.
    helper_args = argparse.Namespace(
        python_deps=args.python_deps, sdk_root=args.sdk_root,
        source_repo=training_repo,
    )
    initializer = HELPERS.load_sdk(helper_args)
    domain = args.domain_id if args.domain_id is not None else secrets.randbelow(151) + 40
    initializer(domain, "lo")
    bridge = HELPERS.DDSBridge(model, data, indices)

    spec_source = ACTION_GO_ROOT / ("actions/allmimic/config/generic_allmimic_paoxiao.yaml"
                                   if args.variant == "allmimic" else
                                   "actions/uni_supermimic/config/generic_uni_supermimic.yaml")
    spec = yaml.safe_load(spec_source.read_text())
    spec.setdefault("runtime", {}).update(duration_sec=0, publish_hz=500)
    spec["robot"]["params"] = str(params_path)
    spec["model"]["path"] = str(package / "policy.onnx")
    spec["reference"].update(path=str(reference_dir / "reference.csv"), policy_step_source="lowstate_tick",
                             sim_lockstep_sequence=True, end_behavior="hold", yaw_alignment="none",
                             trace=args.trace)
    spec["handoff"].update(activation_blend_sec=0.0, damping_on_stop=False,
                           cmd_topic="rt/user_lowcmd", state_topic="rt/lowstate")
    spec["safety"] = {"profile": args.safety_profile, "enabled": True, "finite_check": True,
                      "joint_limit_enabled": args.safety_profile == "hardware",
                      "preserve_pd_torque": args.variant == "allmimic" and args.safety_profile == "hardware",
                      "max_position_step_enabled": False,
                      "lowstate_timeout_sec": 5.0}
    spec_path = log_dir / "simulation_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    ready_path, start_path = log_dir / "ready", log_dir / "start"
    command = [str(player), "--net", "lo", "--domain-id", str(domain), "--spec", str(spec_path),
               "--params", str(params_path), "--model", str(package / "policy.onnx"),
               "--trajectory", str(reference_dir / "reference.csv"), "--hz", "50", "--publish-hz", "500",
               "--activation-blend-sec", "0", "--ready-file", str(ready_path),
               "--start-file", str(start_path), "--no-damping-on-stop"]

    stop = threading.Event()
    events = queue.SimpleQueue()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    traces = []
    first_frame = 0
    last_frame = motion.num_frames - 1
    paused = False
    reference_mode = args.reference
    process = None
    player_log = None
    reset_count = 0
    reset_index = 0
    episode_steps = 0
    previous_sequence = 0
    substeps_left = 0
    active_command = None
    handed_off = False
    wait_since = time.monotonic()
    next_state = 0.0
    next_control = 0.0
    next_status = 0.0
    next_step = 0.0
    next_sync = 0.0
    control_offset = 0
    wall_started = time.monotonic()
    last_loop = wall_started
    playback_wall = 0.0
    force_max = 0.0
    error_joint_squared = 0.0
    error_root_sum = 0.0
    error_orientation_sum = 0.0
    metric_samples = 0
    stats = {"policy_steps": 0, "physics_steps": 0, "manual_resets": 0,
             "manual_pauses": 0, "finite": True, "automatic_reset": False,
             "automatic_pause": False, "clip": clip, "clip_frames": motion.num_frames,
             "control_dt": 0.02, "physics_dt": float(model.opt.timestep),
             "inference_backend": "ActionGo C++ ONNX Runtime via loopback Unitree DDS",
             "player_command": command, "checkpoint": checkpoint,
             "trace_enabled": args.trace, "reference_end_behavior": "hold",
             "safety_profile": args.safety_profile}

    reference_data = mujoco.MjData(model)
    if not args.headless:
        import mujoco.viewer
        reference_scene = mujoco.MjvScene(model, maxgeom=max(1000, 2 * model.ngeom))
        reference_options = mujoco.MjvOption()
        reference_options.geomgroup[3:] = 0
    else:
        reference_scene = None
        reference_options = None

    def restart():
        nonlocal process, player_log, reset_count, reset_index, episode_steps, previous_sequence
        nonlocal substeps_left, active_command, handed_off, wait_since, next_step
        stop_process(process)
        if player_log:
            player_log.close()
        reset_count += 1
        reset_index = reset_count - 1
        ready_path.unlink(missing_ok=True)
        start_path.unlink(missing_ok=True)
        mujoco.mj_resetData(model, data)
        set_motion_state(mujoco, torch, mat_to_quat, model, data, indices, motion, first_frame)
        bridge.reset_queue()
        episode_steps = 0
        previous_sequence = 0
        substeps_left = 0
        active_command = None
        handed_off = False
        wait_since = time.monotonic()
        next_step = wait_since
        player_log = (log_dir / f"player-{reset_count}.log").open("w", buffering=1)
        event_path = log_dir / f"player-{reset_count}.jsonl"
        process = subprocess.Popen(command + ["--event-log", str(event_path)],
                                   stdout=player_log, stderr=subprocess.STDOUT)
        print(f"ActionGo player pid={process.pid}; clip={clip}; DDS=lo/{domain}", flush=True)

    def update_reference(viewer):
        destination = viewer.user_scn
        destination.ngeom = 0
        frame = min(episode_steps, last_frame)
        position = motion.root_p[frame].numpy().astype(np.float64).copy()
        if reference_mode == "beside":
            position[1] += 1.6
        if reference_mode != "off":
            root_q = indices["root_q"]
            reference_data.qpos[:] = model.qpos0
            reference_data.qpos[root_q:root_q + 3] = position
            reference_data.qpos[root_q + 3:root_q + 7] = mat_to_quat(motion.root_R[frame]).numpy()
            reference_data.qpos[indices["q"]] = motion.q[frame].numpy()
            mujoco.mj_forward(model, reference_data)
            mujoco.mjv_updateScene(model, reference_data, reference_options, None, viewer.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, reference_scene)
            append_reference_geoms(mujoco, destination, reference_scene, indices["robot_geoms"],
                                   np.asarray([0.0, 0.8, 1.0, 0.45], dtype=np.float32))
            for index in range(destination.ngeom):
                destination.geoms[index].objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                destination.geoms[index].objid = -1
        root = data.qpos[indices["root_q"]:indices["root_q"] + 3]
        midpoint = root if reference_mode == "off" else (position + root) * 0.5
        viewer.cam.lookat[:] = [midpoint[0], midpoint[1], max(0.7, midpoint[2])]

    try:
        restart()
        viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(
            model, data, key_callback=events.put)
        with viewer_context as viewer:
            if viewer:
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.5, 135, -15
                viewer.opt.geomgroup[3:] = 0
            print("SPACE: pause/resume | BACKSPACE: manual reset | G: beside/ghost/off", flush=True)
            while not stop.is_set() and (viewer is None or viewer.is_running()):
                now = time.monotonic()
                if handed_off and not paused:
                    playback_wall += now - last_loop
                last_loop = now
                if args.duration_sec and data.time >= args.duration_sec - 1e-9:
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"ActionGo player exited {process.returncode}; inspect {player_log.name}")
                if now >= next_state:
                    bridge.publish()
                    next_state = now + 0.002
                if args.control_file and now >= next_control:
                    if args.control_file.is_file():
                        with args.control_file.open() as stream:
                            stream.seek(control_offset)
                            while True:
                                offset = stream.tell()
                                line = stream.readline()
                                if not line.endswith("\n"):
                                    stream.seek(offset)
                                    break
                                if line.strip():
                                    events.put(json.loads(line))
                            control_offset = stream.tell()
                    next_control = now + 0.02
                render_due = viewer is not None and now >= next_sync
                with viewer.lock() if viewer else nullcontext():
                    while not events.empty():
                        key = events.get()
                        if isinstance(key, dict):
                            action = key.get("command")
                            if action == "pause":
                                key = 32
                            elif action == "resume":
                                key = 32 if paused else None
                            elif action == "reset":
                                key = 259
                            elif action is not None:
                                raise ValueError(f"Unknown simulation command: {action}")
                        if key == 32:
                            paused = not paused
                            stats["manual_pauses"] += int(paused)
                            next_step = now
                            wait_since = now
                            print("Paused" if paused else "Playing", flush=True)
                        elif key == 259:
                            restart()
                            if viewer:
                                viewer.perturb.active = 0
                                viewer.perturb.active2 = 0
                            stats["manual_resets"] += 1
                        elif key in (ord("G"), ord("g")):
                            modes = ("beside", "ghost", "off")
                            reference_mode = modes[(modes.index(reference_mode) + 1) % len(modes)]
                    if not handed_off and ready_path.is_file() and bridge.hold_messages:
                        start_path.touch()
                        handed_off = True
                        wait_since = now
                        next_step = now
                    if handed_off and not paused:
                        if substeps_left == 0:
                            active_command = bridge.pop()
                            if active_command is not None:
                                sequence = active_command["sequence"]
                                if sequence != previous_sequence + 1:
                                    raise RuntimeError("ActionGo DDS command sequence is not consecutive")
                                previous_sequence = sequence
                                frame = min(sequence - 1, last_frame)
                                if args.trace:
                                    root_q, root_v = indices["root_q"], indices["root_v"]
                                    traces.append({
                                        "reset_index": reset_index,
                                        "sequence": sequence,
                                        "reference_frame": frame,
                                        "tick_ms": int(round(data.time * 1000)),
                                        "q": data.qpos[indices["q"]].copy(),
                                        "qd": data.qvel[indices["v"]].copy(),
                                        "root_p": data.qpos[root_q:root_q + 3].copy(),
                                        "root_quat": data.qpos[root_q + 3:root_q + 7].copy(),
                                        "root_w_local": data.qvel[root_v + 3:root_v + 6].copy(),
                                        "target_q": active_command["q"].copy(),
                                        "kp": active_command["kp"].copy(),
                                        "kd": active_command["kd"].copy(),
                                        "tau": active_command["tau"].copy(),
                                        "target_qd": active_command["dq"].copy(),
                                    })
                                bridge.apply(active_command)
                                substeps_left = 4
                                wait_since = now
                                stats["policy_steps"] += 1
                        if substeps_left and (args.no_realtime or now >= next_step):
                            mujoco.mj_step(model, data)
                            substeps_left -= 1
                            stats["physics_steps"] += 1
                            next_step += model.opt.timestep
                            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                                stats["finite"] = False
                                raise RuntimeError("MuJoCo state became nonfinite")
                            force_max = max(force_max, float(np.max(np.abs(data.actuator_force[indices["actuators"]]))))
                            if substeps_left == 0:
                                mujoco.mj_forward(model, data)
                                episode_steps += 1
                                frame = min(episode_steps, last_frame)
                                q_error = data.qpos[indices["q"]] - motion.q[frame].numpy()
                                error_joint_squared += float(np.mean(q_error * q_error))
                                root_q = indices["root_q"]
                                position_error = data.qpos[root_q:root_q + 3] - motion.root_p[frame].numpy()
                                error_root_sum += float(np.linalg.norm(position_error))
                                rotation = np.empty(9)
                                mujoco.mju_quat2Mat(rotation, data.qpos[root_q + 3:root_q + 7])
                                rotation_delta = rotation.reshape(3, 3).T @ motion.root_R[frame].numpy()
                                angle = np.arccos(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
                                error_orientation_sum += float(angle)
                                metric_samples += 1
                                bridge.publish()
                                wait_since = now
                    if not paused and now - wait_since > 10.0:
                        raise TimeoutError(f"Timed out waiting for ActionGo command; inspect {player_log.name}")
                    if render_due:
                        if paused:
                            mujoco.mjv_applyPerturbPose(model, data, viewer.perturb, 1)
                            mujoco.mj_forward(model, data)
                        update_reference(viewer)
                        next_sync = now + 1.0 / args.fps
                if render_due:
                    viewer.sync()
                if now >= next_status:
                    status = {"tick_ms": int(round(data.time * 1000)), "policy_steps": stats["policy_steps"],
                              "paused": paused, "manual_resets": stats["manual_resets"],
                              "realtime_factor": (stats["physics_steps"] * float(model.opt.timestep) / playback_wall
                                                  if playback_wall > 0 else None)}
                    temp_status = log_dir / "status.tmp"
                    temp_status.write_text(json.dumps(status))
                    temp_status.replace(log_dir / "status.json")
                    next_status = now + 0.2
                time.sleep(0.0001 if args.no_realtime else 0.0005)
    except Exception as error:
        stats["error"] = str(error)
        raise
    finally:
        stop_process(process)
        if player_log:
            player_log.close()
        elapsed = time.monotonic() - wall_started
        stats.update(simulated_seconds=stats["physics_steps"] * float(model.opt.timestep),
                     wall_seconds=elapsed, playback_wall_seconds=playback_wall,
                     realtime_factor=(stats["physics_steps"] * model.opt.timestep / playback_wall
                                      if playback_wall > 0 else None),
                     last_policy_sequence=previous_sequence, reference_frame=min(episode_steps, last_frame),
                     dds_messages=bridge.messages, dds_duplicate_messages=bridge.duplicates,
                     max_abs_actuator_force_nm=force_max,
                     player_exit_code=process.returncode if process else None,
                     automatic_fall_reset=False, automatic_fall_pause=False)
        stats["passed"] = ("error" not in stats and stats["finite"] and stats["policy_steps"] > 0
                           and stats["physics_steps"] > 0)
        if metric_samples:
            stats["joint_rmse_rad"] = float(np.sqrt(error_joint_squared / metric_samples))
            stats["mean_root_position_error_m"] = error_root_sum / metric_samples
            stats["mean_root_orientation_error_rad"] = error_orientation_sum / metric_samples
        if traces:
            np.savez_compressed(log_dir / "trace.npz", **{
                key: np.asarray([sample[key] for sample in traces]) for key in traces[0]
            })
        (log_dir / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2), flush=True)
        print(f"ActionGo MuJoCo output: {log_dir}", flush=True)
    return 0 if stats["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
