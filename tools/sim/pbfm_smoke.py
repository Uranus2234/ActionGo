import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import threading
import time

import mujoco
import numpy as np
import yaml

from action_go_sim_remote import NativeActionGoBridge
from pbfm_depth import DepthHistory, DepthPublisher, MujocoDepthCamera


ROOT = Path(__file__).resolve().parents[2]


class PbfmSimulationBridge(NativeActionGoBridge):
    def apply_command(self):
        applied = super().apply_command()
        if applied:
            velocity = np.clip(self.data.qvel[self.dof_adr], -2 * self.velocity_limits,
                               2 * self.velocity_limits)
            lower = np.maximum(-self.effort_limits,
                               self.effort_limits * (-1 - velocity / self.velocity_limits))
            upper = np.minimum(self.effort_limits,
                               self.effort_limits * (1 - velocity / self.velocity_limits))
            self.data.ctrl[self.motor_actuator_ids] = np.clip(
                self.data.ctrl[self.motor_actuator_ids], lower, upper)
        return applied


class KeyboardBridge(PbfmSimulationBridge):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.keyboard_lock = threading.Lock()
        self.speed = 0.0
        self.lateral = 0.0
        self.turn = 0.0
        self.held_keys = set()
        self.start_requested = False
        self.exit_requested = False

    def key_event(self, key, pressed):
        with self.keyboard_lock:
            if pressed:
                self.held_keys.add(key)
            else:
                self.held_keys.discard(key)
            if pressed and key in (257, 335):
                self.start_requested = True
            elif pressed and key == ord("X"):
                self.exit_requested = True
            self.speed = float(ord("W") in self.held_keys) - 0.3 * (ord("S") in self.held_keys)
            if ord("W") in self.held_keys and ord("S") in self.held_keys:
                self.speed = 0.0
            self.lateral = 0.4 * ((ord("Q") in self.held_keys) - (ord("E") in self.held_keys))
            self.turn = float((ord("A") in self.held_keys) - (ord("D") in self.held_keys))
            if 32 in self.held_keys:
                self.speed = self.lateral = self.turn = 0.0

    def clear_keys(self):
        with self.keyboard_lock:
            self.held_keys.clear()
            self.speed = self.lateral = self.turn = 0.0

    def set_axes(self, forward, lateral, turn):
        axes = np.asarray([forward, lateral, turn], dtype=np.float64)
        if not np.isfinite(axes).all():
            raise ValueError("replay joystick axes must be finite")
        forward, lateral, turn = np.clip(axes, -1, 1)
        with self.keyboard_lock:
            self.held_keys.clear()
            self.speed = float(forward * (1.0 if forward >= 0 else 0.3))
            self.lateral = float(lateral * 0.4)
            self.turn = float(turn)

    def _joystick_mask_and_axes(self):
        with self.keyboard_lock:
            speed_axis = self.speed / (1.0 if self.speed >= 0 else 0.3)
            return 0, [-self.lateral / 0.4, -self.turn, 0.0, speed_axis]

    def status(self):
        with self.keyboard_lock:
            mode = "WALK" if self.speed or self.lateral else "TURN" if self.turn else "STOP / standing context"
            actual_speed = np.linalg.norm(self.data.qvel[:2])
            return (f"{mode}\ncmd_v={self.speed:+.1f}  cmd_side={self.lateral:+.1f} m/s"
                    f"  turn_axis={self.turn:+.2f}\nactual speed={actual_speed:.2f} m/s")


class KeyboardViewer:
    def __init__(self, model, data, bridge):
        import glfw

        self.glfw = glfw
        self.model, self.data, self.bridge = model, data, bridge
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        self.window = glfw.create_window(1280, 800, "ActionGo P-BFM | Hold keys to move", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("MuJoCo keyboard window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.cam = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.texts = []
        self.last_cursor = None
        self.last_click_time = -1.0
        self.last_click_position = None
        glfw.set_key_callback(self.window, lambda window, key, scancode, action, mods:
                              bridge.key_event(key, action != glfw.RELEASE))
        glfw.set_window_focus_callback(self.window, self.focus_changed)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_scroll_callback(self.window, lambda window, horizontal, vertical:
                                 mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                                       0, -0.05 * vertical, self.scene, self.cam))

    def focus_changed(self, window, focused):
        if not focused:
            self.bridge.clear_keys()
            self.perturb.active = 0
            self.data.xfrc_applied[:] = 0
            self.last_cursor = None

    def select_body(self, horizontal, vertical):
        width, height = self.glfw.get_window_size(self.window)
        if width <= 0 or height <= 0:
            return
        point = np.zeros(3, dtype=np.float64)
        geom_id = np.full(1, -1, dtype=np.int32)
        flex_id = np.full(1, -1, dtype=np.int32)
        skin_id = np.full(1, -1, dtype=np.int32)
        body_id = mujoco.mjv_select(self.model, self.data, self.option, width / height,
                                   horizontal / width, 1 - vertical / height, self.scene,
                                   point, geom_id, flex_id, skin_id)
        self.perturb.active = 0
        self.perturb.select = max(0, body_id)
        self.perturb.skinselect = int(skin_id[0])
        if body_id > 0:
            rotation = self.data.xmat[body_id].reshape(3, 3)
            self.perturb.localpos[:] = rotation.T @ (point - self.data.xpos[body_id])

    def mouse_button(self, window, button, action, mods):
        glfw = self.glfw
        if action == glfw.RELEASE:
            self.perturb.active = 0
            self.data.xfrc_applied[:] = 0
            return
        horizontal, vertical = glfw.get_cursor_pos(window)
        self.last_cursor = (horizontal, vertical)
        control = bool(mods & glfw.MOD_CONTROL)
        now = glfw.get_time()
        double_click = (button == glfw.MOUSE_BUTTON_LEFT and self.last_click_position is not None and
                        now - self.last_click_time < 0.3 and
                        np.hypot(horizontal - self.last_click_position[0],
                                 vertical - self.last_click_position[1]) < 5)
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.last_click_time = now
            self.last_click_position = (horizontal, vertical)
        if double_click or control and button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            self.select_body(horizontal, vertical)
        if control and self.perturb.select > 0:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.perturb.active = mujoco.mjtPertBit.mjPERT_ROTATE
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self.perturb.active:
                mujoco.mjv_initPerturb(self.model, self.data, self.scene, self.perturb)

    def apply_perturbation(self):
        self.data.xfrc_applied[:] = 0
        if self.perturb.active:
            mujoco.mjv_applyPerturbForce(self.model, self.data, self.perturb)

    def mouse_move(self, window, horizontal, vertical):
        previous = self.last_cursor
        self.last_cursor = (horizontal, vertical)
        if previous is None:
            return
        height = max(1, self.glfw.get_window_size(window)[1])
        if self.perturb.active:
            shifted = any(self.glfw.get_key(window, key) == self.glfw.PRESS
                          for key in (self.glfw.KEY_LEFT_SHIFT, self.glfw.KEY_RIGHT_SHIFT))
            if self.perturb.active == mujoco.mjtPertBit.mjPERT_TRANSLATE:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shifted else mujoco.mjtMouse.mjMOUSE_MOVE_V
            else:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shifted else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            mujoco.mjv_movePerturb(self.model, self.data, action,
                                  (horizontal - previous[0]) / height,
                                  (vertical - previous[1]) / height, self.scene, self.perturb)
            return
        if self.glfw.get_mouse_button(window, self.glfw.MOUSE_BUTTON_LEFT) != self.glfw.PRESS:
            return
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                             (horizontal - previous[0]) / height,
                             (vertical - previous[1]) / height, self.scene, self.cam)

    def poll_events(self):
        self.glfw.poll_events()

    def is_running(self):
        return not self.glfw.window_should_close(self.window)

    def set_texts(self, texts):
        self.texts = texts

    def sync(self):
        self.glfw.make_context_current(self.window)
        width, height = self.glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.option, self.perturb, self.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        for font_scale, position, title, text in self.texts:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, position, viewport, title, text, self.context)
        self.glfw.swap_buffers(self.window)

    def close(self):
        self.bridge.clear_keys()
        self.perturb.active = 0
        self.data.xfrc_applied[:] = 0
        self.context.free()
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()


def configure_camera(camera):
    camera.lookat[:] = [0, 0, 0.85]
    camera.distance = 2.6
    camera.azimuth = 135
    camera.elevation = -15


def add_stairs(scene, step_height, width, heading_degrees=0.0, flat_run=1.8):
    step_depth = 0.3
    num_steps = 5
    platform_length = 0.8
    segments = []
    for level in range(1, num_steps + 1):
        segments.append((flat_run + (level - 1) * step_depth, step_depth, level * step_height))
    platform_start = flat_run + num_steps * step_depth
    segments.append((platform_start, platform_length, num_steps * step_height))
    descent_start = platform_start + platform_length
    for level in range(1, num_steps):
        segments.append((descent_start + (level - 1) * step_depth,
                         step_depth, (num_steps - level) * step_height))
    for index, (start, length, height) in enumerate(segments):
        heading = np.deg2rad(heading_degrees)
        center = start + length / 2
        scene.worldbody.add_geom(name=f"actiongo_stair_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                 pos=[center * np.cos(heading), center * np.sin(heading), height / 2],
                                 quat=[np.cos(heading / 2), 0, 0, np.sin(heading / 2)],
                                 size=[length / 2, width / 2, height / 2], group=0,
                                 friction=[1, 0.005, 0.0001],
                                 rgba=[0.44, 0.52, 0.6, 1] if index % 2 == 0 else [0.65, 0.58, 0.43, 1])
    return {"type": "stairs", "flat_run": flat_run, "step_height": step_height,
            "step_depth": step_depth, "num_steps": num_steps, "platform_length": platform_length,
            "width": width, "end_x": descent_start + (num_steps - 1) * step_depth,
            "heading_degrees": heading_degrees}


def apply_tracking_initial_state(data, bridge, initial):
    pose = initial["initialPose"]
    state = initial["initialState"]
    quaternion = np.asarray(pose["rootQuaternionWxyz"], dtype=np.float64)
    yaw = np.arctan2(2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                     1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2))
    alignment = np.asarray([np.cos(-yaw / 2), 0, 0, np.sin(-yaw / 2)])
    aligned = np.empty(4)
    mujoco.mju_mulQuat(aligned, alignment, quaternion)
    data.qpos[:3] = [0, 0, pose["rootPosition"][2]]
    data.qpos[3:7] = aligned
    data.qpos[bridge.qpos_adr] = pose["jointPosition"]
    velocity = np.asarray(state["rootLinearVelocityWorld"])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, alignment)
    data.qvel[:3] = rotation.reshape(3, 3) @ velocity
    data.qvel[3:6] = state["rootAngularVelocityBody"]
    data.qvel[bridge.dof_adr] = state["jointVelocity"]


def render_video(model, frames, path):
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    configure_camera(camera)
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1280x720", "-r", "30", "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            for position in frames:
                data.qpos[:] = position
                mujoco.mj_forward(model, data)
                camera.lookat[:2] = data.qpos[:2]
                camera.lookat[2] = max(0.85, data.qpos[2] + 0.1)
                renderer.update_scene(data, camera=camera)
                encoder.stdin.write(renderer.render().tobytes())
    finally:
        encoder.stdin.close()
        returncode = encoder.wait(timeout=30)
    if returncode != 0:
        raise RuntimeError(f"video encoder failed: {returncode}")


def main():
    parser = argparse.ArgumentParser(description="P-BFM loopback DDS / MuJoCo smoke")
    parser.add_argument("--source", type=Path, required=True, help="P-BFM training repository")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--player", type=Path, default=ROOT / "build/generic_sensor_policy_player")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Seconds per run; 0 keeps the viewer open and restarts tracking at its end")
    parser.add_argument("--sim-dt", type=float, default=0.005,
                        help="PD integration step; upstream training uses 0.005 seconds")
    parser.add_argument("--domain-id", type=int, default=197)
    parser.add_argument("--drop-depth-after", type=float, help="Seconds after start; expect stale exit and damping")
    parser.add_argument("--viewer", action="store_true", help="Show a live MuJoCo window")
    parser.add_argument("--keyboard", action="store_true", help="Keyboard control in viewer; Enter starts, X exits")
    parser.add_argument("--keyboard-script", type=Path, help="Replay timed key events for a measured stop test")
    parser.add_argument("--idle-context", type=Path)
    parser.add_argument("--video", action="store_true", help="Render simulation states to output/pbfm.mp4 after inference")
    parser.add_argument("--context", help="Imported fixed preset ID, e.g. forward08-arms-spread")
    parser.add_argument("--tracking", type=Path, help="Tracking playback directory; keyboard mode uses only its initial state")
    parser.add_argument("--terrain", choices=["flat", "stairs"], default="flat")
    parser.add_argument("--step-height", type=float, default=0.12)
    parser.add_argument("--stair-width", type=float, default=6.0)
    parser.add_argument("--stair-approach", type=float, default=1.8,
                        help="Distance in meters from the initial root to the first stair edge")
    parser.add_argument("--start-on-platform", action="store_true",
                        help="Start at the center of the top stair platform")
    parser.add_argument("--terrain-heading-deg", type=float, default=0.0)
    args = parser.parse_args()
    if args.keyboard and not args.viewer:
        parser.error("--keyboard requires --viewer")
    if args.keyboard_script and (args.keyboard or args.duration <= 0):
        parser.error("--keyboard-script requires a positive duration, without --keyboard")
    if args.duration < 0 or not np.isfinite(args.duration):
        parser.error("--duration must be finite and nonnegative")
    if args.stair_approach < 0 or not np.isfinite(args.stair_approach):
        parser.error("--stair-approach must be finite and nonnegative")
    if args.start_on_platform and args.terrain != "stairs":
        parser.error("--start-on-platform requires --terrain stairs")
    if args.duration == 0 and (not args.viewer or args.video):
        parser.error("--duration 0 requires --viewer and cannot buffer --video")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    keyboard_control = args.keyboard or args.keyboard_script is not None
    if args.spec is None:
        args.spec = ROOT / "actions/pbfm/config" / ("generic_pbfm_joystick.yaml" if keyboard_control else "generic_pbfm.yaml")
    args.output.mkdir(parents=True, exist_ok=False)
    spec = yaml.safe_load(args.spec.read_text())
    if args.context:
        spec["context"]["path"] = f"context_{args.context}.yaml"
    for section, key in [("model", "path"), ("robot", "params"), ("context", "path")]:
        if key not in spec[section]:
            continue
        spec[section][key] = str((args.spec.parent / spec[section][key]).resolve())
    if args.tracking:
        if args.context:
            raise ValueError("Choose --tracking or --context")
        spec["context"]["path"] = str((args.tracking / "context.yaml").resolve())
    if keyboard_control and args.context:
        parser.error("keyboard control selects directional contexts; --context is for preset playback")
    key_events = json.loads(args.keyboard_script.read_text()) if args.keyboard_script else []
    if keyboard_control:
        if spec["context"].get("command_source") != "wireless_remote":
            parser.error("keyboard control requires a wireless_remote spec")
        for key in ("idle_path",):
            if key in spec["context"]:
                spec["context"][key] = str((args.spec.parent / spec["context"][key]).resolve())
        for key in ("directional_paths", "turn_paths"):
            for direction, path in spec["context"].get(key, {}).items():
                spec["context"][key][direction] = str((args.spec.parent / path).resolve())
        if args.idle_context:
            spec["context"]["idle_path"] = str(args.idle_context.resolve())
    runtime_spec = args.output.resolve() / "runtime_spec.yaml"
    runtime_spec.write_text(yaml.safe_dump(spec, sort_keys=False))
    params = yaml.safe_load((args.spec.parent / spec["robot"]["params"]).read_text())
    robot = params["robot"]
    simulation = params["simulation"]
    xml = args.source / "pbfm/data/robots/g1_mjlab/g1_29dof.xml"
    scene = mujoco.MjSpec.from_file(str(xml))
    scene.worldbody.add_geom(name="actiongo_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                             size=[50, 50, 0.5], pos=[0, 0, -0.5], group=0, friction=[1, 0.005, 0.0001],
                             rgba=[0.3, 0.35, 0.4, 1])
    scene.worldbody.add_light(pos=[0, -3, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    terrain = {"type": "flat", "floor_geometry": "box", "floor_surface_z": 0.0}
    if args.terrain == "stairs":
        terrain = add_stairs(scene, args.step_height, args.stair_width,
                            args.terrain_heading_deg, flat_run=args.stair_approach)
        terrain["floor_geometry"] = "box"
    spawn_offset = np.zeros(3)
    if args.start_on_platform:
        platform_center = (terrain["flat_run"] + terrain["num_steps"] * terrain["step_depth"]
                           + terrain["platform_length"] / 2)
        heading = np.deg2rad(terrain["heading_degrees"])
        spawn_offset[:] = [platform_center * np.cos(heading), platform_center * np.sin(heading),
                           terrain["num_steps"] * terrain["step_height"]]
    model = scene.compile()
    model.opt.timestep = args.sim_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    model.opt.tolerance = 1e-8
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    data = mujoco.MjData(model)
    ChannelFactoryInitialize(args.domain_id, "lo")
    bridge = KeyboardBridge(model, data) if keyboard_control else PbfmSimulationBridge(model, data)
    model.dof_armature[bridge.dof_adr] = simulation["armature"]
    model.dof_frictionloss[bridge.dof_adr] = simulation["frictionLoss"]
    efforts = np.asarray(simulation["effortLimit"])
    model.actuator_ctrlrange[bridge.motor_actuator_ids] = np.stack([-efforts, efforts], axis=1)
    bridge.ctrl_low = -efforts
    bridge.ctrl_high = efforts
    bridge.effort_limits = efforts
    bridge.velocity_limits = np.asarray(simulation["velocityLimit"])
    mujoco.mj_setConst(model, data)
    data.qpos[bridge.qpos_adr] = robot["default_angles"]
    if args.tracking:
        apply_tracking_initial_state(data, bridge, json.loads((args.tracking / "initial_state.json").read_text()))
    data.qpos[:3] += spawn_offset
    mujoco.mj_forward(model, data)
    camera = MujocoDepthCamera(model)
    shm_name = "actiongo_pbfm_smoke_" + str(args.output.stat().st_ino)
    depth = DepthPublisher(shm_name)
    ready = args.output.resolve() / "ready"
    start = args.output.resolve() / "start"
    trace = args.output.resolve() / "trace.jsonl"
    log = (args.output / "player.log").open("w")
    command = [str(args.player.resolve()), "--spec", str(runtime_spec), "--net", "lo",
               "--depth-source", "shm",
               "--domain-id", str(args.domain_id), "--depth-shm-name", shm_name,
               "--duration-sec", str(args.duration), "--ready-file", str(ready),
               "--start-file", str(start),
               "--event-log", str(args.output.resolve() / "events.jsonl")]
    if args.duration > 0:
        command.extend(["--trace-file", str(trace)])
    process = None
    viewer = None
    video_frames = []
    next_video_time = 0.0
    next_viewer_time = 0.0
    started = None
    minimum_height = float(data.qpos[2])
    maximum_height = minimum_height
    maximum_forward_x = float(data.qpos[0])
    steps = 0
    next_depth = 0.0
    origin = time.monotonic()
    deadline = origin
    result = {"terrain": terrain, "context_path": spec["context"].get("path", "realtime_udp"),
              "context_source": spec["context"].get("source", "cache"),
              "initial_root_position": data.qpos[:3].tolist(), "spawn_offset": spawn_offset.tolist()}
    if args.start_on_platform:
        print(f"[simulation] platform spawn: root={data.qpos[:3].tolist()}, surface_z={spawn_offset[2]}",
              flush=True)
    last_depth_time = origin
    episode = 1
    completed_simulation_time = 0.0
    next_key_event = 0
    next_sample_time = 0.0
    samples = []
    joint_samples = []
    try:
        if args.viewer:
            from mujoco import viewer as mujoco_viewer

            viewer = KeyboardViewer(model, data, bridge) if args.keyboard else mujoco_viewer.launch_passive(model, data)
            configure_camera(viewer.cam)
            viewer.sync()
        depth.write(camera.capture(model, data))
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while args.duration == 0 or args.keyboard or time.monotonic() - origin < args.duration + 20:
            now = time.monotonic()
            if args.keyboard:
                viewer.poll_events()
            if process.poll() is not None:
                if args.duration != 0 or process.returncode != 0 or not args.tracking or not viewer.is_running():
                    break
                completed_simulation_time += float(data.time)
                episode += 1
                ready.unlink(missing_ok=True)
                start.unlink(missing_ok=True)
                with bridge.lock:
                    bridge.usercmd = None
                mujoco.mj_resetData(model, data)
                apply_tracking_initial_state(data, bridge,
                    json.loads((args.tracking / "initial_state.json").read_text()))
                data.qpos[:3] += spawn_offset
                mujoco.mj_forward(model, data)
                depth.history = DepthHistory()
                depth.write(camera.capture(model, data))
                started = None
                deadline = now
                next_depth = now + 0.02
                last_depth_time = now
                print(f"[simulation] episode {episode}: restarting reference and physics", flush=True)
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if args.keyboard and bridge.exit_requested:
                process.terminate()
                process.wait(timeout=5)
                break
            if viewer is not None and not viewer.is_running():
                process.terminate()
                process.wait(timeout=5)
                break
            bridge.publish_lowstate()
            dropping = (args.drop_depth_after is not None and started is not None
                        and now - started >= args.drop_depth_after)
            if now >= next_depth and not dropping:
                depth.write(camera.capture(model, data))
                next_depth = now + 0.02
                last_depth_time = now
            if started is None and ready.exists():
                with bridge.lock:
                    hold = bridge.usercmd
                if hold is not None and (not args.keyboard or bridge.start_requested):
                    error = float(np.max(np.abs(hold.q - data.qpos[bridge.qpos_adr])))
                    if error > 1e-4:
                        raise RuntimeError(f"prestart command is not current-pose hold: {error}")
                    result["prestart_hold_max_error"] = error
                    bridge.set_command_mode("user")
                    start.write_text("start\n")
                    started = now
            if started is not None:
                while next_key_event < len(key_events) and data.time >= key_events[next_key_event]["time"]:
                    event = key_events[next_key_event]
                    if "axes" in event:
                        bridge.set_axes(*event["axes"])
                    else:
                        bridge.key_event(ord(event["key"]), event["pressed"])
                    next_key_event += 1
                bridge.apply_command()
                if args.keyboard:
                    viewer.apply_perturbation()
                previous_time = float(data.time)
                mujoco.mj_step(model, data)
                if data.time <= previous_time:
                    raise RuntimeError("MuJoCo reset after unstable integration; reduce --sim-dt")
                steps += 1
                minimum_height = min(minimum_height, float(data.qpos[2]))
                maximum_height = max(maximum_height, float(data.qpos[2]))
                maximum_forward_x = max(maximum_forward_x, float(data.qpos[0]))
                if keyboard_control and args.duration > 0 and data.time >= next_sample_time:
                    samples.append([data.time, *data.qpos[:3], *data.qvel[:3],
                                    bridge.speed, bridge.lateral, bridge.turn, *data.qvel[3:6], *data.qpos[3:7]])
                    joint_samples.append(data.qpos[bridge.qpos_adr].copy())
                    next_sample_time += 0.02
                if args.video and data.time >= next_video_time:
                    video_frames.append(data.qpos.copy())
                    next_video_time += 1 / 30
            if viewer is not None and now >= next_viewer_time:
                viewer.cam.lookat[:2] = data.qpos[:2]
                viewer.cam.lookat[2] = max(0.85, data.qpos[2] + 0.1)
                if args.keyboard:
                    viewer.set_texts([
                        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         "ActionGo P-BFM | " + ("RUNNING" if started is not None else
                                                "READY - ENTER to start" if ready.exists() else "Loading policy..."),
                         "HOLD W/S move | A/D turn | Q/E sidestep\nRELEASE to stop | SPACE stop | X exit\n"
                         "Double-click select | CTRL+RMB drag | CTRL+LMB rotate | SHIFT horizontal\n" + bridge.status())])
                viewer.sync()
                next_viewer_time = now + 1 / 30
            deadline += model.opt.timestep
            time.sleep(max(0.0, deadline - time.monotonic()))
        if process.poll() is None:
            raise RuntimeError("P-BFM smoke timed out before player exit")
        if viewer is not None and (not viewer.is_running() or args.keyboard and bridge.exit_requested):
            result["user_closed"] = True
        elif args.drop_depth_after is None:
            if process.returncode != 0:
                raise RuntimeError(f"player failed ({process.returncode}); see {args.output / 'player.log'}")
        else:
            log.flush()
            output = (args.output / "player.log").read_text()
            if process.returncode == 0 or "stale" not in output:
                raise RuntimeError("depth dropout did not cause stale failure")
            with bridge.lock:
                final_command = bridge.usercmd
            if final_command is None or np.any(final_command.kp != 0) or np.any(final_command.kd != 3):
                raise RuntimeError("depth dropout did not finish with damping LowCmd")
            result.update(stale_exit_and_damping=True,
                          last_depth_to_exit_sec=time.monotonic() - last_depth_time)
        policy_steps = 0
        max_raw_action = 0.0
        max_heading_error = 0.0
        max_turn_axis = 0.0
        command_yaw_range = [float("inf"), -float("inf")]
        with trace.open() if trace.exists() else nullcontext(()) as stream:
            for line in stream:
                frame = json.loads(line)
                policy_steps += 1
                max_raw_action = max(max_raw_action, float(np.max(np.abs(frame["action"]))))
                if "command_yaw" in frame:
                    error = frame["command_yaw"] - frame["current_yaw"]
                    max_heading_error = max(max_heading_error, abs(float(np.rad2deg(np.arctan2(np.sin(error), np.cos(error))))))
                    max_turn_axis = max(max_turn_axis, abs(frame["command_axes"][2]))
                    command_yaw_range[0] = min(command_yaw_range[0], frame["command_yaw"])
                    command_yaw_range[1] = max(command_yaw_range[1], frame["command_yaw"])
        if ((args.duration > 0 and policy_steps == 0) or steps == 0) and not result.get("user_closed"):
            raise RuntimeError("smoke completed without active policy / physics steps")
        result.update(policy_steps=policy_steps, physics_steps=steps,
                      simulated_duration=completed_simulation_time + float(data.time),
                      episodes=episode, trace_enabled=args.duration > 0, min_pelvis_height=minimum_height,
                      final_pelvis_height=float(data.qpos[2]),
                      max_pelvis_height=maximum_height, max_forward_x=maximum_forward_x,
                      final_position=data.qpos[:3].tolist(),
                      max_raw_action=max_raw_action if args.duration > 0 else None,
                      finite_state=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
                      simulation_dt=float(model.opt.timestep),
                      integrator="implicitfast", motor_model="speed_dependent_dc",
                      upstream_training_dt=simulation["physicsDt"],
                      scope="loopback DDS, terrain raycast, upstream robot parameters, runtime safety filter")
        if keyboard_control and policy_steps:
            result.update(max_heading_error_deg=max_heading_error, max_turn_axis=max_turn_axis,
                          command_yaw_range_rad=command_yaw_range)
        if samples:
            samples = np.asarray(samples)
            np.savez_compressed(args.output / "movement.npz", time=samples[:, 0],
                                position=samples[:, 1:4], velocity=samples[:, 4:7], command=samples[:, 7:10],
                                angular_velocity=samples[:, 10:13], quaternion_wxyz=samples[:, 13:17],
                                joint_position=joint_samples, joint_names=np.asarray(robot["joint_order"]))
            result["stop_windows"] = []
            moving = np.any(samples[:, 7:10] != 0, axis=1)
            releases = np.flatnonzero(moving[:-1] & ~moving[1:]) + 1
            speed = np.linalg.norm(samples[:, 4:6], axis=1)
            for release in releases:
                next_press = np.flatnonzero(moving[release:])
                end = release + next_press[0] if len(next_press) else len(samples)
                settled = None
                for index in range(release, end - 24):
                    if np.all(speed[index:index + 25] < 0.08):
                        settled = index
                        break
                late = np.flatnonzero(samples[release:end, 0] >= samples[release, 0] + 1) + release
                result["stop_windows"].append({
                    "release_sim_time": float(samples[release, 0]),
                    "settle_below_008_mps_for_05_sec": None if settled is None else
                        float(samples[settled, 0] - samples[release, 0]),
                    "displacement_after_release_m": float(np.linalg.norm(samples[end - 1, 1:3] - samples[release, 1:3])),
                    "late_mean_speed_mps": float(speed[late].mean()) if len(late) else None,
                    "late_displacement_m": float(np.linalg.norm(samples[late[-1], 1:3] - samples[late[0], 1:3]))
                        if len(late) else None})
        events = [json.loads(line) for line in (args.output / "events.jsonl").read_text().splitlines()]
        metrics = [event for event in events if event["event"] == "runtime_metrics"]
        if metrics:
            result["runtime_metrics"] = metrics[-1]
        if not result["finite_state"]:
            raise RuntimeError("nonfinite simulation state")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        if viewer is not None:
            viewer.close()
        depth.close()
        log.close()
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.video:
        path = args.output / "pbfm.mp4"
        render_video(model, video_frames, path)
        print(f"Video: {path.resolve()}")


if __name__ == "__main__":
    main()
