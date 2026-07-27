#!/usr/bin/env python3
"""Interactive MuJoCo preview for action_go's BFM-Zero stand action.

This is a simulation-only companion for runtime/players/generic_policy_player.cpp. It reuses
`actions/bfm0/config/BFMZero.yaml` and the `z_file` configured in `actions/bfm0/config/BFMZero.yaml`, builds the
same 721-dim BFM-Zero actor input, runs the ONNX policy, then applies the
resulting q_target through a MuJoCo PD controller.

Default behavior opens a MuJoCo viewer and runs until the viewer window is
closed. No video is recorded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

# Interactive viewer needs an OpenGL platform. If an X server is available,
# GLFW is usually the right choice. If the user explicitly set MUJOCO_GL,
# respect it.
os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


ACTION_GO_ROOT = Path(__file__).resolve().parents[2]
BFM_ZERO_ROOT = ACTION_GO_ROOT.parent / "BFM-Zero"

NUM_DOFS = 29
Z_DIM = 256
HISTORY_STEPS = 4
POLICY_OBS_DIM = 465
ACTOR_OBS_DIM = 721


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive MuJoCo preview for action_go BFM-Zero inference")
    p.add_argument("--xml", default=str(BFM_ZERO_ROOT / "data/robots/g1/scene_29dof_freebase.xml"))
    p.add_argument("--model", default=str(ACTION_GO_ROOT / "actions/bfm0/model/FBcprAuxModel.onnx"))
    p.add_argument("--params", default=str(ACTION_GO_ROOT / "actions/bfm0/config/BFMZero.yaml"))
    p.add_argument("--z", default=None, help="latent txt; defaults to z_file in --params")
    p.add_argument("--duration", type=float, default=0.0, help="simulation seconds; 0 means run until window closes")
    p.add_argument("--sim-dt", type=float, default=0.002)
    p.add_argument("--policy-hz", type=float, default=50.0)
    p.add_argument("--target-interp-sec", type=float, default=0.02, help="interpolate new policy q targets over this time; 0 disables")
    p.add_argument("--safety-demo", action="store_true", help="start limp, press SPACE to enable IMU safety: fallen -> BFM-Zero recovery -> walkrun-zero")
    p.add_argument("--fall-angle-deg", type=float, default=57.3, help="safety fallen threshold; RoboJuDo uses 1.0 rad ~= 57.3 deg")
    p.add_argument("--recover-angle-deg", type=float, default=20.0, help="upright threshold before returning to walkrun-zero")
    p.add_argument("--fall-hold-ms", type=float, default=60.0, help="fallen tilt debounce time")
    p.add_argument("--recover-hold-ms", type=float, default=1200.0, help="upright stable time before switching back")
    p.add_argument("--walkrun-grace-sec", type=float, default=2.0, help="after switching to simulated walkrun-zero, ignore fall re-trigger for this many seconds so the transition is visible")
    p.add_argument("--auto-start-sec", type=float, default=-1.0, help="safety-demo headless helper: auto press SPACE after this many sim seconds; negative disables")
    p.add_argument("--headless", action="store_true", help="run physics/policy without opening the viewer; for smoke tests only")
    p.add_argument("--print-every", type=int, default=50, help="print every N policy/demo steps; 0 disables logs")
    return p.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def load_vec(path_or_values, expected: int, name: str) -> np.ndarray:
    arr = np.asarray(path_or_values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected:
        raise ValueError(f"{name} dim mismatch: got {arr.shape[0]}, expected {expected}")
    return arr


def quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-6:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        q = q / n
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    # Same as quat_rotate_inverse(q, [0,0,-1]): R(q)^T * g_world.
    return quat_to_matrix_wxyz(quat_wxyz).T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)


def tilt_deg_from_quat(quat_wxyz: np.ndarray) -> float:
    gravity = projected_gravity(quat_wxyz)
    return float(np.degrees(np.arccos(np.clip(-gravity[2], -1.0, 1.0))))


def roll_history(history: np.ndarray, value: np.ndarray) -> np.ndarray:
    history = np.roll(history, 1, axis=0)
    history[0] = value
    return history


class BFMZeroMujocoPreview:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        cfg = load_yaml(args.params)
        self.default = load_vec(cfg["default_angles"], NUM_DOFS, "default_angles")
        self.action_scale = load_vec(cfg["action_scale"], NUM_DOFS, "action_scale")
        self.kp = load_vec(cfg["kp"], NUM_DOFS, "kp")
        self.kd = load_vec(cfg["kd"], NUM_DOFS, "kd")
        self.lower = load_vec(cfg.get("joint_pos_lower_limit", [-1e6] * NUM_DOFS), NUM_DOFS, "joint_pos_lower_limit")
        self.upper = load_vec(cfg.get("joint_pos_upper_limit", [1e6] * NUM_DOFS), NUM_DOFS, "joint_pos_upper_limit")
        self.action_rescale = float(cfg.get("action_rescale", 5.0))
        z_path = Path(args.z) if args.z else Path(args.params).parent / cfg.get("z_file", "bfm_zero_goal_getup_z.txt")
        self.z_path = z_path
        self.z = np.loadtxt(z_path, dtype=np.float32).reshape(-1)
        if self.z.shape[0] != Z_DIM:
            raise ValueError(f"z dim mismatch: {self.z.shape}")

        self.session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.model = mujoco.MjModel.from_xml_path(args.xml)
        self.model.opt.timestep = args.sim_dt
        self.data = mujoco.MjData(self.model)
        self.reset_model_pose()

        self.last_action = np.zeros(NUM_DOFS, dtype=np.float32)
        self.hist_prev_actions = np.zeros((HISTORY_STEPS, NUM_DOFS), dtype=np.float32)
        self.hist_base_ang_vel = np.zeros((HISTORY_STEPS, 3), dtype=np.float32)
        self.hist_dof_pos = np.zeros((HISTORY_STEPS, NUM_DOFS), dtype=np.float32)
        self.hist_dof_vel = np.zeros((HISTORY_STEPS, NUM_DOFS), dtype=np.float32)
        self.hist_projected_gravity = np.zeros((HISTORY_STEPS, 3), dtype=np.float32)
        self.q_target = self.default.copy()
        self.q_target_from = self.default.copy()
        self.q_target_to = self.default.copy()
        self.q_target_interp_start = 0.0

        self.torque_limits = np.abs(self.model.actuator_ctrlrange[6 : 6 + NUM_DOFS, 1]).astype(np.float32)
        self.torque_limits[self.torque_limits <= 0] = 1e6
        self.policy_counter = 0
        self.demo_step_counter = 0

        # Safety demo state machine:
        #   LIMP          : no policy/no torque, robot is allowed to collapse.
        #   WALKRUN_ZERO  : stand-in for encrypted official walkrun; action=0 -> default pose PD.
        #   RECOVERY      : BFM-Zero policy drives get-up/stand.
        #   BFM_ZERO      : normal preview mode, always run BFM-Zero.
        self.demo_state = "LIMP" if args.safety_demo else "BFM_ZERO"
        self.start_requested = False
        self.force_recovery_requested = False
        self.reset_requested = False
        self.force_walkrun_requested = False
        self.fall_since: float | None = None
        self.recover_since: float | None = None
        self.last_demo_state_print = -1.0
        self.walkrun_grace_until = 0.0

        print(
            f"[bfm_zero_mujoco] model={args.model}\n"
            f"[bfm_zero_mujoco] xml={args.xml}\n"
            f"[bfm_zero_mujoco] z={self.z_path}\n"
            f"[bfm_zero_mujoco] z_source={cfg.get('z_source', '<unknown>')}\n"
            f"[bfm_zero_mujoco] z_norm={np.linalg.norm(self.z):.4f}, input={self.input_name}, output={self.output_name}"
        )
        if self.args.safety_demo:
            print(
                "[safety_demo] initial state=LIMP: no policy output/no torque, let the robot collapse.\n"
                "[safety_demo] Press SPACE to enable monitor: tilt fall -> BFM-Zero recovery -> WALKRUN_ZERO.\n"
                "[safety_demo] Keys: SPACE=start monitor, G=force BFM-Zero, W=force walkrun-zero, R=reset limp."
            )

    def reset_model_pose(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[7 : 7 + NUM_DOFS] = self.default
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_demo_limp(self) -> None:
        self.reset_model_pose()
        self.data.ctrl[:] = 0.0
        self.last_action[:] = 0.0
        self.hist_prev_actions[:] = 0.0
        self.hist_base_ang_vel[:] = 0.0
        self.hist_dof_pos[:] = 0.0
        self.hist_dof_vel[:] = 0.0
        self.hist_projected_gravity[:] = 0.0
        self.q_target = self.default.copy()
        self.q_target_from = self.default.copy()
        self.q_target_to = self.default.copy()
        self.q_target_interp_start = float(self.data.time)
        self.demo_state = "LIMP"
        self.start_requested = False
        self.force_recovery_requested = False
        self.force_walkrun_requested = False
        self.fall_since = None
        self.recover_since = None
        self.walkrun_grace_until = 0.0
        print(f"[safety_demo] RESET -> LIMP at t={self.data.time:.2f}s")

    def base_tilt_deg(self) -> float:
        return tilt_deg_from_quat(self.data.qpos[3:7].astype(np.float32).copy())

    def set_zero_policy_target(self) -> None:
        # Stand-in for official encrypted walkrun in sim: policy output is all zeros,
        # so target is just default_angles with the same PD gains.
        self.last_action[:] = 0.0
        self.q_target_from = self.current_q_target().copy()
        self.q_target_to = self.default.astype(np.float32).copy()
        self.q_target_interp_start = float(self.data.time)
        self.q_target = self.q_target_to

    def switch_demo_state(self, state: str, reason: str) -> None:
        old = self.demo_state
        if old == state:
            return
        self.demo_state = state
        self.fall_since = None
        self.recover_since = None
        if state == "WALKRUN_ZERO":
            self.set_zero_policy_target()
            # Only suppress re-trigger after a successful recovery, so pressing
            # SPACE on an already-fallen robot still enters BFM-Zero immediately.
            self.walkrun_grace_until = (
                float(self.data.time) + max(0.0, float(self.args.walkrun_grace_sec))
                if old == "RECOVERY"
                else 0.0
            )
        elif state == "RECOVERY":
            # Start BFM-Zero from the current fallen pose; keep histories as-is/zeroed.
            self.q_target_from = self.current_q_target().copy()
            self.q_target_to = self.q_target_from.copy()
            self.q_target_interp_start = float(self.data.time)
        print(
            f"[safety_demo] STATE {old} -> {state} at t={self.data.time:.2f}s "
            f"tilt={self.base_tilt_deg():.1f}deg base_z={self.data.qpos[2]:.3f} reason={reason}"
        )

    def handle_demo_requests(self) -> None:
        if self.reset_requested:
            self.reset_requested = False
            self.reset_demo_limp()
            return
        if self.start_requested and self.demo_state == "LIMP":
            self.start_requested = False
            self.switch_demo_state("WALKRUN_ZERO", "SPACE/start monitor")
        if self.force_recovery_requested:
            self.force_recovery_requested = False
            self.switch_demo_state("RECOVERY", "G/force recovery")
        if self.force_walkrun_requested:
            self.force_walkrun_requested = False
            self.switch_demo_state("WALKRUN_ZERO", "W/force walkrun-zero")

    def safety_demo_tick(self) -> None:
        self.handle_demo_requests()
        if not self.args.safety_demo:
            return

        t = float(self.data.time)
        tilt = self.base_tilt_deg()
        fall_hold = max(0.0, float(self.args.fall_hold_ms)) / 1000.0
        recover_hold = max(0.0, float(self.args.recover_hold_ms)) / 1000.0

        if self.demo_state == "LIMP":
            return

        if self.demo_state == "WALKRUN_ZERO":
            if t < self.walkrun_grace_until:
                self.fall_since = None
                return
            if tilt >= self.args.fall_angle_deg:
                if self.fall_since is None:
                    self.fall_since = t
                if t - self.fall_since >= fall_hold:
                    self.switch_demo_state(
                        "RECOVERY",
                        f"tilt {tilt:.1f}deg >= {self.args.fall_angle_deg:.1f}deg for {self.args.fall_hold_ms:.0f}ms",
                    )
            else:
                self.fall_since = None
            return

        if self.demo_state == "RECOVERY":
            if tilt <= self.args.recover_angle_deg:
                if self.recover_since is None:
                    self.recover_since = t
                if t - self.recover_since >= recover_hold:
                    self.switch_demo_state(
                        "WALKRUN_ZERO",
                        f"tilt {tilt:.1f}deg <= {self.args.recover_angle_deg:.1f}deg for {self.args.recover_hold_ms:.0f}ms",
                    )
            else:
                self.recover_since = None

    @property
    def dof_pos(self) -> np.ndarray:
        return self.data.qpos[7 : 7 + NUM_DOFS].astype(np.float32).copy()

    @property
    def dof_vel(self) -> np.ndarray:
        return self.data.qvel[6 : 6 + NUM_DOFS].astype(np.float32).copy()

    def build_actor_obs(self) -> np.ndarray:
        dof_pos_minus_default = self.dof_pos - self.default
        dof_vel = self.dof_vel
        base_quat = self.data.qpos[3:7].astype(np.float32).copy()  # MuJoCo freejoint quaternion is wxyz.
        base_ang_vel = self.data.qvel[3:6].astype(np.float32).copy() * 0.25
        gravity = projected_gravity(base_quat).astype(np.float32)

        self.hist_prev_actions = roll_history(self.hist_prev_actions, self.last_action)
        self.hist_base_ang_vel = roll_history(self.hist_base_ang_vel, base_ang_vel)
        self.hist_dof_pos = roll_history(self.hist_dof_pos, dof_pos_minus_default)
        self.hist_dof_vel = roll_history(self.hist_dof_vel, dof_vel)
        self.hist_projected_gravity = roll_history(self.hist_projected_gravity, gravity)

        policy_obs = np.concatenate(
            [
                dof_pos_minus_default,
                dof_vel,
                gravity,
                base_ang_vel,
                self.last_action,
                self.hist_prev_actions.reshape(-1),
                self.hist_base_ang_vel.reshape(-1),
                self.hist_dof_pos.reshape(-1),
                self.hist_dof_vel.reshape(-1),
                self.hist_projected_gravity.reshape(-1),
            ]
        ).astype(np.float32)
        if policy_obs.shape[0] != POLICY_OBS_DIM:
            raise RuntimeError(f"policy obs dim {policy_obs.shape[0]} != {POLICY_OBS_DIM}")
        actor_obs = np.concatenate([policy_obs, self.z]).astype(np.float32)
        if actor_obs.shape[0] != ACTOR_OBS_DIM:
            raise RuntimeError(f"actor obs dim {actor_obs.shape[0]} != {ACTOR_OBS_DIM}")
        return actor_obs

    def current_q_target(self) -> np.ndarray:
        interp_sec = max(0.0, float(self.args.target_interp_sec))
        if interp_sec <= 1e-6:
            return self.q_target_to
        alpha = np.clip((self.data.time - self.q_target_interp_start) / interp_sec, 0.0, 1.0)
        return ((1.0 - alpha) * self.q_target_from + alpha * self.q_target_to).astype(np.float32)

    def policy_step(self) -> None:
        obs = self.build_actor_obs()[None, :]
        action = self.session.run([self.output_name], {self.input_name: obs})[0].astype(np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action * self.action_rescale
        next_q_target = np.clip(self.default + self.last_action * self.action_scale, self.lower, self.upper)
        self.q_target_from = self.current_q_target().copy()
        self.q_target_to = next_q_target.astype(np.float32)
        self.q_target_interp_start = float(self.data.time)
        self.q_target = self.q_target_to
        self.policy_counter += 1
        if self.args.print_every > 0 and self.policy_counter % self.args.print_every == 0:
            print(
                f"[bfm_zero_mujoco] t={self.data.time:.2f}s state={self.demo_state} "
                f"base_z={self.data.qpos[2]:.3f} tilt={self.base_tilt_deg():.1f}deg "
                f"action_norm={np.linalg.norm(self.last_action):.3f}"
            )

    def sim_step(self) -> None:
        self.data.ctrl[:] = 0.0
        if not (self.args.safety_demo and self.demo_state == "LIMP"):
            q_cmd = self.current_q_target()
            torque = self.kp * (q_cmd - self.dof_pos) - self.kd * self.dof_vel
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl[6 : 6 + NUM_DOFS] = torque
        mujoco.mj_step(self.model, self.data)

    def run(self) -> None:
        if self.args.headless:
            self.run_headless()
        else:
            self.run_viewer()

    def run_headless(self) -> None:
        duration = self.args.duration if self.args.duration > 0 else (8.0 if self.args.safety_demo else 2.0)
        steps = int(round(duration / self.args.sim_dt))
        policy_interval = max(1, int(round((1.0 / self.args.policy_hz) / self.args.sim_dt)))
        for step in range(steps):
            if self.args.safety_demo and self.args.auto_start_sec >= 0 and self.data.time >= self.args.auto_start_sec:
                self.start_requested = True
                self.args.auto_start_sec = -1.0
            if self.args.safety_demo:
                self.safety_demo_tick()
                if step % policy_interval == 0 and self.demo_state == "RECOVERY":
                    self.policy_step()
            else:
                if step % policy_interval == 0:
                    self.policy_step()
            self.sim_step()
            if self.args.safety_demo and self.args.print_every > 0 and step % max(1, int(self.args.print_every)) == 0:
                print(
                    f"[safety_demo] t={self.data.time:.2f}s state={self.demo_state} "
                    f"tilt={self.base_tilt_deg():.1f}deg base_z={self.data.qpos[2]:.3f}"
                )
        print(
            f"[bfm_zero_mujoco] headless done: t={self.data.time:.2f}s state={self.demo_state} "
            f"base_z={self.data.qpos[2]:.3f} tilt={self.base_tilt_deg():.1f}deg"
        )

    def run_viewer(self) -> None:
        policy_interval = max(1, int(round((1.0 / self.args.policy_hz) / self.args.sim_dt)))
        duration = self.args.duration
        start_wall = time.time()

        def key_callback(keycode: int) -> None:
            if keycode == ord(" "):
                self.start_requested = True
            elif keycode in (ord("g"), ord("G")):
                self.force_recovery_requested = True
            elif keycode in (ord("w"), ord("W")):
                self.force_walkrun_requested = True
            elif keycode in (ord("r"), ord("R")):
                self.reset_requested = True

        print(
            "[bfm_zero_mujoco] interactive viewer opened. Close the window to stop.\n"
            "[bfm_zero_mujoco] You can rotate/pan/zoom with the mouse; use MuJoCo's perturb/drag controls to disturb bodies."
        )
        if self.args.safety_demo:
            print(
                "[safety_demo] Controls: SPACE=start safety monitor, G=force BFM-Zero, "
                "W=force walkrun-zero, R=reset limp."
            )
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -12.0
            viewer.cam.azimuth = 180.0
            step = 0
            while viewer.is_running():
                loop_start = time.time()
                if duration > 0 and self.data.time >= duration:
                    break
                if self.args.safety_demo:
                    self.safety_demo_tick()
                    if step % policy_interval == 0 and self.demo_state == "RECOVERY":
                        self.policy_step()
                    if self.args.print_every > 0 and step % max(1, int(self.args.print_every)) == 0:
                        print(
                            f"[safety_demo] t={self.data.time:.2f}s state={self.demo_state} "
                            f"tilt={self.base_tilt_deg():.1f}deg base_z={self.data.qpos[2]:.3f}"
                        )
                else:
                    if step % policy_interval == 0:
                        self.policy_step()
                self.sim_step()
                viewer.cam.lookat[:] = self.data.qpos[:3]
                viewer.sync()
                step += 1

                # Keep approximately real-time. Manual dragging/interaction still
                # works because the passive viewer is synced every sim step.
                elapsed = time.time() - loop_start
                sleep_time = self.args.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print(
            f"[bfm_zero_mujoco] viewer closed: wall={time.time() - start_wall:.1f}s "
            f"sim={self.data.time:.2f}s state={self.demo_state} tilt={self.base_tilt_deg():.1f}deg"
        )


def main() -> None:
    BFMZeroMujocoPreview(parse_args()).run()


if __name__ == "__main__":
    main()
