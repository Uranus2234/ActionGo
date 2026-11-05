import argparse
import json
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
import yaml

from action_go_sim_remote import NativeActionGoBridge


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "actions/tracking/ufo_duobox_trial"


class TrialBridge(NativeActionGoBridge):
    def __init__(self, model, data, simulation):
        super().__init__(model, data, require_torque_motors=True)
        self.efforts = np.asarray(simulation["effort_limit"])
        self.velocities = np.asarray(simulation["velocity_limit"])
        model.dof_armature[self.dof_adr] = simulation["armature"]
        model.dof_frictionloss[self.dof_adr] = simulation["friction"]
        self.ctrl_low = -self.efforts
        self.ctrl_high = self.efforts
        model.actuator_ctrlrange[self.motor_actuator_ids] = np.stack([-self.efforts, self.efforts], axis=1)

    def apply_command(self):
        applied = super().apply_command()
        velocity = np.clip(self.data.qvel[self.dof_adr], -2 * self.velocities, 2 * self.velocities)
        lower = np.maximum(-self.efforts, -self.efforts * (1 + velocity / self.velocities))
        upper = np.minimum(self.efforts, self.efforts * (1 - velocity / self.velocities))
        self.data.ctrl[self.motor_actuator_ids] = np.clip(self.data.ctrl[self.motor_actuator_ids], lower, upper)
        return applied


def main():
    parser = argparse.ArgumentParser(description="Loopback-only C++ UFO inference and finite handoff check")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", choices=["reference", "default"], default="reference")
    parser.add_argument("--stop-after", type=float)
    parser.add_argument("--drop-state-after", type=float)
    parser.add_argument("--domain-id", type=int, default=191)
    args = parser.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    args.output.mkdir(parents=True, exist_ok=False)
    params = yaml.safe_load((PACKAGE / "config/Robot.yaml").read_text())
    scene = mujoco.MjSpec.from_file(str(args.scene.resolve()))
    scene.worldbody.add_geom(name="trial_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[50, 50, 0.1], friction=[1, 0.005, 0.0001])
    model = scene.compile()
    model.opt.timestep = 0.002
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    ChannelFactoryInitialize(args.domain_id, "lo")
    bridge = TrialBridge(model, data, params["simulation"])
    mujoco.mj_setConst(model, data)
    if args.initial == "reference":
        reference = np.load(PACKAGE / "motions/duobox_050_ma_reference.npz")["reference_qpos"]
        data.qpos[:7] = reference[0, :7]
        data.qpos[bridge.qpos_adr] = reference[0, 7:]
        mujoco.mj_differentiatePos(model, data.qvel, 0.02, reference[0], reference[1])
    else:
        data.qpos[bridge.qpos_adr] = params["robot"]["default_angles"]
    mujoco.mj_forward(model, data)
    ready, start, complete, release = [args.output.resolve() / name for name in ("ready", "start", "complete", "release")]
    command = [str(ROOT / "build/generic_policy_player"), "--net", "lo", "--domain-id", str(args.domain_id),
               "--spec", str(PACKAGE / "config/generic_ufo_duobox_050_ma.yaml"),
               "--ready-file", str(ready), "--start-file", str(start),
               "--handoff-complete-file", str(complete), "--handoff-release-file", str(release),
               "--event-log", str(args.output.resolve() / "events.jsonl"), "--no-damping-on-stop"]
    started = completed = None
    stopped = False
    heights = []
    positions = []
    with (args.output / "player.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        origin = deadline = time.monotonic()
        try:
            while time.monotonic() - origin < 45:
                now = time.monotonic()
                if process.poll() is not None:
                    break
                dropping = args.drop_state_after is not None and started is not None and now - started >= args.drop_state_after
                if not dropping:
                    bridge.publish_lowstate()
                if started is None and ready.exists() and bridge.usercmd is not None:
                    hold_error = float(np.max(np.abs(bridge.usercmd.q - data.qpos[bridge.qpos_adr])))
                    if hold_error > 1e-4:
                        raise RuntimeError(f"prestart hold mismatch: {hold_error}")
                    bridge.set_command_mode("user")
                    start.touch()
                    started = now
                if started is not None:
                    bridge.apply_command()
                    mujoco.mj_step(model, data)
                    heights.append(float(data.qpos[2]))
                    positions.append(data.qpos.copy())
                    if not np.isfinite(data.qpos).all():
                        raise RuntimeError("nonfinite simulation state")
                    if args.stop_after is not None and not stopped and now - started >= args.stop_after:
                        process.terminate()
                        stopped = True
                if complete.exists() and completed is None:
                    completed = now
                if completed is not None and now - completed >= 0.3 and not release.exists():
                    if process.poll() is not None:
                        raise RuntimeError("player exited before release")
                    release.touch()
                deadline += model.opt.timestep
                time.sleep(max(0, deadline - time.monotonic()))
            else:
                raise RuntimeError("trial timeout")
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    result = {"initial": args.initial, "returncode": process.returncode,
              "prestart_hold_error": hold_error if started else None,
              "minimum_root_height": min(heights) if heights else None,
              "final_root_height": heights[-1] if heights else None,
              "sim_seconds": float(data.time), "finite_completion": completed is not None,
              "completion_wall_seconds": completed - started if completed else None,
              "release_barrier_checked": release.exists(), "stop_after": args.stop_after,
              "drop_state_after": args.drop_state_after}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output / "rollout.npz", qpos=np.array(positions))
    print(json.dumps(result, indent=2))
    if args.drop_state_after is not None:
        assert process.returncode != 0
    elif args.stop_after is not None:
        assert process.returncode == 0 and stopped
    else:
        assert process.returncode == 0 and result["finite_completion"] and result["release_barrier_checked"]


if __name__ == "__main__":
    main()
