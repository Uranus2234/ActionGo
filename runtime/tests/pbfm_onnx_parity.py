import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnxruntime as ort
import yaml


ROOT = Path(__file__).resolve().parents[2]


def make_scenario(name, params, model, latent, pbfm=False, velocity=None,
                  heading_valid=False, heading_relative=(1, 0)):
    robot = yaml.safe_load(params.read_text())["robot"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(3927)
    defaults = np.asarray(robot["default_angles"], dtype=np.float32)
    dimensions = {"actions": 29, "base_ang_vel": 3, "dof_pos": 29, "dof_vel": 29, "projected_gravity": 3}
    history = {key: np.zeros((4, dimension), dtype=np.float32) for key, dimension in dimensions.items()}
    last_action = np.zeros(29, dtype=np.float32)
    latent = np.asarray(latent, dtype=np.float32)
    velocity = np.asarray(velocity if velocity is not None else [0, 0, 0], dtype=np.float32)
    frames = []
    for step in range(24):
        position = defaults + generator.normal(0, 0.025, 29).astype(np.float32)
        joint_velocity = generator.normal(0, 0.1, 29).astype(np.float32)
        gyro = generator.normal(0, 0.15, 3).astype(np.float32)
        gravity = np.asarray([0.02 * np.sin(step), 0.02 * np.cos(step), -np.sqrt(1 - 0.02 ** 2)], dtype=np.float32)
        state = np.concatenate([position - defaults, joint_velocity, gravity, gyro * np.float32(0.25)])
        values = {"actions": last_action, "base_ang_vel": gyro * np.float32(0.25),
                  "dof_pos": position - defaults, "dof_vel": joint_velocity, "projected_gravity": gravity}

        def append_history():
            for key, value in values.items():
                history[key][1:] = history[key][:-1].copy()
                history[key][0] = value

        if not pbfm:
            append_history()
        historical = np.concatenate([history[key].reshape(-1) for key in sorted(history)])
        record = {"q": position, "dq": joint_velocity, "gravity": gravity, "gyro": gyro,
                  "state": state, "last_action": last_action, "history_actor": historical}
        if pbfm:
            yaw = np.float32(step * 0.07)
            error = np.arctan2(heading_relative[1], heading_relative[0]) - yaw
            heading = np.asarray([1 - np.cos(error), np.sin(error)], dtype=np.float32) * heading_valid
            depth = generator.uniform(0.05, 1.0, (1, 8, 18, 32)).astype(np.float32)
            feed = {"state": state[None], "last_action": last_action[None], "history_actor": historical[None],
                    "heading": heading[None], "velocity_command": velocity[None], "depth_image": depth,
                    "z": latent[None]}
            action, estimated = session.run(None, feed)
            record.update(yaw=float(yaw), heading=heading, depth=depth.reshape(-1), estimated_velocity=estimated[0])
            append_history()
        else:
            actor = np.concatenate([state, last_action, historical, latent])
            action = session.run(None, {"actor_obs": actor[None]})[0]
            record["actor_obs"] = actor
        action = action[0]
        scaled = np.float32(5) * np.clip(action, -1, 1)
        target = np.clip(defaults + np.asarray(robot["action_scale"], dtype=np.float32) * scaled,
                         np.asarray(robot["joint_pos_lower_limit"], dtype=np.float32),
                         np.asarray(robot["joint_pos_upper_limit"], dtype=np.float32))
        record.update(action=action, scaled_action=scaled, target_q=target)
        frames.append(record)
        last_action = scaled
    return {"name": name, "variant": "pbfm" if pbfm else "bfm_zero", "model": str(model),
            "default_angles": defaults, "action_scale": robot["action_scale"],
            "lower": robot["joint_pos_lower_limit"], "upper": robot["joint_pos_upper_limit"],
            "z": latent, "velocity_command": velocity, "heading_valid": heading_valid,
            "heading_relative": heading_relative, "frames": frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-binary", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for package, params in [("bfm0", "BFMZero.yaml"), ("ufo", "UFO.yaml")]:
        folder = ROOT / "actions" / package
        spec_name = "generic_bfm_zero.yaml" if package == "bfm0" else "generic_ufo_stand.yaml"
        spec = yaml.safe_load((folder / "config" / spec_name).read_text())
        latent = np.loadtxt(folder / "config" / spec["observation"]["latent_file"], dtype=np.float32)
        cases.append(make_scenario(package, folder / "config" / params,
                                   folder / "model/FBcprAuxModel.onnx", latent))
    folder = ROOT / "actions/pbfm"
    for filename in ["context_arms-up.yaml", "context_forward08-arms-spread.yaml", "context_rotate-inplace.yaml"]:
        context = yaml.safe_load((folder / "config" / filename).read_text())
        cases.append(make_scenario(filename, folder / "config/PBFM.yaml", folder / "model/policy.onnx",
                                   context["z"], pbfm=True, velocity=context["velocity_command"],
                                   heading_valid=context["heading_valid"],
                                   heading_relative=context["heading_relative_xy"]))
    with tempfile.TemporaryDirectory(prefix="actiongo-pbfm-parity-") as temporary:
        fixture = Path(temporary) / "replay.json"
        fixture.write_text(json.dumps(cases, default=lambda value: value.tolist()))
        subprocess.run([str(args.test_binary.resolve()), str(fixture)], check=True)


if __name__ == "__main__":
    main()
