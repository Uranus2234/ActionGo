import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml


MOTION = "duobox_g1_export__04_sparring_050_ma_free_liu_free__ma"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FlatActor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.normalizer = model._obs_normalizer
        self.actor = model._actor
        self.std = model.cfg.actor_std

    def forward(self, actor_obs):
        observation = self.normalizer({
            "state": actor_obs[:, :64],
            "last_action": actor_obs[:, 64:93],
            "history_actor": actor_obs[:, 93:465],
        })
        return self.actor(observation, actor_obs[:, 465:], self.std).mean.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxing-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.boxing_root.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(root))
    from boxing_bfm.config import Recipe, activate_repo
    from boxing_bfm.expert import ExpertBuffer, encode_window
    from boxing_bfm.train import ready_records

    recipe = Recipe.load(root / "configs/ufo_mjlab_4gpu_week.json")
    source = activate_repo(recipe.source_repo())
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModel
    from humanoidverse.agents.envs.humanoidverse_mjlab import _g1_mjlab_mode15_actuator_params
    import mujoco

    torch.set_num_threads(2)
    torch.manual_seed(4728)
    checkpoint = root / "runs/ufo_mjlab_4gpu_week/checkpoint"
    model = FBcprAuxModel.load(str(checkpoint / "model"), device="cpu", strict=False)
    model.eval().requires_grad_(False)
    assert model.cfg.seq_length == 8 and model.cfg.archi.z_dim == 256
    record = next(record for record in ready_records(recipe, None) if record["motion_name"] == MOTION)
    assert record["split"] == "validation" and record["control_frames"] >= 507
    environment = yaml.safe_load((root / "runs/ufo_mjlab_4gpu_week/rank_0/environment.yaml").read_text())
    robot = environment["robot"]
    joints = robot["dof_names"]
    training = yaml.safe_load((source / "configs/robots/g1_29dof.yaml").read_text())["training"]
    control = training["control"]
    defaults = np.array([training["init_state"]["default_joint_angles"][joint] for joint in joints], dtype=np.float32)
    gains = {field: [next(float(value) for pattern, value in control[field].items() if pattern in joint)
                     for joint in joints] for field in ("stiffness", "damping")}
    effort = robot["dof_effort_limit_list"]
    scales = [0.25 * limit / gain for limit, gain in zip(effort, gains["stiffness"])]
    scene = source / "humanoidverse/data/robots/g1_mjlab/g1_29dof.xml"
    physics = mujoco.MjModel.from_xml_path(str(scene))
    ranges = np.array([physics.joint(joint).range for joint in joints])
    actuator = _g1_mjlab_mode15_actuator_params(joints)
    params = {
        "schema_version": 1,
        "robot": {"motor_count": 29, "joint_order": joints,
                  "default_angles": defaults.tolist(), "kp": gains["stiffness"], "kd": gains["damping"],
                  "action_scale": scales, "torque_limits": actuator["effort_limit"],
                  "joint_pos_lower_limit": ranges[:, 0].tolist(), "joint_pos_upper_limit": ranges[:, 1].tolist()},
        "policy": {"num_actions": 29, "actor_obs_dim": 721, "policy_obs_dim": 465},
        "observation": {"z_dim": 256, "history_steps": 4},
        "action": {"action_rescale": 5.0}, "safety": {},
        "simulation": actuator,
    }
    for directory in ("config", "model", "motions"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    (output / "config/Robot.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    expert = ExpertBuffer([record], {record["pool"]: 1.0}, 8, 4728, defaults)
    with torch.no_grad():
        observation = expert.observation(0, np.arange(1, 507))
        encoded = model.backward_map(observation)
        latents = model.project_z(torch.stack([encoded[start:start + 8].mean(0) for start in range(499)]))
        for frame in (0, 1, 100, 498):
            torch.testing.assert_close(latents[frame], encode_window(model, expert, 0, frame, 8))
        np.save(output / "motions/duobox_050_ma_latents.npy", latents.numpy())
        actor = FlatActor(model).eval()
        sample = torch.randn(1, 721)
        sample[:, 465:] = latents[0]
        actor_path = output / "model/actor.onnx"
        torch.onnx.export(actor, sample, str(actor_path), input_names=["actor_obs"],
                          output_names=["action"], opset_version=17, dynamo=False)
        inputs = torch.randn(32, 721)
        inputs[:, 465:] = latents[:32]
        expected = torch.cat([actor(row[None]) for row in inputs])
    recording = np.load(root / f"runs/review_20261105/duobox/recordings/{MOTION}.npz")
    replay = {"name": "ufo_duobox_previous_only", "variant": "bfm_zero",
              "history_timing": "previous_only", "model": "../model/actor.onnx",
              "default_angles": defaults.tolist(), "action_scale": scales,
              "lower": ranges[:, 0].tolist(), "upper": ranges[:, 1].tolist(),
              "z": latents[0].tolist(), "heading_relative": [0, 0],
              "heading_valid": False, "velocity_command": [0, 0, 0], "frames": [],
              "actor_cases": [{"actor_obs": observation.tolist(), "action": action.tolist()}
                              for observation, action in zip(inputs.numpy(), expected.numpy())]}
    history = {field: np.zeros((4, width), dtype=np.float32) for field, width in
               (("actions", 29), ("base_ang_vel", 3), ("dof_pos", 29), ("dof_vel", 29), ("projected_gravity", 3))}
    last_action = np.zeros(29, dtype=np.float32)
    with torch.no_grad():
        for frame_index in range(16):
            position = recording["actual_qpos"][frame_index, 7:].astype(np.float32)
            velocity = ((recording["actual_qpos"][frame_index + 1, 7:] - position) / 0.02).astype(np.float32)
            quaternion = recording["actual_qpos"][frame_index, 3:7]
            gravity = np.empty(3)
            mujoco.mju_rotVecQuat(gravity, np.array([0., 0., -1.]), quaternion * np.array([1, -1, -1, -1]))
            gravity = gravity.astype(np.float32)
            gyro = np.array([0.1 * frame_index, -0.03 * frame_index, 0.07], dtype=np.float32)
            state = np.concatenate([position - defaults, velocity, gravity, gyro * 0.25])
            historical = np.concatenate([history[field].reshape(-1) for field in sorted(history)])
            observation = {"state": torch.from_numpy(state[None]),
                           "last_action": torch.from_numpy(last_action[None]),
                           "history_actor": torch.from_numpy(historical[None])}
            action = model.act(observation, latents[:1], mean=True)[0].numpy()
            scaled = 5 * np.clip(action, -1, 1)
            target = np.clip(defaults + np.array(scales) * scaled, ranges[:, 0], ranges[:, 1])
            actor_obs = np.concatenate([state, last_action, historical, latents[0].numpy()])
            replay["frames"].append({"q": position.tolist(), "dq": velocity.tolist(),
                "gravity": gravity.tolist(), "gyro": gyro.tolist(), "state": state.tolist(),
                "last_action": last_action.tolist(), "history_actor": historical.tolist(),
                "actor_obs": actor_obs.tolist(), "action": action.tolist(),
                "scaled_action": scaled.tolist(), "target_q": target.tolist()})
            fields = {"actions": last_action, "base_ang_vel": gyro * 0.25,
                      "dof_pos": position - defaults, "dof_vel": velocity, "projected_gravity": gravity}
            for field, value in fields.items():
                history[field][1:] = history[field][:-1].copy()
                history[field][0] = value
            last_action = scaled
    (output / "motions/replay_fixture.json").write_text(json.dumps([replay]) + "\n")
    np.savez_compressed(output / "motions/duobox_050_ma_reference.npz",
                        reference_qpos=recording["reference_qpos"][:500],
                        joint_names=np.array(joints), dt=np.array(0.02))
    metadata = {
        "checkpoint": "runs/ufo_mjlab_4gpu_week/checkpoint/model/model.safetensors",
        "checkpoint_sha256": sha256(checkpoint / "model/model.safetensors"),
        "checkpoint_transitions": 55694336,
        "source_revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "exporter_sha256": sha256(Path(__file__)),
        "boxing_files_sha256": {str(path.relative_to(root)): sha256(path) for path in (
            root / "boxing_bfm/expert.py", root / "boxing_bfm/evaluation.py",
            root / "boxing_bfm/runtime.py", root / "configs/ufo_mjlab_4gpu_week.json")},
        "source_files_sha256": {str(path.relative_to(source)): sha256(path) for path in (
            source / "humanoidverse/agents/envs/humanoidverse_mjlab.py",
            source / "humanoidverse/agents/fb/model.py", source / "configs/robots/g1_29dof.yaml", scene)},
        "motion": MOTION, "split": "validation", "motion_sha256": sha256(Path(record["motion_file"])),
        "start_frame": 0, "action_frames": 499, "reference_frames": 500,
        "dt": 0.02, "future_window": 8, "latent_norm": 16.0,
        "history_timing": "previous_only", "learned_observation_normalization": "embedded in actor.onnx",
        "action_scale_source": "0.25 * effective config.robot.dof_effort_limit_list / kp",
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "assets": {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                   if path.is_file() and path.name != "manifest.json"
                   and path.suffix in (".onnx", ".npz", ".npy", ".yaml", ".json")},
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
