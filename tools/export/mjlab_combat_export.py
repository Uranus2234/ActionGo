import argparse
import copy
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import onnx
import torch
import yaml
from onnx.reference import ReferenceEvaluator


RAW_DIM = 261
REFERENCE_DIM = 83


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MjlabCombatPolicy(torch.nn.Module):
    def __init__(self, actor, defaults):
        super().__init__()
        self.actor = actor
        self.register_buffer("defaults", defaults.clone())

    def observation(self, raw):
        from combat_tracker.temporal_reference import HybridReferenceWindow, build_hybrid_features
        from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_euler_xyz, quat_inv, quat_mul

        joints, velocities = raw[:, :29], raw[:, 29:58]
        pelvis = raw[:, 58:62]
        pelvis = pelvis / pelvis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        zeros = torch.zeros_like(joints[:, 0])
        yaw = quat_from_euler_xyz(zeros, zeros, joints[:, 12])
        roll = quat_from_euler_xyz(joints[:, 13], zeros, zeros)
        pitch = quat_from_euler_xyz(zeros, joints[:, 14], zeros)
        torso = quat_mul(pelvis, quat_mul(yaw, quat_mul(roll, pitch)))
        reference = raw[:, 94:260].reshape(-1, 2, REFERENCE_DIM)
        valid = torch.cat((torch.ones_like(raw[:, 260:261]), raw[:, 260:261]), dim=1) > 0.5
        window = HybridReferenceWindow(
            reference[:, :, :29], reference[:, :, 29:58], reference[:, :, 58:61],
            reference[:, :, 61:65], reference[:, :, 65:68], reference[:, :, 68:71],
            reference[:, :, 71:83].reshape(-1, 2, 4, 3), valid,
        )
        rotation = matrix_from_quat(quat_mul(quat_inv(torso), window.anchor_quat_w[:, 0]))
        prefix = torch.cat((window.joint_pos[:, 0], window.joint_vel[:, 0],
                            rotation[:, :, :2].reshape(-1, 6), raw[:, 62:65],
                            joints - self.defaults, velocities, raw[:, 65:94]), dim=-1)
        return torch.cat((prefix, build_hybrid_features(window, torso, (0, 2), 0.02)), dim=-1)

    def forward(self, obs):
        return self.actor(self.observation(obs))


def reference_rows(bank, anchor, bodies):
    return torch.cat((bank.joint_pos, bank.joint_vel, bank.body_pos_w[:, anchor],
                      bank.body_quat_w[:, anchor], bank.body_lin_vel_w[:, anchor],
                      bank.body_ang_vel_w[:, anchor], bank.body_pos_w[:, bodies].flatten(1)), dim=-1).cpu()


def make_spec():
    return {
        "name": "mjlab_combat_4gpu_92000",
        "runtime": {"template": "motion_conditioned_tracking", "variant": "mjlab_combat",
                    "inference_hz": 50, "publish_hz": 500, "onnx_threads": 2,
                    "stats_interval": 50, "dry_run_duration_sec": 1.0},
        "robot": {"params": "MjlabCombat.yaml", "motor_count": 29},
        "model": {"path": "policy.onnx", "input_names": ["obs"], "output_names": ["actions"],
                  "obs_input": "obs", "action_output": "actions"},
        "reference": {"type": "mjlab_combat_csv_v1", "path": "reference.csv", "yaw_alignment": "initial",
                      "end_behavior": "hold", "policy_step_source": "lowstate_tick", "policy_tick_ms": 20,
                      "sim_lockstep_sequence": True, "trace": True},
        "observation": {"builder": "mjlab_combat", "obs_dim": RAW_DIM},
        "action": {"type": "target_q", "action_dim": 29, "add_default": True, "scale_from_params": "action_scale"},
        "handoff": {"cmd_topic": "rt/user_lowcmd", "state_topic": "rt/lowstate", "mode_machine": 5,
                    "activation_blend_sec": 0.0, "damping_on_stop": False},
        "safety": {"profile": "simulation_parity", "enabled": True, "finite_check": True,
                   "joint_limit_enabled": False, "max_position_step_enabled": False,
                   "lowstate_timeout_sec": 10.0},
    }


def load_environment(training_repo, checkpoint, motion, device):
    sys.path.insert(0, str(training_repo / "src"))
    import mjlab.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    task = "Combat-Tracking-Mixed-Unitree-G1-No-State-Estimation-Future40ms"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.auto_reset = False
    cfg.events.pop("encoder_bias", None)
    command_cfg = cfg.commands["motion"]
    command_cfg.motion_file = str(motion.resolve())
    command_cfg.initial_checkpoint = ""
    command_cfg.joint_position_range = (0.0, 0.0)
    command_cfg.collect_success = False
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapper = RslRlVecEnvWrapper(env)
    runner_cfg = load_rl_cfg(task)
    runner_cfg.upload_model = False
    runner = load_runner_cls(task)(wrapper, asdict(runner_cfg), device=device)
    runner.load_for_inference(str(checkpoint))
    return env, runner


@torch.inference_mode()
def export(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    env, runner = load_environment(args.training_repo, args.checkpoint, args.motion, args.device)
    try:
        command = env.command_manager.get_term("motion")
        contract = runner.contract
        if contract["reference"]["offsets_steps"] != [0, 2] or contract["reference"]["append_width"] != 105:
            raise ValueError("This variant requires the trained current/+40ms hybrid reference contract")
        actor = copy.deepcopy(runner.alg._raw_actor).cpu().eval().as_onnx(verbose=False)
        defaults = env.scene["robot"].data.default_joint_pos[0].cpu()
        packed = MjlabCombatPolicy(actor, defaults).eval()
        rows = reference_rows(command.bank, command.motion_anchor_body_index, command.task_body_indexes)
        np.savetxt(args.output / "reference.csv", rows.numpy(), delimiter=",", fmt="%.9g")
        np.testing.assert_array_equal(np.loadtxt(args.output / "reference.csv", delimiter=",", dtype=np.float32), rows.numpy())
        bank = command.bank
        qpos = torch.cat((bank.body_pos_w[:, 0], bank.body_quat_w[:, 0], bank.joint_pos), dim=-1).cpu().numpy()
        qvel = torch.cat((bank.body_lin_vel_w[:, 0], bank.body_ang_vel_w[:, 0], bank.joint_vel), dim=-1).cpu().numpy()
        np.savez_compressed(args.output / "motion.npz", qpos=qpos, qvel=qvel)
        samples = []
        observation_error = action_error = 0.0
        for frame in sorted({0, min(20, len(rows) - 1), len(rows) // 2, len(rows) - 2, len(rows) - 1}):
            ids = torch.zeros(1, dtype=torch.long, device=args.device)
            command.reset_to_motion(ids, 0, frame)
            env.sim.forward()
            env.scene.update(env.step_dt)
            command._update_command()
            observations = env.observation_manager.compute(update_history=True)
            robot = env.scene["robot"]
            raw = torch.cat((robot.data.joint_pos[0], robot.data.joint_vel[0], robot.data.root_link_quat_w[0],
                             observations["actor"][0, 64:67], env.action_manager.action[0],
                             rows[frame].to(args.device), rows[min(frame + 2, len(rows) - 1)].to(args.device),
                             torch.tensor([float(frame + 2 < len(rows))], device=args.device)))[None].cpu()
            expected_obs = observations["actor"].cpu()
            actual_obs = packed.observation(raw)
            torch.testing.assert_close(actual_obs, expected_obs, atol=3e-5, rtol=1e-4)
            expected_action = runner.get_inference_policy(args.device)(observations).cpu()
            torch.testing.assert_close(packed(raw), expected_action, atol=5e-5, rtol=1e-4)
            observation_error = max(observation_error, float((actual_obs - expected_obs).abs().max()))
            action_error = max(action_error, float((packed(raw) - expected_action).abs().max()))
            samples.append(raw)
        torch.onnx.export(packed, samples[0], str(args.output / "policy.onnx"), input_names=["obs"],
                          output_names=["actions"], opset_version=18, dynamo=False)
        model = onnx.load(str(args.output / "policy.onnx"))
        onnx.checker.check_model(model, full_check=True)
        evaluator = ReferenceEvaluator(model)
        onnx_error = 0.0
        for sample in samples:
            expected = packed(sample).numpy()
            actual = evaluator.run(None, {"obs": sample.numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=1e-4)
            onnx_error = max(onnx_error, float(np.abs(actual - expected).max()))
        model = env.sim.mj_model
        joints = [model.joint(index).name.split("/")[-1] for index in range(1, model.njnt)]
        if joints != contract["joint_names"] or model.nq != 36 or model.nv != 35:
            raise ValueError("Training robot joint order must match the SDK G1 order")
        actuators = [int(np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]) for joint in range(1, 30)]
        action = env.action_manager.get_term("joint_pos")
        scales = action._scale[0].cpu().tolist()
        params = {"schema_version": 1, "policy": {"num_obs": RAW_DIM, "num_actions": 29},
                  "robot": {"motor_count": 29, "policy_joint_order": joints, "joint_map": list(range(29)),
                            "default_angles": defaults.tolist(), "action_scale": scales,
                            "kp": model.actuator_gainprm[actuators, 0].tolist(),
                            "kd": (-model.actuator_biasprm[actuators, 2]).tolist(),
                            "torque_limits": model.actuator_forcerange[actuators, 1].tolist(),
                            "joint_pos_lower_limit": model.jnt_range[1:, 0].tolist(),
                            "joint_pos_upper_limit": model.jnt_range[1:, 1].tolist()}}
        (args.output / "MjlabCombat.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
        (args.output / "generic_mjlab_combat.yaml").write_text(yaml.safe_dump(make_spec(), sort_keys=False))
        mujoco.mj_saveModel(model, str(args.output / "scene.mjb"))
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        manifest = {"variant": "mjlab_combat", "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_iteration": saved["iter"], "checkpoint_sha256": sha256(args.checkpoint),
                    "training_repo": str(args.training_repo.resolve()), "policy_contract": contract,
                    "motion": str(args.motion.resolve()), "motion_sha256": sha256(args.motion),
                    "reference_rows": len(rows), "reference_width": REFERENCE_DIM,
                    "raw_input_width": RAW_DIM, "actor_input_width": 259,
                    "raw_input_fields": [["q_sdk", 29], ["dq_sdk", 29], ["pelvis_quat_wxyz", 4],
                                         ["pelvis_gyro", 3], ["last_action", 29], ["current_reference", 83],
                                         ["future_reference_40ms", 83], ["future_valid", 1]],
                    "reference_fields": [["joint_pos", 29], ["joint_vel", 29], ["anchor_pos_w", 3],
                                         ["anchor_quat_wxyz", 4], ["anchor_lin_vel_w", 3], ["anchor_ang_vel_w", 3],
                                         ["task_body_pos_w", 12]],
                    "policy_sha256": sha256(args.output / "policy.onnx"),
                    "reference_sha256": sha256(args.output / "reference.csv"),
                    "reuse": ["mjlab task/robot/PD/normalizer/MLP", "MotionBank", "build_hybrid_features",
                              "mjlab quaternion math", "Phase2TrackingRunner.load_for_inference"],
                    "validation": {"mjlab_observation_max_abs_error": observation_error,
                                   "mjlab_action_max_abs_error": action_error, "onnx_max_abs_error": onnx_error,
                                   "samples": len(samples), "boundary_frames_included": True}}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "iteration": saved["iter"], "validation": manifest["validation"]}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    export(parser.parse_args())
