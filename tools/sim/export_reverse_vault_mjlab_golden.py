#!/usr/bin/env python3
"""Export the formal Reverse Vault flat MJB, startup state, and golden trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort
import torch
import yaml

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--robot-params", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--task",
        default="Mjlab-Tracking-ReverseVault-Flat-Unitree-G1-No-State-Estimation",
        help="Formal MJLab task whose model/solver/contact configuration produced the policy",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_list(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().numpy().tolist()


def optional_sim_tensor(data: Any, name: str) -> list[Any] | None:
    value = getattr(data, name, None)
    if value is None:
        return None
    return tensor_list(value[0])


def configure_environment(args: argparse.Namespace) -> Any:
    cfg = load_env_cfg(args.task, play=False)
    cfg.scene.num_envs = 1
    cfg.events = {}
    cfg.terminations = {}
    actor = cfg.observations["actor"]
    actor.enable_corruption = False
    actor.terms.pop("motion_anchor_pos_b", None)
    actor.terms.pop("base_lin_vel", None)
    motion = cfg.commands["motion"]
    motion.motion_file = str(args.motion)
    motion.sampling_mode = "start"
    motion.pose_range = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    motion.velocity_range = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    motion.joint_position_range = (0.0, 0.0)
    return cfg


def write_exact_mjb(
    env: ManagerBasedRlEnv, qpos: np.ndarray, qvel: np.ndarray, path: Path
) -> None:
    model = env.sim.mj_model
    keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "init_state"
    )
    if keyframe_id < 0:
        if model.nkey != 1:
            raise RuntimeError("exact export requires a unique startup keyframe")
        keyframe_id = 0
    model.key_qpos[keyframe_id] = qpos
    model.key_qvel[keyframe_id] = qvel
    mujoco.mj_saveModel(model, str(path), None)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = configure_environment(args)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    observations, _ = env.reset(seed=args.seed)
    actor_obs = observations["actor"]
    if tuple(actor_obs.shape) != (1, 154):
        raise RuntimeError(f"formal actor observation must be [1,154], got {actor_obs.shape}")

    startup_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
    startup_qvel = env.sim.data.qvel[0].detach().cpu().numpy().copy()
    startup_ctrl = env.sim.data.ctrl[0].detach().cpu().numpy().copy()
    startup_act = env.sim.data.act[0].detach().cpu().numpy().copy()
    exact_mjb = args.output_dir / "reverse_vault_flat_24000_exact.mjb"
    write_exact_mjb(env, startup_qpos, startup_qvel, exact_mjb)
    np.savez(
        args.output_dir / "startup_state.npz",
        qpos=startup_qpos,
        qvel=startup_qvel,
        ctrl=startup_ctrl,
        act=startup_act,
    )

    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]
    robot_params = yaml.safe_load(args.robot_params.read_text(encoding="utf-8"))[
        "robot"
    ]
    default_q = np.asarray(robot_params["default_angles"], dtype=np.float32)
    action_scale = np.asarray(robot_params["action_scale"], dtype=np.float32)
    motion_term = env.command_manager.get_term("motion")

    trace_path = args.output_dir / "native_golden.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        for policy_step in range(args.steps):
            obs_numpy = actor_obs.detach().cpu().numpy().astype(np.float32)
            reference_step = int(motion_term.time_steps[0].item())
            output_values = session.run(
                None,
                {
                    "obs": obs_numpy,
                    "time_step": np.asarray([[reference_step]], dtype=np.float32),
                },
            )
            outputs = dict(zip(output_names, output_values, strict=True))
            raw_action = outputs["actions"][0].astype(np.float32)
            raw_target = default_q + action_scale * raw_action
            terms = {
                name.removeprefix("actor-"): value
                for name, value in env.observation_manager.get_active_iterable_terms(0)
                if name.startswith("actor-")
            }
            before_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
            before_qvel = env.sim.data.qvel[0].detach().cpu().numpy().copy()
            action_tensor = torch.as_tensor(
                raw_action[None], device=env.device, dtype=torch.float32
            )
            next_observations, _, _, _, _ = env.step(action_tensor)
            record = {
                "policy_step": policy_step,
                "reference_step": reference_step,
                "sim_time_before": float(policy_step * env.step_dt),
                "qpos_before": before_qpos.tolist(),
                "qvel_before": before_qvel.tolist(),
                "observation_terms": terms,
                "raw_obs": obs_numpy[0].tolist(),
                "raw_action": raw_action.tolist(),
                "raw_target": raw_target.tolist(),
                "data_ctrl_after": optional_sim_tensor(env.sim.data, "ctrl"),
                "actuator_force_after": optional_sim_tensor(
                    env.sim.data, "actuator_force"
                ),
                "qfrc_actuator_after": optional_sim_tensor(
                    env.sim.data, "qfrc_actuator"
                ),
                "qpos_after": tensor_list(env.sim.data.qpos[0]),
                "qvel_after": tensor_list(env.sim.data.qvel[0]),
                "reference_joint_pos": tensor_list(motion_term.joint_pos[0]),
                "reference_joint_vel": tensor_list(motion_term.joint_vel[0]),
                "reference_body_quat_w": tensor_list(motion_term.body_quat_w[0]),
            }
            trace.write(json.dumps(record, separators=(",", ":")) + "\n")
            actor_obs = next_observations["actor"]

    manifest = {
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "motion": str(args.motion.resolve()),
        "motion_sha256": sha256(args.motion),
        "mjb": str(exact_mjb.resolve()),
        "mjb_sha256": sha256(exact_mjb),
        "startup_state": str((args.output_dir / "startup_state.npz").resolve()),
        "trace": str(trace_path.resolve()),
        "seed": args.seed,
        "steps": args.steps,
        "timestep": float(env.physics_dt),
        "decimation": int(round(env.step_dt / env.physics_dt)),
        "obs_dim": int(actor_obs.shape[-1]),
        "action_dim": int(raw_action.shape[-1]),
        "initial_reference_step": 1,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    env.close()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
