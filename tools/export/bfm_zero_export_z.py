#!/usr/bin/env python3
"""Export BFM-Zero latent prompts for action_go.

The regular ``reward``/``goal`` modes read the author's exported latent banks
without changing BFM-Zero.  The ``stand-reward`` mode mirrors the author's
reward-wrapper path (model.reward_wr_inference), but computes the reward in this
action_go script so BFM-Zero source files stay untouched.

Examples:
  # List original goal keys
  tools/export/bfm_zero_export_z.py --mode goal --list

  # Export a get-up goal latent
  tools/export/bfm_zero_export_z.py --mode goal \
    --key fallAndGetUp1_subject4_2193 \
    --output actions/bfm0/config/bfm_zero_goal_getup_z.txt

  # Export the original reward idle latent used by reward mode
  tools/export/bfm_zero_export_z.py --mode reward --key move-ego-0-0 --z-id 0

  # Infer a fixed standing reward latent:
  #   R = (head_height = 1.2m) ∧ (base_vel = 0m/s)
  ../BFM-Zero/.venv/bin/python tools/export/bfm_zero_export_z.py \
    --mode stand-reward --output actions/bfm0/config/bfm_zero_reward_head1p2_base0_z.txt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import joblib
import numpy as np


ACTION_GO_ROOT = Path(__file__).resolve().parents[2]
BFM_ZERO_ROOT = ACTION_GO_ROOT.parent / "BFM-Zero"


STAND_REWARD_MODE = "stand-reward"


def _path_or_none(value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else ACTION_GO_ROOT / value


def _float_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def default_custom_buffer() -> Path:
    """Pick a replay buffer without modifying BFM-Zero.

    The public model/checkpoint in this workspace has weights but no
    buffer.hdf5.  When that happens we use the adjacent
    new_model_for_training_code_inference replay buffer and, unless explicitly
    overridden, the matching new_model checkpoint/ONNX in action_go config.
    """
    candidates = [
        BFM_ZERO_ROOT / "model/checkpoint/buffers/train",
        BFM_ZERO_ROOT / "model/checkpoint/buffers/train_reduced",
        BFM_ZERO_ROOT / "new_model_for_training_code_inference/checkpoint/buffers/train",
    ]
    for candidate in candidates:
        if (candidate / "buffer.hdf5").exists() and (candidate / "config.json").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a usable BFM-Zero replay buffer. Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def _read_pairwise_by_time(dataset: Any, t_idx: np.ndarray, env_idx: np.ndarray, *, time_offset: int = 0) -> np.ndarray:
    """Read dataset[t_i + time_offset, env_i] for many pairs efficiently.

    h5py cannot fancy-index two axes pairwise.  Group by time index and read one
    full environment row at a time; this is simple and avoids loading the whole
    26GB buffer into RAM.
    """
    t_idx = np.asarray(t_idx, dtype=np.int64) + int(time_offset)
    env_idx = np.asarray(env_idx, dtype=np.int64)
    if t_idx.shape != env_idx.shape:
        raise ValueError(f"t_idx/env_idx shape mismatch: {t_idx.shape} vs {env_idx.shape}")
    if np.any(t_idx < 0) or np.any(t_idx >= dataset.shape[0]):
        raise IndexError(f"time indices out of range for dataset {dataset.name}: [{t_idx.min()}, {t_idx.max()}]")

    out = np.empty((len(t_idx), *dataset.shape[2:]), dtype=dataset.dtype)
    for t in np.unique(t_idx):
        mask = np.flatnonzero(t_idx == t)
        # Read all envs at this time step, then NumPy-index pairwise env ids.
        row = dataset[int(t)]
        out[mask] = row[env_idx[mask]]
    return out


def _sample_valid_transition_indices(h5: Any, buffer_cfg: dict[str, Any], num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    capacity, num_envs = h5["qpos"].shape[:2]
    if capacity < 2:
        raise ValueError(f"buffer capacity must be >=2, got {capacity}")
    rng = np.random.default_rng(seed)
    cursor_end = None
    if buffer_cfg.get("_is_full", False) and buffer_cfg.get("_idx") is not None:
        # Original TrajectoryDictBufferMultiDim treats cursor-1 as an episode end.
        cursor_end = (int(buffer_cfg["_idx"]) - 1) % capacity

    ts: list[np.ndarray] = []
    envs: list[np.ndarray] = []
    total = 0
    while total < num_samples:
        remaining = num_samples - total
        candidate_count = max(4096, int(remaining * 1.5))
        # Avoid the physical last row so t+1 stays in-range.  This intentionally
        # drops a tiny fraction of possible wraparound transitions.
        cand_t = rng.integers(0, capacity - 1, size=candidate_count, dtype=np.int64)
        cand_env = rng.integers(0, num_envs, size=candidate_count, dtype=np.int64)
        valid = np.ones(candidate_count, dtype=bool)
        if cursor_end is not None:
            valid &= cand_t != cursor_end
        if "truncated" in h5:
            truncated = _read_pairwise_by_time(h5["truncated"], cand_t, cand_env).reshape(-1).astype(bool)
            valid &= ~truncated
        if not np.any(valid):
            continue
        keep_t = cand_t[valid][:remaining]
        keep_env = cand_env[valid][:remaining]
        ts.append(keep_t)
        envs.append(keep_env)
        total += len(keep_t)

    return np.concatenate(ts), np.concatenate(envs)


def _load_hdf5_reward_samples(buffer_dir: Path, num_samples: int, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    buffer_dir = Path(buffer_dir)
    with (buffer_dir / "config.json").open("r") as f:
        buffer_cfg = json.load(f)
    with h5py.File(buffer_dir / "buffer.hdf5", "r") as h5:
        t_idx, env_idx = _sample_valid_transition_indices(h5, buffer_cfg, num_samples, seed)
        next_obs = {
            "state": _read_pairwise_by_time(h5["observation-state"], t_idx, env_idx, time_offset=1).astype(np.float32),
            "privileged_state": _read_pairwise_by_time(
                h5["observation-privileged_state"], t_idx, env_idx, time_offset=1
            ).astype(np.float32),
        }
        # The BFM backward map only consumes state+privileged_state, but keeping
        # the shape checks here catches accidental buffer/model mismatches early.
        if next_obs["state"].shape[-1] != 64:
            raise ValueError(f"expected observation-state dim 64, got {next_obs['state'].shape}")
        if next_obs["privileged_state"].shape[-1] != 463:
            raise ValueError(f"expected observation-privileged_state dim 463, got {next_obs['privileged_state'].shape}")
        qpos = _read_pairwise_by_time(h5["qpos"], t_idx, env_idx, time_offset=1).astype(np.float32)
        qvel = _read_pairwise_by_time(h5["qvel"], t_idx, env_idx, time_offset=1).astype(np.float32)
        action = _read_pairwise_by_time(h5["action"], t_idx, env_idx, time_offset=0).astype(np.float32)
    return next_obs, qpos, qvel, action


def _tolerance(value: np.ndarray | float, *, bounds: tuple[float, float], margin: float, value_at_margin: float = 0.01) -> np.ndarray:
    """dm_control-style tolerance fallback for the custom reward."""
    try:
        from dm_control.utils import rewards as dm_rewards

        return dm_rewards.tolerance(value, bounds=bounds, margin=margin, value_at_margin=value_at_margin, sigmoid="gaussian")
    except Exception:
        value_arr = np.asarray(value, dtype=np.float32)
        lower, upper = bounds
        in_bounds = (value_arr >= lower) & (value_arr <= upper)
        distance = np.where(value_arr < lower, lower - value_arr, np.where(value_arr > upper, value_arr - upper, 0.0))
        if margin <= 0:
            return in_bounds.astype(np.float32)
        # Gaussian chosen so reward ~= value_at_margin when distance == margin.
        scale = np.sqrt(-2.0 * np.log(max(value_at_margin, 1e-12)))
        rew = np.exp(-0.5 * (scale * distance / margin) ** 2)
        return np.where(in_bounds, 1.0, rew).astype(np.float32)


def _compute_head_base_stand_reward(
    xml_path: Path,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    *,
    head_height: float,
    base_vel: float,
    height_tolerance: float,
    height_margin: float,
    base_vel_tolerance: float,
    base_vel_margin: float,
    base_vel_source: str,
) -> tuple[np.ndarray, dict[str, float]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq != qpos.shape[-1] or model.nv != qvel.shape[-1]:
        raise ValueError(
            f"XML qpos/qvel dims ({model.nq}, {model.nv}) do not match buffer ({qpos.shape[-1]}, {qvel.shape[-1]})"
        )
    head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head")
    if head_geom_id < 0:
        raise ValueError(f"XML has no geom named 'head': {xml_path}")
    com_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_link_subtreelinvel")
    if base_vel_source == "com" and com_sensor_id < 0:
        raise ValueError(f"XML has no torso_link_subtreelinvel sensor for --base-vel-source=com: {xml_path}")

    data = mujoco.MjData(model)
    rewards = np.empty((qpos.shape[0], 1), dtype=np.float32)
    head_values = np.empty((qpos.shape[0],), dtype=np.float32)
    speed_values = np.empty((qpos.shape[0],), dtype=np.float32)
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        if model.nu:
            data.ctrl[:] = 0.0
            n_ctrl = min(model.nu, action.shape[-1])
            data.ctrl[:n_ctrl] = action[i, :n_ctrl]
        mujoco.mj_forward(model, data)
        head_z = float(data.geom(head_geom_id).xpos[2])
        if base_vel_source == "com":
            start = model.sensor_adr[com_sensor_id]
            end = start + model.sensor_dim[com_sensor_id]
            vel_vec = data.sensordata[start:end].copy()
        else:
            # MuJoCo freejoint qvel is [linear xyz, angular xyz].
            vel_vec = data.qvel[:3].copy()
        speed = float(np.linalg.norm(vel_vec))
        head_values[i] = head_z
        speed_values[i] = speed
        height_reward = _tolerance(
            head_z,
            bounds=(head_height - height_tolerance, head_height + height_tolerance),
            margin=height_margin,
            value_at_margin=0.01,
        )
        vel_reward = _tolerance(
            speed,
            bounds=(max(0.0, base_vel - base_vel_tolerance), base_vel + base_vel_tolerance),
            margin=base_vel_margin,
            value_at_margin=0.01,
        )
        # Logical conjunction is represented as product, as in the author's
        # reward functions (stand_reward * dont_move * ...).
        rewards[i, 0] = float(height_reward * vel_reward)

    stats = {
        "reward_min": float(np.min(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "reward_max": float(np.max(rewards)),
        "head_min": float(np.min(head_values)),
        "head_mean": float(np.mean(head_values)),
        "head_max": float(np.max(head_values)),
        "speed_min": float(np.min(speed_values)),
        "speed_mean": float(np.mean(speed_values)),
        "speed_max": float(np.max(speed_values)),
    }
    return rewards, stats


def infer_stand_reward_z(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    # Keep rendering/headless imports deterministic if MuJoCo/dm_control are used.
    os.environ.setdefault("MUJOCO_GL", "egl")
    if str(BFM_ZERO_ROOT) not in sys.path:
        sys.path.insert(0, str(BFM_ZERO_ROOT))
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

    buffer_dir = args.buffer or default_custom_buffer()
    buffer_dir = buffer_dir if buffer_dir.is_absolute() else BFM_ZERO_ROOT / buffer_dir
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        checkpoint = checkpoint if checkpoint.is_absolute() else BFM_ZERO_ROOT / checkpoint
    elif "new_model_for_training_code_inference" in str(buffer_dir):
        checkpoint = BFM_ZERO_ROOT / "new_model_for_training_code_inference/checkpoint"
    else:
        checkpoint = BFM_ZERO_ROOT / "model/checkpoint"
    xml_path = args.xml or (BFM_ZERO_ROOT / "humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml")
    xml_path = xml_path if xml_path.is_absolute() else BFM_ZERO_ROOT / xml_path

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    print(f"[stand-reward] checkpoint={checkpoint}")
    print(f"[stand-reward] buffer={buffer_dir}")
    print(f"[stand-reward] xml={xml_path}")
    print(f"[stand-reward] samples={args.num_samples} seed={args.seed}")
    next_obs_np, qpos, qvel, action = _load_hdf5_reward_samples(buffer_dir, args.num_samples, args.seed)
    reward_np, stats = _compute_head_base_stand_reward(
        xml_path,
        qpos,
        qvel,
        action,
        head_height=args.head_height,
        base_vel=args.base_vel,
        height_tolerance=args.height_tolerance,
        height_margin=args.height_margin,
        base_vel_tolerance=args.base_vel_tolerance,
        base_vel_margin=args.base_vel_margin,
        base_vel_source=args.base_vel_source,
    )
    print(
        "[stand-reward] reward stats: "
        f"min={stats['reward_min']:.6f} mean={stats['reward_mean']:.6f} max={stats['reward_max']:.6f}"
    )
    print(
        "[stand-reward] head_z stats: "
        f"min={stats['head_min']:.3f} mean={stats['head_mean']:.3f} max={stats['head_max']:.3f}; "
        f"base_speed stats: min={stats['speed_min']:.3f} mean={stats['speed_mean']:.3f} max={stats['speed_max']:.3f}"
    )
    if stats["reward_max"] <= 1e-6:
        raise RuntimeError("custom stand reward is all zeros; loosen tolerances or check XML/buffer compatibility")

    model = load_model_from_checkpoint_dir(checkpoint, device=args.device)
    model.to(args.device)
    model.eval()
    next_obs = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in next_obs_np.items()}
    reward = torch.as_tensor(reward_np, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        z = model.reward_wr_inference(next_obs=next_obs, reward=reward).reshape(-1).float().cpu().numpy()
    if z.shape[0] != 256:
        raise ValueError(f"inferred latent should be 256-D, got {z.shape}")
    meta = {
        "checkpoint": str(checkpoint),
        "buffer": str(buffer_dir),
        "xml": str(xml_path),
        "stats": stats,
    }
    return z.astype(np.float32), meta


def load_bank(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return joblib.load(f)


def default_bank(mode: str) -> Path:
    if mode == "reward":
        return BFM_ZERO_ROOT / "model/reward_inference/reward_locomotion.pkl"
    if mode == "goal":
        return BFM_ZERO_ROOT / "model/goal_inference/goal_reaching.pkl"
    raise ValueError(mode)


def select_z(bank: dict[str, Any], mode: str, key: str, z_id: int) -> np.ndarray:
    if key not in bank:
        available = "\n  ".join(bank.keys())
        raise KeyError(f"{mode} key not found: {key}\nAvailable:\n  {available}")

    if mode == "reward":
        # Original rl_policy/bfm_zero.py reward branch does:
        #   v = z_dict_raw[reward]
        #   selected list appends v[z_id]
        #   inputs concatenate selected_z[self.z_index]
        value = bank[key]
        if z_id < 0 or z_id >= len(value):
            raise IndexError(f"z_id={z_id} out of range for reward {key}; len={len(value)}")
        z = np.asarray(value[z_id], dtype=np.float32)
    else:
        # Original goal branch directly concatenates z_dict[goal_name].
        z = np.asarray(bank[key], dtype=np.float32)

    z = z.reshape(-1)
    if z.shape[0] != 256:
        raise ValueError(f"latent should flatten to 256 values, got shape {z.shape}")
    return z.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["reward", "goal", STAND_REWARD_MODE], required=True)
    p.add_argument("--bank", type=Path, default=None, help="override latent bank pkl path for reward/goal modes")
    p.add_argument("--key", help="reward name or goal name")
    p.add_argument("--z-id", type=int, default=0, help="reward z id; ignored for goal")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--list", action="store_true", help="list available keys and exit (reward/goal modes)")

    # Custom reward inference, kept in action_go to avoid editing BFM-Zero source.
    p.add_argument("--checkpoint", type=Path, default=None, help="BFM-Zero checkpoint dir; default: ../BFM-Zero/model/checkpoint")
    p.add_argument("--buffer", type=Path, default=None, help="replay buffer dir containing buffer.hdf5; auto-detected by default")
    p.add_argument("--xml", type=Path, default=None, help="MuJoCo XML for reward relabeling; default: author's noadditional_actuators scene")
    p.add_argument("--device", default="cpu", help="torch device for reward_wr_inference")
    p.add_argument("--torch-threads", type=int, default=0, help="set torch CPU threads for custom inference; 0 leaves default")
    p.add_argument("--num-samples", type=int, default=150_000, help="number of replay states used by stand-reward")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head-height", type=float, default=1.2, help="target head height in meters")
    p.add_argument("--base-vel", type=float, default=0.0, help="target base speed in m/s")
    p.add_argument("--height-tolerance", type=float, default=0.03, help="full-reward band around head height target")
    p.add_argument("--height-margin", type=float, default=0.6, help="Gaussian falloff margin for head height")
    p.add_argument("--base-vel-tolerance", type=float, default=0.05, help="full-reward band around base speed target")
    p.add_argument("--base-vel-margin", type=float, default=0.3, help="Gaussian falloff margin for base speed")
    p.add_argument("--base-vel-source", choices=["root", "com"], default="root", help="root uses qvel[:3]; com uses torso_link_subtreelinvel")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == STAND_REWARD_MODE:
        if args.list:
            raise SystemExit("--list is only valid for reward/goal bank modes")
        z, meta = infer_stand_reward_z(args)
        output = args.output
        if output is None:
            output = (
                ACTION_GO_ROOT
                / "actions"
                / "bfm0"
                / f"bfm_zero_reward_head{_float_tag(args.head_height)}_base{_float_tag(args.base_vel)}_z.txt"
            )
        if not output.is_absolute():
            output = ACTION_GO_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output, z.reshape(1, -1), fmt="%.9g")
        print(f"mode={args.mode}")
        print(f"reward=R=(head_height={args.head_height:g}m) ∧ (base_vel={args.base_vel:g}m/s)")
        print(f"checkpoint={meta['checkpoint']}")
        print(f"buffer={meta['buffer']}")
        print(f"output={output}")
        print(f"shape={z.shape} norm={np.linalg.norm(z):.6f}")
        return

    bank_path = args.bank or default_bank(args.mode)
    bank = load_bank(bank_path)
    if not isinstance(bank, dict):
        raise TypeError(f"expected dict latent bank, got {type(bank)} from {bank_path}")

    if args.list:
        for key, value in bank.items():
            arr = np.asarray(value)
            print(f"{key}\tshape={arr.shape}")
        return

    if not args.key:
        raise SystemExit("--key is required unless --list is used")

    z = select_z(bank, args.mode, args.key, args.z_id)
    output = args.output
    if output is None:
        safe_key = args.key.replace("/", "_").replace(" ", "_")
        suffix = f"_z{args.z_id}" if args.mode == "reward" else ""
        output = ACTION_GO_ROOT / "actions" / "bfm0" / "config" / f"bfm_zero_{args.mode}_{safe_key}{suffix}.txt"
    if not output.is_absolute():
        output = ACTION_GO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, z.reshape(1, -1), fmt="%.9g")

    print(f"mode={args.mode}")
    print(f"bank={bank_path}")
    print(f"key={args.key}")
    if args.mode == "reward":
        print(f"z_id={args.z_id}")
    print(f"output={output}")
    print(f"shape={z.shape} norm={np.linalg.norm(z):.6f}")


if __name__ == "__main__":
    main()
