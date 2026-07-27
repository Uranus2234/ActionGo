#!/usr/bin/env python3
"""Prepare TWIST offline .pkl motion references for Generic Motion Tracking Runtime.

`generic_motion_tracking_player` performs policy inference in C++/ONNX Runtime. This
helper only converts TWIST/GMR motion .pkl files into a small text cache of
33-float `mimic_obs` frames, matching deploy_real/server_high_level_motion_lib.py.

It intentionally avoids importing Isaac Gym so it can run on deployment hosts.
"""
import argparse
import os
import pickle
import sys
from types import ModuleType

import numpy as np
import torch


class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real is not None:
            self.__dict__.update(real.__dict__)


def patch_numpy_core_pickle_compat():
    # Some TWIST/GMR pickles produced with NumPy 2.x reference numpy._core.
    core = getattr(np, "_core", getattr(np, "core", None))
    sys.modules.setdefault("numpy._core", FakeModule("numpy._core", core))
    sys.modules.setdefault("numpy._core.multiarray", FakeModule("numpy._core.multiarray", getattr(core, "multiarray", None)))


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def smooth(x: torch.Tensor, box_pts: int) -> torch.Tensor:
    box = torch.ones(box_pts, dtype=x.dtype, device=x.device) / float(box_pts)
    num_channels = x.shape[1]
    x_reshaped = x.T.unsqueeze(0)
    smoothed = torch.nn.functional.conv1d(
        x_reshaped,
        box.view(1, 1, -1).expand(num_channels, 1, -1),
        groups=num_channels,
        padding=box_pts // 2,
    )
    return smoothed.squeeze(0).T


def quat_conjugate(q):
    return torch.cat((-q[..., 0:3], q[..., 3:4]), dim=-1)


def quat_mul(q1, q2):
    # xyzw convention, matching TWIST pose.utils.torch_utils.
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack((x, y, z, w), dim=-1)


def quat_diff(q0, q1):
    return quat_mul(q1, quat_conjugate(q0))


def quat_to_exp_map(q):
    qw = torch.clamp(q[..., 3], -1.0, 1.0)
    sin_theta = torch.sqrt(torch.clamp(1.0 - qw * qw, min=0.0))
    angle = normalize_angle(2.0 * torch.acos(qw))
    axis = q[..., 0:3] / torch.clamp(sin_theta.unsqueeze(-1), min=1e-5)
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1.0
    mask = torch.abs(sin_theta) > 1e-5
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    axis = torch.where(mask.unsqueeze(-1), axis, default_axis)
    return angle.unsqueeze(-1) * axis


def slerp(q0, q1, t):
    cos_half_theta = torch.sum(q0 * q1, dim=-1)
    neg_mask = cos_half_theta < 0
    q1 = torch.where(neg_mask.unsqueeze(-1), -q1, q1)
    cos_half_theta = torch.abs(cos_half_theta).unsqueeze(-1)
    cos_half_theta = torch.clamp(cos_half_theta, -1.0, 1.0)
    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(torch.clamp(1.0 - cos_half_theta * cos_half_theta, min=0.0))
    t = t.unsqueeze(-1)
    ratio_a = torch.sin((1.0 - t) * half_theta) / torch.clamp(sin_half_theta, min=1e-6)
    ratio_b = torch.sin(t * half_theta) / torch.clamp(sin_half_theta, min=1e-6)
    new_q = ratio_a * q0 + ratio_b * q1
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1.0, q0, new_q)
    return new_q


def euler_from_quaternion_xyzw(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def quat_rotate_inverse_xyzw(q, v):
    q_w = q[:, 3]
    q_vec = q[:, 0:3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * (2.0 * q_w).unsqueeze(-1)
    c = q_vec * (2.0 * torch.sum(q_vec * v, dim=-1, keepdim=True))
    return a - b + c


class SingleMotionLib:
    def __init__(self, motion_file: str, device: torch.device):
        with open(motion_file, "rb") as f:
            motion_data = pickle.load(f)
        self.device = device
        self.fps = float(motion_data["fps"])
        self.dt = 1.0 / self.fps
        self.root_pos = torch.tensor(motion_data["root_pos"], dtype=torch.float32, device=device)
        self.root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float32, device=device)
        self.dof_pos = torch.tensor(motion_data["dof_pos"], dtype=torch.float32, device=device)
        self.num_frames = int(self.root_pos.shape[0])
        self.length = self.dt * (self.num_frames - 1)
        self.root_pos_delta = self.root_pos[-1] - self.root_pos[0]
        self.root_pos_delta[-1] = 0.0

        root_vel = torch.zeros_like(self.root_pos)
        root_vel[:-1] = self.fps * (self.root_pos[1:] - self.root_pos[:-1])
        root_vel[-1] = root_vel[-2]
        self.root_vel = smooth(root_vel, 19)

        root_ang_vel = torch.zeros_like(self.root_pos)
        root_drot = quat_diff(self.root_rot[:-1], self.root_rot[1:])
        root_ang_vel[:-1] = self.fps * quat_to_exp_map(root_drot)
        root_ang_vel[-1] = root_ang_vel[-2]
        self.root_ang_vel = smooth(root_ang_vel, 19)

    def calc_motion_frame(self, times: torch.Tensor):
        motion_loop_num = torch.floor(times / self.length)
        times = times - motion_loop_num * self.length
        phase = torch.clamp(times / self.length, 0.0, 1.0)
        frame_idx0 = (phase * (self.num_frames - 1)).long()
        frame_idx1 = torch.minimum(frame_idx0 + 1, torch.tensor(self.num_frames - 1, device=self.device))
        blend = phase * (self.num_frames - 1) - frame_idx0.float()
        root_pos = (1.0 - blend.unsqueeze(-1)) * self.root_pos[frame_idx0] + blend.unsqueeze(-1) * self.root_pos[frame_idx1]
        root_pos = root_pos + motion_loop_num.unsqueeze(-1) * self.root_pos_delta
        root_rot = slerp(self.root_rot[frame_idx0], self.root_rot[frame_idx1], blend)
        root_vel = self.root_vel[frame_idx0]
        root_ang_vel = self.root_ang_vel[frame_idx0]
        dof_pos = (1.0 - blend.unsqueeze(-1)) * self.dof_pos[frame_idx0] + blend.unsqueeze(-1) * self.dof_pos[frame_idx1]
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos


def build_mimic_obs(motion_lib: SingleMotionLib, t_step: int, control_dt: float, tar_obs_steps: torch.Tensor):
    device = motion_lib.device
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = (tar_obs_steps * control_dt + motion_times).flatten()

    root_pos, root_rot, root_vel, root_ang_vel, dof_pos = motion_lib.calc_motion_frame(obs_motion_times)
    roll, pitch, yaw = euler_from_quaternion_xyzw(root_rot)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    root_vel = quat_rotate_inverse_xyzw(root_rot, root_vel).reshape(1, -1, 3)
    root_ang_vel = quat_rotate_inverse_xyzw(root_rot, root_ang_vel).reshape(1, -1, 3)
    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])

    # Insert zero wrist-roll slots at full 25-DoF indices 19 and 24.
    dof_pos_with_wrist = torch.zeros(25, device=device, dtype=dof_pos.dtype).reshape(1, 1, 25)
    wrist_ids = [19, 24]
    other_ids = [i for i in range(25) if i not in wrist_ids]
    dof_pos_with_wrist[..., other_ids] = dof_pos
    dof_pos = dof_pos_with_wrist

    mimic_obs = torch.cat(
        (root_pos[..., 2:3], roll, pitch, yaw, root_vel, root_ang_vel[..., 2:3], dof_pos),
        dim=-1,
    )[:, 0:1]
    return mimic_obs.reshape(-1).detach().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twist-root", default=os.environ.get("TWIST_ROOT", ""), help="Kept for CLI compatibility; not imported by this helper")
    ap.add_argument("--motion-pkl", required=True)
    ap.add_argument("--out", required=True, help="Output .csv cache path")
    ap.add_argument("--robot", default="g1", choices=["g1"])
    ap.add_argument("--steps", default="1", help="Comma-separated TWIST target observation steps")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    patch_numpy_core_pickle_compat()
    motion_pkl = os.path.abspath(os.path.expanduser(args.motion_pkl))
    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = torch.device(args.device)
    motion_lib = SingleMotionLib(motion_pkl, device=device)
    tar_obs_steps = torch.tensor([int(x.strip()) for x in args.steps.split(",") if x.strip()], dtype=torch.float32, device=device)
    control_dt = 1.0 / motion_lib.fps
    num_steps = int(motion_lib.length / control_dt)
    if args.max_frames > 0:
        num_steps = min(num_steps, args.max_frames)
    if num_steps <= 0:
        raise RuntimeError(f"motion has no usable frames: {motion_pkl}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            f"# twist_mimic_csv_v1 fps={motion_lib.fps:.9g} dt={control_dt:.9g} frames={num_steps} "
            f"dims=33 source={motion_pkl}\n"
        )
        for t in range(num_steps):
            obs = build_mimic_obs(motion_lib, t, control_dt, tar_obs_steps)
            if obs.shape[0] != 33:
                raise RuntimeError(f"expected 33-float mimic obs, got {obs.shape}")
            f.write(",".join(f"{float(x):.9g}" for x in obs.tolist()))
            f.write("\n")

    print(f"[twist_prepare_motion] wrote {num_steps} frames to {out_path}")


if __name__ == "__main__":
    main()
