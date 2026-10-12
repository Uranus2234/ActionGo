#!/usr/bin/env python3
"""Convert an InstinctMJ motion with the official IsaacLab G1 replay path.

MuJoCo and PhysX do not use the same body ordering. Consequently this tool
does not copy body arrays from the source NPZ. It creates the CSV accepted by
IsaacLab's ``scripts/csv_to_npz.py`` and lets IsaacLab replay the 29-DoF pose
through its own G1 articulation and forward kinematics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


SOURCE_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

ISAACLAB_BODY_NAMES = (
    "pelvis", "left_hip_pitch_link", "pelvis_contour_link", "right_hip_pitch_link",
    "waist_yaw_link", "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link",
    "right_knee_link", "head_link", "left_shoulder_pitch_link", "logo_link",
    "right_shoulder_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "LL_FOOT", "LR_FOOT", "left_elbow_link", "right_elbow_link", "left_wrist_roll_link",
    "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link", "left_rubber_hand", "right_rubber_hand",
)


def _load_source(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as source:
        required = {"fps", "joint_pos", "body_pos_w", "body_quat_w", "joint_names", "body_names"}
        missing = required.difference(source.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        names = [str(name) for name in np.asarray(source["joint_names"]).reshape(-1)]
        if names != list(SOURCE_JOINT_NAMES):
            missing_joints = [name for name in SOURCE_JOINT_NAMES if name not in names]
            if missing_joints:
                raise ValueError(f"Missing G1 joints: {missing_joints}")
            indexes = [names.index(name) for name in SOURCE_JOINT_NAMES]
        else:
            indexes = list(range(len(SOURCE_JOINT_NAMES)))

        body_names = [str(name) for name in np.asarray(source["body_names"]).reshape(-1)]
        if "pelvis" not in body_names:
            raise ValueError("Source motion has no pelvis body")
        pelvis_index = body_names.index("pelvis")
        joint_pos = np.asarray(source["joint_pos"], dtype=np.float32)[:, indexes]
        root_pos = np.asarray(source["body_pos_w"], dtype=np.float32)[:, pelvis_index]
        root_quat_wxyz = np.asarray(source["body_quat_w"], dtype=np.float32)[:, pelvis_index]
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
            raise ValueError(f"joint_pos must have shape (frames, 29), got {joint_pos.shape}")
        if root_pos.shape != (joint_pos.shape[0], 3) or root_quat_wxyz.shape != (joint_pos.shape[0], 4):
            raise ValueError("pelvis position/quaternion frame count does not match joint_pos")
        if not np.isfinite(np.concatenate((root_pos, root_quat_wxyz, joint_pos), axis=1)).all():
            raise ValueError("source motion contains NaN or Inf")
        quat_norm = np.linalg.norm(root_quat_wxyz, axis=1)
        if not np.allclose(quat_norm, 1.0, atol=2.0e-3):
            raise ValueError("pelvis quaternions are not normalized")
        fps = float(np.asarray(source["fps"]).reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be positive and finite, got {fps}")

    csv = np.concatenate((root_pos, root_quat_wxyz[:, [1, 2, 3, 0]], joint_pos), axis=1)
    return csv, np.asarray([fps], dtype=np.float32), fps


def convert(input_path: Path, output_path: Path, isaaclab_repo: Path, python_executable: str) -> None:
    csv_data, fps_array, fps = _load_source(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="instinct_isaaclab_") as temporary_dir:
        temporary = Path(temporary_dir)
        csv_path = temporary / "motion.csv"
        isaaclab_output = temporary / "motion.npz"
        np.savetxt(csv_path, csv_data, delimiter=",", fmt="%.9g")
        command = [
            python_executable,
            str(isaaclab_repo / "scripts/csv_to_npz.py"),
            "--input_file", str(csv_path),
            "--input_fps", str(round(fps)),
            "--output_fps", str(round(fps)),
            "--robot", "g1",
            "--output_file", str(isaaclab_output),
            "--headless",
        ]
        environment = os.environ.copy()
        source_package = isaaclab_repo / "source/whole_body_tracking"
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(source_package) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        subprocess.run(command, cwd=isaaclab_repo, env=environment, check=True)

        with np.load(isaaclab_output, allow_pickle=False) as converted:
            output = {key: converted[key] for key in converted.files}
        required_output = {
            "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w",
        }
        missing_output = required_output.difference(output)
        if missing_output:
            raise KeyError(f"IsaacLab converter output is missing keys: {sorted(missing_output)}")
        frame_count = output["joint_pos"].shape[0]
        if output["joint_pos"].shape[1:] != (len(ISAACLAB_JOINT_NAMES),):
            raise ValueError(f"IsaacLab joint_pos has unexpected shape: {output['joint_pos'].shape}")
        if output["body_pos_w"].shape[1:] != (len(ISAACLAB_BODY_NAMES), 3):
            raise ValueError(f"IsaacLab body_pos_w has unexpected shape: {output['body_pos_w'].shape}")
        output["fps"] = fps_array
        output["motion_start_mask"] = np.asarray([True, *([False] * (frame_count - 1))], dtype=np.bool_)
        output["joint_names"] = np.asarray(ISAACLAB_JOINT_NAMES)
        output["body_names"] = np.asarray(ISAACLAB_BODY_NAMES)
        np.savez(output_path, **output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="InstinctMJ source motion NPZ")
    parser.add_argument("output", type=Path, help="IsaacLab motion NPZ")
    parser.add_argument(
        "--isaaclab-repo",
        type=Path,
        default=Path(os.environ.get("WHOLE_BODY_TRACKING_ROOT", ".")),
        help="whole_body_tracking checkout containing scripts/csv_to_npz.py",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("ISAACLAB_PYTHON", "python"),
        help="IsaacLab Python executable",
    )
    args = parser.parse_args()
    convert(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.isaaclab_repo.expanduser().resolve(),
        args.python,
    )
    print(f"[INFO] wrote IsaacLab motion: {args.output}")


if __name__ == "__main__":
    main()
