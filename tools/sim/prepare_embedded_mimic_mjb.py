#!/usr/bin/env python3
"""Write an embedded-mimic reference frame into an existing MuJoCo keyframe."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mjb", type=Path, required=True)
    parser.add_argument("--output-mjb", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--robot-params", type=Path, required=True)
    parser.add_argument("--reference-step", type=int, default=0)
    parser.add_argument("--pelvis-body-index", type=int, default=0)
    parser.add_argument("--keyframe", default="init_state")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_joint(model: mujoco.MjModel, name: str) -> int:
    for candidate in (name, f"robot/{name}"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
        if joint_id >= 0:
            return joint_id
    raise RuntimeError(f"joint not found: {name}")


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_binary_path(str(args.input_mjb))
    keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe
    )
    if keyframe_id < 0:
        raise RuntimeError(f"keyframe not found: {args.keyframe}")

    params = yaml.safe_load(args.robot_params.read_text(encoding="utf-8"))
    joint_order = params["robot"]["policy_joint_order"]
    if len(joint_order) != 29:
        raise RuntimeError("expected 29 policy joints")

    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]
    required_outputs = {
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    missing = required_outputs.difference(output_names)
    if missing:
        raise RuntimeError(f"model outputs are missing: {sorted(missing)}")
    outputs = dict(
        zip(
            output_names,
            session.run(
                None,
                {
                    "obs": np.zeros((1, 154), dtype=np.float32),
                    "time_step": np.asarray(
                        [[args.reference_step]], dtype=np.float32
                    ),
                },
            ),
            strict=True,
        )
    )

    key_qpos = model.key_qpos[keyframe_id]
    key_qvel = model.key_qvel[keyframe_id]
    free_joint_id = next(
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    )
    qpos_address = model.jnt_qposadr[free_joint_id]
    qvel_address = model.jnt_dofadr[free_joint_id]
    body_index = args.pelvis_body_index
    key_qpos[qpos_address : qpos_address + 3] = outputs["body_pos_w"][0, body_index]
    key_qpos[qpos_address + 3 : qpos_address + 7] = outputs["body_quat_w"][0, body_index]
    key_qvel[qvel_address : qvel_address + 3] = outputs["body_lin_vel_w"][0, body_index]
    key_qvel[qvel_address + 3 : qvel_address + 6] = outputs["body_ang_vel_w"][0, body_index]

    joint_pos = outputs["joint_pos"][0]
    joint_vel = outputs["joint_vel"][0]
    if joint_pos.shape != (29,) or joint_vel.shape != (29,):
        raise RuntimeError(
            f"unexpected reference joint shapes: {joint_pos.shape}, {joint_vel.shape}"
        )
    for index, joint_name in enumerate(joint_order):
        joint_id = resolve_joint(model, joint_name)
        key_qpos[model.jnt_qposadr[joint_id]] = joint_pos[index]
        key_qvel[model.jnt_dofadr[joint_id]] = joint_vel[index]

    args.output_mjb.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(args.output_mjb), None)
    print(f"output={args.output_mjb}")
    print(f"output_sha256={sha256(args.output_mjb)}")
    print(f"model_sha256={sha256(args.model)}")
    print(f"reference_step={args.reference_step}")
    print(f"pelvis_pos={key_qpos[qpos_address:qpos_address + 3].tolist()}")
    print(f"pelvis_quat_wxyz={key_qpos[qpos_address + 3:qpos_address + 7].tolist()}")


if __name__ == "__main__":
    main()
