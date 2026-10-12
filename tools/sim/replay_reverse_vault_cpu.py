#!/usr/bin/env python3
"""Run deterministic no-DDS Reverse Vault CPU lockstep replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjb", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--robot-params", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def axis_quaternion(angle: float, axis: str) -> np.ndarray:
    cosine = np.cos(0.5 * angle)
    sine = np.sin(0.5 * angle)
    if axis == "x":
        return np.asarray([cosine, sine, 0.0, 0.0], dtype=np.float32)
    if axis == "y":
        return np.asarray([cosine, 0.0, sine, 0.0], dtype=np.float32)
    return np.asarray([cosine, 0.0, 0.0, sine], dtype=np.float32)


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def resolve_joint(model: mujoco.MjModel, name: str) -> int:
    for candidate in (name, f"robot/{name}"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
        if joint_id >= 0:
            return joint_id
    raise RuntimeError(f"joint not found: {name}")


def main() -> None:
    args = parse_args()
    params = yaml.safe_load(args.robot_params.read_text(encoding="utf-8"))["robot"]
    order = params["policy_joint_order"]
    default_q = np.asarray(params["default_angles"], dtype=np.float32)
    action_scale = np.asarray(params["action_scale"], dtype=np.float32)
    model = mujoco.MjModel.from_binary_path(str(args.mjb))
    data = mujoco.MjData(model)
    keyframe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    mujoco.mj_resetDataKeyframe(model, data, keyframe)
    mujoco.mj_forward(model, data)

    joint_ids = [resolve_joint(model, name) for name in order]
    qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    qvel_addresses = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    actuator_ids = []
    for joint_id in joint_ids:
        matches = [
            actuator_id
            for actuator_id in range(model.nu)
            if int(model.actuator_trnid[actuator_id, 0]) == joint_id
        ]
        if len(matches) != 1:
            raise RuntimeError(f"joint {joint_id} actuator matches: {matches}")
        actuator_ids.append(matches[0])
    gyro_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel"
    )
    gyro_address = int(model.sensor_adr[gyro_id])

    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]
    golden_records = [json.loads(line) for line in args.golden.read_text().splitlines()]
    previous_action = np.zeros(29, dtype=np.float32)
    maxima = {
        "obs": 0.0,
        "action": 0.0,
        "target": 0.0,
        "ctrl": 0.0,
        "qpos_after": 0.0,
        "qvel_after": 0.0,
    }
    first_divergence = None
    trace = []

    for policy_step, golden in enumerate(golden_records[: args.steps]):
        reference_step = int(golden["reference_step"])
        reference_values = session.run(
            None,
            {
                "obs": np.zeros((1, 154), dtype=np.float32),
                "time_step": np.asarray([[reference_step]], dtype=np.float32),
            },
        )
        reference = dict(zip(output_names, reference_values, strict=True))
        joint_q = np.asarray(data.qpos[qpos_addresses], dtype=np.float32)
        joint_dq = np.asarray(data.qvel[qvel_addresses], dtype=np.float32)
        pelvis_quaternion = np.asarray(data.qpos[3:7], dtype=np.float32)
        torso_quaternion = quaternion_multiply(
            pelvis_quaternion,
            quaternion_multiply(
                axis_quaternion(float(joint_q[12]), "z"),
                quaternion_multiply(
                    axis_quaternion(float(joint_q[13]), "x"),
                    axis_quaternion(float(joint_q[14]), "y"),
                ),
            ),
        )
        reference_quaternion = reference["body_quat_w"][0, 7]
        anchor = (
            quaternion_matrix(torso_quaternion).T
            @ quaternion_matrix(reference_quaternion)
        )[:, :2].reshape(-1)
        raw_obs = np.concatenate(
            [
                reference["joint_pos"][0],
                reference["joint_vel"][0],
                anchor,
                np.asarray(
                    data.sensordata[gyro_address : gyro_address + 3],
                    dtype=np.float32,
                ),
                joint_q - default_q,
                joint_dq,
                previous_action,
            ]
        ).astype(np.float32)
        outputs = dict(
            zip(
                output_names,
                session.run(
                    None,
                    {
                        "obs": raw_obs[None],
                        "time_step": np.asarray(
                            [[reference_step]], dtype=np.float32
                        ),
                    },
                ),
                strict=True,
            )
        )
        raw_action = outputs["actions"][0].astype(np.float32)
        raw_target = default_q + action_scale * raw_action
        for motor_index, actuator_id in enumerate(actuator_ids):
            data.ctrl[actuator_id] = raw_target[motor_index]

        obs_error = float(
            np.max(np.abs(raw_obs - np.asarray(golden["raw_obs"], dtype=np.float32)))
        )
        action_error = float(
            np.max(
                np.abs(raw_action - np.asarray(golden["raw_action"], dtype=np.float32))
            )
        )
        target_error = float(
            np.max(
                np.abs(raw_target - np.asarray(golden["raw_target"], dtype=np.float32))
            )
        )
        golden_ctrl = np.asarray(golden["data_ctrl_after"], dtype=np.float32)
        ctrl_error = float(np.max(np.abs(np.asarray(data.ctrl) - golden_ctrl)))
        for _ in range(4):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        qpos_error = float(
            np.max(
                np.abs(
                    np.asarray(data.qpos)
                    - np.asarray(golden["qpos_after"], dtype=np.float64)
                )
            )
        )
        qvel_error = float(
            np.max(
                np.abs(
                    np.asarray(data.qvel)
                    - np.asarray(golden["qvel_after"], dtype=np.float64)
                )
            )
        )
        errors = {
            "obs": obs_error,
            "action": action_error,
            "target": target_error,
            "ctrl": ctrl_error,
            "qpos_after": qpos_error,
            "qvel_after": qvel_error,
        }
        for name, value in errors.items():
            maxima[name] = max(maxima[name], value)
        if first_divergence is None and (
            obs_error > 1e-6 or action_error > 1e-5 or ctrl_error > 1e-7
        ):
            first_divergence = {"policy_step": policy_step, "errors": errors}
        trace.append(
            {
                "policy_step": policy_step,
                "reference_step": reference_step,
                "errors": errors,
                "right_knee_ctrl": float(data.ctrl[actuator_ids[9]]),
                "right_knee_force": float(data.actuator_force[actuator_ids[9]]),
            }
        )
        previous_action = raw_action

    report = {
        "steps": len(trace),
        "max_errors": maxima,
        "first_tensor_or_control_divergence": first_divergence,
        "trace": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "trace"}, indent=2))


if __name__ == "__main__":
    main()
