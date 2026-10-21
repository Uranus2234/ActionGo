import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


def export_goal_context(source, output, forward_speed=0.0, lateral_speed=0.0):
    package = source / "model/public_demo"
    config = json.loads((package / "config.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(source / "pbfm/data/robots/g1_mjlab/g1_29dof.xml"))
    data = mujoco.MjData(model)
    defaults = np.asarray(config["defaultJointPosition"], dtype=np.float32)
    for name, position in zip(config["jointNames"], defaults):
        data.qpos[model.joint(name).qposadr[0]] = position
    mujoco.mj_forward(model, data)
    links = ["pelvis", *["torso_link" if name == "waist_pitch_joint" else
                         name.removesuffix("_joint") + "_link" for name in config["jointNames"]]]
    positions = [data.body(name).xpos.copy() for name in links]
    rotations = [data.body(name).xmat.reshape(3, 3).copy() for name in links]
    torso = links.index("torso_link")
    positions.append(positions[torso] + rotations[torso] @ [0, 0, 0.35])
    rotations.append(rotations[torso])
    privileged = np.concatenate([[data.qpos[2]],
                                 np.asarray(positions[1:]).reshape(-1) - np.tile(data.qpos[:3], 30),
                                 np.concatenate([rotation[:, [0, 2]].T.reshape(-1) for rotation in rotations]),
                                 np.tile([forward_speed, lateral_speed, 0], 31), np.zeros(93)]).astype(np.float32)
    state = np.concatenate([np.zeros(58), [0, 0, -1], np.zeros(3)]).astype(np.float32)
    session = ort.InferenceSession(str(package / "backward_encoder.onnx"), providers=["CPUExecutionProvider"])
    latent = session.run(["z"], {"state": state[None], "privileged_state": privileged[None]})[0][0]
    if latent.shape != (256,) or not np.isfinite(latent).all() or not np.isclose(np.linalg.norm(latent), 16, atol=1e-4):
        raise ValueError("Invalid goal latent from public backward encoder")
    context = {"contract": "reward_goal_yaw_command_v1", "z": latent.tolist(),
               "heading_relative_xy": [1, 0], "heading_valid": True, "velocity_command": [forward_speed, lateral_speed, 1],
               "source": "P-BFM model/public_demo/backward_encoder.onnx; default joint pose with constant target translation",
               "feature_source": "pages 671fbafe71a5f58bab77eeb5d969051fd2d283f4 interactive/assets/motion-import-uImOWE3z.js backwardFeatures/motionKinematics",
               "checkpoint_step": config["checkpointStep"],
               "target_root_height": float(data.qpos[2]), "target_joint_position": defaults.tolist(),
               "target_root_velocity": [forward_speed, lateral_speed, 0],
               "target_joint_and_angular_velocity": "zero"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(context, sort_keys=False))
    return context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a G1 posture/translation goal using the official P-BFM backward encoder")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward-speed", type=float, default=0.0)
    parser.add_argument("--lateral-speed", type=float, default=0.0)
    parser.add_argument("--keyboard-contexts", action="store_true", help="Export idle/forward/backward/left/right contexts to --output directory")
    args = parser.parse_args()
    if not np.isfinite([args.forward_speed, args.lateral_speed]).all():
        parser.error("target speeds must be finite")
    if args.keyboard_contexts:
        for name, forward, lateral in [("idle", 0, 0), ("forward", 1.2, 0), ("backward", -0.3, 0),
                                        ("left", 0, 0.4), ("right", 0, -0.4)]:
            export_goal_context(args.source, args.output / f"context_keyboard_{name}.yaml", forward, lateral)
    else:
        export_goal_context(args.source, args.output, args.forward_speed, args.lateral_speed)
    print(f"Goal context output: {args.output}")
