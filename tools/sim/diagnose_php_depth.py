import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import onnxruntime as ort
import yaml

from php_camera_geometry import estimate_lower_plane

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Inspect PHP depth traces without robot control.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--actiongo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with args.trace.open() as source:
        for line in source:
            records.append(yaml.load(line, Loader=yaml.CSafeLoader))
    if not records:
        raise ValueError("trace contains no samples")
    record = records[-1]
    depth = np.asarray(record["depth"], dtype=np.float32).reshape(58, 87)
    metres = (depth + 0.5) * 2.7 + 0.3
    figure, axis = plt.subplots(figsize=(10, 6))
    picture = axis.imshow(metres, vmin=0.3, vmax=3, cmap="turbo_r", origin="upper")
    axis.set_title("Actual PHP input: optical-axis depth (m); clipped at 3 m")
    axis.set_xlabel("Image column")
    axis.set_ylabel("Image row (top to bottom)")
    figure.colorbar(picture, ax=axis, label="Depth (m)")
    figure.tight_layout()
    figure.savefig(args.output / "depth.png", dpi=160)
    plt.close(figure)

    model_root = args.actiongo_root / "actions/perception/php/model"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    encoder = ort.InferenceSession(str(model_root / "depth_backbone.onnx"), options)
    actor = ort.InferenceSession(str(model_root / "student.onnx"), options)
    robot = yaml.safe_load((args.actiongo_root / "actions/perception/php/config/PHPParams.yaml").read_text())["robot"]
    responses = {}
    for command in (0, 1, 6):
        outputs = {}
        latents = {}
        for name, image in (("actual", depth), ("all_far", np.full_like(depth, 0.5))):
            latent = encoder.run(["depth_latent"], {"depth_image": image[None]})[0]
            observation = np.asarray(record["obs"], dtype=np.float32)[None].copy()
            observation[:, 93:108] = 0
            observation[:, 93 + command] = 1
            observation[:, 108:] = latent.reshape(1, 32)
            outputs[name] = actor.run(["actions"], {
                "obs": observation, "time_step": np.zeros((1, 1), dtype=np.float32)
            })[0].reshape(29)
            latents[name] = latent
        difference = np.abs(outputs["actual"] - outputs["all_far"])
        joint = int(np.argmax(difference))
        responses[str(command)] = {
            "depth_latent_max_difference": float(np.max(np.abs(latents["actual"] - latents["all_far"]))),
            "action_max_difference_rad": float(difference[joint]),
            "action_max_difference_joint": robot["policy_joint_order"][joint],
            "leg_action_max_difference_rad": float(difference[:12].max()),
            "left_knee_difference_rad": float(difference[3]),
            "right_knee_difference_rad": float(difference[9]),
        }
    velocities = np.abs(np.asarray([sample["dq"] for sample in records]))
    sample_index, joint_index = np.unravel_index(np.argmax(velocities), velocities.shape)
    planes = [plane for sample in records[len(records) // 2:]
              if (plane := estimate_lower_plane(sample)) is not None]
    result = {
        "samples": len(records),
        "observed_commands": sorted({sample["command"] for sample in records}),
        "depth_row_medians_metres": np.median(metres, axis=1).tolist(),
        "peak_joint_velocity_rad_s": float(velocities[sample_index, joint_index]),
        "peak_joint_velocity_joint": robot["policy_joint_order"][joint_index],
        "lower_plane_fit": {
            "accepted_samples": len(planes),
            "median": {key: float(np.median([plane[key] for plane in planes]))
                       for key in planes[0]} if planes else None,
            "assumption": "Lower image side quarters are a horizontal floor. Relative pitch is approximate for small torso roll.",
            "official_camera_pitch_relative_to_torso_deg": 27,
            "training_pitch_randomization_deg": 3,
        },
        "counterfactual_actual_vs_all_far": responses,
        "limitation": "Single-step sensitivity with identical proprioception and previous action; not a locomotion or stair acceptance test.",
    }
    (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "depth_row_medians_metres"}, indent=2))


if __name__ == "__main__":
    main()
