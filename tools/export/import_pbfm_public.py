import argparse
import json
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Import the official P-BFM public-demo inference package")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "actions/pbfm")
    parser.add_argument("--hardware-params", type=Path, default=ROOT / "actions/ufo/config/UFO.yaml")
    args = parser.parse_args()
    config = json.loads((args.source / "config.json").read_text())
    contexts = json.loads((args.source / "contexts.json").read_text())
    if config["format"] != "direct_depth_full_v1" or config["controlDt"] != 0.02:
        raise ValueError("Unsupported P-BFM export format or control period")
    hardware = yaml.safe_load(args.hardware_params.read_text())["robot"]
    if config["jointNames"] != hardware["joint_order"]:
        raise ValueError("P-BFM joint order differs from the selected G1 hardware parameters")
    robot = {
        "motor_count": len(config["jointNames"]),
        "joint_order": config["jointNames"],
        "default_angles": config["defaultJointPosition"],
        "kp": config["kp"], "kd": config["kd"], "action_scale": config["actionScale"],
        "torque_limits": [min(source, limit) for source, limit in
                          zip(config["effortLimit"], hardware["torque_limits"])],
        "joint_pos_lower_limit": [interval[0] for interval in config["jointRanges"]],
        "joint_pos_upper_limit": [interval[1] for interval in config["jointRanges"]],
    }
    params = {
        "schema_version": 1, "robot": robot,
        "policy": {"num_actions": 29, "actor_obs_dim": 5334},
        "action": {"action_rescale": config["actionMultiplier"]},
        "simulation": {key: config[key] for key in
                       ("effortLimit", "velocityLimit", "armature", "frictionLoss", "physicsDt")},
        "provenance": {
            "source": "Roboparty/P-BFM pages public_demo/config.json",
            "checkpoint_step": config["checkpointStep"],
            "hardware_limit_source": "existing ActionGo UFO torque limits, capped by P-BFM effortLimit",
        },
    }
    output_config = args.output / "config"
    output_model = args.output / "model"
    output_config.mkdir(parents=True, exist_ok=True)
    output_model.mkdir(parents=True, exist_ok=True)
    (output_config / "PBFM.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    for preset in contexts["presets"]:
        context = {
            "contract": preset["contextContract"], "preset": preset["id"],
            "checkpoint_step": contexts["checkpointStep"], "z": preset["z"],
            "heading_relative_xy": preset["headingRelativeXY"],
            "heading_valid": preset["headingValid"], "velocity_command": preset["velocityCommand"],
            "source": preset["source"],
        }
        (output_config / f"context_{preset['id']}.yaml").write_text(yaml.safe_dump(context, sort_keys=False))
    shutil.copyfile(args.source / "policy.onnx", output_model / "policy.onnx")
    shutil.copyfile(args.source / "LICENSE", output_model / "LICENSE")
    print(f"Imported checkpoint {config['checkpointStep']} and {len(contexts['presets'])} fixed contexts into {args.output}")


if __name__ == "__main__":
    main()
