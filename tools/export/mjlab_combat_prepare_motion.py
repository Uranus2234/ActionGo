"""Prepare another standardized G1 NPZ for the same exported mjlab combat policy."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from mjlab_combat_export import reference_rows, sha256


def prepare(args):
    sys.path.insert(0, str(args.training_repo.resolve() / "src"))
    from combat_tracker.motion_bank import MotionBank

    base = args.package.resolve()
    manifest = json.loads((base / "manifest.json").read_text())
    contract = manifest["policy_contract"]
    if manifest["variant"] != "mjlab_combat" or sha256(base / "policy.onnx") != manifest["policy_sha256"]:
        raise ValueError("Invalid mjlab combat source package")
    bank = MotionBank(base / "unused.json", contract["joint_names"], contract["tracked_body_names"],
                      1.0 / contract["step_dt"], motion_file=args.motion)
    if bank.model_sha256 != contract["model_sha256"]:
        raise ValueError("External motion robot model differs from the trained robot")
    anchor = bank.body_names.index(contract["anchor_body_name"])
    bodies = [bank.body_names.index(name) for name in contract["reference"]["task_body_names"]]
    rows = reference_rows(bank, anchor, bodies).numpy()
    args.output.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.output / "reference.csv", rows, delimiter=",", fmt="%.9g")
    np.testing.assert_array_equal(np.loadtxt(args.output / "reference.csv", delimiter=",", dtype=np.float32), rows)
    root = bank.body_names.index("pelvis")
    qpos = torch.cat((bank.body_pos_w[:, root], bank.body_quat_w[:, root], bank.joint_pos), dim=-1).numpy()
    qvel = torch.cat((bank.body_lin_vel_w[:, root], bank.body_ang_vel_w[:, root], bank.joint_vel), dim=-1).numpy()
    np.savez_compressed(args.output / "motion.npz", qpos=qpos, qvel=qvel)
    for name in ("policy.onnx", "scene.mjb", "MjlabCombat.yaml"):
        (args.output / name).symlink_to(base / name)
    spec = yaml.safe_load((base / "generic_mjlab_combat.yaml").read_text())
    (args.output / "generic_mjlab_combat.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    output_manifest = copy.deepcopy(manifest)
    output_manifest.update(motion=str(args.motion.resolve()), motion_sha256=sha256(args.motion),
                           reference_rows=len(rows), reference_sha256=sha256(args.output / "reference.csv"),
                           base_package=str(base), same_policy_weights=True)
    (args.output / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "frames": len(rows),
                      "policy_sha256": manifest["policy_sha256"], "same_policy_weights": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
