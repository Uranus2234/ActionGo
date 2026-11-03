#!/usr/bin/env python3
"""Export TWIST TorchScript policy (.pt) to ONNX for action_go generic runtime."""
import argparse
import os
from pathlib import Path
import torch


def main():
    action_go_root = Path(__file__).resolve().parents[2]
    twist_root = os.environ.get("TWIST_ROOT", "")
    default_pt = str(Path(twist_root) / "assets/twist_general_motion_tracker.pt") if twist_root else None
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=default_pt,
                    help="TorchScript checkpoint; defaults to $TWIST_ROOT/assets/twist_general_motion_tracker.pt")
    ap.add_argument("--out", default=str(action_go_root / "actions/tracking/twist/model/twist_general_motion_tracker.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    if not args.pt:
        ap.error("--pt is required when TWIST_ROOT is not set")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model = torch.jit.load(args.pt, map_location="cpu")
    model.eval()
    dummy = torch.zeros(1, 1155, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["obs"],
        output_names=["action"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
