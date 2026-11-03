#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import numpy as np


ACTION_GO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ACTION_GO_ROOT.parent / "UFO/model/g1_policy/tracking_inference_mjlab/zs_7.pkl"
DEFAULT_OUTPUT = ACTION_GO_ROOT / "actions/tracking/ufo/config/ufo_tracking_zs7.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a UFO tracking latent sequence for ActionGo")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--window-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.asarray(joblib.load(args.source), dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 256 or source.shape[0] == 0:
        raise ValueError(f"expected [frames,256] source, got {source.shape}")
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")

    target_norm = float(np.linalg.norm(source[0]))
    output = np.empty_like(source)
    for frame in range(source.shape[0]):
        window = source[frame : frame + args.window_size]
        discounts = args.gamma ** np.arange(window.shape[0], dtype=np.float32)
        discounts /= np.sum(discounts)
        latent = np.sum(window * discounts[:, None], axis=0)
        output[frame] = latent / (np.linalg.norm(latent) + 1e-8) * target_norm

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output.astype(np.float32))
    print(
        f"saved {args.output}: shape={output.shape} gamma={args.gamma} "
        f"window_size={args.window_size} norm={target_norm:.6f}"
    )


if __name__ == "__main__":
    main()
