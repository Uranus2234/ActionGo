#!/usr/bin/env python3
"""Crop OmniXtreme motion.npz by frame range.

The OmniXtreme motion file stores most arrays as [frames, ...].  This tool
slices every array whose first dimension equals joint_pos.shape[0] and preserves
metadata arrays such as fps unchanged.

Frame convention is Python slicing: [start_frame, end_frame).  To keep frames
0..950 inclusive, use --start-frame 0 --end-frame 951.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crop OmniXtreme motion.npz by frame range")
    p.add_argument("--input", "-i", required=True, help="Input motion.npz")
    p.add_argument("--output", "-o", required=True, help="Output cropped .npz")
    p.add_argument("--start-frame", type=int, default=0, help="Inclusive start frame, default 0")
    p.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive end frame. Example: 951 keeps frames 0..950 inclusive.",
    )
    p.add_argument(
        "--end-frame-inclusive",
        type=int,
        default=None,
        help="Inclusive end frame convenience option. Mutually exclusive with --end-frame.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    p.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed. Default uses uncompressed np.savez for fastest loading.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()

    if args.end_frame is not None and args.end_frame_inclusive is not None:
        raise SystemExit("--end-frame and --end-frame-inclusive are mutually exclusive")
    if args.end_frame_inclusive is not None:
        end = args.end_frame_inclusive + 1
    else:
        end = args.end_frame

    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {out}; pass --overwrite to replace")

    with np.load(inp, allow_pickle=False) as data:
        if "joint_pos" not in data:
            raise SystemExit("input does not contain required key 'joint_pos'")
        total_frames = int(data["joint_pos"].shape[0])
        start = int(args.start_frame)
        if end is None:
            end = total_frames
        end = int(end)
        if start < 0 or end < 0 or start >= end or end > total_frames:
            raise SystemExit(
                f"invalid frame range [{start}, {end}); valid range is [0, {total_frames}] and start < end"
            )

        cropped: Dict[str, np.ndarray] = {}
        sliced_keys = []
        kept_keys = []
        for key in data.files:
            arr = data[key]
            if arr.ndim >= 1 and int(arr.shape[0]) == total_frames:
                cropped[key] = arr[start:end].copy()
                sliced_keys.append(key)
            else:
                cropped[key] = arr.copy()
                kept_keys.append(key)

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.compressed:
        np.savez_compressed(out, **cropped)
    else:
        np.savez(out, **cropped)

    print(f"[omnixtreme_crop_motion] input={inp}")
    print(f"[omnixtreme_crop_motion] output={out}")
    print(f"[omnixtreme_crop_motion] frames {total_frames} -> {end - start} using slice [{start}, {end})")
    if sliced_keys:
        print("[omnixtreme_crop_motion] sliced: " + ", ".join(sliced_keys))
    if kept_keys:
        print("[omnixtreme_crop_motion] kept: " + ", ".join(kept_keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
