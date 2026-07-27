#!/usr/bin/env python3
"""Replay or synthesize the upstream GEAR-SONIC PICO protocol-v3 pose stream."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq

HEADER_SIZE = 1280


def dtype_name(value: np.ndarray) -> str:
    names = {
        np.dtype("float32"): "f32",
        np.dtype("float64"): "f64",
        np.dtype("int32"): "i32",
        np.dtype("int64"): "i64",
        np.dtype("bool"): "bool",
        np.dtype("uint8"): "u8",
    }
    dtype = np.dtype(value.dtype)
    if dtype not in names:
        raise ValueError(f"unsupported dtype: {dtype}")
    return names[dtype]


def pack_pose(fields: dict[str, np.ndarray], topic: str) -> bytes:
    metadata = []
    payload = []
    for name, raw in fields.items():
        value = np.ascontiguousarray(raw)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        metadata.append({"name": name, "dtype": dtype_name(value), "shape": list(value.shape)})
        payload.append(value.tobytes())
    header = json.dumps(
        {"v": 3, "endian": "le", "count": 1, "fields": metadata},
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ValueError(f"header is too large: {len(header)} > {HEADER_SIZE}")
    return topic.encode("utf-8") + header.ljust(HEADER_SIZE, b"\0") + b"".join(payload)


def synthetic_window(first_frame: int, count: int) -> dict[str, np.ndarray]:
    smpl_joints = np.zeros((count, 24, 3), dtype=np.float32)
    # A small upright skeleton is preferable to a fully collapsed reference for smoke tests.
    smpl_joints[:, :, 2] = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    body_quat = np.zeros((count, 4), dtype=np.float32)
    body_quat[:, 0] = 1.0
    return {
        "smpl_pose": np.zeros((count, 21, 3), dtype=np.float32),
        "smpl_joints": smpl_joints,
        "body_quat_w": body_quat,
        "joint_pos": np.zeros((count, 29), dtype=np.float64),
        "joint_vel": np.zeros((count, 29), dtype=np.float64),
        "frame_index": np.arange(first_frame, first_frame + count, dtype=np.int64),
        "toggle_data_collection": np.asarray([False], dtype=bool),
    }


def load_recordings(path: Path) -> list[dict[str, np.ndarray]]:
    files = sorted(path.glob("pose_*.npz")) if path.is_dir() else [path]
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"no PICO pose recordings found at {path}")
    packets = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            packets.append({name: np.asarray(data[name]) for name in data.files})
    return packets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://*:5556")
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--recording", type=Path, help="pose_*.npz file or directory from the upstream streamer")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--frames-per-packet", type=int, default=5)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 runs until interrupted")
    args = parser.parse_args()
    if args.hz <= 0 or args.frames_per_packet <= 0:
        parser.error("--hz and --frames-per-packet must be positive")

    recordings = load_recordings(args.recording.expanduser()) if args.recording else []
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    print(f"[sonic_pico_replay] PUB {args.bind} topic={args.topic} protocol=v3")
    time.sleep(0.5)

    period = 1.0 / args.hz
    started = time.monotonic()
    sequence = 0
    try:
        while args.duration_sec <= 0 or time.monotonic() - started < args.duration_sec:
            tick = time.monotonic()
            fields = recordings[sequence % len(recordings)] if recordings else synthetic_window(
                sequence, args.frames_per_packet
            )
            socket.send(pack_pose(fields, args.topic))
            sequence += 1
            remaining = tick + period - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
    print(f"[sonic_pico_replay] sent={sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
