#!/usr/bin/env python3
"""Replay or synthesize upstream GEAR-SONIC PICO pose/planner streams."""

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


def pack_message(fields: dict[str, np.ndarray], topic: str, version: int) -> bytes:
    metadata = []
    payload = []
    for name, raw in fields.items():
        value = np.ascontiguousarray(raw)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        metadata.append({"name": name, "dtype": dtype_name(value), "shape": list(value.shape)})
        payload.append(value.tobytes())
    header = json.dumps(
        {"v": version, "endian": "le", "count": 1, "fields": metadata},
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ValueError(f"header is too large: {len(header)} > {HEADER_SIZE}")
    return topic.encode("utf-8") + header.ljust(HEADER_SIZE, b"\0") + b"".join(payload)


def planner_fields(args: argparse.Namespace) -> dict[str, np.ndarray]:
    return {
        "mode": np.asarray([args.planner_mode], dtype=np.int32),
        "movement": np.asarray(args.movement, dtype=np.float32),
        "facing": np.asarray(args.facing, dtype=np.float32),
        "speed": np.asarray([args.speed], dtype=np.float32),
        "height": np.asarray([args.height], dtype=np.float32),
    }


def command_fields(*, start: bool, stop: bool, planner: bool) -> dict[str, np.ndarray]:
    return {
        "start": np.asarray([start], dtype=np.uint8),
        "stop": np.asarray([stop], dtype=np.uint8),
        "planner": np.asarray([planner], dtype=np.uint8),
    }


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
    parser.add_argument("--mode", choices=("planner", "pose"), default="pose")
    parser.add_argument("--planner-mode", type=int, choices=range(27), default=2)
    parser.add_argument("--movement", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.3, 0.0, 0.0))
    parser.add_argument("--facing", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(1.0, 0.0, 0.0))
    parser.add_argument("--speed", type=float, default=-1.0)
    parser.add_argument("--height", type=float, default=-1.0)
    parser.add_argument("--recording", type=Path, help="pose_*.npz file or directory from the upstream streamer")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--frames-per-packet", type=int, default=5)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 runs until interrupted")
    args = parser.parse_args()
    if args.hz <= 0 or args.frames_per_packet <= 0:
        parser.error("--hz and --frames-per-packet must be positive")
    if args.mode == "planner" and args.recording:
        parser.error("--recording is only valid with --mode pose")
    if not np.all(np.isfinite((*args.movement, *args.facing, args.speed, args.height))):
        parser.error("planner command values must be finite")

    recordings = load_recordings(args.recording.expanduser()) if args.recording else []
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    data_topic = "planner" if args.mode == "planner" else args.topic
    data_version = 1 if args.mode == "planner" else 3
    print(f"[sonic_pico_replay] PUB {args.bind} mode={args.mode} topic={data_topic} protocol=v{data_version}")
    time.sleep(0.5)

    data = pack_message(planner_fields(args), "planner", 1) if args.mode == "planner" else None
    control = pack_message(
        command_fields(start=True, stop=False, planner=args.mode == "planner"), "command", 1
    )
    # Match the manager ordering: publish one valid data packet before handing over control.
    if data is not None:
        socket.send(data)
    else:
        socket.send(pack_message(synthetic_window(0, args.frames_per_packet), args.topic, 3))
    socket.send(control)

    period = 1.0 / args.hz
    started = time.monotonic()
    sequence = 0
    try:
        while args.duration_sec <= 0 or time.monotonic() - started < args.duration_sec:
            tick = time.monotonic()
            if args.mode == "planner":
                socket.send(data)
            else:
                fields = recordings[sequence % len(recordings)] if recordings else synthetic_window(
                    sequence, args.frames_per_packet
                )
                socket.send(pack_message(fields, args.topic, 3))
            # A real operator sends this after the player is listening. Repeating the
            # idempotent state makes unattended replay robust to PUB/SUB slow joiners.
            if sequence % max(1, int(round(args.hz))) == 0:
                socket.send(control)
            sequence += 1
            remaining = tick + period - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        socket.send(pack_message(command_fields(start=False, stop=True, planner=True), "command", 1))
        time.sleep(0.02)
        socket.close()
    print(f"[sonic_pico_replay] sent={sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
