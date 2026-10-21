import argparse
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np

from pbfm_reference import PoseResampler, ReferenceEncoder


def packet(context, sequence, capture_unix_sec):
    return json.dumps({"contract": "bfm_realtime_context_v1", "sequence": sequence,
                       "capture_unix_sec": capture_unix_sec, **context}, allow_nan=False).encode()


def replay(args, encoder):
    motion = json.loads(args.replay.read_text())
    if motion["schema"] != "pbfm-g1-motion-v1" or set(motion["jointNames"]) != set(encoder.joint_names):
        raise ValueError("Replay requires the official pbfm-g1-motion-v1 format and all 29 named joints")
    remap = [motion["jointNames"].index(name) for name in encoder.joint_names]
    rate = float(motion["fps"])
    if not 1 <= rate <= 240:
        raise ValueError("Invalid replay sample rate")
    origin = time.monotonic()
    for index, frame in enumerate(motion["frames"]):
        target = origin + index / rate
        time.sleep(max(0, target - time.monotonic()))
        pose = np.concatenate([frame["rootPosition"], frame["rootQuaternionWxyz"],
                               np.asarray(frame["jointPosition"])[remap]])
        yield index / rate, pose, time.time()


def pico(args, encoder):
    sys.path.insert(0, str(args.gmr.resolve()))
    from general_motion_retargeting import GeneralMotionRetargeting, XRobotStreamer

    retarget = GeneralMotionRetargeting(src_human="xrobot", tgt_robot="unitree_g1",
                                       actual_human_height=args.human_height, verbose=False)
    streamer = XRobotStreamer()
    joint_addresses = [retarget.model.joint(name).qposadr[0] for name in encoder.joint_names]
    if retarget.model.nq != 36 or len(set(joint_addresses)) != 29:
        raise ValueError("GMR G1 joint layout differs from P-BFM")
    last_stamp = None
    origin_stamp = None
    last_new = time.monotonic()
    started = last_new
    try:
        print("[pbfm-pico] waiting for XRoboToolkit full-body tracking", flush=True)
        while True:
            poses, _, _, _, stamp = streamer.get_raw_body_data()
            now = time.monotonic()
            if poses is None or stamp == last_stamp:
                timeout = args.startup_timeout if last_stamp is None else args.stale_sec
                if now - last_new > timeout:
                    raise RuntimeError("XRoboToolkit body stream missing or stale")
                time.sleep(0.001)
                continue
            capture_time = time.time()
            stamp = int(stamp)
            if stamp <= 0 or (last_stamp is not None and stamp < last_stamp):
                raise RuntimeError("XRoboToolkit body timestamp reset; restart and recalibrate")
            if last_stamp is not None and (stamp - last_stamp) * 1e-9 > args.stale_sec:
                raise RuntimeError("XRoboToolkit body stream has a stale gap")
            values = np.asarray(poses, dtype=np.float64)
            if values.shape != (24, 7) or not np.isfinite(values).all():
                raise ValueError("XRoboToolkit must supply 24 finite XYZ + XYZW body poses")
            if not np.allclose(np.linalg.norm(values[:, 3:], axis=1), 1, atol=1e-3):
                raise ValueError("Invalid XRoboToolkit body rotations")
            body = {name: [values[index, :3].tolist(), values[index, [6, 3, 4, 5]].tolist()]
                    for index, name in enumerate(streamer.body_joint_names)}
            body = streamer.coordinate_transform_unity_data(body)
            qpos = retarget.retarget(body)
            pose = np.concatenate([qpos[:7], qpos[joint_addresses]])
            last_stamp, last_new = stamp, now
            if origin_stamp is None:
                origin_stamp = stamp
            yield (stamp - origin_stamp) * 1e-9, pose, capture_time
            if args.duration and now - started >= args.duration:
                return
    finally:
        streamer.close()


def main():
    parser = argparse.ArgumentParser(description="XRoboToolkit -> GMR G1 -> P-BFM backward encoder -> ActionGo")
    parser.add_argument("--source", type=Path, required=True, help="P-BFM repository containing model/public_demo and G1 XML")
    parser.add_argument("--gmr", type=Path, default=Path.home() / "resource/GMR")
    parser.add_argument("--human-height", type=float, default=1.7, help="Actual operator height in metres; used by GMR")
    parser.add_argument("--host", default="127.0.0.1", help="ActionGo receiver host (robot address for hardware)")
    parser.add_argument("--port", type=int, default=15112)
    parser.add_argument("--stale-sec", type=float, default=0.2)
    parser.add_argument("--startup-timeout", type=float, default=30)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--replay", type=Path, help="Replay an already retargeted G1 JSON through the same online encoder")
    parser.add_argument("--record", type=Path, help="Save sent contexts as JSONL")
    args = parser.parse_args()
    if not np.isfinite([args.human_height, args.stale_sec, args.startup_timeout, args.duration]).all() or not (
            args.human_height > 0 and args.stale_sec > 0 and args.startup_timeout > 0 and args.duration >= 0):
        parser.error("Height and timeouts must be positive; duration must be nonnegative")
    encoder = ReferenceEncoder(args.source)
    resampler = PoseResampler(max_gap=args.stale_sec)
    destination = (socket.gethostbyname(args.host), args.port)
    stream = replay(args, encoder) if args.replay else pico(args, encoder)
    record = args.record.open("w") if args.record else None
    sequence = 0
    print(f"[pbfm-pico] P-BFM checkpoint=364822528 destination={destination} reference_delay=0.34s", flush=True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            for timestamp, pose, capture_time in stream:
                newest_context = None
                for sample_time, sample_pose in resampler.push(timestamp, pose):
                    context = encoder.push(sample_time, sample_pose)
                    if context is not None:
                        newest_context = context
                if newest_context is None:
                    continue
                if time.time() - capture_time > args.stale_sec:
                    raise RuntimeError("Pico retargeting/encoding exceeded freshness deadline")
                encoded = packet(newest_context, sequence, capture_time)
                sender.sendto(encoded, destination)
                if record:
                    record.write(encoded.decode() + "\n")
                if sequence % 50 == 0:
                    print(f"[pbfm-pico] sequence={sequence} z_norm={np.linalg.norm(newest_context['z']):.3f} "
                          f"velocity={newest_context['velocity_command'][:2]}", flush=True)
                sequence += 1
    finally:
        stream.close()
        if record:
            record.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
