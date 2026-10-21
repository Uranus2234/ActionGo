import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess

import yaml


PUBLIC_REVISION = "671fbafe71a5f58bab77eeb5d969051fd2d283f4"


def main():
    parser = argparse.ArgumentParser(description="Import official P-BFM time-varying contexts")
    parser.add_argument("--source", type=Path, required=True, help="P-BFM Git repository with pages objects")
    parser.add_argument("--motion", default="lafan-full-walk1-subject5")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, help="Override walking speed in m/s, together with --heading-deg")
    parser.add_argument("--heading-deg", type=float, help="Fixed heading relative to starting yaw, together with --speed")
    args = parser.parse_args()
    if (args.speed is None) != (args.heading_deg is None):
        parser.error("--speed and --heading-deg must be provided together")
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.speed is not None and (not math.isfinite(args.speed) or args.speed < 0
                                   or not math.isfinite(args.heading_deg)):
        parser.error("speed must be finite and nonnegative; heading must be finite")
    asset = f"interactive/pbfm/tracking/{args.motion}.json.gz"
    compressed = subprocess.check_output(["git", "-C", str(args.source), "show", f"{PUBLIC_REVISION}:{asset}"])
    motion = json.loads(gzip.decompress(compressed))
    if motion["checkpointStep"] != 364822528 or motion["dt"] != 0.02:
        raise ValueError("Tracking context does not match the public P-BFM checkpoint / 50 Hz contract")
    if motion["referencePoseTiming"] != "post-action":
        raise ValueError("Unsupported reference pose timing")
    frames = []
    for frame in motion["frames"][:args.frames]:
        velocity = frame["velocityHeading0"]
        frames.append({"z": frame["z"], "heading_relative_xy": frame["headingRelativeXY"],
                       "heading_valid": frame["headingValid"],
                       "reference_velocity_heading0_xy": velocity[:2], "velocity_valid": bool(velocity[2])})
        if args.speed is not None:
            heading = [math.cos(math.radians(args.heading_deg)), math.sin(math.radians(args.heading_deg))]
            frames[-1].update(heading_relative_xy=heading, heading_valid=True,
                              reference_velocity_heading0_xy=[args.speed * value for value in heading],
                              velocity_valid=True)
    if not frames:
        raise ValueError("Tracking import requires at least one frame")
    args.output.mkdir(parents=True, exist_ok=False)
    context = {"contract": "tracking_heading0_yaw_velocity_v1", "dt": motion["dt"],
               "motion": motion["id"], "checkpoint_step": motion["checkpointStep"],
               "source": f"Roboparty/P-BFM {PUBLIC_REVISION}:{asset}",
               "frames": frames}
    if args.speed is not None:
        context["command_override"] = {"speed_mps": args.speed, "heading_deg": args.heading_deg,
                                       "source": "official interactive heading/speed override semantics"}
    (args.output / "context.yaml").write_text(yaml.safe_dump(context, sort_keys=False))
    initial = {key: motion[key] for key in ["initialPose", "initialState"]}
    initial.update(context_contract=context["contract"], motion=motion["id"],
                   frames=len(frames), dt=motion["dt"], source=context["source"])
    (args.output / "initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
    print(f"Imported {motion['id']}: {len(frames)} frames, {len(frames) * motion['dt']:.2f} seconds")


if __name__ == "__main__":
    main()
