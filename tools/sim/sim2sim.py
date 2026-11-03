#!/usr/bin/env python3
"""Run ActionGo policies in one MuJoCo scene with a mjlab WALKRUN controller."""

import sys
from pathlib import Path

from action_go_sim_remote import main


if __name__ == "__main__":
    profile = Path(__file__).resolve().parents[2] / "config/sim2sim.yaml"
    raise SystemExit(main(["--profile", str(profile), *sys.argv[1:]]))
