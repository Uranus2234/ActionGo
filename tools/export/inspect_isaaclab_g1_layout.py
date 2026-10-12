#!/usr/bin/env python3
"""Print the runtime joint and body ordering of the WBT IsaacLab G1 asset."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class LayoutSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
    scene = InteractiveScene(LayoutSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    print("ISAACLAB_G1_JOINT_NAMES=" + repr(list(robot.joint_names)), flush=True)
    print("ISAACLAB_G1_BODY_NAMES=" + repr(list(robot.body_names)), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
