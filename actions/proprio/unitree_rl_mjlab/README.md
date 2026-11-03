# unitree_rl_mjlab walkrun (simulation)

`sim_walkrun` uses the exported unitree_rl_mjlab velocity ONNX policy as the
unitree_mujoco replacement for the real robot's official `walkrun` controller.

Default ONNX:

```text
actions/proprio/unitree_rl_mjlab/model/policy.onnx
```

Source checkpoint:

```text
$UNITREE_RL_MJLAB_ROOT/logs/rsl_rl/g1_velocity/<run>/model_10000.pt
```

Quick checks:

```bash
# Generic Policy Runtime. The spec points to the exported ONNX in this action directory.
./build/generic_policy_player \
  --spec actions/proprio/unitree_rl_mjlab/config/generic_unitree_rl_mjlab.yaml \
  --check-only

# Unified sim remote interface: same remote bindings as config/remote_actions.yaml.
/usr/bin/python3 tools/sim/action_go_sim_remote.py --interface lo --domain-id 0
```

The unified sim entry defaults to `--sim-dt 0.002`; walkrun/action LowCmd
publication defaults to `1 / --sim-dt` so the native bridge runs a 500 Hz
low-level loop like the robot.

In simulation, action players are adapted from `rt/user_lowcmd` to `rt/lowcmd`.
The unified sim interface starts in passive damping, waits for `Start`, then lets
IMU safety trigger `bfm_zero_stand` before switching to this simulated walkrun.
