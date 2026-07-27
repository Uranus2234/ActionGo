# Holosoma WBT action_go notes

Holosoma WBT 当前动作：

```text
holosoma_dance
```

资源：

```text
actions/holosoma/model/ppo_g1_29dof_dancing.onnx
actions/holosoma/config/HolosomaWBT.yaml
actions/holosoma/holosoma_wbt.yaml
```

`holosoma_dance` 使用 `prestart_before_userctrl: true`：先启动 player 发布当前姿态 hold，写 `ready_file`，再由 action_go 切 UserCtrl 并写 `start_file` 开始策略，可减少切换空窗。

`holosoma_dance` 在 `holosoma_wbt.yaml` 中设置了 `max_frames: 1020`，并传给 player 的 `--motion-end-step 1020`，不会完整跳完整段舞。

Keep only two paths:

1. **Official MuJoCo validation**: Holosoma simulator/bridge + action_go C++ ONNX player.
2. **Real robot**: action_go C++ ONNX player via `rt/user_lowcmd`.

No standalone kinematic playback, no manual motion-rate scaling, no direct control from ONNX `joint_pos`.

## Official MuJoCo validation

Terminal 1:

```bash
cd "$HOLOSOMA_ROOT"
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

Terminal 2:

```bash
cd /path/to/action_go
tools/sim/holosoma_wbt_official_mujoco_player.sh
```

In the MuJoCo window, use the official WBT flow: press `8`/`9` to lower/remove the gantry, let the robot stabilize, then press `Enter` in Terminal 2.  The C++ player first runs the policy with `time_step` pinned at the start frame, then starts the motion clip.

## Real robot command

Single-action bringup, without remote safety state machine:

```bash
cd /path/to/action_go
./build/action_go_test --net enP8p1s0 --config config/motion_actions.yaml --action holosoma_dance
```

Recommended remote bringup, with IMU safety enabled by `config/remote_actions.yaml`:

```bash
cd /path/to/action_go
./build/action_go_remote_test --net enP8p1s0 --config config/motion_actions.yaml --remote-map config/remote_actions.yaml
```

Remote binding is `R1+F1: holosoma_dance`.  If `safety_enable: true`, tilt beyond the configured threshold interrupts any current action, including Holosoma, and starts `bfm_zero_stand`; after recovery it switches back to official `walkrun`.

Holosoma is configured with `prestart_before_userctrl: true`: the C++ player is launched first, publishes a safe current-pose hold to `rt/user_lowcmd`, reports ready, then action_go calls `SwitchToUserCtrl()` and writes `start_file`.  After UserCtrl is active, the player briefly interpolates to embedded motion frame 0, starts policy inference with the embedded motion frame pinned, then starts the motion clip.  This removes the UserCtrl handoff vacuum without assuming pre-UserCtrl commands moved the robot.

## Implementation notes

- ONNX inputs: `obs`, `time_step`.
- ONNX control output path matches official Holosoma WBT:
  `q_target = default_dof_angles + actions * action_scale`.
- ONNX `joint_pos`, `joint_vel`, `ref_quat_xyzw` are embedded reference terms used for the next observation, not direct motor targets.
- Observation flatten order matches official `BasePolicy` sorted terms:
  `actions, base_ang_vel, dof_pos, dof_vel, motion_command, motion_ref_ori_b`.
- Startup matches official WBT state order: init to motion frame 0 -> policy
  active with `motion_clip_progressing=false` -> motion clip progressing.  Do
  not replace this with direct kinematic playback.
- `--max-target-jump` warns on `q_target` versus current-state tracking error.
  It also warns on discontinuities between consecutive policy targets, but does
  not abort; killing the policy on a large recovery command removes the
  balancing controller.  Hard recovery is handled by `action_go_remote_test`
  IMU safety.
- Hardware timing: one `time_step` per 50 Hz policy tick.
- Official MuJoCo timing: `--use-sim-time` reads Holosoma's ZMQ sim clock on port 5555.
