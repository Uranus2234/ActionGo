# Simulation

仿真和真机最容易混淆的是 DDS topic：

```text
真机 UserCtrl:     rt/user_lowcmd
unitree_mujoco:    rt/lowcmd
LowState:          rt/lowstate
```

真机配置不要把 UserCtrl 动作发到 `rt/lowcmd`。

## BFM-Zero MuJoCo preview

```bash
python3 tools/sim/bfm_zero_mujoco_preview.py --safety-demo
```

无界面：

```bash
python3 tools/sim/bfm_zero_mujoco_preview.py \
  --headless \
  --safety-demo \
  --duration 6 \
  --auto-start-sec 1.0
```

## unitree_mujoco 中单独测 BFM policy

如果使用 `unitree_mujoco`，不要走 `SwitchToUserCtrl`。直接跑 player，并发到 `rt/lowcmd`：

```bash
./build/generic_policy_player \
  --net lo \
  --cmd-topic rt/lowcmd \
  --state-topic rt/lowstate \
  --spec actions/bfm0/config/generic_bfm_zero.yaml \
  --duration-sec 10
```

## unitree_mujoco 中测试 UFO

推荐使用 ActionGo 的 native bridge。它会直接加载
`$UNITREE_MUJOCO_ROOT/unitree_robots/g1/scene_29dof.xml`，并保持与真机一致的
`rt/user_lowcmd` / `rt/lowcmd` command gate：

```bash
cd /path/to/action_go

/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --interface lo \
  --domain-id 0 \
  --no-use-joystick \
  --auto-action ufo_idle
```

无界面 smoke test：

```bash
/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --headless \
  --interface lo \
  --domain-id 0 \
  --no-use-joystick \
  --auto-action ufo_idle \
  --auto-exit-sec 6
```

可用 UFO 动作：

```text
ufo_idle
ufo_forward_slow
ufo_move_left
ufo_spin_arms
ufo_tracking_zs7
ufo_stand
```

Tracking 模式：

```bash
/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --interface lo \
  --domain-id 0 \
  --no-use-joystick \
  --auto-action ufo_tracking_zs7
```

统一遥控模式下，`Start` 选择 `bfm_zero_stand`，`Left+B` 选择 `ufo_stand`。
IMU 有效且倾角不低于 30° 时 ARM 立即请求恢复，否则只启用动作和监控。
恢复站稳后切回 walkrun；后续摔倒默认先阻尼，等待再次 ARM，显式启用
`safety_auto_recovery` 时才自动恢复。`Left+A` 启动 UFO tracking。

没有手柄时直接使用 MuJoCo viewer 键盘：

```text
S      ARM 选择 bfm_zero_stand；IMU 有效且倾角 >=30° 时恢复，对应 Start
B      ARM 选择 ufo_stand；IMU 有效且倾角 >=30° 时恢复，对应 Left+B
T      启动 ufo_tracking_zs7
G      启动 ufo_stand
V      重新启动命令行 `--auto-action` 指定的动作
W      停止当前动作并进入模拟 walkrun
P      停止当前动作并进入 passive
R      重置仿真并自动进入模拟 walkrun
SPACE  暂停/继续
```

正常顺序是按 `S` 选择 BFM stand，或按 `B` 选择 UFO stand；倾倒时等待恢复并切回
walkrun，直立时 ARM 不启动恢复，之后按 `T` 启动 tracking。当前动作运行时需要先按 `W` 或 `P` 停止，
才能启动另一个动作。

无手柄测试倾倒后的人工 ARM 恢复请求；ARM 时 IMU 倾角须达到恢复阈值：

```bash
/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --headless \
  --no-use-joystick \
  --auto-arm-sec 0.2 \
  --auto-exit-sec 22 \
  --interface lo \
  --domain-id 0
```

上面的 `--auto-arm-sec` 测试 Start/BFM 入口。测试 Left+B/UFO 入口：

```bash
/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --headless \
  --no-use-joystick \
  --auto-arm-alt-sec 0.2 \
  --auto-exit-sec 18 \
  --interface lo \
  --domain-id 0
```

`bfm_zero_stand` 和 `ufo_stand` 当前均已注册，可通过不同 ARM 按键选择。

如果使用 stock `unitree_mujoco` bridge 分两个终端启动，则 player 必须显式发布
到 `rt/lowcmd`。不要修改 UFO 真机 spec 中的 `rt/user_lowcmd`，只使用命令行覆盖：

```bash
./build/generic_policy_player \
  --net lo \
  --cmd-topic rt/lowcmd \
  --state-topic rt/lowstate \
  --spec actions/ufo/config/generic_ufo_idle.yaml \
  --duration-sec 10
```

## 统一遥控仿真入口

`tools/sim/action_go_sim_remote.py` 对齐真机 `action_go_remote_test` 的外壳/按键逻辑：

- 启动后默认 `SHELL_IDLE + passive damping`：机器人不进 walkrun，会保持阻尼并倒地
- `Start`：ARM 选择 BFM；IMU 有效且倾角 >=30° 时运行 `bfm_zero_stand`
- `Left+B`：ARM 选择 UFO；IMU 有效且倾角 >=30° 时运行 `ufo_stand`
- 站稳后切到仿真 walkrun；后续摔倒默认阻尼并回 SHELL_IDLE，等待再次 ARM
- `safety_auto_recovery: true` 才允许摔倒阻尼后自动恢复；恢复失败阻尼，不自动重试
- 无效或超过 500 ms 的 IMU 会重置恢复站稳计时；动作自身的 `safety_enable` 与真机一致
- `Select+Start`：回到 SHELL_IDLE，并进入仿真 passive damping
- 低层 emergency / exit 组合默认禁用，常规遥控停止只使用 `Select+Start`
- 具体动作绑定继续读 `config/remote_actions.yaml`

区别是：真机的官方 `walkrun` 在仿真中由
`build/generic_policy_player` 启动的 unitree_rl_mjlab velocity policy 替代。
Instinct 暂时不接入统一仿真入口：`Up+Y -> g1_parkour` 仍保留给真机
supervisor，但在 `tools/sim/action_go_sim_remote.py` 中会被过滤禁用。
默认模型：

```text
actions/unitree_rl_mjlab/model/policy.onnx
```

推荐用 system Python 启动仿真 supervisor，因为 unitree_mujoco 的 CycloneDDS Python 依赖通常装在 `/usr/bin/python3`。walkrun player 本身是 C++ binary，默认直接使用已导出的 ONNX；如果手动传 `.pt`，则仍会尝试解析同目录导出的 ONNX 做推理：

```bash
/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --interface lo \
  --domain-id 0
```

默认 `--sim-dt 0.002`，低层 LowState/LowCmd gate 按 500Hz 同步仿真步进；
walkrun 和 action player 的 `--publish-hz` 默认都会跟随 `1 / --sim-dt`。
默认 `--bridge native` 会保留真机 topic 语义：walkrun 发 `rt/lowcmd`，
action_go players 发 `rt/user_lowcmd`，由仿真 gate 在切换时选择当前生效命令。

无手柄/无界面 smoke test：

```bash
timeout 8s /usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --headless \
  --no-use-joystick \
  --walkrun-zero-command
```

单独检查 walkrun player：

```bash
./build/generic_policy_player \
  --spec actions/unitree_rl_mjlab/config/generic_unitree_rl_mjlab.yaml \
  --check-only
```

如果一定要换成导出的 ONNX，可传：

```bash
/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --walkrun-model /path/to/policy.onnx \
  --walkrun-params /path/to/params/env.yaml
```

## ProtoMotions viewer

```bash
python3 tools/sim/protomotions_unitree_mujoco_viewer.py \
  --motion actions/protomotions/motions/output_walk.50fps.npz
```

## Holosoma 官方 MuJoCo player

```bash
tools/sim/holosoma_wbt_official_mujoco_player.sh
```
