# UFO

UFO 是当前 ActionGo 中启用的 Unitree G1 29DoF latent-conditioned policy。
它通过 `generic_policy_player` 部署，支持固定 reward prompt、时序 tracking
latent，以及用于倾倒恢复的固定 stand prompt。

`actions/tracking/bfm0` 与 `actions/tracking/ufo` 当前同时注册。`Start` 选择 `bfm_zero_stand`，
`Left+B` 选择 `ufo_stand`；IMU 有效且倾角不低于 30° 时 ARM 立即运行所选恢复，
否则只启用动作和监控。两者站稳后都会切回 walkrun。默认后续倾倒只进入阻尼，
等待再次 ARM；显式启用自动恢复时才自动运行所选 stand backend。

## 模型协议

```text
actor_obs[1,721] -> action[1,29]
```

输入由以下部分组成：

```text
465D robot state/history + 256D latent
```

UFO 与 BFM-Zero 使用相同的 tensor 协议和 observation 拼接方式，但 UFO
有独立的 ONNX、PD gains、action scale 和关节限制，不能直接复用
`actions/tracking/bfm0/config/BFMZero.yaml`。

模型来源：

```text
HF repo: xuewang/ufo-g1-policy
HF revision: fc5efb7b3bb7a82270abaeabb0cf3c1194f1c7e6
Policy sha256: e0a67a6e7d40e6bb50253da4f4e25451556de2dba6446ebaa05f424aee118067
```

ActionGo 包内已经包含运行所需的 ONNX、latent 和配置；普通使用者不需要再次
下载模型。

## 可用动作

| 动作名 | Latent 来源 | 用途 |
|---|---|---|
| `ufo_idle` | `move-ego-0-0` | 静止站立 prompt |
| `ufo_stand` | `move-ego-0-0` | IMU safety 倾倒恢复和手动站立 |
| `ufo_forward_slow` | `move-ego-low0.5-0-0` | 低速前进 |
| `ufo_move_left` | `move-ego-90-0.3` | 向左移动 |
| `ufo_spin_arms` | `spin-arms-5-l-l` | 原地旋转并摆臂 |
| `ufo_tracking_zs7` | 发布模型中的 `zs_7` | 循环 tracking 动作 |

`ufo_tracking_zs7` 包含 6573 帧 256D latent，运行时按 50Hz 推进并循环。
导出时保持 UFO 原始参数：

```text
gamma=0.8
window_size=3
latent norm=16
```

## 编译

```bash
cd /path/to/action_go
cmake -S . -B build
cmake --build build --parallel "$(nproc)"
```

至少应生成：

```text
build/generic_policy_player
build/action_go_remote_test
build/policy_spec_schema_check
```

## 部署前检查

全量 schema 检查：

```bash
./build/policy_spec_schema_check --all
```

单独检查 tracking：

```bash
./build/policy_spec_schema_check \
  --spec actions/tracking/ufo/config/generic_ufo_tracking_zs7.yaml

./build/generic_policy_player \
  --spec actions/tracking/ufo/config/generic_ufo_tracking_zs7.yaml \
  --check-only
```

单独检查 stand：

```bash
./build/generic_policy_player \
  --spec actions/tracking/ufo/config/generic_ufo_stand.yaml \
  --check-only
```

## unitree_mujoco 仿真

推荐使用 ActionGo native bridge，它保持与真机一致的 `rt/user_lowcmd` command
gate，不需要修改 UFO spec。

无手柄、带 MuJoCo viewer：

```bash
cd /path/to/action_go

/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --interface lo \
  --domain-id 0 \
  --no-use-joystick
```

Viewer 键盘：

| 按键 | 功能 |
|---|---|
| `S` | ARM 选择 BFM；IMU 有效且倾角 ≥30° 时恢复，对应遥控器 `Start` |
| `B` | ARM 选择 UFO；IMU 有效且倾角 ≥30° 时恢复，对应遥控器 `Left+B` |
| `T` | 启动 `ufo_tracking_zs7` |
| `G` | 启动 `ufo_stand` |
| `W` | 停止当前动作并进入模拟 walkrun |
| `P` | 停止当前动作并进入 passive |
| `R` | 重置仿真 |
| `SPACE` | 暂停或继续 |

正常顺序：

```text
启动 viewer
  -> 按 S 选择 BFM stand，或按 B 选择 UFO stand
  -> 对应 stand 恢复并切回 walkrun
  -> 恢复后按 T 启动 tracking
```

当前有动作运行时，先按 `W` 或 `P` 停止，再启动另一个动作。

无界面 tracking smoke test：

```bash
/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --headless \
  --no-use-joystick \
  --auto-action ufo_tracking_zs7 \
  --auto-exit-sec 6 \
  --interface lo \
  --domain-id 0
```

无界面倾倒恢复测试：

```bash
/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --headless \
  --no-use-joystick \
  --auto-arm-sec 0.2 \
  --auto-exit-sec 18 \
  --interface lo \
  --domain-id 0
```

上述命令使用 Start/BFM backend。测试 Left+B/UFO backend：

```bash
/usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --headless \
  --no-use-joystick \
  --auto-arm-alt-sec 0.2 \
  --auto-exit-sec 18 \
  --interface lo \
  --domain-id 0
```

## 真机运行

真机必须使用 `rt/user_lowcmd`，不要把 UFO spec 改成 `rt/lowcmd`。

1. 确认机器人已进入官方 `walkrun` 或 passive/damping。
2. 确认实际 DDS 网卡名。
3. 启动 ActionGo supervisor。

```bash
cd /path/to/action_go

ACTION_GO_NET=enP8p1s0 tools/launch/start_safe.sh
```

如果机器人网卡不是 `enP8p1s0`，替换为实际接口，例如 `eth0`。

遥控器操作：

| 按键 | 功能 |
|---|---|
| `Start` | ARM；IMU 有效且倾角 ≥30° 时运行 `bfm_zero_stand` |
| `Left + B` | ARM；IMU 有效且倾角 ≥30° 时运行 `ufo_stand` |
| `Left + A` | 启动 `ufo_tracking_zs7` |
| `R2 + A` | 停止动作，切回官方 walkrun，返回 shell |
| `Select + Start` | 停止动作，切入 passive/damping，返回 shell |

## 倾角 Safety

默认配置位于 `config/remote_actions.yaml`：

```yaml
safety_enable: true
safety_auto_recovery: false
shell_start: Start
shell_start_action: bfm_zero_stand
shell_start_alt: Left+B
shell_start_alt_action: ufo_stand
safety_action: bfm_zero_stand
safety_return_mode: walkrun
fall_angle_deg: 57.3
recover_angle_deg: 30.0
fall_hold_ms: 60
recover_hold_ms: 1200
safety_cooldown_ms: 1000
```

`safety_action` 是回退值；正常遥控流程会使用 ARM 按键选择的 backend。

执行流程：

```text
按 Start
  -> ARM ActionGo
  -> IMU 有效且倾角 >=30° 时运行 bfm_zero_stand，否则仅 enable

按 Left+B
  -> ARM ActionGo
  -> IMU 有效且倾角 >=30° 时运行 ufo_stand，否则仅 enable

倾角 >= 57.3° 持续 60ms
  -> 中断当前动作或官方 walkrun
  -> 进入阻尼，默认等待操作员再次 ARM
  -> 自动恢复开启时才自动启动当前选择的 stand backend

恢复运行中
  -> 新鲜 IMU 倾角 <=30° 稳定 1200ms
  -> 切回官方 walkrun
```

`ufo_stand` 设置 `safety_enable: false`，避免恢复动作递归触发自身。
MuJoCo viewer 中也可以按 `G` 手动测试 UFO stand。

## Tracking Latent 再生成

如果重新从 UFO artifact 下载了 `zs_7.pkl`，可执行：

```bash
python3 tools/export/ufo_export_tracking_latent.py
```

默认输入：

```text
../UFO/model/g1_policy/tracking_inference_mjlab/zs_7.pkl
```

默认输出：

```text
actions/tracking/ufo/config/ufo_tracking_zs7.npy
```

也可以指定路径：

```bash
python3 tools/export/ufo_export_tracking_latent.py \
  --source /path/to/zs_7.pkl \
  --output actions/tracking/ufo/config/ufo_tracking_zs7.npy
```

## 文件结构

```text
actions/tracking/ufo/
  ufo.yaml
  README.md
  model/FBcprAuxModel.onnx
  config/UFO.yaml
  config/generic_ufo_*.yaml
  config/ufo_reward_*_z.txt
  config/ufo_tracking_zs7.npy
```

## 真机安全要求

在无支撑真机运行前，必须依次完成：

1. ONNX `--check-only`。
2. unitree_mujoco tracking 和倾倒恢复测试。
3. 吊装或支撑条件下的 `ufo_stand` 测试。
4. 验证 `R2 + A` 和 `Select + Start` 停止路径。
5. 确认物理急停可用。

不要仅凭 MuJoCo 成功就直接进行无保护真机测试。
