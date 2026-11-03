# BFM-Zero

BFM-Zero 用于摔倒后的恢复站立。默认动作名：

```text
bfm_zero_stand
```

资源：

```text
actions/tracking/bfm0/model/FBcprAuxModel.onnx
actions/tracking/bfm0/config/BFMZero.yaml
actions/tracking/bfm0/config/bfm_zero_reward_head1p2_base0_z.txt
```

## 配置

BFM-Zero 已迁移到 Generic Policy Runtime，动作入口在
`actions/tracking/bfm0/bfm_zero.yaml` 中：

```yaml
actions:
  generic_policy:
    bfm_zero_stand:
      params: config/generic_bfm_zero.yaml
```

## Latent Prompt

BFM-Zero 在 ActionGo 中仍归入 `single_actor_state`。它的 observation 包含一个
固定的 256D latent `z`，由 `generic_bfm_zero.yaml` 的
`observation.latent_file` 指向：

```text
actions/tracking/bfm0/config/bfm_zero_reward_head1p2_base0_z.txt
```

这个 `z` 可以理解为 fixed behavior prompt / behavior embedding。虽然 BFM-Zero
训练侧的 latent 空间可以编码 motion、goal 或 reward 语义，当前 runtime 不在线读取
motion source，也不逐帧跟踪 reference motion，因此不归入
`motion_conditioned_tracking`。

真机必须使用：

```text
rt/user_lowcmd
```

不要在真机上发到：

```text
rt/lowcmd
```

`rt/lowcmd` 主要用于 `unitree_mujoco` 仿真。

## 模型检查

```bash
./build/generic_policy_player \
  --spec actions/tracking/bfm0/config/generic_bfm_zero.yaml \
  --check-only
```

## 手动执行

推荐通过 supervisor 执行：

```bash
./build/action_go_remote_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --remote-map config/remote_actions.yaml
```

`Start` 选择 BFM 并 ARM；IMU 有效且倾角不低于 `30°` 时立即请求恢复，
否则只启用动作和监控。`Left+B` 使用同样规则选择 UFO。
如需单独测试 BFM，使用 `action_go_test --action bfm_zero_stand`。

## Safety 与恢复

默认 `safety_auto_recovery: false`，IMU safety 触发后的流程是：

```text
摔倒
  -> 官方阻尼，回 SHELL_IDLE
  -> 操作员按 Start 人工请求 bfm_zero_stand
  -> 新鲜 IMU 倾角 <=30° 稳定 1200 ms
  -> walkrun
```

显式设为 `safety_auto_recovery: true` 时，摔倒阻尼后会自动运行当前选择的恢复策略。
恢复期间忽略普通摔倒角度中断，payload 失败则阻尼并停止重试。

BFM-Zero 的 `bfm_zero_stand` 已启用回切 hot-standby：收到切回 `walkrun` 请求后，
先调用官方 7111 接口，同时继续发布最后一条经安全过滤的 BFM 命令 0.6 秒，随后才退出
player。这样可以覆盖官方 walkrun 完成 ownership 接管的启动延迟，避免中间低刚度空窗。

如果切回 walkrun 时仍短暂软掉，优先检查：

1. `generic_policy_player` command 是否带 `--no-damping-on-stop`
2. `recover_hold_ms` 是否太短，建议 `3000` 或更大
3. `recover_angle_deg` 是否太宽，建议真机可试 `10.0`
4. 官方 walkrun 是否在 0.6 秒内完成接管；若固件启动更慢，再按实测延长
   `actions/tracking/bfm0/bfm_zero.yaml` 的 `switch_back_overlap_sec`

示例：

```yaml
recover_angle_deg: 10.0
recover_hold_ms: 3000
```

## 仿真测试

项目自带 MuJoCo preview：

```bash
python3 tools/sim/bfm_zero_mujoco_preview.py --safety-demo
```

无界面 smoke test：

```bash
python3 tools/sim/bfm_zero_mujoco_preview.py \
  --headless \
  --safety-demo \
  --duration 6 \
  --auto-start-sec 1.0
```
