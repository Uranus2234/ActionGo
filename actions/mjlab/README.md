# MJLab 策略包

本目录用于 ActionGo 接入由 MJLab 导出的策略。当前注册两套 Reverse Vault：

| 动作 | checkpoint | ONNX | SHA-256 |
| --- | --- | --- | --- |
| `reverse_vault_flat_mjlab` | flat `model_28000.pt` | `model/reverse_vault_flat_28000.onnx` | `d3fb86b8da4263e5c66cc671e00c5f26ebaf3473bf319c24c9ec437584b0de8a` |
| `reverse_vault_terrain_mjlab` | terrain `model_20000.pt` | `model/reverse_vault_terrain_20000.onnx` | `47ea43c940bfc82a77ab4f8696964118026ac9eb3286656a7d915e0247b90e25` |
| `dance1_subject2_mjlab` | LAFAN1 flat `model_29999.pt` | `model/dance1_subject2_29999.onnx` | `2173f598f295483f86c47079822e53b5d344c98305b6508de60308e4c3be0e63` |

这些 ONNX 都由 MJLab `MotionTrackingOnPolicyRunner.save()` 在保存对应 checkpoint
后立即导出，不是按 tensor shape 猜出的兼容模型。原先通过仿真验收的
`reverse_vault_flat_24000.onnx` 继续保留为回归基线，但不再由真机动作引用。

共同模型契约：

- 输入：`obs[1,154]`、`time_step[1,1]`
- 输出：`actions[1,29]` 以及 ONNX 内嵌的 368 帧参考动作
- 目标关节：`default_q + action_scale * actions`
- 控制频率：50 Hz 推理、500 Hz 目标发布
- 启动：不插值，先保持 current-pose 两个 policy 周期（约 40 ms），首个有效参考步为 1
- joint order、default pose、PD、action scale 和 observation metadata 完全一致

## 配置边界

- `config/generic_reverse_vault.yaml`：平地最新模型的真机配置。
- `config/generic_reverse_vault_terrain.yaml`：地形最新模型的真机配置。
- `config/generic_reverse_vault_sim.yaml`：平地仿真数值对齐配置。
- `config/generic_reverse_vault_terrain_sim.yaml`：地形仿真数值对齐配置。
- `config/generic_dance1_subject2.yaml`：LAFAN1 `dance1_subject2` 真机配置。
- `config/generic_dance1_subject2_sim.yaml`：该动作的 C++ MuJoCo lockstep 配置。

真机配置发布到 `rt/user_lowcmd`，保留 hardware finite/PD/stale-state safety 和
supervisor handoff，但依据官方 MJLab deploy 语义关闭公共机械关节位置
target clamp。该 clamp 会改写策略输出的 PD setpoint；官方是直接发布
`processed_actions()`，由执行器/电机力矩约束实际响应。仿真配置发布到
`rt/lowcmd`，使用 LowState tick 锁步并同样关闭会改变训练侧 target 的
joint-limit/max-step clamp；仿真配置不得用于真机。

## 静态检查

```bash
for spec in \
  actions/mjlab/config/generic_reverse_vault.yaml \
  actions/mjlab/config/generic_reverse_vault_terrain.yaml \
  actions/mjlab/config/generic_reverse_vault_sim.yaml \
  actions/mjlab/config/generic_reverse_vault_terrain_sim.yaml \
  actions/mjlab/config/generic_dance1_subject2.yaml \
  actions/mjlab/config/generic_dance1_subject2_sim.yaml; do
  ./build/policy_spec_schema_check --spec "$spec"
  ./build/generic_policy_player --spec "$spec" --check-only
done
```

## 真机遥控流程

启动方式与 BeyondMimic 完全相同：

```bash
ACTION_GO_NET=enP8p1s0 tools/launch/start_safe.sh
```

推荐按键顺序：

1. 程序启动后处于 `SHELL_IDLE`，不会接管机器人。
2. 按一次 `Start` 选择 BFM recovery backend 并 ARM；也可以用 `Left+B`
   选择 UFO backend。ARM 只启用动作和 IMU 监控，不立即启动 stand 或接管控制。
3. 新鲜 IMU 倾角不低于 `57.3°` 持续 `60 ms` 后，safety 自动阻尼并回到
   `SHELL_IDLE`；自动 recovery 默认关闭。再次 ARM 只重新启用监控。
   显式启用自动 recovery 后才在阻尼后运行所选 stand；倾角不高于 `30°`
   稳定 `1200 ms` 后切回 walkrun。恢复或其他动作运行期间，新动作按键会被忽略。
4. 平地动作：先按住 `Up`，再按一下 `X`，随后松开两键。
5. 地形动作：先按住 `Up`，再按一下 `A`，随后松开两键。

```text
Up+X         -> reverse_vault_flat_mjlab
Up+A         -> reverse_vault_terrain_mjlab
Left+Y       -> dance1_subject2_mjlab
Select+Start -> 立即停止当前动作，切官方阻尼并回 SHELL_IDLE
R2+A         -> 立即停止当前动作，切官方 walkrun 并回 SHELL_IDLE
```

组合键在程序中按 bitmask 判断，本身不要求先后顺序；上面的“先方向键、后面键”是为了
现场操作一致而规定的推荐顺序。

两个动作与 BeyondMimic 使用相同的生命周期：player 先在后台启动并校验模型，发布
current-pose hold 并写 `ready_file`；supervisor 等待 ready（默认最多 5 s），再停止
官方控制、切换 UserCtrl，最后写 `start_file`。`start_file` 到达后 player 仍保持当前
姿态两个 50 Hz policy 周期（约 40 ms），不推进 reference、不释放第一帧 target，随后
才开始正式推理。动作结束后保持末帧完成 handoff，以 0.30 秒 overlap 切回官方
`walkrun`，收到 release 后 player 才退出。

## C++ SDK2 MuJoCo 仿真

仿真使用 C++ `unitree_mujoco` bridge 和 C++ `generic_policy_player`，不依赖
`unitree_sdk2py`。

```bash
# flat，默认 nhfield=0
MJLAB_ACTION_VARIANT=flat ./actions/mjlab/unitree_sdk2_mujoco.sh

# terrain，默认 nhfield=1
MJLAB_ACTION_VARIANT=terrain ./actions/mjlab/unitree_sdk2_mujoco.sh

# dance1_subject2（6574 帧，约 131.5 秒）
MJLAB_ACTION_VARIANT=dance1_subject2 ./actions/mjlab/unitree_sdk2_mujoco.sh
```

viewer 启动后保持暂停；按 `Space` 才同时释放物理和策略。
