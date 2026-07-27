# ActionGo 新读者与新模型适配指南

这份文档面向第一次接触 ActionGo 的读者，目标是说明三件事：

```text
如何启动框架
如何运行已有动作
如何把一个新的 ONNX 全身运动控制 policy 接入统一 runtime
```

核心原则是：优先写 `policy_spec.yaml` 和 action YAML，尽量不要为每个模型新增专用 C++ player。只有现有模板无法表达新模型的输入、输出、传感器或多模型 pipeline 时，才扩展 C++ runtime。

ActionGo 的四类部署形态由集中 `PolicyContract` 描述，代码位置为：

```text
runtime/include/action_go/runtime/policy_contract.h
runtime/src/policy_contract.cpp
```

该 contract 负责把 `runtime.template`、`runtime.variant`、推荐 generic player、输入来源和输出解码方式对应起来。适配新模型时，应先确认它能落入现有 contract；只有需要新增可复用部署形态时，才扩展 contract 和 generic runtime。

可执行的 Unitree G1 约束由 `PolicyProtocolConstraint` 描述，代码位置为：

```text
runtime/include/action_go/runtime/policy_protocol.h
runtime/src/policy_protocol.cpp
```

该 protocol 层集中记录 G1 29DoF、obs/action dim、ONNX I/O、多 session 名称和 TWIST/Humanoid-GPT/OmniXtreme 等固定输入协议。不要把这些字段隐藏在新 player 里；应写入 spec/params，并让 `policy_spec_schema_check` 校验。

## 1. 启动与使用框架

默认构建：

```bash
cmake -S . -B build
cmake --build build --target action_go_all -j
```

默认会生成：

```text
build/action_go_remote_test
build/generic_policy_player
build/generic_motion_tracking_player
build/generic_sensor_policy_player
build/generic_composite_policy_player
```

`generic_composite_policy_player` 用于 OmniXtreme 这类 composite pipeline 策略，默认随
`action_go_all` 构建，并可在 CPU/auto provider 下做 `--check-only`。如果真机需要
TensorRT，可以用 TensorRT-enabled ONNX Runtime 单独重建该 target：

```bash
cmake -S . -B build_omni -DACTION_GO_ONNXRUNTIME_DIR=$HOME/onnxruntime-trt
cmake --build build_omni --target generic_composite_policy_player -j
```

真机前先 dry-run，检查 supervisor、action registry、player 路径和命令展开：

```bash
ACTION_GO_NET=lo tools/launch/start_dry_run.sh
```

在机器人上，把 `lo` 换成 DDS 通信网卡，例如：

```bash
ACTION_GO_NET=enp130s0 tools/launch/start_dry_run.sh
```

dry-run 不切 UserCtrl，也不发布真实 `LowCmd`。它只用于确认启动路径和配置是否能加载。

推荐真机入口：

```bash
tools/launch/start_safe.sh
```

动作按键来自：

```text
config/remote_actions.yaml
```

动作定义来自：

```text
config/motion_actions.yaml
actions/<policy>/<policy>.yaml
```

## 2. 先给新模型分类

适配新模型前，先判断它属于哪类 runtime template。

| 模板 | 适用模型 | 入口 |
|---|---|---|
| `single_actor_state` | 单 actor，输入本体状态、command 或固定 latent prompt，输出 action | `generic_policy_player` |
| `embedded_mimic_time` | 模型内部包含 mimic/reference，运行时输入 `obs + time_step` | `generic_policy_player` |
| `motion_conditioned_tracking` | 输入状态 + motion command / reference cache / tracking target | `generic_motion_tracking_player` |
| `depth_conditioned_actor` | depth encoder + actor | `generic_sensor_policy_player` |
| `object_conditioned_actor` | actor 依赖 object pose | `generic_sensor_policy_player` |
| `composite_motion_policy` | 多模型组合 pipeline，例如 base / residual / FK | `generic_composite_policy_player` |

现有参考配置：

```text
single_actor_state              actions/bfm0/config/generic_bfm_zero.yaml
single_actor_state              actions/unitree_rl_mjlab/config/generic_unitree_rl_mjlab.yaml
embedded_mimic_time             actions/beyondmimic/config/generic_beyondmimic.yaml
embedded_mimic_time             actions/holosoma/config/generic_holosoma.yaml
motion_conditioned_tracking     actions/protomotions/config/generic_protomotions.yaml
motion_conditioned_tracking     actions/twist/config/generic_twist.yaml
depth_conditioned_actor         actions/instinct/config/generic_instinct_parkour.yaml
object_conditioned_actor        actions/steadytray/config/generic_steadytray.yaml
composite_motion_policy         actions/omnixtreme/config/generic_omnixtreme.yaml
```

如果新模型能归入上述模板，就不要新增专用 C++ player。优先新增 spec 和 action 配置。
如果需要新增模板，应同步更新 `PolicyContract`、对应 generic player 的校验逻辑以及 `runtime_hardening_test` 中的 contract 测试。

## 3. 新建策略目录

推荐目录结构：

```text
actions/<new_policy>/
  <new_policy>.yaml
  README.md
  config/
    generic_<new_policy>.yaml
    <PolicyParams>.yaml
  model/
    policy.onnx
```

如果是 motion-conditioned tracking policy，再加入 motion 或 command cache：

```text
actions/<new_policy>/motions/
```

如果是 sensor policy，需要在 README 里说明依赖来源，例如 RealSense、AprilTag、UDP、ZMQ、Redis 或其他感知流。

## 4. 编写 generic spec

一个最小的 state-policy spec 如下：

```yaml
name: generic_<new_policy>
runtime:
  template: single_actor_state
  variant: <new_policy>
  inference_hz: 50
  publish_hz: 500
  onnx_threads: 2
  stats_interval: 50
robot:
  params: <PolicyParams>.yaml
  motor_count: 29
observation:
  builder: <existing_builder_or_variant>
  obs_dim: <N>
model:
  path: ../model/policy.onnx
  input_names: [obs]
  output_names: [actions]
  obs_input: obs
  action_output: actions
action:
  type: target_q
  add_default: true
handoff:
  cmd_topic: rt/user_lowcmd
  state_topic: rt/lowstate
  mode_machine: 5
  damping_on_stop: false
```

注意：新模型不是“只对齐维度”就能部署。必须同时确认训练时和部署时的语义一致：

```text
观测项顺序
history 长度
归一化参数
关节顺序
默认关节位置
kp / kd
action scale
action clip
关节限幅
输出语义
控制频率
传感器或 motion source 的时间同步方式
```

如果这些语义不一致，模型可能通过 ONNX shape 检查，但真机输出仍然是错误甚至危险的。
同时，`robot.motor_count`、`observation.obs_dim`、history 长度、`action.action_dim`、
motion / sensor source 维度也应显式写入 spec 或 policy 参数文件。ActionGo 会用这些字段
校验 ONNX I/O 和运行时缓存，避免把 G1 29DoF、154D mimic obs、TWIST 1155D input 等维度
藏在 player 代码中。

## 5. 注册 action

新增：

```text
actions/<new_policy>/<new_policy>.yaml
```

普通 generic policy 示例：

```yaml
actions:
  generic_policy:
    my_new_action:
      safety_enable: true
      params: config/generic_<new_policy>.yaml
      command: "{action_go_root}/build/generic_policy_player --net {net} --cmd-topic rt/user_lowcmd --state-topic rt/lowstate --spec {params} --ready-file {ready_file} --start-file {start_file} --no-damping-on-stop"
```

如果是 motion-conditioned tracking policy，命令入口换成：

```text
build/generic_motion_tracking_player
```

如果是 sensor policy，命令入口换成：

```text
build/generic_sensor_policy_player
```

如果是 OmniXtreme 类 composite policy，命令入口换成：

```text
build/generic_composite_policy_player
```

然后在 `config/motion_actions.yaml` 中加入 action file：

```yaml
action_files:
  my_new_policy: actions/<new_policy>/<new_policy>.yaml
```

不需要改 `config/remote_actions.yaml`，除非你要给新动作绑定遥控器按键。一个 action 可以先注册、check-only、dry-run，确认没问题后再绑定按键。

## 6. 真机前验证顺序

先构建：

```bash
cmake --build build --target action_go_all -j
```

先跑全量 spec schema checker：

```bash
./build/policy_spec_schema_check --all
```

运行对应的 check-only：

```bash
./build/generic_policy_player \
  --spec actions/<new_policy>/config/generic_<new_policy>.yaml \
  --check-only
```

motion-conditioned / sensor / composite 策略使用对应入口：

```bash
./build/generic_motion_tracking_player \
  --spec actions/<new_policy>/config/generic_<new_policy>.yaml \
  --check-only

./build/generic_sensor_policy_player \
  --spec actions/<new_policy>/config/generic_<new_policy>.yaml \
  --check-only

./build/generic_composite_policy_player \
  --spec actions/<new_policy>/config/generic_<new_policy>.yaml \
  --check-only
```

再检查全量 action registry：

```bash
ACTION_GO_NET=lo tools/launch/start_dry_run.sh
```

需要保留 bringup 或实验诊断日志时，打开 supervisor/payload JSONL：

```bash
ACTION_GO_EVENT_LOG=/tmp/actiongo_supervisor.jsonl \
  ACTION_GO_NET=lo tools/launch/start_dry_run.sh
```

真机前必须确认：

- `--check-only` 能加载模型、参数、input name、output name 和 tensor size。
- `policy_spec_schema_check --all` 能校验 template、variant、G1 protocol dims、ONNX I/O、joint map、kp/kd 和 limits。
- `start_dry_run.sh` 能加载全量 action files。
- policy 只通过 generic player 发布命令，最终 `LowCmd` 会经过 `CommandSafetyFilter`。
- 真实动作启用 `ready_file` / `start_file` handoff。
- RealSense、AprilTag、TensorRT、ZMQ、Redis 等依赖缺失时，必须在切 UserCtrl 前失败。
- `ACTION_GO_EVENT_LOG` 打开后，supervisor/payload JSONL 能记录 handoff、first-command、runtime metrics 和 failure events。

## 7. 什么时候需要改 C++

只有现有模板无法表达新模型时才改 C++。典型情况：

- 需要新的 observation builder。
- 需要新的 sensor source 或 realtime transport。
- 模型是新的多阶段推理图。
- action 输出不是已有的 target joint position / joint target / supported command form。

需要改 C++ 时，优先把可复用逻辑放到：

```text
runtime/include/action_go/runtime/
runtime/src/
```

policy-specific glue 要尽量窄。不要因为接入一个新模型就新增一个长期维护的专用 player。

## 8. 最小验收清单

硬件测试前按这个清单检查：

```text
[ ] 新模型已归入最接近的现有 template。
[ ] spec 路径能从 actions/<new_policy>/config/ 正确解析。
[ ] ONNX input/output names 与导出的模型一致。
[ ] observation 语义与训练时一致。
[ ] joint order 和 action scale 已确认。
[ ] kp/kd 和 joint limits 已配置。
[ ] policy_spec_schema_check --all 通过。
[ ] --check-only 通过。
[ ] start_dry_run.sh 能加载全量动作。
[ ] ACTION_GO_EVENT_LOG dry-run 能生成 supervisor/payload JSONL。
[ ] 没有误改 remote key binding。
[ ] sensor 或 realtime 依赖缺失时能提前失败。
```
