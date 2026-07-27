# BeyondMimic

BeyondMimic 用于固定时长动作，例如踢球、全身动作、躺倒。

资源目录：

```text
actions/beyondmimic/
```

当前常用动作：

```text
kick_right
hlqk
wbst
lie_down_beyondmimic
```

## 配置示例

```yaml
actions:
  generic_policy:
    kick_right:
      params: config/generic_beyondmimic.yaml
      model: model/kick_right.onnx
      max_frames: 200
      command: "{action_go_root}/build/generic_policy_player --net {net} --cmd-topic rt/user_lowcmd --state-topic rt/lowstate --spec {params} --model {model} --max-frames {max_frames} --ready-file {ready_file} --start-file {start_file} --no-damping-on-stop"
```

说明：

- BeyondMimic 使用 `embedded_mimic_time` 模板，模型输入为 `obs + time_step`，内部 reference 输出只用于观测递推。
- `max_frames` 是动作时长或终止条件。
- 真机 topic 使用 `rt/user_lowcmd`。
- 默认使用 `prestart_before_userctrl`：先启动 player 完成 ONNX/DDS/lowstate 初始化并持续发布当前姿态 hold，再切 UserCtrl，减少切入动作时的 LowCmd 真空期。
- 动作结束后根据 `switch_back_mode` 回到官方状态。

## 单动作测试

```bash
./build/action_go_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --action kick_right
```

等待遥控器确认：

```bash
./build/action_go_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --action kick_right \
  --remote-control
```

## 遥控执行

```bash
./build/action_go_remote_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --remote-map config/remote_actions.yaml
```

按键：

```text
Start        -> ARM
Select+Start -> 任意状态切官方阻尼，回到 SHELL_IDLE
R2+Down      -> kick_right
Left+X       -> hlqk
F1+A         -> wbst
L2+Down      -> lie_down_beyondmimic
```

`lie_down_beyondmimic` 结束后通常切到 `passive`，用于让机器人躺下并回 shell idle。


## 常见 ONNX Runtime 版本问题

如果 player 在创建 ready file 前退出，并看到类似：

```text
libonnxruntime.so.1: version `VERS_1.19.2' not found
```

说明 `generic_policy_player` 编译时链接的是一个 ORT 版本（例如
`~/onnxruntime-trt-1.19.2`），但运行时动态库搜索路径却先加载了另一个版本
（例如仓库内 `thirdparty/onnxruntime-linux-aarch64-1.22.0`）。随后出现的
`prestart ready timeout status=256` 只是进程提前退出后的连带现象。

推荐修复方式是依赖每个 player 自己的 build RPATH，而不是全局指定一个
ORT。`tools/launch/start_safe.sh` / `start_unsafe.sh` / `start_dry_run.sh` 默认会
清理 `LD_LIBRARY_PATH` 中的 ONNX Runtime 路径，避免旧工程的 ORT 抢先加载。

手动检查 split build：

```bash
cd ~/action_go
env -u LD_LIBRARY_PATH ldd build/generic_policy_player | grep onnxruntime
# 应指向: ~/action_go/thirdparty/onnxruntime-linux-aarch64-1.22.0/...
```

如果所有 player 都是用同一个自定义 ORT 编译的，才使用全局 prepend 模式：

```bash
export ACTION_GO_ORT_MODE=prepend
export ACTION_GO_ORT=$HOME/onnxruntime-trt-1.19.2
bash tools/launch/start_safe.sh
```
