# SONIC WBC

本包默认把 NVIDIA GR00T-WholeBodyControl 的 SONIC v1.1 encoder、decoder 和 V2 locomotion
planner 原生接入 ActionGo。v1.1 使用 robot-heading-normalized target orientation，并在训练时
加入 wrist-pose augmentation。PICO 采集、标定和 SMPL 解算继续由原项目的
`pico_manager_thread_server.py` 完成；ActionGo 会按需托管该进程，并消费 manager 的
`command`、`planner` 与 protocol-v3 SMPL `pose` topic。VLA PolicyServer、planner frozen
upper body、VR-3PT 和 Dex3 双手控制不在当前支持边界内。

模型 contract：

```text
encoder: obs_dict[1,1751] -> encoded_tokens[1,64]
decoder: obs_dict[1,994]  -> action[1,29]
planner: context[1,4,36] + mode/velocity/direction -> mujoco_qpos[1,64,36]
target:  default_q + remap(action) * per_joint_action_scale
```

上游 original release 的 1762D encoder、decoder、observation config 和
`config/generic_sonic.yaml` 继续保留用于回退；稳定动作名默认使用
`config/generic_sonic_v1_1.yaml`。两版 TensorRT cache 必须分目录保存。

离线检查：

```bash
./build/generic_motion_tracking_player \
  --spec actions/sonic/config/generic_sonic_v1_1.yaml \
  --input offline-csv \
  --trajectory actions/sonic/motions/squat_001__A359 \
  --ep cpu --check-only
```

PICO planner 检查（会实际加载并执行第三个 ONNX session；当前仅允许本机 endpoint）：

```bash
./build/generic_motion_tracking_player \
  --spec actions/sonic/config/generic_sonic_v1_1.yaml \
  --input pico-zmq --zmq-connect \
  --zmq-endpoint tcp://127.0.0.1:5556 --zmq-topic pose \
  --ep cpu --check-only
```

普通实时 reference 检查（兼容 G1 v1 与 PICO SMPL v3，不启用 planner 状态机）：

```bash
./build/generic_motion_tracking_player \
  --spec actions/sonic/config/generic_sonic_v1_1.yaml \
  --input realtime-zmq --zmq-connect \
  --zmq-endpoint tcp://127.0.0.1:5556 --zmq-topic pose \
  --require-realtime --input-timeout-sec 3.0 --stale-sec 0.1 \
  --ep cpu --check-only
```

真机使用 TensorRT 时在目标板载机器从 ONNX 构建 engine；不要复制仓库外已有的
`.trt` cache。默认保持 FP32，只有与官方 replay 达到误差门槛后才使用 `--trt-fp16`。
可让 motion player 单独链接板载 GPU ORT，而不改变其他 player 的运行库：

```bash
cmake -S . -B build \
  -DACTION_GO_MOTION_ONNXRUNTIME_DIR=$HOME/onnxruntime-trt
cmake --build build --parallel --target generic_motion_tracking_player
```

实时包使用官方 1280-byte JSON header。G1 v1 需要 `joint_pos[N,29]`、
`joint_vel[N,29]`、`body_quat[_w][N,4]`；PICO v3 需要原 sender 直接发布的
`smpl_joints[N,24,3]`、`smpl_pose[N,21,3]`、`body_quat_w[N,4]`、
`joint_pos/joint_vel[N,29]` 和 `frame_index[N]`。浮点字段支持 f32/f64，附带的 bool/u8
元数据会按 header 安全跳过。frame gap、sender 重启或 stride 变化触发 reference window
reset。无实时 sender 时可去掉 `--require-realtime` 做纯 I/O/shape 检查，此时使用 identity
G1 reference，不代表 PICO 链路已经通过。

语义来源与映射：

| field / semantic | 上游来源 | 提取值 | ActionGo 字段 / 实现 | 置信度 |
|---|---|---|---|---|
| encoder / decoder I/O | `sonic_v1_1/model_{encoder,decoder}.onnx` | 1751→64，994→29 | `model.sessions.*` | SHA256 与 HF LFS etag 一致 |
| encoder layout | `policy/sonic_v1_1/observation_config.yaml` | 删除 11 个 root-z 槽；G1 10×step5，SMPL/wrist 10×step1 | `config/observation_config_sonic_v1_1.yaml`、`SonicObservationBuilder` | 上游配置核对 |
| history | `GatherHis*` + `StateLogger::GetLatest(..., false)` | 10 帧 oldest→newest | `observation.history_len` | 上游代码核对 |
| quaternion / anchor | `GatherMotionAnchorOrientationMutiFrame(..., orientation_mode=1)` | 初始 heading 对齐后，仅用当前 robot yaw 的逆旋转归一化；wxyz、矩阵前两列按行展开 | `SonicObservationBuilder` v1.1 layout | 单元测试覆盖 |
| joint remap / decode | 官方 G1 deploy mapping 与 policy params | 29DoF，default + remap(action) × scale | `SONIC_V1_1.yaml`、`sonic_decode_target` | 单元测试覆盖 |
| realtime protocol | `zmq_packed_message_subscriber.hpp`、`streamed_motion_merger.hpp`、`zmq_planner_sender.py` | G1 v1 / SMPL v2-v3、planner/command v1、1280-byte header | `parse_sonic_zmq_packet`、`parse_sonic_{planner,command}_packet` | parser 单测覆盖 |
| planner I/O 与时序 | `localmotion_kplanner.hpp`、`localmotion_kplanner_tensorrt.hpp` | 4×36 context、30→50 Hz、step2 lookahead、8-frame blend | `SonicPlanner` | helper 单测覆盖；真模型 check-only |

尚未完成 v1.1 官方 deploy replay 的 encoder token / decoder action / target q `1e-4` 数值验收，
也未完成板载 TensorRT cold/warm cache、20 ms deadline、PICO 实流 MuJoCo 和真机吊架实验；这些均为
接管真机前的 blocker，不能由 `--check-only` 替代。

## 原版 PICO Streamer

ActionGo 托管上游 PICO manager，但不复制 Pico SDK、标定或人体解算。首次
部署仍需在 Jetson 上完成一次上游 PICO 环境安装：

```bash
cd /path/to/GR00T-WholeBodyControl
SKIP_SIM_AND_UNITREE=1 bash install_scripts/install_pico.sh
```

之后无需切换到 GR00t 目录或另开终端。`sonic_pico_teleop` 会自动查找
`$HOME/resource/GR00T-WholeBodyControl` 并用其中的 `.venv_teleop` 启动 manager；非默认
路径只需在启动 ActionGo 前设置：

```bash
export ACTION_GO_GR00T_ROOT=/path/to/GR00T-WholeBodyControl
```

默认使用 CPU 且不创建可视化窗口，适合 Jetson。诊断时可设置
`ACTION_GO_SONIC_PICO_VIS=1`，已有手动 streamer 时可设置
`ACTION_GO_SONIC_PICO_AUTOSTART=0`。`ACTION_GO_SONIC_PICO_CUDA=1` 通常没有必要。
ActionGo 只清理由自己启动的 manager；`R2+A` 会结束 SONIC action 及其托管 manager 并
返回 `walkrun`。

打开 PICO 的 XRoboToolkit 应用并连接 Jetson 后，站到上游要求的标定姿态，按
**A+B+X+Y**。manager 会标定并从 OFF 进入 PLANNER，ActionGo 收到首个合法 planner 包、
start command、fresh LowState 且三个模型完成 warmup 后才进入 standby/ready；supervisor
随后才 handoff。按 **A+X** 可在 PLANNER 与 POSE 间切换，切换时 ActionGo 重置 reference
heading/history，避免沿用前一种输入语义。

PLANNER 中左摇杆控制移动方向和幅度，右摇杆左右控制 facing yaw；**A+B** 向后切换模式，
**X+Y** 向前切换模式。上游当前 PICO manager 可选 `0..19`：idle、slow walk、walk、run、
squat、双膝跪、单膝跪、俯卧、crawl、idle/walk boxing、左右/随机直拳、肘部爬行、左右勾拳、
前跳、stealth walk 和 injured walk。模型 contract 保留 `0..26`，但 manager 当前不暴露
`20..26`。左摇杆回中时 manager 会发送 idle。

不要进入 **B+Y** frozen upper body 或按左摇杆进入 VR-3PT；这些 planner 扩展以及 Dex3
字段会被 ActionGo 明确拒绝。reference 超过 100 ms 后保持最后安全目标，累计 500 ms 后
异常退出。PICO 的 **A+B+X+Y** 可正常 stop；Unitree 遥控器 **R2+A** 的优先级更高，在
等待 PICO、standby、startup ramp 和 tracking 任意阶段都会退出 SONIC 并返回 `walkrun`。

`planner_sonic.onnx` 约 739 MB，只在 `pico-zmq` 路径加载；离线和普通 `realtime-zmq`
不会承担这部分算力/显存。上游 TensorRT 在 Jetson Orin 上记录约 1--2 ms/次、按模式以
1--10 Hz 重规划；首次 TensorRT engine 构建会明显更慢，并额外占用约 747 MB cache 与运行
缓冲。当前仍需完成目标 Jetson 上 warm-cache deadline 和显存验收后才能视为真机已验证。

### MuJoCo

sender 与 ActionGo 在同一台机器时：

```bash
ACTION_GO_SONIC_EP=cpu /usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --unitree-mujoco-root /path/to/unitree_mujoco \
  --unitree-sdk2py-root /path/to/unitree_sdk2_python \
  --interface lo --domain-id 0 --no-use-joystick \
  --auto-action sonic_pico_teleop
```

没有 Pico 时，可在另一个终端发布 synthetic planner/command v1 做完整 planner 状态机测试：

```bash
/usr/bin/python3 tools/remote/sonic_pico_zmq_replay.py \
  --mode planner --planner-mode 2 --movement 0.2 0 0 --duration-sec 30
```

要测试 POSE 协议则改用 `--mode pose`。synthetic 输入只验证协议、planner 推理和
DDS/MuJoCo 闭环，不用于真机，也不代表运动质量。

### 真机

当前 ZMQ reference endpoint 仅允许 loopback/本机 IPC。PICO sender 在另一台电脑时的
远程 reference 传输需要后续认证协议，ActionGo 会在打开非本机 endpoint 时直接拒绝，
因此不要把下面旧版远程示例当作当前公开版本的支持路径。

本机 sender 示例：

```bash
export ACTION_GO_SONIC_ZMQ_ENDPOINT=tcp://127.0.0.1:5556
export ACTION_GO_SONIC_EP=tensorrt
ACTION_GO_NET=lo tools/launch/start_safe.sh
```

确认 sender 已进入 POSE 且持续出流后，Unitree 遥控器先按 `Start` 进入运行态，再按
`F2+Y` 启动 `sonic_pico_teleop`。`R2+A` 停止并返回官方 `walkrun`，`Select+Start`
进入官方 damping。player 在收到首个合法 v3 reference、fresh LowState、模型 warmup
全部完成前不会创建 ready file，也不会转移控制权。ready 后仍持续检查 PICO freshness；
handoff 前断流会撤销 ready，3 秒 startup ramp 内断流会中止接管并回到 `walkrun`。
`R2+A` 在等待 PICO、standby、startup ramp 和正常 tracking 任一阶段都可以取消动作并
回到官方 `walkrun`。

## 无 Pico 测试

当前默认 checkpoint 没有速度 command 输入，因此不能作为 `vx/vy/yaw` 连续 locomotion
policy 测试。没有 Pico 时可以完整测试离线序列跟踪：ActionGo 从当前姿态用 3 秒过渡到
SONIC default pose，随后跟踪官方 `squat_001__A359` reference，序列结束后切回官方
`walkrun`。

推荐先做无 LowCmd 检查：

```bash
./build/generic_motion_tracking_player \
  --spec actions/sonic/config/generic_sonic_v1_1.yaml \
  --input offline-csv \
  --trajectory actions/sonic/motions/squat_001__A359 \
  --ep cpu --duration-sec 1 --dry-run --offline-dry-run-state
```

真机 supervisor 启动后，先确认机器人在官方 walkrun 中稳定，再按 `Start` 启用外壳，按 `F2+X` 启动
`sonic_offline_squat`。`R2+A` 随时中止当前动作并返回官方 `walkrun`，`Select+Start`
进入官方 damping；`F2+Y` 用于上述 `sonic_pico_teleop`。
