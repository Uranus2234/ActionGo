# SONIC WBC

本包把 NVIDIA GR00T-WholeBodyControl 中的默认 SONIC encoder/decoder 原生接入
ActionGo。板载只运行 `model_encoder.onnx` 和 `model_decoder.onnx`，不包含 VLA
PolicyServer、kinematic planner 或 Dex3 双手控制。PICO 采集、标定和 SMPL 解算继续由
原项目的 `pico_manager_thread_server.py` 完成；ActionGo 直接消费其 protocol-v3 SMPL
`pose` 输出并执行 SONIC SMPL encoder mode。

模型 contract：

```text
encoder: obs_dict[1,1762] -> encoded_tokens[1,64]
decoder: obs_dict[1,994]  -> action[1,29]
target:  default_q + remap(action) * per_joint_action_scale
```

离线检查：

```bash
./build/generic_motion_tracking_player \
  --spec actions/sonic/config/generic_sonic.yaml \
  --input offline-csv \
  --trajectory actions/sonic/motions/squat_001__A359 \
  --ep cpu --check-only
```

实时 ZMQ 检查（兼容 G1 v1 与原 PICO streamer 的 SMPL v3；当前仅允许本机 endpoint）：

```bash
./build/generic_motion_tracking_player \
  --spec actions/sonic/config/generic_sonic.yaml \
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
| encoder / decoder I/O | `policy/release/model_{encoder,decoder}.onnx` | 1762→64，994→29 | `model.sessions.*` | 已校验 SHA256 与官方文件一致 |
| encoder layout | `policy/release/observation_config.yaml` | G1 mode 10×step5；SMPL mode 10×step1 | `config/observation_config.yaml`、`SonicObservationBuilder` | 上游代码核对 |
| history | `GatherHis*` + `StateLogger::GetLatest(..., false)` | 10 帧 oldest→newest | `observation.history_len` | 上游代码核对 |
| quaternion / anchor | `motion_data_reader.hpp`、`GatherMotionAnchorOrientationMutiFrame` | wxyz、矩阵前两列按行展开 | `SonicObservationBuilder` | 单元测试覆盖 |
| joint remap / decode | 官方 G1 deploy mapping 与 policy params | 29DoF，default + remap(action) × scale | `SONIC.yaml`、`sonic_decode_target` | 单元测试覆盖 |
| realtime protocol | `zmq_packed_message_subscriber.hpp`、`streamed_motion_merger.hpp`、`zmq_planner_sender.py` | G1 v1 / SMPL v2-v3、1280-byte header、滑动窗口 | `parse_sonic_zmq_packet` | v1/v3 parser 测试覆盖 |

尚未完成官方 deploy replay 的 encoder token / decoder action / target q `1e-4` 数值验收，
也未完成板载 TensorRT cold/warm cache、20 ms deadline、PICO 实流 MuJoCo 和真机吊架实验；这些均为
接管真机前的 blocker，不能由 `--check-only` 替代。

## 原版 PICO Streamer

ActionGo 使用上游 streaming-only POSE 路径，不复制 Pico SDK、标定或人体解算。先在
`GR00T-WholeBodyControl` 根目录启动原 sender：

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --vis_vr3pt --vis_smpl
```

按原项目顺序站到标定姿态，按 PICO **A+B+X+Y** 完成标定，再按 **A+X** 进入 POSE。
此时 `pose` topic 持续发布 v3 SMPL window。ActionGo 不处理 planner/VR_3PT/Dex3：
planner mode 只会使 pose 流停止；由于 ActionGo 保留更严格的 stale safety，停流 100 ms
后保持最后安全目标，累计 500 ms 后退出当前 action。

### MuJoCo

sender 与 ActionGo 在同一台机器时：

```bash
ACTION_GO_SONIC_EP=cpu /usr/bin/python3 tools/sim/action_go_sim_remote.py \
  --unitree-mujoco-root /path/to/unitree_mujoco \
  --unitree-sdk2py-root /path/to/unitree_sdk2_python \
  --interface lo --domain-id 0 --no-use-joystick \
  --auto-action sonic_pico_teleop
```

没有 Pico 时，可在另一个终端发布同格式 synthetic v3 流做链路测试：

```bash
/usr/bin/python3 tools/remote/sonic_pico_zmq_replay.py --duration-sec 30
```

synthetic reference 只验证协议、推理和 DDS/MuJoCo 闭环，不用于真机，也不代表运动质量。

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
  --spec actions/sonic/config/generic_sonic.yaml \
  --input offline-csv \
  --trajectory actions/sonic/motions/squat_001__A359 \
  --ep cpu --duration-sec 1 --dry-run --offline-dry-run-state
```

真机 supervisor 启动后，先按 `Start` 进入运行态并确认 stand 稳定，再按 `F2+X` 启动
`sonic_offline_squat`。`R2+A` 随时中止当前动作并返回官方 `walkrun`，`Select+Start`
进入官方 damping；`F2+Y` 用于上述 `sonic_pico_teleop`。
