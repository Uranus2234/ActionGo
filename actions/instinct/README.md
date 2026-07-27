# Instinct Parkour

Instinct Parkour 是带深度输入的策略。默认动作：

```text
g1_parkour
```

模型目录配置在：

```text
config/motion_actions.yaml
```

默认路径：

```text
actions/instinct
```

其中：

```text
actions/instinct/model/              # parkour 模型
actions/instinct/config/             # parkour 配置
actions/instinct/stand_onboard/model # stand 模型
actions/instinct/stand_onboard/config
```

## 依赖

`generic_sensor_policy_player` 需要 OpenCV。RealSense 真机深度输入需要 `librealsense2`。
真机执行默认还会订阅 `rt/secondary_imu`，和官方 `instinct_onboard` 一样优先使用 torso IMU；如果 1 秒内没有收到 torso IMU，supervisor 不会切入 UserCtrl。

如果 CMake 提示：

```text
OpenCV not found; generic_sensor_policy_player target is disabled
```

说明本机不会编译该 player。

## 模型检查

```bash
./build/generic_sensor_policy_player --spec actions/instinct/config/generic_instinct_parkour.yaml \
  --logdir actions/instinct \
  --check-only
```

## prestart 与深度相机启动保护

`config/motion_actions.yaml` 给所有 `instinct_parkour` 动作启用了 `prestart_before_userctrl`：

```text
启动 player -> 初始化 RealSense/ONNX/DDS -> 发布当前姿态 hold -> ready -> 再切 UserCtrl
```

所以按 `Up+Y` 后，RealSense 深度相机、torso IMU 和 ONNX 会在官方 walkrun 仍然接管时先初始化。检查失败时不会进入 UserCtrl，不会启动策略，并会请求保持/切回官方 `walkrun`；检查成功后才切 UserCtrl，减少软下去的真空期。

进入 UserCtrl 后不再用手写的 `startup-step-size` 拉姿态，而是先执行 official ColdStartAgent-style 过渡到 `default_joint_pos`，然后默认直接进入 parkour 模型。默认流程：

```text
Up+Y -> 启动 Instinct player
     -> official ColdStartAgent-style 过渡到 default_joint_pos
     -> 直接进入 parkour 模型
```

这里的 cold-start 使用官方默认 `startup_step_size=0.2`、`kpkd_factor=2.0` 语义，并在 default pose 保持 `0.1s`，让速度先降下来。`actions/instinct/stand_onboard` 仍保留为备用模型；如果以后需要恢复官方 `cold_start -> stand -> parkour` 流程，可在动作命令里重新加入 `--standdir {action_dir}/stand_onboard --stand-until-l1`。

Instinct 的低层发送也按官方方式执行：`publish_hz=50`、`target_interp_sec=0`、`target_rate_limit=0`。不要在 parkour 上额外做 target-rate-limit，否则策略会以为自己已经发出了大动作，但机器人目标被限速滞后，容易越修越摆。

## 关节顺序

`actions/instinct/model` 和 `actions/instinct/stand_onboard/model` 是官方提供的 InstinctLab onboard checkpoint，不是新 InstinctMJ/MuJoCo 导出的 checkpoint。它们的观测、动作、`default_joint_pos`、`action_scale`、KP/KD 都按旧版 `instinct_onboard` 的 shoulder-first 顺序解释：

```text
left_shoulder_pitch, right_shoulder_pitch, waist_pitch,
left_shoulder_roll,  right_shoulder_roll,  waist_roll,
left_shoulder_yaw,   right_shoulder_yaw,   waist_yaw,
left_elbow, right_elbow,
left_hip_pitch, right_hip_pitch,
...
```

因此 `runtime/src/instinct_parkour_runtime.cpp` 里的 `kSimJointNames/kJointMap/kJointSigns/kJointHigh/kJointLow/kTorqueLimits`，以及备用 Python player `tools/sim/instinct_parkour_player.py` 里的同名常量，都必须保持 shoulder-first。误改成 InstinctMJ 的 waist-first 顺序后，stand 常见表现是“刚开始站住约一秒，然后手、腿、身体开始乱晃/发散”。

手动只检查相机：

```bash
./build/generic_sensor_policy_player --spec actions/instinct/config/generic_instinct_parkour.yaml \
  --depth-camera-check-only \
  --require-depth-camera \
  --depth-camera-timeout-sec 1.0 \
  --min-depth-camera-frames 1
```

## 真机 dry-run

只跑 RealSense + DDS 输入 + ONNX，不发 LowCmd：

```bash
./build/generic_sensor_policy_player --spec actions/instinct/config/generic_instinct_parkour.yaml \
  --net enP8p1s0 \
  --logdir actions/instinct \
  --dds-dry-run \
  --depth-source realsense \
  --require-torso-imu \
  --duration-sec 5 \
  --stats-interval 25
```

固定速度命令为 0：

```bash
./build/generic_sensor_policy_player --spec actions/instinct/config/generic_instinct_parkour.yaml \
  --net enP8p1s0 \
  --logdir actions/instinct \
  --dds-dry-run \
  --depth-source realsense \
  --require-torso-imu \
  --command-source zero \
  --duration-sec 5 \
  --stats-interval 25
```

如果 dry-run 的诊断里出现类似：

```text
qerr_default=0.741@right_elbow_joint
target_err=0.989@right_elbow_joint
```

表示当前官方 walkrun 姿态离 Instinct 训练默认姿态较远；真机执行时会先执行 official cold-start，再直接进入 parkour。

## 真机执行

推荐通过 supervisor：

```bash
./build/action_go_remote_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --remote-map config/remote_actions.yaml
```

按键：

```text
Start     -> ARM
Up+Y      -> g1_parkour
```
