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

真机建议使用强制 RealSense 检查的构建方式：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DACTION_GO_ONNXRUNTIME_DIR="$PWD/thirdparty/onnxruntime-linux-aarch64-1.22.0" \
  -DACTION_GO_REQUIRE_REALSENSE=ON \
  -DACTION_GO_REALSENSE_ROOT="$HOME/librealsense/install"

cmake --build build --parallel "$(nproc)" --target \
  generic_sensor_policy_player action_go_remote_test
```

注意第二步必须是 `cmake --build build ...`，不是在源码根目录执行
`make`。如果 CMake 已显示 `librealsense2 enabled`，但随后的 `make`
报 `No makefile found`，新二进制并没有生成，启动时仍会用到旧的无
RealSense player。正确二进制启动时会显示：

```text
[generic_sensor_policy_player] build features: realsense=enabled
```

如果 CMake 提示：

```text
OpenCV not found; generic_sensor_policy_player target is disabled
```

说明本机不会编译该 player。

### Jetson Orin 本地安装 librealsense

机器人端是 Jetson Orin（ARM64/JetPack）。不要使用 x86_64 预编译包，也不要求通过
`apt install` 安装 `librealsense2-dev`。推荐从 GitHub 拉取固定版本，在用户目录完成
编译和本地安装：

```bash
export PATH=/usr/local/cuda/bin:$PATH

cd "$HOME"
git clone --depth 1 --branch v2.57.5 \
  https://github.com/IntelRealSense/librealsense.git

cmake -S "$HOME/librealsense" \
  -B "$HOME/librealsense/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/librealsense/install" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TOOLS=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_WITH_DDS=OFF \
  -DBUILD_WITH_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DFORCE_RSUSB_BACKEND=ON

cmake --build "$HOME/librealsense/build" -j4
cmake --install "$HOME/librealsense/build"
```

Orin 的 CUDA 架构是 SM 8.7。`FORCE_RSUSB_BACKEND=ON` 使用用户态 USB backend，
避免修改 Jetson 内核的 UVC 驱动。如果 JetPack 没有提供
`/usr/local/cuda/bin/nvcc`，可暂时改为 `BUILD_WITH_CUDA=OFF`；相机仍能运行，但
librealsense 的相关处理会走 CPU。

安装 USB 权限规则后重新插拔相机：

```bash
sudo install -m 644 \
  "$HOME/librealsense/config/99-realsense-libusb.rules" \
  /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

重新配置 ActionGo，显式指定本地 librealsense CMake package：

```bash
cd "$HOME/action_go"

cmake -S . -B build \
  -Drealsense2_DIR="$HOME/librealsense/install/lib/cmake/realsense2"
cmake --build build --target generic_sensor_policy_player -j4
```

CMake 输出必须包含：

```text
generic_sensor_policy_player: librealsense2 enabled
```

正常情况下，ActionGo 的构建 RPATH 或当前运行环境可以找到该动态库，不需要每次启动
都手动设置 `LD_LIBRARY_PATH`。先检查链接结果：

```bash
ldd build/generic_sensor_policy_player | grep realsense
```

只有运行时出现 `librealsense2.so: cannot open shared object file` 或 `ldd` 显示
`not found` 时，才临时运行：

```bash
export LD_LIBRARY_PATH="$HOME/librealsense/install/lib:${LD_LIBRARY_PATH:-}"
```

需要永久生效时，将上面这行 `export` 加入 `~/.bashrc`，然后执行：

```bash
source ~/.bashrc
```

用 `lsusb -t` 确认 RealSense 所在 USB 链路显示 `5000M`；如果是 `480M`，深度流
可能出现明显延迟或掉帧。

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

官方 [`instinct_onboard/scripts/g1_parkour.py`](https://github.com/project-instinct/instinct_onboard/blob/b6965a4202d8dceaf163dd95edb7742ed5d65e38/scripts/g1_parkour.py#L182)
每 `0.02s` 调用一次 `main_loop_callback`，在同一次 callback 中执行
`ParkourAgent.step()` 并提交电机目标（旧版 `send_action()`，新版
`send_target_joint_state()`）；`UnitreeRosNode._publish_motor_cmd()` 随即发布 LowCmd。
因此官方 parkour 的推理和正常命令发布目标频率都是 `50 Hz`，没有独立的
`500 Hz` 命令重发线程；RealSense 的 `60 Hz` 是相机采集频率。

ActionGo 保持 `inference_hz=50`、`publish_hz=50`、`target_interp_sec=0` 和
`target_rate_limit=0`，沿用远端已验证的独立推理／发布线程，发布最新策略目标。
独立线程的调度方式与官方同一 callback 不同；频率一致不代表两者的实际时序已经等价验证。
不要在 parkour 上额外做 target-rate-limit，否则策略会
以为自己已经发出了大动作，但机器人目标被限速滞后，容易越修越摆。

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
