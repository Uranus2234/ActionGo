# PHP 深度感知策略使用指南

本包在 ActionGo 中运行 [PHP Parkour](https://github.com/amazon-far/php_parkour)
的深度学生策略，适用于 Unitree G1 29DoF。策略由深度编码器和动作 ONNX 两个模型组成，
通过 `generic_sensor_policy_player` 推理；动作名为 `php_locomotion`，遥控器绑定为 `Up+B`。

推荐顺序：安装模型与依赖 → 仿真 → 真机只读检查 → 有外部保护的运动测试。
目前真机阶梯尚未完成验收，切入策略时偶发失去支撑仍在排查；只读检查通过不代表运动测试通过。

除特别说明外，以下命令均在 ActionGo 仓库根目录的 Linux Bash 终端执行。

## 1. 安装依赖和模型

依赖包括 C++17、CMake、Unitree SDK2、ONNX Runtime、yaml-cpp、Eigen、OpenCV、
libzmq，以及真机深度采集所需的 librealsense2。版本和安装要求见
[依赖说明](../../DEPENDENCIES.md)；Jetson 的 RealSense 安装可参考
[Instinct 使用说明](../instinct/README.md)。

将官方 PHP 仓库放在 ActionGo 同级目录，使用其下载器获取配套模型：

```bash
git clone --recurse-submodules https://github.com/amazon-far/php_parkour.git ../php_parkour
python3 ../php_parkour/scripts/download_assets.py student \
  --destination "$PWD/actions/php/model"
```

已有 PHP 仓库时跳过 clone。模型目录必须包含以下两个文件，替换模型时应使用配套的一对：

```text
actions/php/model/depth_backbone.onnx
actions/php/model/student.onnx
```

## 2. 编译

Jetson/aarch64 示例：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DACTION_GO_ONNXRUNTIME_DIR="$PWD/thirdparty/onnxruntime-linux-aarch64-1.22.0" \
  -DACTION_GO_REQUIRE_REALSENSE=ON \
  -DACTION_GO_REALSENSE_ROOT="$HOME/librealsense/install"
cmake --build build --parallel 2 --target \
  generic_sensor_policy_player generic_policy_player action_go_remote_test policy_spec_schema_check
```

在 x86_64 本机编译时，将 ONNX Runtime 目录改为
`thirdparty/onnxruntime-linux-x64-1.22.0`。RealSense 安装在其他位置时修改
`ACTION_GO_REALSENSE_ROOT`；仅做仿真且未安装相机依赖时，可关闭
`ACTION_GO_REQUIRE_REALSENSE` 并省略该安装路径。

真机运行的 player 必须显示 `realsense=enabled`。基本检查：

```bash
source tools/launch/ort_env.sh
action_go_configure_ort_env "$PWD" "php_check"
./build/policy_spec_schema_check --spec actions/php/config/generic_php.yaml
./build/generic_sensor_policy_player --spec actions/php/config/generic_php.yaml --check-only
```

## 3. 先在 MuJoCo 中验证

ActionGo 仿真使用官方 Holosoma 的 G1 模型、阶梯和深度插件，动作推理由本包的 C++ player 执行。
安装伴随脚本、Python 环境及运行命令见 [ActionGo 仿真安装指南](integration/README.md)。
官方 Python 推理流程另见 [PHP sim2sim 指南](https://github.com/amazon-far/php_parkour/blob/main/wbt_training/DEPLOY.md)。

完成仿真安装后，在同级 `php_parkour` 根目录运行：

```bash
bash run_actiongo_sim.sh --actiongo-root ../action_go
```

该伴随入口自动切入 PHP、前进、站稳并切回仿真 walkrun。仿真 walkrun 使用
`unitree_rl_mjlab` 策略，因此还需要它的 ONNX 模型。
完整阶梯验收要求实际通过全部 12 个障碍、无跌倒、站稳并完成切回；绕行到终点不算通过。
仿真结果不能替代真机验收。

## 4. 相机和真机只读检查

PHP 相机应按官方 D435i 配置安装，相对 `torso_link` 的位置为
`(0.01, 0.01, 0.44) m`，RPY 为 `(1, 27, 1)°`，即主要向下俯约 27°。
沿用其他策略的安装时必须重新核对；内参投影不能补偿错误的安装位置或角度。

配置请求 `848×480` 深度、约 10Hz 策略采样；设备不支持该分辨率的 10Hz profile 时，
会尝试 30Hz 采集后降采样。策略还需要新鲜的 LowState 和 `rt/secondary_imu` 躯干 IMU。

关闭其他占用 RealSense 的程序，然后在机器人 ActionGo 根目录执行：

```bash
ACTION_GO_NET=eth0 bash tools/launch/check_php.sh
```

将 `eth0` 替换为实际承载机器人 DDS 的网卡，不能仅按 SSH 连接所用的网卡选择。
脚本依次检查 PHP 配置、模型、相机，并执行 10 秒真实传感器输入推理。
整个检查不切换 UserCtrl、不发布 LowCmd，但会占用相机，不能与策略或独立相机监控同时运行。

日志默认写入 `outputs/php_hardware_check/`。自定义构建目录和输出目录：

```bash
ACTION_GO_NET=eth0 PHP_BUILD_DIR=build-php-dryrun \
  PHP_LOG_DIR=outputs/php_check bash tools/launch/check_php.sh
```

## 5. 启动真机与遥控操作

机器人先在官方 WALKRUN 中稳定站立，摇杆回中，并准备外部支撑保护。启动外壳：

```bash
mkdir -p outputs
ACTION_GO_NET=eth0 ACTION_GO_EVENT_LOG="$PWD/outputs/php_supervisor.jsonl" \
  bash tools/launch/start_safe.sh
```

按以下顺序操作：

1. 按一次 `Start` 启用外壳。
2. 按一次 `Up+B` 请求启动 PHP。
3. 等待 `startup transition done` 和 `start real DDS`，再给方向指令。

外壳先预启动策略，等待相机、状态和模型就绪，再切换 UserCtrl 并发送 start 信号。
请通过外壳启动，以保留 ready/start 握手和切回逻辑。

| 操作 | 功能 |
|---|---|
| 左摇杆前推 | 低速前进，command 1；不需要按 R1 |
| 按住 R1 并前推左摇杆 | 高速前进，command 6；松开 R1 回到低速 |
| 左摇杆左右或斜向 | 选择离散方向 |
| 右摇杆左右 | 选择离散左右方向 |
| 两个摇杆都回中 | 站立，command 0 |
| R2+A | 停止 PHP、切回官方 WALKRUN，外壳返回空闲；再次启动先按 Start |
| Select+Start | 切官方阻尼；使用前确保有外部支撑 |

PHP 使用离散方向和速度档，摇杆幅度不连续调节速度，右摇杆也不是连续偏航速度。
以 `policy_command=编号(方向/档位)` 日志确认实际输入。

## 6. 策略运行时实时查看深度和角度

在本机 ActionGo 根目录运行，下列 IP 和路径为示例，替换为自己的机器人配置：

```bash
PHP_ROBOT_HOST=unitree@192.168.3.162 \
  PHP_ROBOT_ROOT=/home/unitree/action_go \
  PHP_ROBOT_NET=eth0 \
  bash tools/remote/monitor_php_depth.sh
```

本机 Python 需要 NumPy、Pillow 和 Tk，机器人 Python 需要 NumPy。
可用 `PHP_MONITOR_PYTHON=/path/to/python` 指定本机解释器。

窗口显示策略实际使用的 `58×87` 处理后深度图和结合 IMU 的地面拟合角度，最高 10Hz。
它通过旁路读取运行中的策略，不另开相机、不启动策略、不发布 LowCmd。
可先打开窗口等待策略帧；关闭窗口不会停止策略。同一机器人只打开一个旁路窗口。

角度估算要求拟合区域内可见水平地面。安装角应在机器人静止站立时核对，
箱面、遮挡、运动和蹲姿都可能影响估算结果。

## 7. 常见问题与诊断

| 现象 | 检查方法 |
|---|---|
| `Device or resource busy` | 关闭独立深度采集、相机角度采集等相机占用程序；运行中的策略应使用上述旁路监控 |
| `Couldn't resolve requests` | 确认深度 profile 和 librealsense；使用支持 10Hz/30Hz 自动选择的当前 player |
| prestart ready timeout | 查找超时前的 player ERROR；先执行只读检查，确认模型、深度和躯干 IMU 就绪 |
| 深度窗口没有画面 | 确认 PHP player 正在运行且包含旁路发布功能，核对 SSH 地址，关闭重复旁路窗口 |
| 前推但不走 | 检查 `policy_command` 是否从 stand 变为前进，以及是否已经完成启动交接 |
| 遇障碍没有预期抬腿 | 查看策略实际深度，核对安装角、位置、遮挡和障碍尺寸；用相同场景仿真对照 |
| 刚切入就失去支撑 | 停止运动测试，保留切入前后完整日志，并采集下面的只读命令诊断 |

在机器人另一个终端记录命令和关节状态：

```bash
mkdir -p outputs/php_startup_watch
python3 tools/sim/diagnose_php_motion.py --net eth0 --duration 60 \
  --output outputs/php_startup_watch/lowcmd.jsonl
```

此工具需要 Unitree SDK2 Python 包；仅订阅 DDS，不占相机、不发命令、不接管控制。
它记录命令间隔、kp/kd、力矩及关节状态。订阅频率不等于发布频率；
判断阻尼或命令中断需要结合 kp、kd、tau 和数据新鲜度。

## 配置与进一步阅读

- [真机配置](config/generic_php.yaml)：RealSense 输入及推理/发布频率。
- [仿真配置](config/generic_php_sim.yaml)：SHM 深度输入；与真机共用模型和控制语义。
- [机器人参数](config/PHPParams.yaml)：关节顺序、默认姿态、增益和限位。
- [动作入口](php.yaml)：player 启动及 ready/start 交接。
- [历史部署与实验记录](../../docs/php_deployment_history.md)：旧测量、实现来源和验证记录。
