# P-BFM：行为 context 与深度条件推理

首期接入使用 `depth_conditioned_actor / pbfm`，入口为已有的
`generic_sensor_policy_player`，动作名为 `generic_policy:pbfm_arms_up`。
本体观测、历史容器、动作解码和控制循环与 BFM-Zero / UFO 共用；P-BFM 的
七输入 ONNX、完整 context 和深度时间栈单独实现。真机入口新增下述方向键组合。

## Pico 实时全身遥操（2026-10-21）

`Down+X` 启动 `pbfm_pico_teleop`，使用同一个公开 P-BFM checkpoint `364822528`。
PC 端直接读取 XRoboToolkit 的 24 个身体姿态，经 GMR 的 `xrobot_to_g1.json` 全身 IK
生成 G1 参考姿态，按关节名重排，再用 P-BFM 自身的 backward encoder 在线生成 z。
机器人端用既有 `generic_sensor_policy_player` 接收 z、heading、velocity，仍使用真实机器人
状态、历史和深度运行 P-BFM 完整 actor。没有加载 SONIC actor、encoder 或 planner。
原 `Down+A` loco、`Down+B` dance 和其他策略按键保留。

在连接 Pico 的 PC 上启动 XRoboToolkit PC Service，开启全身追踪并完成其身体标定，
保持地面为 Z=0。脚本的 `--human-height` 填操作者实际身高。发送到本机仿真时用
`--host 127.0.0.1`，发送到机器人时用对应机器人地址：

```bash
bash tools/launch/pbfm_pico_sender.sh --host 192.168.3.162 --human-height 1.70
```

脚本默认使用 `$HOME/resource/GMR`、`$HOME/algo/fighting/P-BFM` 和 conda 的 `gmr` 环境。
可通过 `ACTION_GO_GMR_ROOT`、`ACTION_GO_PBFM_SOURCE`、`ACTION_GO_PBFM_PICO_PYTHON` 指定。
PC 需要 numpy/scipy、MuJoCo、mink/GMR、XRoboToolkit SDK 和 onnxruntime；P-BFM 源目录必须
带 `model/public_demo/{config.json,backward_encoder.onnx}` 与 G1 XML/mesh。
这些依赖在 PC 运行，机器人无需安装 GMR。双方系统时钟偏差须小于 0.2 秒，
UDP 15112 须可达。每次重启 sender 后，重新进入该遥操动作以重置序号及朝向对齐。

机器人重新编译并完成相机/DDS 检查后使用既有 `tools/launch/start_unsafe.sh` 或
`start_safe.sh`；先 `Start`，再 `Down+X`。启动时先等新鲜人体条件与深度，成功推理后才
创建 ready，再按实际开始时的机器人 yaw 和人体 yaw 对齐。`R2+A` 返回官方 walkrun，
`Select+Start` 切阻尼。人体数据超 0.2 秒、序号/采集时间冻结、非法 latent 或深度断流会
中止策略；接管前失败撤销 ready，异常仍尝试共享阻尼退出，supervisor 保留 walkrun 回退。

在线编码采用官方网页的 50 Hz FK、31 body（含 torso 上方 0.35 m 的虚拟 head）、
64D state、463D heading-local privileged state、sigma=2/radius=8 速度滤波，
以及未来 1–8 帧 embedding 均值并归一化到范数 16。
为了提供这些未来数据，参考缓存固定引入 **0.34 秒延迟**，另加 IK/网络/控制耗时；
首次需约 0.52 秒填满 27 帧。freshness 检查使用最新采集时间，不把该已声明的参考延迟
误判为相机或人体断流。人体静止时继续发送新采样的站姿条件，不播放任何离线 gait。

只读与仿真检查：

```bash
./build/generic_sensor_policy_player --spec actions/pbfm/config/generic_pbfm_pico.yaml --check-only
python runtime/tests/pbfm_realtime_test.py --player build/generic_sensor_policy_player
python runtime/tests/pbfm_reference_test.py --source /path/to/P-BFM
bash tools/launch/pbfm_pico_sender.sh --replay /path/to/P-BFM/model/public_demo/motion-example.json
```

replay 只用于把已重定向的 G1 运动送过相同在线编码/UDP 链路，不能作为真人 Pico 验收。
`--record /tmp/pbfm_pico.jsonl` 可记录发出的条件，player 的 `--trace-file` 可核对实际 actor 输入。
官方网页特征/latent 数值对照及本地 lifecycle 测试已执行；Pico 实流、实际操作者重定向效果、
实机动作和视频同等效果仍待验收，不能由模型 shape 或只读测试推断。

2026-10-21 已用同一 sender 的 G1 参考回放完成4秒 UDP→完整 actor→DDS→MuJoCo
闭环，200次推理、状态有限、无重置，最低 pelvis 0.780m；trace 为 `pico_realtime` 且 z
随参考变化。证据见 `outputs/pbfm_realtime_20261021_183609/`。这段输入为站姿加小幅
左肘运动，不是 Pico 真人数据。本机 SDK 可初始化，但只读探测时没有收到身体追踪流；
需开启 XRoboToolkit 全身追踪并完成标定后，继续实流验收。

| 字段 / 语义 | 来源 | 提取值 | ActionGo 实现 | 状态 |
| --- | --- | --- | --- | --- |
| 人体输入 | GMR `xrobot_utils.py:get_raw_body_data/coordinate_transform_unity_data` | 24×XYZ/XYZW；body timestamp；Unity→Z-up | `tools/teleop/pbfm_pico.py` | 上游代码核对 |
| 全身重定向 | GMR `motion_retarget.py`、`ik_configs/xrobot_to_g1.json` | xrobot→unitree_g1，按实际身高缩放；按关节名转换 | 同上 | 实际操作者标定待验收 |
| backward 特征 | 官方 pages `motion-import-uImOWE3z.js:backwardFeatures/motionKinematics` | state64、privileged463、FK/heading/速度滤波 | `tools/teleop/pbfm_reference.py` | 与原函数数值对照通过 |
| z | 配套 `model/public_demo/backward_encoder.onnx` 和网页 importMotion | 256D、未来八帧均值、范数16 | 同上，PC 在线运行 | 真模型数值对照通过 |
| 接管与断流 | 既有 P-BFM ready/start、共享命令过滤与 supervisor | ready 前验证；开始时双 yaw 对齐；200ms stale | `BfmContextStream`、generic BFM loop | 本地 dry-run 测试；实机待验收 |

## 真机 loco / tracking（2026-10-20）

- `Down+A`：`pbfm_loco`，左摇杆前后/侧移，右摇杆左右转向；松杆切垂臂站立条件，实机停步尚未验收。
  前进按推进幅度给出 0–1.0 m/s 指令，后退上限 0.3 m/s，侧移 0.4 m/s，合速度上限 1.0 m/s。
  保留当前仿真验证过的步态与转向条件，无运行时长限制。
- `Down+B`：`pbfm_tracking_dance`，同一官方 checkpoint `364822528`，播放
  `lafan-full-dance1-subject3` 的 56.52–98.72 秒，共 2110 帧 / 42.2 秒，完整保留 68–78 秒主体。
  保留官方原始 z；heading 和 velocity 按片段初始朝向重表达，不修改速度大小。
  不循环；末帧后立即通知 supervisor 切回 walkrun，交接期间保持最后发布目标，接管并 release 后退出。
  资产位于 `motions/lafan_dance_56_52_98_72/`；其中参考初态仅供仿真，真机从当前姿态交接。
  官方未给出最佳模型排名，此选择依据已有本地 dance 平地/台阶回放。
- 先按 `Start` 启用外壳动作；`R2+A` 退出到官方 walkrun，`Select+Start` 切官方阻尼。
  原有其他组合和 recovery backend 保留。

两个新 spec 默认 `sensor.depth_source: realsense`。RealSense 按 `848×480 Z16@30Hz`
采集，用实际标定内参投影训练的 `36×64` 光线；必须覆盖裁剪后使用的区域。
零深度作无效值填远，C++ 完成 P-BFM 裁剪、Gaussian、uint8 量化和一次 `/255`。
50 Hz 策略时点读取最近真实图像并推进 37 帧历史，30 Hz 相机允许重复采样；
重复读取不刷新真实帧时间戳。至少收到 3 帧才 ready，启动等待最多 5 秒，真实帧超过 0.2 秒未更新则失败。
真实相机 spec 另显式要求 `sensor.realsense.min_valid_fraction: 0.20`：在填远和模糊之前，
按标定光线对应的实际策略裁剪区域统计非零深度。有效比例不足的帧不计入 ready，
不更新 freshness；持续不足则启动失败或触发既有 stale 退出。20% 是本地采集质量下限，
不是训练阈值，也不能证明该比例以上即可稳站。2026-10-20 实机出现空杆前移；
随后的只读采样发现下方中央约 93% 原始深度无效，具体记录见 `docs/pbfm_deployment_20261020.md`。
事件日志新增 `pbfm_command_state`，按运动状态变化及每秒记录原始/死区后三轴、速度指令、
参考相位、模型估计速度及深度统计；模型估计速度不能代替真实位移测量。
start 时重置深度历史为最新有效图像，避免 standby 发布频率改变初始时间栈。

**相机安装与当前 Instinct 完全一致。** `actions/instinct/config/env.yaml` 的 world-convention
WXYZ 四元数转换为 optical XYZW 后，与 P-BFM profile 相同；光轴在 torso frame 为
`[0.66913165,0.00354942,-0.74313541]`，相对躯干前向下俯约 `48°`。
两者 mount position 也相同，为 `[0.0487988662,0.01,0.4378029938]` m。
若现有相机已按该 Instinct 标定安装，可共用；这里的 48° 不等于 PHP 的 28°。
spec 记录的是要求的外参，软件不会替代实际安装标定。

以后传到机器人时，保留本仓库源码、`CMakeLists.txt`、完整 `actions/pbfm/`
（含 `model/policy.onnx`、`config/`、完整 `motions/`）、两个 config registry、
`tools/launch/` 及依赖目录的相对结构。在目标架构重新编译，避免复制本机 x86 二进制到 Jetson。
目标机需安装对应架构的 ONNX Runtime、Unitree SDK、yaml-cpp、OpenCV、libzmq 和 librealsense2。

```bash
cmake -S . -B build -DACTION_GO_REQUIRE_REALSENSE=ON
cmake --build build --target generic_sensor_policy_player policy_spec_schema_check action_go_remote_test --parallel 4
./build/policy_spec_schema_check --all
./build/generic_sensor_policy_player --spec actions/pbfm/config/generic_pbfm_loco.yaml --check-only
./build/generic_sensor_policy_player --spec actions/pbfm/config/generic_pbfm_tracking_dance.yaml --check-only
tools/launch/check_pbfm.sh offline
tools/launch/check_pbfm.sh camera
ACTION_GO_NET=eth0 tools/launch/check_pbfm.sh dds
```

检查脚本所有模式均不发布 LowCmd：`offline` 检查资产/ONNX，`camera` 使用合成机器人状态检查真实相机，
`dds` 订阅实际 LowState 和相机运行推理。通过后，再使用 `tools/launch/start_safe.sh`。
2026-10-20 已部署到机器人上的独立 ActionGo 目录，
完成 aarch64 原生构建、真实相机与真实 LowState 的只读推理检查；未发布 LowCmd。
物理运动、安装外参、长时间板载性能及实机切换仍待验证。部署记录见
`docs/pbfm_deployment_20261020.md`。

该机器人使用 `eth0`，可在独立仓库中运行 `bash tools/launch/start_pbfm.sh`，
自动启用 IMU safety 并将 supervisor 日志写入本仓库 `logs/`；运行后仍需手动按 Start 启用动作。

## 资产与构建

本地开发树包含 P-BFM 代码、配置和派生 context；公开代码快照排除 ONNX、
派生 context、LaFAN 初始状态和 `reference.npz`，也不包含机器人模型或训练依赖。
模型及其衍生 context 遵循上游 CC-BY-NC-4.0，LaFAN 参考运动另遵循其数据许可，
不适用 ActionGo 源码的 Apache-2.0。许可证保留在 `model/LICENSE`。
`reference.npz` 用于重新编码和分析，已有 context 的推理不需要它。

`python3 tools/assets/fetch_assets.py --policy pbfm` 从固定上游版本下载并组装权重，
校验最终 SHA256。context 需按下述导入/编码步骤单独准备，不能把仅下载模型视为完整包。

从已准备好的官方公开推理目录导入，不需要训练框架在线运行：

```bash
python3 tools/export/import_pbfm_public.py --source /path/to/P-BFM/model/public_demo
cmake -S . -B build
cmake --build build --target generic_sensor_policy_player policy_spec_schema_check bfm_policy_test --parallel 4
./build/policy_spec_schema_check --all
./build/generic_sensor_policy_player --spec actions/pbfm/config/generic_pbfm.yaml --check-only
```

源代码版本 `cafdfbdd71640a39b7f091589239d6501ac7f00d`；公开权重来自官方
`pages` 分支 `671fbafe71a5f58bab77eeb5d969051fd2d283f4`，checkpoint step
`364822528`。权重约 144 MiB，本地已导入，许可证为随包保留的 CC-BY-NC-4.0。
完整 ONNX 内含 normalizer、depth CNN/GRU、速度估计器及 Actor，不能在外部重复归一化。

当前机器的 Conda `libstdc++` 比 OpenCV 依赖旧；若报 `GLIBCXX_3.4.30 not found`，
可对相应命令设置 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`。
这是本机运行环境处理方式，不是板载部署要求。

## 关键语义

| 语义 | 上游来源 | ActionGo 对应 |
| --- | --- | --- |
| state：q-default、dq、gravity、gyro×0.25 | `envs/mjlab/observations.py` | `BfmObservationHistory`，64D |
| 按字段分组的此前四帧，读取后再加入当前帧 | `envs/mjlab/observations.py`、`history.py` | `history_timing: previous_only`，372D；BFM/UFO 保留原采样时点 |
| 上次 scaled action；目标 q=default+scale×5×clip(raw,-1,1) | `envs/mjlab/core.py`、公开 `config.json` | 共享 `decode_bfm_action`；不把安全过滤后的 q 写回历史 |
| 29 关节顺序、默认姿态、PD 和缩放 | 公开 `config.json` | 独立 `PBFM.yaml`；不直接复用 BFM-Zero 的腕关节参数 |
| heading 相对启动 yaw，速度在当前 yaw frame | `agents/inference_context.py`、公开 `contexts.json` | context 同时保存 z、heading、velocity validity；正式 start 时锁定 yaw |
| 50 Hz 策略；保守硬件力矩上限 | `config.json`、已有 `UFO.yaml` | 运行时取两者上限的较小值；上游仿真参数单独保留 |
| optical-Z 预处理和时间栈 | `perception/instinct_direct_depth.py`、`depth_camera.py` | `tools/sim/pbfm_depth.py`、spec 的 sensor profile |

五个官方固定 preset 已生成 `context_*.yaml`，默认是 `arms-up`。更换 context 时修改
`context.path`，保持该 checkpoint 配套的 z 与命令元数据一起切换。
`[0,0,1]` 表示有效零速度指令，`[0,0,0]` 表示未指定；不能混用。
没有为公开 preset 添加站立或恢复能力的额外承诺。

## 自然行走与速度、航向条件

之前的 `forward08-arms-spread` 是明确要求张臂的 reward preset。自然摆臂行走使用
官方 LaFAN tracking context。公开目录缺少散文件时，可从本地 P-BFM Git 仓库中
固定的 pages 提交读取缓存；导入器直接保留每帧已编码的 z，不重复均值或投影。

```bash
python3 tools/export/import_pbfm_tracking.py --source /path/to/P-BFM \
  --motion lafan-full-walk1-subject5 --frames 700 \
  --speed 0.6 --heading-deg 0 --output outputs/pbfm_walk_context
PYTHONPATH=/path/to/unitree_sdk2_python python tools/sim/pbfm_smoke.py \
  --source /path/to/P-BFM --tracking outputs/pbfm_walk_context \
  --output outputs/pbfm_walk --duration 14 --viewer --video
./build/generic_sensor_policy_player --spec outputs/pbfm_walk/runtime_spec.yaml --check-only
```

省略 `--speed / --heading-deg` 时跟随原始参考的速度和转弯；同时指定两项时，保留
行走 z，覆盖为相对启动 yaw 的固定速度与航向。这对应官方交互网页的 heading/speed
条件覆盖语义；在线键盘控制另见下节。
官方展示了遥操视频，但公开主仓库未发现完整的实机遥操采集与部署链路。
不能把当前 cache 播放称为实时全身遥操。

### MuJoCo 在线键盘控制

离线编码器需要 `numpy`、`scipy`、`mujoco`、`onnxruntime` 和 `PyYAML`；在线仍只运行 actor。

```bash
python3 tools/export/export_pbfm_idle.py --source /path/to/P-BFM \
  --keyboard-contexts --output actions/pbfm/config
python3 tools/export/export_pbfm_walk_cycle.py --source /path/to/P-BFM \
  --symmetric --swing-foot-pitch 0.1 --speed 1.0 --turn-rate 0.6 \
  --turn-gait-amplitude 0.6 --output actions/pbfm/config/joystick_walk
python3 tools/export/export_pbfm_walk_cycle.py --source /path/to/P-BFM \
  --start 5894 --length 58 --speed -0.3 --symmetric --swing-foot-pitch 0.1 \
  --output actions/pbfm/config/joystick_backward
./build/generic_sensor_policy_player \
  --spec actions/pbfm/config/generic_pbfm_joystick.yaml --check-only
PYTHONPATH=/path/to/unitree_sdk2_python python tools/sim/pbfm_smoke.py \
  --source /path/to/P-BFM \
  --output outputs/pbfm_keyboard --duration 0 --viewer --keyboard --terrain stairs
```

点击 MuJoCo 窗口后，Enter 开始；按住 W 前进（目标 1.0 m/s）、S 后退（0.3 m/s），
按住 A/D 持续左/右转（目标航向积分速率 0.6 rad/s），Q/E 左/右侧移（0.4 m/s）。松开对应键立即归零该轴，
窗口失去焦点时所有轴归零；相反方向同时按下抵消，按键重复不累加速度。
空格按住时停止，X 退出。松键会同时切换站立 z 和有效零速度条件，实际停稳存在动力学响应时间。
键盘模式使用 GLFW 的按下/松开事件；左键拖动旋转视角，滚轮缩放。
双击左键选中身体部件；Ctrl+右键拖动施力，Ctrl+左键旋转施加力矩，Shift 切换水平操作。
Ctrl 点击也会直接选中鼠标下的部件；松开鼠标或窗口失去焦点即撤销扰动力。
拖拽在 Enter 启动后通过物理外力作用于机器人，不直接覆盖关节角。
窗口分别显示速度指令和 MuJoCo 实测水平速度，开始前物理状态冻结，运行时长从 start 开始计时。
示例不限时。W/S 使用重新编码的前进/后退周期步态，Q/E 使用方向匹配的固定 goal context。
运行过程中不冻结关节、不清零物理速度，站立由同一个策略闭环执行。

该入口复用 `LowState.wireless_remote`，没有新增网络端口。spec 显式配置
`context.command_source=wireless_remote` 和四项正值上限 `max_forward_speed`、
`max_backward_speed`、`max_lateral_speed`、`max_yaw_rate`；默认仍从 context 读取命令。
轴映射与既有 Unitree 一致：LY 前进、−LX 左侧移、−RX 左转。运行时在 50 Hz
积分目标 yaw，heading 与速度条件按目标航向转换。键盘默认与真实摇杆动作
`generic_policy:pbfm_joystick` 使用同一 `generic_pbfm_joystick.yaml`；既有遥控按键映射不变。
前进 LY 在死区外直接映射为 `LY * 1.0 m/s`，例如 0.25/0.5/1.0 对应目标 0.25/0.5/1.0 m/s；
5% 死区内为零。`max_translation_speed=1.0` 也限制斜向合速度。
`proportional_speed=true` 按幅度混合 idle 与行走 latent 并恢复范数 16，避免小幅输入仍执行满幅步态。
这是一米每秒的指令上限，物理实际速度会有跟踪误差。入口另外配置
`context.idle_path` 与 `context.command_deadband=0.05`：全部轴处于死区时切换站立 context，
转向时 `max_heading_lead_rad=0.5` 限制目标领先当前航向的角度；转向轴刚归零时锁定当前航向，
取消尚未完成的转角，后续站立或直行持续保持该目标，不随物理漂移反复重置。
`context.directional_paths` 显式提供 forward/backward/left/right 的固定或周期 context，
W/S/Q/E 选择相应 z，斜向按相对指令强度加权并恢复原范数。
`context.turn_paths` 提供左/右转向踏步序列；仅转向时也持续抬脚，转速由转向轴推进目标航向。
小幅转向在同相位左右转向 latent 间插值并归一化，保留共同踏步条件，避免混合 idle 后站死。
左右转向参考由 0.6 倍行走幅度、零根平移、
正负 yaw 角速度及其 `omega × relative_position` link 速度重新编码。
全部轴归零时保持站立。缺失站立、方向或转向资产会在 ready 前失败。
方向序列要求 `dt=0.02`，平移或配置了转向序列的转向输入会推进并循环；全部松键暂停相位。
恢复行走时继续此前相位。普通 tracking 仍在末帧结束。
未配置 `idle_path` 的既有 remote spec 仍只覆盖 heading/velocity。
命令输出仍通过共享安全过滤。仅在 loopback MuJoCo 中验证，实机遥操未验证。

这些 goal 资产不是官方五个 preset。Q/E 的 `export_pbfm_idle.py` 使用配套官方
`backward_encoder.onnx`，按官方网页 `motionKinematics/backwardFeatures` 构造默认关节姿态、
模型初始根高度；Q/E 目标平移速度依次为 `(0,0.4)`、`(0,-0.4)` m/s，
站立为零。所有目标关节速度和角速度为零，31 link 的位置、X/Z 轴 6D 朝向及速度构成 463D privileged state。
离线生成归一化 256D z；在线无需第二个 encoder session。资产 YAML 记录来源与目标姿态。
相机与 actor 仍执行真实物理闭环；输入目标角速度为零不能保证机身零偏航。

此前 W 也使用默认姿态加匀速平移的固定 goal，造成小碎步及前臂前举。
现在 `export_pbfm_walk_cycle.py` 从官方 `lafan-full-walk1-subject5` 的第 682 帧起
提取 54 帧、1.08 秒的行走周期，进行周期平滑和左右半周期镜像平均，移除根 yaw
变化及侧向根平移，保留动态屈膝、抬脚和摆臂。按官方 64D state / 463D privileged
特征顺序重建关节与 link 速度，经配套 backward encoder、未来八帧均值和范数 16
归一化后写入 `joystick_walk/context.yaml`，目标前向速度为 1.0 m/s。旧 `keyboard_walk` 1.2 m/s 资产保留。
此为适配后的参考，不是未经修改的官方 cache。
摆动期参考踝角另作脚尖下垂修正，目标脚俯仰 0.1 rad，并限制在上游关节范围内；
修正后的 FK 和速度重新进入 backward encoder，不在控制输出端覆盖关节。
`joystick_walk/idle.yaml` 采用默认腿部和 1.2 rad 肘角编码自然垂臂站立；
不改模型输出、PD 默认角或仿真物理状态。`reference.npz` 保存编码前的参考运动。

S 此前使用默认站姿加负速度的固定 goal，实测后退较慢且步幅小。现从同一官方
LaFAN 文件第 5894 帧起提取 58 帧的真实后退片段，去转向、对称及周期化后，
按目标 −0.3 m/s 经官方 encoder 重新生成 `joystick_backward/context.yaml`。
参考按 50 Hz 循环，松 S 切回原站立条件；负 LY 幅度仍连续映射到 0–0.3 m/s 的后退指令。
此为本地参考适配，并非官方现成的后退 preset。真实后方地形感知未验证。

可重放按键并测量实际停步，而非仅检查指令是否归零：

```bash
PYTHONPATH=/path/to/unitree_sdk2_python python tools/sim/pbfm_smoke.py \
  --source /path/to/P-BFM \
  --keyboard-script tools/sim/pbfm_keyboard_straight.json --terrain stairs \
  --duration 30 --video --output outputs/pbfm_keyboard_stop
./build/generic_sensor_policy_player --spec outputs/pbfm_keyboard_stop/runtime_spec.yaml --check-only
```

`movement.npz` 保存仿真时间、位置、物理速度、四元数、实际关节角与按键指令，`summary.json` 报告每次松键的收步时间、
位移、一秒后速度、最大航向误差和转向轴。`trace.jsonl` 保存 `motion_state`、`command_axes`、
`command_yaw` 与 `reference_step`，可区分按键误输入、目标航向变化和模型跟踪误差。
此前 0.6 m/s 指令的本地 30 秒零转向测试中，转向轴和目标 yaw 全程为零，最大实际偏航为平地 8.29°、台阶 11.36°；
两次松键均收步，仍不能宣称精确航向/位置控制。此前 0.8 m/s 目标复测的平均水平速度约
平地 0.654、台阶 0.625 m/s，松键约 0.84–0.96 秒收步。证据见 `docs/pbfm_inference_plan.md`。

sequence 契约为 `tracking_heading0_yaw_velocity_v1`，`dt=0.02`；每帧包含 `z`、
`heading_relative_xy`、`heading_valid`、`reference_velocity_heading0_xy`、`velocity_valid`。
普通播放时 z、heading、velocity 在 start 后使用同一索引推进；速度每步从启动 yaw 坐标旋转到
当前 yaw 坐标，heading 也按启动 yaw 对齐。最后一帧后结束推理，不循环、不重复末帧推理；
新真机动作保持最后发布目标至 supervisor release，无 complete/release 参数时直接结束。
`--tracking` 仅在仿真开始时设置官方初态，之后全部由 C++ 模型
通过 DDS 控制物理仿真。此 LaFAN 片段开头约 3 秒包含张臂准备姿态。
原始参考转弯较多；固定航向覆盖虽改善前进效果，仍有侧漂，不代表精确速度跟踪。

## 深度与接管

原有仿真/外部输入为 POSIX SHM `actiongo_pbfm_depth` 中连续的 float32 `[8,18,32]`，
取值 `[0,1]`，按旧到新排列。ZMQ PUB 心跳端点为
`ipc:///tmp/actiongo_pbfm_depth.heartbeat`，JSON 包含递增 `seq` 和 `state: OK`。
生产端应在 50 Hz 控制时点更新历史；消费端直接读取已组成的时间栈，不自行追加历史。

profile `optical_z_maxcrop18_reset_v2` 使用 `[36,64]` optical-Z 米制图像，裁剪
`[top=18,bottom=0,left=16,right=16]`，无效值填 2.5 m，3×3 sigma=1 Gaussian
reflect 边界，归一化后量化为 uint8。37 帧历史按 age
`[35,30,25,20,15,10,5,0]` 抽样，首次有效图像填满历史，固定 delay=0。
ONNX 输入仅在最后除以 255 一次。真实相机的零值无效像素应先按无效值处理。

相机固定在 `torso_link`，位置和 optical quaternion 来自上游 profile；仿真
`MujocoDepthCamera` 使用相同 pixel-center、双 FOV 内参及 range→optical-Z 转换。
光线投射时按官方网页排除 `torso_link` 上的 `head_link` 外壳 mesh，随后恢复 geom group；
否则相机位于外壳内部，输入会被约 1.17 cm 的壳体完全遮挡。
不兼容 PHP/Instinct 现有的其他深度 profile。

模型和实时深度完成一次推理后才写 ready；等待 start 时保持当前姿态并持续检查深度。
超时 0.2 秒会退出，已有 publisher 时走共享异常阻尼；LowCmd 经过
`CommandSafetyFilter`。文件深度仅用于 check-only / dry-run；不允许发布真实命令。
本 variant 暂不支持事务 preview，保留 ready/start 热备及 switch-back。

## 可复现验证

```bash
ctest --test-dir build -R '^(bfm_policy_test|pbfm_depth_test|pbfm_onnx_parity|pbfm_lifecycle_test|generic_preview_integration)$' --output-on-failure
python3 runtime/tests/pbfm_depth_test.py --source /path/to/P-BFM
ACTION_GO_NET=lo tools/launch/start_dry_run.sh
```

测试 Python 需 NumPy、PyYAML、ONNX Runtime、pyzmq；上游深度对照另需 PyTorch。
对照测试直接执行上游的纯预处理函数与配置定义，跳过依赖 MJLab/Warp 的相机导入。
120 步 Python/C++ 回放覆盖 BFM-Zero、UFO 和三个 P-BFM context 的 feed、raw action、
estimated velocity、scaled action 和 target q，FP32 容差 `atol=rtol=2e-4`。
生命周期测试覆盖缺深度、正常 ready/start、待启动断流、活动断流、未归一化深度和文件发布拒绝。

完整闭环使用 MuJoCo、CycloneDDS、Unitree Python SDK 和 pyzmq：

```bash
PYTHONPATH=/path/to/unitree_sdk2_python python tools/sim/pbfm_smoke.py \
  --source /path/to/P-BFM --output outputs/pbfm_smoke --duration 10
PYTHONPATH=/path/to/unitree_sdk2_python python tools/sim/pbfm_smoke.py \
  --source /path/to/P-BFM --output outputs/pbfm_dropout --duration 3 --drop-depth-after 0.6
PYTHONPATH=/path/to/unitree_sdk2_python python tools/sim/pbfm_smoke.py \
  --source /path/to/P-BFM --output outputs/pbfm_visual --duration 12 --viewer --video
PYTHONPATH=/path/to/unitree_sdk2_python python tools/sim/pbfm_smoke.py \
  --source /path/to/P-BFM --output outputs/pbfm_stairs --duration 14 --viewer --video \
  --context forward08-arms-spread --terrain stairs --step-height 0.12 --stair-width 6
```

输出目录必须尚不存在；默认网络 `lo`、DDS domain 197。脚本复用现有
`NativeActionGoBridge`，先检查实际 LowCmd 为 current-pose hold，再发 start，随后由
C++ 策略驱动自由基座 G1 与光线投射深度。输出包含 trace、事件、日志和 summary。
`--viewer` 打开实时窗口；`--video` 保存运行时采集的姿态，再离线渲染为 720p/30 FPS
`pbfm.mp4`，需要系统 `ffmpeg`。视频渲染不占用实时推理循环。
`--context` 选择已导入的 preset，并在输出目录写入路径已解析的运行 spec。
`--terrain stairs` 在前方 1.8 m 后放置五级台阶、0.8 m 平台与下行台阶，默认
阶高 0.12 m、踏面深 0.30 m、通道宽 6 m。台阶同时参与碰撞与实时深度光线投射。
`--stair-approach 0.30` 将初始根部到第一级台阶前沿的水平距离设为 30 cm，默认 1.8 m。
`--start-on-platform` 配合 `--terrain stairs` 从顶部平台中央开始，每轮重置也返回平台；保留参考初态的姿态和离地高度。
`--terrain-heading-deg` 可旋转台阶朝向（默认 0°，沿 +X），例如 LaFAN run 首段可用 150°。
使用 `--viewer --duration 0` 可持续运行到关闭窗口；tracking 片段结束后自动重启策略、
深度历史并重置机器人到参考初态，再播放下一轮。每轮仍保留 ready/start 验证；
这属于多轮仿真演示，不代表无重置的长时稳定运行。此模式关闭逐帧 trace，且不能与
缓存整段姿态的 `--video` 同用，以免无限增长内存和深度 trace。
这是独立的直行演示地形，未复现上游完整训练地形生成器。
默认物理步长 0.005 秒，implicitfast / Newton，策略保持 50 Hz；逐子步执行官方的
速度相关 DC 电机力矩包络。平地使用顶面 z=0 的实体 box，与官方场景的平地几何类别一致。
此前的无限 plane、Euler 积分和固定力矩截断不能视为官方物理复现。
脚本会检测自动重置并失败，不能把重置后的有限状态算作通过。
这仍是软件闭环验证，未复现全部训练随机化或动力学；ActionGo 仍保留目标关节限位、
插值、激活混合和共享安全过滤，不能称为逐步完全等价的官方执行器。

## 官方推理入口与仿真排查

官方原生入口为 `python -m pbfm.inference.tracking`（另有 goal / reward），例如：

```bash
python -m pbfm.inference.tracking --model-folder /path/to/run \
  --motion-list 20 --save-mp4 --terrain-family low_stairs_down --difficulty-row 7
```

该入口依赖 MJLab / MuJoCo-Warp、原始 checkpoint 目录与 motion 数据；本地只有公开 ONNX
导出资产，尚不能执行这一入口。可直接运行的官方公开闭环是
[交互网页](https://roboparty.github.io/P-BFM/interactive/)（MuJoCo WASM + ONNX）。
网页仍有单独的 browser adaptation 验证边界，不等同于原生训练环境的完整复现。

独立原生 Python 对照工具从固定官方 pages 提交提取机器人和场景，不经过 DDS：

```bash
PYTHONPATH=/path/to/unitree_sdk2_python python tools/sim/diagnose_pbfm_sim2sim.py \
  --source /path/to/P-BFM --scene public --duration 24 --video \
  --output outputs/pbfm_official_contract
```

可用 `--scene plane` / `box` 对照地面，`--cases occluded_camera interpolated joint_clamp legacy_physics`
逐项排查；这些是本地复现与消融，不是调用官方原生 CLI。
此次相机修复后，单独使用 plane 仍跌倒；在同一原生对照中仅改成 box，24 秒最低骨盆
从 0.092 m 提升至 0.550 m，未出现低于 0.45 m 的阶段。该证据说明接触几何差异会改变结果，
未进一步断言 MuJoCo 内部的具体接触算法原因。
修正后的 ActionGo 实际 DDS 跑步闭环完成 1200 帧 / 24 秒，最低骨盆 0.565 m，未再跌倒；
视频与日志见本地 `outputs/pbfm_run_sim2sim_fixed/`。断深度仍实际发送 `kp=0,kd=3` 后退出，
见 `outputs/pbfm_dropout_sim2sim_fixed/`。单次短片成功不代表长时、全地形或实机稳定性。

实测记录及当前待办见 `docs/pbfm_inference_plan.md`。
真实相机标定、板载长时性能、吊架及实机切换、Pico 真人实流遥操验收仍待完成。
