# P-BFM 独立真机仓库部署记录（2026-10-20）

## 仓库与配置

SSH 为 `unitree@192.168.3.162`，机器人是 aarch64 Jetson，用户目录 `/path/to/robot-user`。
新增独立 Git 仓库 `/path/to/action_go_pbfm`，从本地 feature 分支的 bundle 克隆，
分支保留 `feature/generic-policy-runtime`，起点为 `7ca789c`。
叠加本地当前可部署源码、配置与 P-BFM 改动；改动保留为未提交状态，未创建提交或推送 GitHub。
不覆盖既有 `/path/to/action_go`，不变更现有 systemd 服务。

模型和 motion 资产另行传输，包含完整 P-BFM `policy.onnx`、loco 条件与官方 36 秒 dance 条件，
以及当前 registry 使用的其他策略资产。仿真输出、视频、日志、本机 build 和本地研究草稿未传输。
部署文件列表在新仓库 `logs/deployment_manifest.json`。

键位：`Down+A` 为 `pbfm_loco`，`Down+B` 为 `pbfm_tracking_dance`。
loco 前进按摇杆幅度映射 0–1 m/s 指令，后退上限 0.3 m/s，松杆切回站立条件。
tracking 使用相同 checkpoint 的官方 LaFAN dance 前 1800 帧 / 36 秒，末帧保持至 supervisor release。
退出沿用 `R2+A` 回官方 walkrun、`Select+Start` 切官方阻尼。

## 原生构建

机器人系统 CMake 3.16.3 低于项目要求，采用已安装的 CMake 3.28.6。
使用仓库自带 aarch64 ONNX Runtime 1.22.0；RealSense 使用 `/usr/local` 的 2.50.0，
Unitree SDK2 使用 `/usr/local/lib/cmake/unitree_sdk2`。

```bash
cd /path/to/action_go_pbfm
~/cmake-3.28/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DACTION_GO_REQUIRE_REALSENSE=ON \
  -DACTION_GO_REALSENSE_ROOT=/usr/local \
  -DACTION_GO_ONNXRUNTIME_DIR="$PWD/thirdparty/onnxruntime-linux-aarch64-1.22.0"
~/cmake-3.28/bin/cmake --build build -j4 --target \
  generic_sensor_policy_player generic_policy_player generic_motion_tracking_player \
  generic_composite_policy_player action_go_remote_test policy_spec_schema_check \
  bfm_policy_test depth_image_preprocess
```

检查启动输出必须为 `realsense=enabled`。该构建的 Humanoid-GPT realtime FK 未配置 MuJoCo，
不影响 P-BFM；AprilTag C 库未安装，SteadyTray 的 AprilTag source 尚不可用。
以上两项是其他策略的可选功能限制，不以此次 P-BFM 部署宣称通过。

## 无电机命令检查

D435i 已识别，固件 `05.15.01.55`。真实 profile 为 `848×480 Z16@30Hz`，标定参数
`fx=fy=424.834, ppx=427.099, ppy=243.413`，成功映射到 P-BFM 的 optical-Z `64×36`。
两模式 ONNX check-only 通过。

真实相机 + 合成机器人状态的 5 秒检查完成 250 次策略推理，平均约 14–16 ms，
policy deadline miss 为 0。真实相机 + `eth0` 实际 LowState 的另一组 5 秒检查也完成 250 次推理，
平均约 15.6–16.7 ms，个别推理约 20.5 ms；测试与编译同时进行，不能视为长期性能验收。
两组均使用 `--dry-run`，不建立 LowCmd publisher，进程退出事件记录实际 `publish_count=0`。
不以 dry-run 内部循环的发布尝试计数代替实际 DDS 发布次数。

四个 generic player、supervisor、schema checker 及检查工具均完成目标机原生构建。
目标机严格检查 34 个 spec，无缺失资产；BFM 策略检查及 36 帧 Python/C++ 深度精确对照通过。
`lo` 上的 supervisor dry-run 加载 91 个动作，确认下+A / 下+B 映射正确并正常退出。

```bash
bash tools/launch/check_pbfm.sh offline
bash tools/launch/check_pbfm.sh camera
ACTION_GO_NET=eth0 bash tools/launch/check_pbfm.sh dds
```

相机启动、真实帧 freshness 和只读推理已验证；实际机械安装是否符合 Instinct/P-BFM
相同的约 48° 下俯外参尚未测量。未启动运动控制、未切 UserCtrl、未发送电机命令。
物理行走、停步、倒退、tracking 与官方 walkrun 的交接仍需受控实机验证。

## 使用入口

在机器人已处于合适的受控测试条件时，手动运行：

```bash
cd /path/to/action_go_pbfm
bash tools/launch/start_pbfm.sh
```

该入口默认使用 `eth0`、启用 IMU safety，并生成 `logs/pbfm_supervisor_<时间>.jsonl`。
启动后先按 Start 启用，再按下+A / 下+B；本次部署不自动运行该入口，不安装开机自启动。
configure、build、check-only、camera、DDS 和 schema 日志保存于新仓库 `logs/`。

## 空杆前移排查与深度质量修正

用户随后手动启动 `start_unsafe.sh`，报告按 Down+A 后未推摇杆即向前走；用户确认相机
仍按 Instinct/P-BFM 位置和角度安装。原运行日志确认加载独立仓库的 loco spec，完成约
10.65 秒 / 548 次策略推理，未报告策略或发布 deadline miss。原日志没有遥控轴，不能
据此回溯运行当时的逐帧输入，也不能将本次报告当作实机停步通过。

运行退出后，只读 DDS 采样 5 秒，96 个观测点的前进/侧移/转向轴均为零；真实相机加
LowState 的 3 秒 dry-run 共 150 帧，全部为 idle、参考相位 0、速度指令 `[0,0,1]`，
实际 LowCmd 发布数为 0。未发现当前空杆输入残留；这不是原运动过程的同步输入记录。

相机原始 Z16 另行只读采样 20 帧，下方中央区域零值比例为 92.4%–93.4%，
对应训练光线的最后一帧策略 crop 仅约 8.0% 有效。此前零深度被按预处理契约填为
2.5 米，量化后的策略深度约 75% 像素达到上限，均值约 2.40 米。
发射器开启、laser power 150、自动曝光开启；尚未确定无效深度来自环境还是采集硬件。
机械角度正确不能替代深度有效性检查。

8 秒本地对照中，相同 idle 条件和硬件力矩上限下，模拟地面深度能收稳；固定使用这次
录制的真机深度也未复现持续前移。全远深度出现约 0.12 m/s 的持续后漂。
该对照证明深度改变会影响零指令行为，但未复现用户报告的持续前移，不能认定深度
是唯一原因。静态 idle latent 是本地站姿目标编码，不是已经验收的真机零位移控制器。
数据与对照脚本保留在本地 `outputs/pbfm_neutral_diagnosis_20261020/`。

修正共享 RealSense 链路：两个真实相机 spec 显式设置 `min_valid_fraction: 0.20`，
在填远/模糊前，对标定后策略 crop 的非零原始深度计数。低质量帧不计入三帧 ready，
不刷新有效帧时间戳；启动最多等 5 秒，运行中沿用 0.2 秒 stale 退出及异常阻尼。
这是采集质量下限，不是策略稳定性证明。新增 `pbfm_command_state` 事件，记录每次
运动状态变化及每秒的原始/处理后三轴、速度命令、相位、估计速度和深度统计。
保留共享安全过滤、ready/start、walkrun 回退和其他策略及键位。

用户要求的启动仍为：

```bash
cd /path/to/action_go_pbfm
ACTION_GO_NET=eth0 ACTION_GO_EVENT_LOG=logs/pbfm_unsafe.jsonl bash tools/launch/start_unsafe.sh
```

`start_unsafe.sh` 关闭 IMU supervisor safety，但不绕过上述深度质量和共享命令过滤。
当前坏深度应先通过只读 `ACTION_GO_NET=eth0 bash tools/launch/check_pbfm.sh dds` 排查；
保持姿态并让相机看到有纹理的地面，先确认有效比例恢复。恢复后仍须同步记录真实
输入、深度和位移进行受控停步复测，不能把拒绝坏深度接管描述成前移问题已经修好。

修正后本地 34 个 schema、两个新 spec check-only、BFM/深度/生命周期三个测试通过；
`start_dry_run.sh` 在 lo 加载 91 动作并等待状态后由超时结束。
机器人已完成更新后的 native build 和两 spec offline 检查。真实相机 + LowState 的
质量门槛 dry-run 以 `policy crop valid_fraction=0.097222 required=0.200000` 在 ready 前
退出，ready 文件未创建，实际 publish_count=0。这是坏输入拒绝的验证，没有运动复测。

### 遮挡解除后的重新采样

用户随后说明上一轮采样时相机被遮挡。应将约 93% 无效记录解释为遮挡状态的数据，
不能据此推断相机故障或原运动测试时也存在坏深度，更不能将其认定为空杆前移根因。
遮挡解除后重新以 848×480 Z16@30Hz 连续采集 30 秒，按秒保存 30 份未经填远/模糊的
原始 uint16 PNG 和统计。全图有效比例 96.94%–97.23%，训练光线对应的策略 crop
30 次采样均为 100% 有效，末帧有效深度中位数 1.241 米。
本地原始数据、预览和统计位于 `outputs/pbfm_uncovered_depth_20261020/`，
`latest_raw.png` 保留 uint16，乘 `metadata.json` 的 0.001 得到米。
本轮纯相机采集没有任何控制发布，采集完成已释放相机。
随后真实 LowState 加无遮挡相机的 3 秒只读推理通过，共 150 步，平均约 13.5–13.7 ms，
当前仍为空杆 idle、零平移命令，最后两次每秒统计的深度均值约 1.24 米、远值比例 0。
质量门槛正常放行，实际 publish_count=0；不作为物理稳站效果结论。

## 相机格式协商失败恢复（2026-10-20 16:33）

用户后续 Down+B 的两次启动在 ready 前报
`xioctl(VIDIOC_S_FMT) failed Last Error: Input/output error`。
检查时没有深度监测/策略残留进程，`fuser /dev/video*` 未发现占用；相同 dance spec 的
独立 `--dry-run` 也复现错误。内核日志同期记录 `Failed to set UVC probe control : -32`，
且此前 16:25 有 USB 断开/重新枚举、端点状态及 U1/U2 协商异常。
这定位到相机/UVC 格式协商链路，不能据此确定线材、供电、固件或驱动哪个是底层诱因。
该错误发生在深度质量检查前，与 idle 指令、dance cache 内容或“无效深度比例门槛”无关。

通过 librealsense `device.hardware_reset()` 只重置序列号 `349522072192` 的 D435i，
设备随后重新枚举。重置工具源码在本地
`outputs/pbfm_camera_reopen_20261020/reset_camera.cc`，目标机临时工具为
`/path/to/action_go_pbfm/outputs/reset_pbfm_camera`。
本次未重启机器人、未主动停止 supervisor、未触发运动模式。恢复后的 dance 2 秒只读测试完成
100 次推理，再顺序重新打开 loco/dance 各 1 秒、各 50 步，三次均通过，实际 publish_count 均为 0。
最终检查没有相机占用；此时用户的 supervisor 进程也已退出，可重新按既有入口启动。
保留 ready/start 失败退出机制，没有为运行时增加静默 USB 重置或无限重试。

### 随后复发：运行中 USB 断连

此前重置与三次只读检查仅证明当时可启动，未修复持续稳定性。用户随后再次报告启动
格式协商失败，并确认没有手动拔插；内核记录 16:38:10、16:39:48 的 USB 断开与重新枚举。
后一次实际 loco 运行约 7.7 秒后于 16:43:59.626 因 0.2 秒无新帧退出，最后有效 crop
比例 0.967014；内核同秒出现 xHCI endpoint/TRB 异常、U1/U2 设置失败，16:44:00 USB
断开并重新枚举，随后再次出现格式协商 -32。质量比例正常也会断流，两种错误是这次
USB 链路异常的前后表现。底层是线材/接头/供电还是主控/驱动/固件兼容，仍未区分。

USB 设备 runtime_suspended_time 为 0，不能把 autosuspend 配置为 auto 直接当作根因。
目标机 Jetson L4T R35.3.1 / kernel 5.10.104-tegra，RealSense SDK 2.50.0、固件 05.15.01.55；
SDK 枚举提示的推荐固件为 05.13.00.50，兼容性仅作为待核查项，没有刷固件或改全局 SDK。
再次重置后的只读诊断也报格式协商错误，没有完成计划的 60 秒测试，不能标为通过。
已结束自身诊断，清理不再有相机文件句柄、已记录异常退出但未结束的旧孤儿 payload PID 7063；
当前没有诊断占用相机。未放宽 stale、质量门槛，也未自动恢复运动。后续先通过短 USB 3
线直连/连接口和供电对照定位，再做静置与晃动线材的纯相机连续采集测试。
该次 kernel/payload/只读失败证据保存在本地 `outputs/pbfm_usb_failure_20261020/`。

## Down+B 舞蹈片段更新

按用户选择，将 `Down+B / pbfm_tracking_dance` 替换为官方同一 checkpoint 的
`lafan-full-dance1-subject3` 56.52–98.72 秒，2110 帧 / 42.2 秒，包含原 68–78 秒主体。
新资产位于 `actions/pbfm/motions/lafan_dance_56_52_98_72/`，spec 指向该目录。
保留原始 z；heading 与 velocity 仅转换到新初始朝向。仿真使用的参考初态、
平台位置和循环重置不用于真机。

已同步到 `192.168.3.162:/path/to/action_go_pbfm`，保留原键位与 action 名。
旧 spec 备份在 `logs/pbfm_dance_clip_backup_20261020/generic_pbfm_tracking_dance.yaml`，
旧 36 秒 motion 资产保留。目标机两模式 schema 与 ONNX check-only 通过，新 tracking
确认加载 2110 帧。此次仅更新资产和配置，不需要重新编译。

末帧后立即向 supervisor 发 complete，触发 walkrun 请求；walkrun 确认后沿用
0.30 秒发布重叠，再 release 退出，不循环、不增加站姿等待尾段。
本地共享 lifecycle/controller 测试通过；从默认站姿启动的新片段平地 DDS/MuJoCo 回放
完成 2110 次推理，状态有限，最低 pelvis 0.613 m、结束 0.764 m。
这些检查不代表新片段的物理运动或 walkrun 实机交接验收。此次未启动真机动作。

## 相机断流的软件路径修复（2026-10-21）

用户确认线材和相机正常后，重新审计程序。此前内核 USB 错误只能定位故障层次，
不能排除程序/SDK 的启停路径触发或放大问题；不再将更换线材作为本次处理前提。

本地真实 POSIX 进程回归复现了停止路径的漏洞：supervisor 只等待 shell 退出，
没有继续等待同组的策略子进程，遇到 SIGTERM 处理或清理阻塞可留下孤儿 payload。
正常轮询发现 shell 自行退出时也会丢弃进程组的清理责任。
已修复为：正常停止在原有 400 ms 窗口内检查整个进程组，超时清理剩余成员；
自然退出时清理同组残留并保留 shell 退出状态。强制阻尼停止仍立即终止进程组。
该修复覆盖 supervisor 正常运行时的停止/退出路径，不宣称解决 supervisor 自身被
SIGKILL 后所有孙进程的回收问题。

检查目标机所用 librealsense v2.50.0 的 `src/pipeline/pipeline.cpp` 后确认：
`try_wait_for_frames(..., 100)` 并非严格限制为 100 ms，超时后检测到设备断连时会
在内部执行 stop/start 和重新解析设备。原采集线程退出时先 join，可能因这条隐式
重连路径拖住相机析构。已改用没有该重连分支的 `poll_for_frames`，空队列时进行
可被停止信号唤醒的 5 ms 等待，退出线程后再显式停流。构造期间 start 成功后的
异常也统一走相同清理路径。没有添加自动重置、自动恢复运动或放宽 0.2 秒 stale 门槛。

GenericPolicyPlayer 原先按成员逆序析构，会在关闭 LowState 订阅前销毁回调访问的
状态对象和 mutex，也可能在 DDS 析构阻塞时延迟相机释放。现显式先释放 P-BFM 相机、
关闭订阅和发布，再销毁其余成员；异常路径的安全阻尼仍在析构前执行。

帧新鲜度按取到 SDK 深度帧时的时间计算，避免预处理完成时间掩盖处理延迟。
stale 错误和每秒 command_state 增加 received_age_sec、received_frames、
rejected_frames、last_frame_number 和有效比例，区分没有收到帧与质量筛选拒绝帧。

验证：新 POSIX 回归在修复前复现三项失败，修复后四种场景均通过：子进程忽略
SIGTERM、子进程延迟清理、shell 提前退出、强制停止。controller、BFM、depth、
P-BFM lifecycle 共五项 CTest 通过；34 个 schema 与 loco/dance ONNX check-only
通过。本地常规构建及启用 RealSense SDK 的构建均通过。

当前 SSH 到 192.168.3.162 返回 `No route to host`，这些修复尚未同步/编译到真机。
首次运行中断流与 xHCI 错误是否由这些软件路径触发仍需相机连续采集和重复启停验证，
不能把本地回归通过写成真机持续稳定性已恢复。未启动真机运动。
