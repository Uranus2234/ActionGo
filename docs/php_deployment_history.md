# PHP 部署与实验历史

本文归档原 PHP README 的内容，保存实现来源、测量和阶段性验证记录。
文中的“当前”、本机路径、设备状态和验收结论均属于各次记录时的情况，可能已经过期；
本文不作为当前操作指南，也不代表真机运动已验收。
安装和使用请阅读 [PHP 使用指南](../actions/php/README.md)。

---

# PHP 深度 locomotion 接入

复用 `depth_conditioned_actor/php` 和 `generic_sensor_policy_player`。
主要对照 Instinct 的 DDS、命令安全过滤、当前姿态 hold、ready/start、activation blend
和 supervisor 切回路径。另对照 Holosoma 的 G1 关节参数和 SteadyTray 的感知输入初始化。
按键经用户批准为 `Up+B`，动作名 `php_locomotion`，结束/异常切回 `walkrun`。

## 策略运行期间查看深度与角度

在运行 ActionGo 的本机终端执行：

```bash
bash tools/remote/monitor_php_depth.sh
```

该入口自动连接 `unitree@192.168.3.162` 并同步转发脚本。默认旁路读取
`/tmp/action_go_php_depth.sock`，可在 PHP 启动前或运行中打开；没有策略帧时等待。
显示正在用于推理的 58×87 float32 深度和对应 IMU 的地面拟合角度，最高 10Hz。
不另开 RealSense、不启动策略、不发布 LowCmd；关闭窗口只停止旁路监控。
发送使用非阻塞 Unix datagram，无接收端或接收拥堵时丢弃监控帧，不等待接收端。
同一机器人只能打开一个旁路接收窗口。

本机 Python 需要 NumPy、Pillow 和 Tk，机器人 Python 需要 NumPy。
默认使用同级 PHP 项目的 `.venv-php-sim/bin/python`，否则使用 `python3`；
可用 `PHP_MONITOR_PYTHON` 指定解释器。连接配置可通过 `PHP_ROBOT_HOST`、
`PHP_ROBOT_ROOT` 和 `PHP_ROBOT_NET` 覆盖；默认机器人路径为
`/home/unitree/action_go`、DDS 网卡为 `eth0`。

同级 PHP 原入口 `bash run_php_depth_live.sh --policy` 也可旁路查看。
省略该入口的 `--policy` 是独立相机采集，不能与策略同时使用。
角度估算要求白框内可见水平地面；运动、蹲下和箱面会影响估算，安装角应静止核对。

仅记录遥控器和电机命令、不占相机的启动诊断：

```bash
python3 tools/sim/diagnose_php_motion.py --net eth0 --duration 600 \
  --output outputs/php_startup_watch/lowcmd.jsonl
```

记录包含命令接收间隔、增益、估算合力矩及实测关节状态。订阅端频率不能代替
发布端频率，也不能仅根据 `kp=0` 判定阻尼，需结合 `kd`、`tau` 和命令新鲜度。
启用 `--trace-file` 的推理和最终 LowCmd 记录改为有界队列异步写出，
增加时间戳、状态序号和实际力矩字段；若写盘失败或队列拥堵会报告记录丢失。
目前切入偶发失去支撑及真实阶梯验收仍未解决；本次监控功能不构成运动验收。

## 模型与构建

模型是同级 PHP 项目公开的 `student-assets-v1` 双 ONNX。
本机已放入 `actions/php/model/`；模型文件按仓库规则忽略。
重新获取时在同级 `php_parkour` 执行其下载器，再复制到此包：

```bash
python ../php_parkour/scripts/download_assets.py student
mkdir -p actions/php/model
cp ~/.cache/php-parkour/student/*.onnx actions/php/model/
sudo apt install libzmq3-dev pkg-config
cmake -S . -B build
cmake --build build --target generic_sensor_policy_player policy_spec_schema_check runtime_hardening_test
./build/policy_spec_schema_check --all
./build/generic_sensor_policy_player --spec actions/php/config/generic_php.yaml --check-only
ACTION_GO_NET=lo tools/launch/start_dry_run.sh
```

## 契约与来源

源码来源均为同级 PHP 固定版本 Holosoma：
`thirdparty/holosoma/src/holosoma_inference/holosoma_inference/`。

| 语义 | 来源 | 此包配置或执行语义 |
|---|---|---|
| 双模型 | 发布 ONNX、`policies/depth_distillation.py` | `depth_image[1,58,87]→depth_latent[1,32]`；`obs[1,140]+time_step[1,1]→actions[1,29]` |
| 观测顺序 | `config/config_values/observation.py`、`BasePolicy`、`DepthDistillationPolicy` | 上次 action29、gyro3、q-default29、dq29、torso gravity3、指令15、latent32；scale=1、单帧历史 |
| 时间 | `prepare_obs_for_rl` | 每次输入零；六项 reference 输出不作为控制目标 |
| 关节 | ONNX joint_names、官方 robot 配置 | canonical G1、identity map、sign=1；加载时检查名称一致 |
| 动作 | ONNX action_scale、`rl_inference`、`BasePolicy.policy_action` | scale=1、clip±100、加default q；真机/仿真共用pd_torque，在物理target范围内补偿官方饱和PD力矩 |
| 姿态/增益 | 官方 `g1_29dof_wbt_distillation` | policy default 与 stiff startup 独立保存；policy kp/kd 取官方部署配置 |
| 启动 | Instinct `run_startup_transition` | ready 前持续当前姿态 hold；start 后 1秒现有线性姿态过渡与0.5秒保持，目标为官方 stiff pose，再启用 PHP |
| 重力/角速度 | `_get_anchor_quat`、训练 `terms/wbt.py:get_base_ang_vel` | 真机/仿真均用secondary torso quaternion，不重复腰部补偿；gyro保持LowState pelvis frame |
| 深度 | Holosoma D435i preset、`DepthShmPlugin` | 仿真SHM消费5046个float；真机复用Instinct rs2 pipeline，加PHP裁剪/bicubic antialias/归一化 |
| 失效 | 官方 depth heartbeat | 首帧/heartbeat 必须在 ready 前到达；seq不推进超过0.5秒或非OK状态退出；继承 supervisor 回退 |
| 周期 | 官方 rl_rate、原 ActionGo publish | 推理50Hz、发布500Hz，全部 LowCmd 经 `CommandSafetyFilter` |

共享安全增益上限显式为 kp=350、kd=20，包含官方 stiff startup kp350。
启动使用现有 ActionGo 过渡；与官方 Python 的2秒 smoothstep姿态/增益渐入不同。
这不改变 ready/start 握手，记录此差异以便实机准备阶段评估。

## ActionGo 驱动的 MuJoCo

仿真伴随脚本和本机验证说明一并保存在 `integration/php_parkour.patch`，
应用方法见 `integration/README.md`。

在同级 PHP 仓库运行：

```bash
bash run_actiongo_sim.sh
bash run_actiongo_sim.sh --headless --output "$PWD/outputs/php_stairs_acceptance"
```

复用官方 Holosoma MuJoCo模型/深度插件，以及 ActionGo `NativeActionGoBridge`、
`SimActionManager`。walkrun 发布 `rt/lowcmd`，PHP 发布 `rt/user_lowcmd`。
manager 等 ready 后切门控、停止 walkrun，再写 start；站稳后按同一 manager 切回。
仿真中 walkrun 使用 `unitree_rl_mjlab`，它不等同真机 Unitree walkrun 服务。
场景默认官方阶梯、无吊架，2秒请求PHP，6秒发高速前进及官方离散方向指令。
全部12个障碍后站立4秒，切回walkrun，再观察4秒；最多100秒。

本机结果：28秒平地站立/行走/站稳切回成功；PHP段最大倾角约11.8°，未跌倒。
此前target对比也截断官方结果，已撤回。修正比较后发现阶梯543/1000帧被物理
target限位改变，最大差3.976rad；这会改变PD攀爬力矩。
`generic_php_sim.yaml` 通过显式 `simulation_model_action` 限位保持官方目标语义，
全部命令仍经共享安全过滤，只允许lo/SHM。模型clip范围与物理限位分别表达，
不修改 `PHPParams.yaml` 的物理运动范围。
上述是历史诊断配置；正式与仿真默认现均使用 `pd_torque`，详见下节。
1000个同输入快照官方原方法对照：obs3.87e-7、action9.57e-7、target1.04e-6，kp/kd一致。
仿真配置已完成两次官方全场12/12障碍、无跌倒、站稳切回；录像与逐障碍证据在
同级PHP `outputs/php_full_course_v4/`。11/12绕过障碍的试验明确判失败。
最终入口复测见 `outputs/php_stairs_acceptance_repeat/`，终点倾角2.09°。
官方Warp训练深度与MuJoCo深度同姿态7点几何核对均值差约0.01，保留官方10Hz部署链路。
深度中断的退出/回退单独采集日志，详见同级 PHP 的本机验证说明。

## 当前边界

已完成`lo`仿真与机器人`eth0`只读推理，尚未发布机器人LowCmd。真机YAML使用RealSense与必需torso IMU，
仿真YAML使用SHM与同款secondary IMU消息，控制参数一致。
真实相机安装、标定、DDS时延和电机响应仍须在机器人上实际采集。
不能直接将 Instinct 的深度裁剪、归一化、多帧history应用到PHP。

## 真机同语义控制与感知

`pd_torque` 为共享 `CommandSafetyFilter` 的显式opt-in，其余策略保持原行为。
每次500Hz发布时计算官方 `kp*(q_raw-q)-kd*dq` 并限制到官方torque_limits，
求等价目标角后施加物理关节限位，通过LowCmd.tau补偿截断损失。
若保留原增益无法同时满足目标角与feedforward力矩限制，该次命令采用
kp=kd=0、tau=有界期望力矩。比较指标为发布快照的最终合力矩，不能只比较q。
NaN/Inf、物理目标角限位、LowState时间戳和kp/kd上限继续保留。

相机为D435i Z16 848×480@10Hz，按真实标定内参投影到官方106×60光线，
PHP取样10Hz；crop `[2:,4:-4]`、0.3–3m clamp、PyTorch等价bicubic antialias
到58×87、far-only clamp、低于0.15置far，再做`(d-.3)/2.7-.5`。
RealSense零读数置far，整帧无测距不更新有效时间；至少3个有效帧后就绪。
深度超时0.5秒、torso IMU超时0.2秒、LowState超时0.2秒。
相机实际安装应为torso偏移`(0.01,0.01,0.44)m`、RPY=`(1,27,1)°`；
软件投影不能补偿错误安装。标定视野覆盖不完整会拒绝初始化。

同控制配置全场复测结果见同级PHP `outputs/php_hardware_parity_stairs_v2/`：
12/12、无跌倒、站稳切回；3386个最终LowCmd快照力矩误差最大1.78e-5Nm，
物理目标角与feedforward限位通过。9组5046像素预处理对照最大误差5.97e-7。
最新双IMU全场及断流结果见同级PHP `docs/LOCAL_SIM.md`。
本机以librealsense2.57.5、RSUSB编译了启用相机的路径；因没有相机，
实际相机检查明确失败于`no RealSense depth camera detected`，未伪报硬件通过。

## Jetson 真机入口

沿用已验证Instinct的G1 29DoF、Jetson Orin、默认网卡enP8p1s0与SDK安装。
当前验收机器人DDS实际网卡为`eth0`，通过`ACTION_GO_NET=eth0`指定。
模型按上节下载与复制，librealsense的Jetson构建见`actions/instinct/README.md`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DACTION_GO_ONNXRUNTIME_DIR="$PWD/thirdparty/onnxruntime-linux-aarch64-1.22.0" \
  -DACTION_GO_REQUIRE_REALSENSE=ON \
  -DACTION_GO_REALSENSE_ROOT="$HOME/librealsense/install"
cmake --build build -j4 --target generic_sensor_policy_player action_go_remote_test policy_spec_schema_check
ACTION_GO_NET=eth0 bash tools/launch/check_php.sh
```

必须看到`realsense=enabled`。检查脚本运行schema、模型、无DDS相机检查及
10秒只读DDS推理，不发布LowCmd、不切UserCtrl；日志在`outputs/php_hardware_check/`。
网卡可通过`ACTION_GO_NET`覆盖。只读检查通过后按原有入口运行：

```bash
ACTION_GO_NET=eth0 ACTION_GO_EVENT_LOG=/tmp/actiongo_php.jsonl bash tools/launch/start_safe.sh
```

机器人在现有walkrun稳定站立、摇杆回中时按`Up+B`。首帧/传感器就绪才交接。
摇杆给官方离散方向，R1高速档，松开摇杆为stand；退出沿用原有切回机制。
软件/仿真一致性不代表机器人已完成相机、电机或阶梯验收。

具体操作：官方walkrun稳定站立、摇杆回中，按一次`Start`启用，再按一次`Up+B`，
等待启动姿态过渡完成。左摇杆前推为低速前进（command 1）；按住`R1`再前推为
高速前进（command 6），松开R1回到低速。左右摇杆均回中才是stand（command 0）。
右摇杆触发离散左右方向，不是连续偏航速度；摇杆幅度不连续调节PHP速度。
`R2+A`停止PHP并切回官方walkrun；`Select+Start`切官方阻尼，机器人需要外部保护。

## 相机安装偏差与真机运动未通过（2026-10-10）

用户报告真机晃动、对台阶没有预期抬腿反应。用户确认测试区域为水平地面，
相机沿用Instinct安装。新增3秒只读采集150帧，不发布LowCmd、不接管控制。
对后半段75个状态样本中的图像下部地面拟合，光轴相对水平下俯中位数43.56°，
躯干pitch约-1.75°，因此相机相对躯干下俯约45.30°。重复使用的10Hz图像不应
被视作75个独立相机帧。地面距离约1.260m，逆深度拟合RMSE约0.00337/m。
这是基于水平地面和较小躯干roll的估计，不能替代安装测量。

离线复现（Python环境需要numpy、PyYAML、onnxruntime、matplotlib）：

```bash
python tools/sim/diagnose_php_depth.py outputs/php_terrain_diagnostic/inference.yaml \
  --actiongo-root "$PWD" --output outputs/php_terrain_diagnostic/analysis
```

输出模型实际输入深度图、水平地面假设下的角度估计和固定状态的深度敏感性结果。

PHP官方`G1FlatRsD435iConfig`要求相对`torso_link`偏移(0.01,0.01,0.44)m，
RPY=(1,27,1)°；训练pitch随机范围只有±3°。当前约18°的pitch偏差明显超出
训练安装范围，必须先按官方位置/角度调整并复核，不能凭相机启动和内参投影通过
认定视觉一致。调整安装会影响沿用原相机的其他策略，测试对应策略前需恢复其标定安装。

固定真实状态和上一帧action，仅将模型深度替换成全远平面，stand/low forward/
high forward的腿部action最大差分别约0.227/0.221/0.163rad，说明模型使用了深度，
不证明真实台阶动作正确。当前主要待修复项是安装几何一致性；电机控制动态是否另有
问题仍待安装纠正后的运动数据验证，不能将晃动全部归因于相机或认定已修复。

日志新增实际`policy_command=编号(方向/档位)`及指令变化，另记录当前最大dq和
整个运行期dq峰值的关节名。原`cmd=(vx,vy,yaw)`属于通用遥控显示，不能用于确认
PHP离散指令；原`dq_max`为整个运行期峰值，不表示持续速度。
先复核安装、只读推理，再在外部保护下验证平地稳定和阶梯；真机运动验收仍未通过。

随后用户两次调整安装：第一次相对躯干下俯约24.0°；第二次2秒只读采集100个
状态样本，后半段50个样本的相对躯干下俯中位数约28.04°（光轴相对水平25.83°，
躯干pitch约-2.22°），与官方27°相差约1.04°，已在训练pitch随机范围内。
中央障碍物进入图像下部，因此角度工具改用图像下部左右外侧四分之一区域拟合地面；
最新拟合逆深度RMSE约0.00297/m。仍需水平地面条件，此结果只验证俯仰角，
没有验证相机相对躯干的位置、roll/yaw或真机运动效果。采集LowCmd trace仍为0字节。

## 本机真实输入 dry-run 验收（2026-10-10）

当前D435i连接为USB 2.1，仅支持848×480深度最高10Hz；原30Hz请求无法启动。
配置已固定为官方模型使用的10Hz深度输入，不改变分辨率、标定投影或预处理。
实测标定fx=fy=424.834、ppx=427.099、ppy=243.413，完整覆盖训练裁剪视野。
在Jetson独立`build-php-dryrun`目录编译，已通过PHP双spec/ONNX检查、
相机检查、共享命令安全测试及10秒真实LowState/secondary IMU/深度推理。
该次推理500帧、平均7.625ms、首轮最大10.097ms，LowCmd trace为0字节。
这项验收不接管控制、不验证运动效果，真机阶梯仍待实际运动验收。

复现当前设备检查（不发布LowCmd）：

```bash
ACTION_GO_NET=eth0 PHP_BUILD_DIR=build-php-dryrun \
  PHP_LOG_DIR=outputs/php_hardware_acceptance bash tools/launch/check_php.sh
```

## USB链路变化后的启动修正（2026-10-10）

本次SSH检查机器人已重启，ActionGo/PHP进程均未运行，官方FSM为802。
LowState、secondary IMU及官方LowCmd均有新鲜数据，PHP的user_lowcmd无发布。
机器人源代码、PHP参数和双ONNX与本机逐文件比较一致。

D435i当前枚举为USB 3.2，848×480 Z16支持6/15/30/60/90Hz，不支持10Hz。
旧配置直接请求10Hz会报`Couldn't resolve requests`，不能完成prestart。
RealSense源现在仅在PHP请求10Hz且该profile无法解析时尝试同分辨率30Hz；
只有SDK确认30Hz可用才采用。USB 2支持10Hz时保持原采集配置。
30Hz采集仍由原有采样门限降到约10Hz后进入模型，投影、裁剪、归一化、50Hz推理、
500Hz发布、安全过滤和ready/start交接均未修改。

机器人以30Hz采集完成10秒真实输入dry-run，500帧回放官方策略的观测最大误差
7.04e-7、action最大误差3.89e-7、原始target最大误差3.49e-7，kp/kd一致。
自动profile选择也已完成10秒只读推理，LowCmd trace仍为0字节。
这些结果解决当前相机启动失败，不证明之前晃动或不前进已经解决；
真机运动及阶梯仍未验收。当前机器人应显式使用`ACTION_GO_NET=eth0`，
通用启动脚本默认enP8p1s0在这台机器上不存在。

## 完整trace的控制时序修正（2026-10-10）

原完整PHP YAML trace在推理线程同步序列化5046个深度值后才提交target，机器人
只读实测平均约8.6ms。现改为后台队列写入，LowCmd trace也在后台序列化；推理及
发布线程仅复制快照。原trace字段保留，增加时间戳、state_seq、骨盆quaternion及
躯干IMU年龄。日志队列满或写失败时明确报告不完整，不等待日志磁盘写入。
进一步补充同一LowState快照的电机估计力矩：推理trace的`tau_est`以及最终命令
trace的`state_tau_est`。两者按策略关节顺序及符号记录，不参与观测或命令计算。
`tau_est`是电机反馈估计值，不能当作外部力矩测量；旧抖动记录没有此字段。

新版机器人DDS dry-run完整记录200帧，平均1.771ms、最大2.347ms，LowCmd发布为0，
与官方Python回放action最大误差3.97e-7。模型、深度处理、增益、CommandSafetyFilter
及ready/start交接保持原配置。此项通过的是记录时序和只读推理验证，严重抖动、
不抬腿以及真机阶梯越障尚未验收，不能据此直接恢复unsafe运动测试。

普通文件后台trace的本机全阶梯复测完成12/12、无跌倒、站稳并交还控制，使用
unsafe外壳配置。窄箱体的额外FIFO日志负载对照出现倒地，后台版也存在失败，
因此日志延迟不能被认定为本次严重抖动的唯一根因。详见同级PHP
`docs/REAL_DEPTH_BOX_DIAG.md`中的本轮对照及适用边界。

新增力矩字段的只读验证曾在退出时触发CycloneDDS 0.10.2的`prevent_callbacks`
断言。已将LowState及secondary IMU reader在持有引用时显式`close()`，然后释放
队列和订阅对象，避免DDS回调持有最后引用时执行析构。话题、队列长度与回调未变。
修复后两次3秒DDS dry-run各记录150帧、正常退出、无deadline miss、LowCmd发布为0；
末段平均1.78/1.88ms。新版实际输入官方回放action最大误差3.54e-7；补充字段和
退出修复后的30厘米箱体仿真亦完成越过、站稳及交还控制。新版已安装到机器人入口。
