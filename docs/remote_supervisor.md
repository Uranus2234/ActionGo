# Remote Supervisor

`action_go_remote_test` 是真机推荐入口。它常驻运行，监听官方遥控器、UDP 命令和 IMU safety。

## 启动

推荐脚本启动，网卡集中写在 `tools/launch/start_safe.sh` 顶部：

```bash
cd ~/action_go
tools/launch/start_safe.sh      # 开启 IMU safety
tools/launch/start_unsafe.sh    # 禁用 IMU safety
tools/launch/start_dry_run.sh   # dry-run：不切 UserCtrl，不发 LowCmd
```

默认网卡是 `enP8p1s0`。如需临时覆盖：

```bash
ACTION_GO_NET=eth0 tools/launch/start_safe.sh
ACTION_GO_NET=enP8p1s0 tools/launch/start_safe.sh
```

脚本顶部已经把常见/当前机器网卡名集中成变量：`enP8p1s0`、`eth0`、`usb0`、`usb1`、`wlx782288b51cae`、`lo`、`docker0`。

Dry-run 模式说明：`--dry-run` 不会 `StopMove`、不会切 `UserCtrl`、不会请求官方 `walkrun/passive` 切换；动作 player 会加无 LowCmd dry-run 参数运行，用来检查模型/DDS/相机启动链路。

## tmux 后台 / 开机自启动

`tools/launch/start.sh` 用 tmux 后台运行 `tools/launch/start_safe.sh`：

```bash
sudo apt install tmux
cd ~/action_go
tools/launch/start.sh start
tools/launch/start.sh attach
tools/launch/start.sh logs
tools/launch/start.sh stop
```

安装 systemd 自启动。service 使用 `Type=forking` + PIDFile 跟踪 tmux；如果启动太早导致 DDS/网卡暂时失败，tmux 内会按非零退出码自动重试：

```bash
cd ~/action_go
tools/launch/install_action_go_service.sh --now
```

如果要把网卡固定写进 service：

```bash
ACTION_GO_NET=enP8p1s0 tools/launch/install_action_go_service.sh --now
```

`stop` 会先向 tmux 里的 supervisor 发送 Ctrl-C，让程序按正常退出流程切回 `WALKRUN`；超时才会 kill tmux session。

手动等价命令：

```bash
./build/action_go_remote_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --remote-map config/remote_actions.yaml
```

## Shell 模式

程序启动后默认进入：

```text
SHELL_IDLE
```

此时：

- 不执行动作
- 不启用 IMU safety
- 不切 UserCtrl
- 不接管机器人

按 `Start` 或 `Left+B` 进入 ARM 状态、选择 recovery backend，并启用动作和
IMU safety 监控。IMU 有效且倾角不低于 `recover_angle_deg`（默认 30°）时，
立即启动所选 BFM/UFO recovery，通过既有 ready/start 交接切入 `UserCtrl`。
低于该阈值或 IMU 无效时只 enable，机器人保持当前官方模式。
动作正在运行时 ARM 请求会被忽略。

## 默认遥控按键

```text
Start         -> ARM，选择 bfm_zero_stand；IMU 有效且倾角 >=30° 时立即恢复
Left+B        -> ARM，选择备用 ufo_stand；IMU 有效且倾角 >=30° 时立即恢复
R2+A          -> 停止当前模型，切官方 walkrun，回到 SHELL_IDLE/睡眠态，程序不退出
Select+Start  -> 切官方 passive，回到 SHELL_IDLE，程序不退出

R2+Down       -> kick_right
Left+X        -> hlqk
Left+A        -> ufo_tracking_zs7
F1+A          -> wbst
Up+Y          -> g1_parkour
Right+X       -> proto_walk
Right+Y       -> proto_dance
Right+A       -> allmimic_paoxiao，159000 轮 AllMimic Omni 混合模型
Right+B       -> allmimic_wbst_motive，同一 AllMimic 策略
Down+A        -> pbfm_loco，真实深度摇杆行走
Down+B        -> pbfm_tracking_dance，LaFAN dance 56.52–98.72 秒
Down+X        -> pbfm_pico_teleop，XRoboToolkit 实时全身遥操
R1+F1         -> holosoma_dance
F1+X          -> omni_seg04_wbst_stand
F1+Y          -> omni_seg12_wbst_motive
F1+B          -> omni_seg11_side_flip_roll
L1+Right      -> twist_offline_pkl
L2+Right      -> twist_nokov_realtime
F2+X          -> sonic_offline_squat
F2+Y          -> sonic_pico_teleop
L2+Down       -> lie_down_beyondmimic，结束后回 passive shell
```

按键配置在：

```text
config/remote_actions.yaml
```

Instinct Parkour 额外保护：启动 `g1_parkour` 时会先在官方 walkrun 仍接管的状态下初始化 RealSense、`rt/secondary_imu` torso IMU、ONNX/DDS，并发布当前姿态 hold；ready 后才切 UserCtrl。没有检测到 RealSense 深度相机、没有深度帧或没有 torso IMU 时，不会切 UserCtrl，不会启动策略，并请求保持/回到官方 `walkrun`。切入 UserCtrl 后执行 official ColdStartAgent-style 过渡到 `default_joint_pos`，然后默认直接进入 parkour 模型。
Instinct parkour 保持官方的 `inference_hz=50`、不做策略帧间插值且不限速；LowCmd
由独立发布线程以 `publish_hz=50` 重发最新策略目标，避免推理阻塞命令发布。
当前 `actions/instinct` checkpoint 是官方 InstinctLab onboard 导出，部署时必须使用旧版 `instinct_onboard` 的 shoulder-first 关节顺序；C++ player 已按这个顺序发送和解析 LowCmd/LowState。若改成 InstinctMJ 的 waist-first 顺序，典型现象是 stand 刚开始能顶住一小会儿，随后手、腿、身体乱摆。

P-BFM 三个入口使用 `generic_sensor_policy_player`，保留 ready/start 和 walkrun 回退。
相机位置与 Instinct 配置一致，光轴相对躯干下俯约48°；先运行
`bash tools/launch/check_pbfm.sh offline`。Pico 模式还需 PC 在线 sender，
人体输入和深度均须新鲜才 ready；配置与延迟说明见 `actions/pbfm/README.md`。

## IMU Safety

逻辑：

```text
ARM 后启用 supervisor IMU safety 的动作/官方 walkrun 中
  -> 检测倾角 >= fall_angle_deg 持续 fall_hold_ms
  -> 中断当前动作，切官方阻尼，回到未使能外壳
  -> 默认等待操作员按 Start/Left+B 人工请求恢复
  -> 若 safety_auto_recovery=true，阻尼后自动启动所选 recovery

人工 ARM：IMU 有效且倾角 >= recover_angle_deg
  -> 立即启动所选 recovery，不等待 fall_hold_ms
  -> 恢复期间忽略普通摔倒角度中断；payload 失败则阻尼
  -> 新鲜 IMU 倾角 <= recover_angle_deg 稳定 recover_hold_ms 后切回 walkrun
```

配置：

```yaml
safety_enable: true
safety_auto_recovery: false
safety_action: bfm_zero_stand
safety_return_mode: walkrun
fall_angle_deg: 57.3
recover_angle_deg: 30.0
fall_hold_ms: 60
recover_hold_ms: 1200
safety_cooldown_ms: 1000
```

ARM 与恢复沿用 Vincent `feature` 的行为：人工 ARM 只检查 IMU 有效和恢复角度，
不检查样本年龄，不等待摔倒去抖，也不受 `safety_auto_recovery` 限制；倾角超过
`fall_angle_deg` 仍允许人工恢复。普通摔倒监控跳过无效或超过 500 ms 的 IMU，
保留已有摔倒计时；倾角回落到摔倒阈值以下、禁用动作 safety、冷却期或状态重置
才清空摔倒计时。恢复回 walkrun 的计时遇到无效、过期或高于恢复阈值的 IMU
会清空，必须重新站稳满 1200 ms。恢复失败后进入阻尼，等待操作员再次 ARM。

## Generic 策略启动平滑与发布线程

四个 generic 入口默认读取：

```yaml
handoff:
  activation_blend_sec: 0.10
```

也可用 `--activation-blend-sec` 覆盖，合法范围为 `[0, 5]` 秒，`0` 表示关闭。
普通 ready/start 交权时，player 从最后一条经 `CommandSafetyFilter` 处理的 hold 命令
开始，在 100 ms 内对 `q/kp/kd` 线性过渡到最新策略目标；策略推理和 motion 时间在
blend 期间继续推进。warmup 和 current-pose hold 不消耗这 100 ms。transactional
preview commit 保持 candidate 首帧一致，因此显式旁路 activation blend，并记录
`activation_blend_bypassed=preview_commit`。

BFM、UFO、Unitree、ProtoMotions、TWIST、HumanoidGPT、SteadyTray、SONIC、
Instinct 和 OmniXtreme 使用独立发布线程。BeyondMimic、mjlab 和 Holosoma
保留同一循环中的推理与发布调度；推理耗时会延后该循环的命令发布。
各入口使用相同的 activation blend 接口，但具体启动顺序由策略配置决定。
独立发布路径的 `runtime_metrics` 包含发布次数、deadline miss 和最大间隔；
同循环路径的统计字段以对应 player 实际输出为准。

这些指标和 `activation_blend_started/completed` 只能证明进程内软件命令的顺序与
连续性，不能证明 7110/7111 后固件已经完成物理 ownership 切换。真机仍需结合独立
`rt/user_lowcmd` 时间戳、LowState 姿态/关节速度和可见卡顿进行验收。

如果 BFM-Zero 刚站起来就切 walkrun 后发软，可以增大：

```yaml
recover_hold_ms: 3000   # 或 5000
recover_angle_deg: 10.0
```

## 启动延时

如果希望程序 ready 后等几秒再启用主逻辑：

```bash
./build/action_go_remote_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --remote-map config/remote_actions.yaml \
  --startup-delay-sec 10
```

## UDP 控制

UDP 群控默认关闭；即使传入 `--udp-listen`，默认也只绑定 `127.0.0.1`。
本机 loopback 可以继续使用旧文本协议。任何非 loopback、广播或组播监听都必须
提供仓库外的 HMAC-SHA256 密钥，且密钥至少 32 字节、不能对 other 用户可读：

```bash
install -m 600 /dev/null /run/secrets/actiongo-udp.key
openssl rand -hex 32 > /run/secrets/actiongo-udp.key

./build/action_go_remote_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --remote-map config/remote_actions.yaml \
  --udp-listen --udp-bind 0.0.0.0 --udp-port 45678 \
  --udp-auth-key-file /run/secrets/actiongo-udp.key \
  --udp-auth-key-id lab-a
```

发送端生成带 `version/type/action/issued_at_ms/execute_at_ms/nonce/key_id/signature`
的 JSON。签名覆盖规范化字段；重复 nonce、过期命令、未来超过 60 秒、错误签名和
未注册 action 均拒绝：

```bash
python3 tools/remote/action_go_udp_send.py \
  --host <机器人IP> --action kick_right --start-after-ms 500 \
  --auth-key-file /run/secrets/actiongo-udp.key --auth-key-id lab-a
```

停止/阻尼命令在固定上限 64 的队列中有更高优先级。广播或组播共享一个组密钥，
能够同步控制多台机器人，但不能区分每个发送方；密钥必须在仓库外安全分发和轮换。

```bash
python3 tools/remote/action_go_udp_send.py --host <机器人IP> --cmd emergency \
  --auth-key-file /run/secrets/actiongo-udp.key --auth-key-id lab-a
```

`emergency` / `stop` 兼容命令和 `Select+Start` 都调用 `dampingStop()`：强制停止
当前策略，进入官方 passive/damping 并回到 `SHELL_IDLE`，程序继续监听。
停止不设置急停锁存；重新按 ARM 后可启动动作。`damping_exit` 使用同一停止实现，随后退出程序。

UDP 只能选择 `MotionActionController` 已加载的 action，不能上传 command、路径、
模型、YAML 或 shell 片段。action YAML 的 `command` 本身等同本机可执行代码，必须只
加载经过审查的受信配置。

上面的 HMAC 协议只保护 supervisor command plane。TWIST/Humanoid-GPT 的 realtime
reference 是独立的数据面，当前默认和实际支持范围都限于 loopback；非 loopback
reference 会直接拒绝。其带 sender identity、时间窗口和重放保护的独立认证协议仍是
后续工作，不能把 supervisor 命令的签名当作 reference 数据已经受保护。

## 切回 Walkrun 的完成屏障

正常手动停止在稳定 `Running` 状态下按 `7111 请求 -> API ack -> 300 ms overlap ->
停止 payload` 执行。7111 调用和 overlap 期间不会提前设置 cancel，BFM 等策略会继续
发布。若停止请求与尚未完成的 7110 UserCtrl RPC 竞态，仍会提前 cancel，并由 mode
mutex 保证延迟完成的 7110 后面一定跟随 7111。

BeyondMimic 有限动作到达末帧后不会立即退出。player 会冻结最后一帧经
`CommandSafetyFilter` 处理的 q/kp/kd，以配置的 500 Hz 持续发布并写 complete 文件；
supervisor 收到 7111 成功返回并完成 overlap 后写 release 文件，player 才停止 hold。
7111 失败时状态为 `SwitchBackBlocked`，payload 保持运行，可再次请求正常切回，也可被
emergency/passive damping 立即抢占。

事件中的 `switch_back_ret` 仅表示 RPC 返回；`handoff_overlap_started`、
`handoff_release_sent` 和 payload 的 `handoff_hold_stopped` 用于证明软件命令交接顺序。
它们不表示固件物理接管已确认。当前不使用 `LowState.mode_machine`、`GetFsmId` 或
`GetFsmMode` 作为 release 条件，因为仓库中没有它们与 7111 ownership 的可靠映射。
