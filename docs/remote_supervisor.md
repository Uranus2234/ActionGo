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

按 `Start` 后进入运行态，动作和 safety 才会生效。

## 默认遥控按键

```text
Start         -> ARM，进入运行态，启用动作/safety
R2+A          -> 停止当前模型，切官方 walkrun，回到 SHELL_IDLE/睡眠态，程序不退出
Select+Start  -> 切官方 passive，回到 SHELL_IDLE，程序不退出

R2+Down       -> kick_right
Left+X        -> hlqk
Left+A        -> ufo_tracking_zs7
F1+A          -> wbst
Up+Y          -> g1_parkour
Right+X       -> proto_walk
Right+Y       -> proto_dance
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
Instinct parkour 的 LowCmd 发送频率和目标更新按官方方式设置为 `publish_hz=50`、不插值、不限速。
当前 `actions/instinct` checkpoint 是官方 InstinctLab onboard 导出，部署时必须使用旧版 `instinct_onboard` 的 shoulder-first 关节顺序；C++ player 已按这个顺序发送和解析 LowCmd/LowState。若改成 InstinctMJ 的 waist-first 顺序，典型现象是 stand 刚开始能顶住一小会儿，随后手、腿、身体乱摆。

## IMU Safety

逻辑：

```text
任意动作/官方 walkrun 中
  -> 检测倾角 >= fall_angle_deg 持续 fall_hold_ms
  -> 中断当前动作，不先切回官方
  -> 启动 bfm_zero_stand
  -> 站稳 recover_hold_ms 后切回 walkrun
```

配置：

```yaml
safety_enable: true
safety_action: bfm_zero_stand
safety_return_mode: walkrun
fall_angle_deg: 57.3
recover_angle_deg: 20.0
fall_hold_ms: 60
recover_hold_ms: 200
safety_cooldown_ms: 1000
```

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

UDP 只能选择 `MotionActionController` 已加载的 action，不能上传 command、路径、
模型、YAML 或 shell 片段。action YAML 的 `command` 本身等同本机可执行代码，必须只
加载经过审查的受信配置。

上面的 HMAC 协议只保护 supervisor command plane。TWIST/Humanoid-GPT 的 realtime
reference 是独立的数据面，当前默认和实际支持范围都限于 loopback；非 loopback
reference 会直接拒绝。其带 sender identity、时间窗口和重放保护的独立认证协议仍是
后续工作，不能把 supervisor 命令的签名当作 reference 数据已经受保护。
