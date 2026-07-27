# ActionGo

ActionGo 是面向 Unitree G1 的模板化全身策略运行时：用 YAML 描述策略语义，经统一 player 完成模型校验、热备预启动、当前姿态保持、控制权切换、`LowCmd` 安全过滤和返回官方控制。它不是“张量维度相同即可运行”的自动适配器。

> **真机安全警告：**本项目可以发送底层控制指令，但不是安全认证产品。首次运行必须有物理支撑、隔离空间和可立即进入阻尼的操作员；先核对观测顺序、关节映射、增益、动作缩放和传感器坐标系，再按资产校验、schema、check-only、dry-run、仿真/回放、低风险真机的顺序推进。

英文主页 [README.md](README.md) 是完整的用户入口；本页只保留中文快速路径。

## 安装与构建

要求 CMake 3.22+、C++17、Unitree SDK2、Eigen3、yaml-cpp、zlib 和 OpenSSL。CPU ONNX Runtime 不随源码仓库打包，使用固定版本和 SHA256 的下载器安装：

```bash
tools/deps/fetch_onnxruntime.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

OpenCV 决定是否构建 sensor player；RealSense、AprilTag、MuJoCo、ZMQ、Redis 和 TensorRT 均为按 adapter 选装依赖。

## 模型与预检

代码版首发不再分发许可未确认的 ONNX、运动数据、标签或第三方二进制。请根据 [MODEL_ASSETS.json](MODEL_ASSETS.json) 从权利方取得所需文件并核对 SHA256；`blocked_pending_review` 表示 BYOW（自行提供权重/数据），不代表允许再分发。

```bash
python3 tools/assets/check_model_assets.py
./build/policy_spec_schema_check --all --allow-missing-assets
./build/policy_spec_schema_check --all --installed-only
./build/policy_spec_schema_check --all       # 真机前严格模式
ACTION_GO_NET=lo tools/launch/start_dry_run.sh
```

`--allow-missing-assets` 用于没有第三方资产时审查 adapter 语义；`--installed-only` 只对已安装资产做真实 ONNX I/O 校验；严格模式要求全部资产存在且匹配。

## 真机入口

```bash
ACTION_GO_NET=<机器人网卡> tools/launch/start_safe.sh
```

程序默认停在 `SHELL_IDLE`，不会立即接管。确认机器人已在官方 walkrun 或 damping 后再 ARM。`R2+A` 停止动作并回官方 walkrun，`Select+Start` 强制进入 passive/damping。OmniXtreme、SteadyTray、部分 BeyondMimic/UFO 恢复动作和 unitree_rl_mjlab 默认关闭 supervisor IMU interruption，必须在更强物理保护下单独审核。

### 默认遥控按键

先按 `Start` 或 `Left+B` ARM，再按动作组合键：

| 按键 | 行为 |
|---|---|
| `Start` | 使用 `bfm_zero_stand` ARM，启用动作和 IMU safety |
| `Left+B` | 使用备用 `ufo_stand` recovery backend ARM |
| `R2+A` | 停止当前动作，回官方 walkrun，进入 `SHELL_IDLE` |
| `Select+Start` | 停止当前动作，进入官方 passive/damping 和 `SHELL_IDLE` |
| `R2+Down` | `kick_right` |
| `Left+X` | `hlqk` |
| `Left+A` | `ufo_tracking_zs7` |
| `F1+A` | `wbst` |
| `Up+Y` | `g1_parkour` |
| `Right+X` | `proto_walk` |
| `Right+Y` | `proto_dance` |
| `R1+F1` | `holosoma_dance` |
| `F1+X` | `omni_seg04_wbst_stand` |
| `F1+Y` | `omni_seg12_wbst_motive` |
| `F1+B` | `omni_seg11_side_flip_roll` |
| `L1+Right` | `twist_offline_pkl` |
| `L2+Right` | `twist_nokov_realtime`，使用本机 UDP reference |
| `F2+X` | `sonic_offline_squat` |
| `F2+Y` | `sonic_pico_teleop`，使用本机 PICO ZMQ bridge |
| `L2+Down` | `lie_down_beyondmimic`，结束后进入 passive shell |

[`config/remote_actions.yaml`](config/remote_actions.yaml) 是最终事实来源。每次真机运行前都应核对，因为本地部署可能有意修改或禁用按键。

UDP 群控默认关闭并默认绑定 `127.0.0.1`。空地址、wildcard、非 loopback、广播或组播均按不可信处理并必须配置仓库外的 HMAC-SHA256 密钥；UDP 只能选择已注册 action，不能上传 shell 命令或配置。实时 TWIST/Humanoid-GPT reference 默认也只监听 loopback；独立的远程 reference 鉴权协议尚未实现，因此非 loopback reference 目前会直接拒绝。详见 [远程 supervisor 文档](docs/remote_supervisor.md)。

作者已经在 Unitree G1 真机上完成过运行测试；公开版本暂未附原始事件日志、固件/SDK manifest 和故障注入记录，因此动作矩阵中的状态表示为“作者已测试，证据待归档”，不等同于安全认证或所有异常路径验证。

## 开发者入口

- [新策略接入](docs/new_policy_onboarding.md)
- [运行时架构](docs/runtime_architecture.md)
- [依赖与版本](DEPENDENCIES.md)
- [贡献规范](CONTRIBUTING.md)
- [安全报告](SECURITY.md)

ActionGo 自有源码采用 Apache-2.0；模型、数据、Unitree 软件和第三方项目不自动受该许可证覆盖。
