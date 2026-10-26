# ActionGo

ActionGo 是面向 Unitree G1 的模板化全身策略运行时：用 YAML 描述策略语义，经统一 player 完成模型校验、热备预启动、当前姿态保持、控制权切换、`LowCmd` 安全过滤和返回官方控制。它不是“张量维度相同即可运行”的自动适配器。

> **真机安全警告：**本项目可以发送底层控制指令，但不是安全认证产品。首次运行必须有物理支撑、隔离空间和可立即进入阻尼的操作员；先核对观测顺序、关节映射、增益、动作缩放和传感器坐标系，再按资产校验、schema、check-only、dry-run、仿真/回放、低风险真机的顺序推进。

英文主页 [README.md](README.md) 是完整的用户入口；本页只保留中文快速路径。

## 安装与构建

要求 CMake 3.22+、C++17、Unitree SDK2、Eigen3、yaml-cpp、zlib、OpenSSL 和 nlohmann/json。Ubuntu 可先安装系统开发头文件：

```bash
sudo apt install libeigen3-dev libyaml-cpp-dev zlib1g-dev libssl-dev nlohmann-json3-dev
```

CPU ONNX Runtime 不随源码仓库打包，使用固定版本和 SHA256 的下载器安装：

```bash
tools/deps/fetch_onnxruntime.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DACTION_GO_CODE_ONLY=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

OpenCV 和 libzmq 用于构建 sensor player；RealSense、AprilTag、MuJoCo、Redis 和 TensorRT 为对应 adapter 所需依赖。真机深度构建应启用 `-DACTION_GO_REQUIRE_REALSENSE=ON`。

## 模型与预检

代码版不分发第三方 ONNX、运动数据、标签或预编译二进制。根据 [资产获取说明](docs/model_assets.md) 从上游取得模型或提供匹配的本地导出，再按 [MODEL_ASSETS.json](MODEL_ASSETS.json) 校验。清单已同步本地动作包的 72 个文件，包含 paoxiao、PHP、P-BFM、mjlab、SONIC 原版与 v1.1 及运动数据；来源不明确的 checkpoint 继续标为待补，未在本地找到的 HandoffGuard risk model 需单独提供。`blocked_pending_review` 表示 BYOW，不代表允许再分发。

```bash
python3 tools/assets/check_model_assets.py --policy bfm0  # 严格检查已安装的指定动作包
python3 tools/assets/check_model_assets.py --installed-only
./build/policy_spec_schema_check --all --allow-missing-assets
./build/policy_spec_schema_check --all --installed-only
./build/policy_spec_schema_check --spec actions/bfm0/config/generic_bfm_zero.yaml
ACTION_GO_NET=lo tools/launch/start_dry_run.sh
```

`--allow-missing-assets` 用于没有第三方资产时审查 adapter 语义；schema 的 `--installed-only` 对已安装模型检查真实 ONNX I/O。资产检查的 `--installed-only` 核对现有文件，缺失项仅报告；`--policy` 可重复指定动作包。真机前对所选包运行严格资产、schema 和 player check-only；全部包均已安装时再运行不带筛选的严格检查。没有安装模型时的通过不代表部署已验证。

## 真机入口

Uni-SuperMimic 使用独立的 `uni_supermimic` variant，动作名为 `uni_supermimic_tracking`。
20 万轮模型的本地导出、MuJoCo 推理与接口说明见 [Uni-SuperMimic 适配](actions/uni_supermimic/README.md)。
`Right+A` 对应 `uni_supermimic_paoxiao`，`Right+B` 对应 `uni_supermimic_wbst_motive`。

```bash
ACTION_GO_NET=<机器人网卡> tools/launch/start_safe.sh
```

程序默认停在 `SHELL_IDLE`，不会立即接管。确认机器人已在官方 walkrun 或 damping 后再 ARM。`Start` 选择 BFM，`Left+B` 选择 UFO；按键时 IMU 有效且倾角不低于 `30 deg`，立即通过 ready/start 交接启动所选恢复策略；低于该阈值只启用动作和 IMU 监控，保持官方模式。

普通监控中，倾角不低于 `57.3 deg` 达 `60 ms` 后进入阻尼和 `SHELL_IDLE`。默认 `safety_auto_recovery: false`，需要操作员再次 ARM 请求恢复；人工 ARM 恢复不受自动恢复开关、摔倒去抖或摔倒角度上限限制。显式开启自动恢复后，检测到摔倒会在阻尼后自动运行所选策略。两种恢复路径均在新鲜 IMU 倾角不高于 `30 deg` 稳定 `1200 ms` 后切回 walkrun。恢复策略运行时不参与普通摔倒角度中断，运行失败则进入阻尼。`R2+A` 停止动作并回官方 walkrun，`Select+Start` 强制进入 passive/damping。OmniXtreme、SteadyTray、部分 BeyondMimic/UFO 恢复动作和 unitree_rl_mjlab 默认关闭 supervisor IMU interruption，必须在更强物理保护下单独审核。

### 默认遥控按键

先按 `Start` 或 `Left+B` ARM，再按动作组合键：

| 按键 | 行为 |
|---|---|
| `Start` | 使用 `bfm_zero_stand` ARM；IMU 有效且倾角 ≥30° 时立即启动恢复 |
| `Left+B` | 使用 `ufo_stand` ARM；IMU 有效且倾角 ≥30° 时立即启动恢复 |
| `R2+A` | 停止当前动作，回官方 walkrun，进入 `SHELL_IDLE` |
| `Select+Start` | 停止当前动作，进入官方 passive/damping 和 `SHELL_IDLE` |
| `R2+Down` | `kick_right` |
| `Left+X` | `hlqk`，当前模型为 `paoxiao.onnx` |
| `Left+A` | `ufo_tracking_zs7` |
| `Left+Y` | `dance1_subject2_mjlab` |
| `F1+A` | `wbst` |
| `Up+X` | `reverse_vault_flat_mjlab` |
| `Up+A` | `reverse_vault_terrain_mjlab` |
| `Up+Y` | `g1_parkour` |
| `Up+B` | `php_locomotion` |
| `Down+A` | `pbfm_loco` |
| `Down+B` | `pbfm_tracking_dance` |
| `Down+X` | `pbfm_pico_teleop` |
| `Right+A` | `uni_supermimic_paoxiao`，20 万轮 UniSuperMimic 跟踪 paoxiao |
| `Right+B` | `uni_supermimic_wbst_motive`，同一 UniSuperMimic 策略跟踪 wbst motive |
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

作者已经在 Unitree G1 真机上完成过运行测试；公开版本暂未附每个 adapter 的原始事件日志、固件/SDK manifest 和故障注入记录。英文主页的动作矩阵按已公开证据标注，未归档的项目保留 `Not documented`，不等同于安全认证或所有异常路径验证。

SONIC 当前动作默认使用 v1.1，原版配置仍保留；v1.1 的官方 replay 和板载验收、PHP 真机阶梯验收仍待完成。Bridge/HandoffGuard 开发功能保留，需要另行提供风险模型。详见 [SONIC](actions/sonic/README.md)、[PHP](actions/php/README.md)、[mjlab](actions/mjlab/README.md) 和 [风险切换预览](docs/risk_switch_preview.md)。

## 开发者入口

[P-BFM](actions/pbfm/README.md) 支持真实深度摇杆行走、离线 tracking 和 Pico UDP 全身遥操；
文本 context 随配置提供，ONNX 通过公开模型目录导入，适用上游 CC-BY-NC-4.0。
真机动作与 Pico 人体实流仍待验收。

- [新策略接入](docs/new_policy_onboarding.md)
- [运行时架构](docs/runtime_architecture.md)
- [依赖与版本](DEPENDENCIES.md)
- [贡献规范](CONTRIBUTING.md)
- [安全报告](SECURITY.md)

ActionGo 自有源码采用 Apache-2.0；模型、数据、Unitree 软件和第三方项目不自动受该许可证覆盖。
