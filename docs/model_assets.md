# 模型与运动资产获取

ActionGo 源码仓库保存 runtime、配置和资产清单，第三方模型优先从上游下载。
权重的公开下载地址不等于再分发许可；代码、模型和动作数据可能适用不同条款。
如果需要镜像上游权重，先核对模型文件本身的许可证，并保留许可证和来源。

## 当前本地清单

`MODEL_ASSETS.json` 记录模型、运动、reference、机器人 XML 和标签文件。
原版 SONIC 与 v1.1 分开记录；本轮不处理 UniSuperMimic。
清单里的校验值标识当前部署文件；`upstream_revision: unknown` 表示尚未确认
对应资产版本，不把同级代码仓库的 commit 当成模型版本。

本地没有找到 `actions/handoffguard/model/handoffguard_risk_tcn.onnx`，开发配置
引用的外部路径也不存在。它记录在 `external_assets`，只用于可选 Bridge 实验；
清单没有伪造其校验值或下载地址。Bridge 模型的获取和验收另行处理。

| 包 | 上游或获取方式 | 放置位置与限制 |
|---|---|---|
| BeyondMimic | [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)；当前 paoxiao 为本地 checkpoint，公开下载待补 | `actions/proprio/beyondmimic/model/`；`hlqk` 使用 `hlqk.onnx` |
| BFM-Zero | [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) | `actions/tracking/bfm0/model/FBcprAuxModel.onnx`；固定 latent prompt 已随配置提供 |
| UFO | [UFO G1 权重](https://huggingface.co/xuewang/ufo-g1-policy) | `actions/tracking/ufo/model/FBcprAuxModel.onnx`；tracking 还需要 `config/ufo_tracking_zs7.npy` |
| Instinct | [Instinct](https://project-instinct.github.io/)；取得与旧 instinct_onboard 语义一致的模型对 | `actions/perception/instinct/model/` 和 `stand_onboard/model/`；encoder 与 actor 必须配套 |
| Holosoma | [Holosoma](https://github.com/amazon-far/holosoma) | `actions/proprio/holosoma/model/ppo_g1_29dof_dancing.onnx` |
| Humanoid-GPT | [Humanoid-GPT](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT)；模型已固定 commit 和 SHA256 | `model/pns_wo_priv216.onnx` 可自动下载；`motions/` 和 `assets/` 另行获取 |
| MJLab | [MJLab](https://github.com/mujocolab/mjlab)；[包说明](../actions/proprio/mjlab/README.md) 记录本地训练 checkpoint | `actions/proprio/mjlab/model/`；当前没有这些本地导出的公开下载地址 |
| MJLab Combat | [包说明](../actions/tracking/mjlab_combat/README.md)；本地四卡 92000 轮 checkpoint | `actions/tracking/mjlab_combat/model/policy_92000.onnx` 和修复版 DuoBox 017 / 034 CSV，共用同一策略 |
| OmniXtreme | [OmniXtreme](https://github.com/Perkins729/OmniXtreme) | 三个 ONNX 和 `actions/tracking/omnixtreme/motions/`，包含完整 motion 与各 segment |
| PHP | [student-assets-v1](https://github.com/amazon-far/php_parkour/releases/tag/student-assets-v1) | `actions/perception/php/model/`，两个文件的本地校验值与上游 release manifest 一致 |
| P-BFM | [P-BFM public demo](https://github.com/Roboparty/P-BFM/tree/671fbafe71a5f58bab77eeb5d969051fd2d283f4/interactive)；[包说明](../actions/perception/pbfm/README.md) | `actions/perception/pbfm/model/policy.onnx`，checkpoint `364822528`；分片自动合并校验。NC 派生 context 不进入公开快照，按包说明单独导入 |
| ProtoMotions | [ProtoMotions](https://github.com/NVlabs/ProtoMotions) | `model/unified_pipeline.onnx` 和 `motions/*.npz` |
| SONIC | [GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)；已固定 HF revision | 原版在 `actions/tracking/sonic/model/`，v1.1 在 `model/sonic_v1_1/`；两版共用已记录的 V2 planner |
| SteadyTray | [SteadyTray](https://github.com/AllenHuangGit/SteadyTray) | 可下载上游 `model/policy_9999.pt`；部署用 ONNX 仍需转换验证；标签源图与 printable PDF 也已记录 |
| TWIST | [TWIST](https://github.com/YanjieZe/TWIST) | 可下载上游 `model/twist_general_motion_tracker.pt`；部署用 ONNX 仍需转换验证，motion 另行获取 |
| unitree_rl_mjlab | [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)；当前为本地导出 | `actions/proprio/unitree_rl_mjlab/model/policy.onnx`；公开 checkpoint 标识待补 |

这些来源说明项目关系，不保证上游当前默认下载就是清单对应的文件。
取不到相同版本时，需要先恢复训练和原部署语义，重新验证配置；不能仅按文件名替换。

## 统一下载器

`tools/assets/fetch_assets.py` 从清单中的 `downloads` 配方读取固定 URL、
Git/HF revision 和 SHA256。目前覆盖 11 个策略族的 18 个文件：16 个可直接部署的
ONNX，以及 TWIST、SteadyTray 的 2 个原始 TorchScript checkpoint。

| 可下载策略族 | 内容 |
|---|---|
| `bfm0`、`ufo`、`holosoma`、`humanoid_gpt`、`protomotions` | 各自匹配现有清单的 ONNX |
| `sonic` | 原版和 v1.1 的 encoder/decoder，以及共用 planner，共 5 个 ONNX |
| `omnixtreme` | 3 个 ONNX |
| `php` | 从固定 release 压缩包提取 2 个 ONNX |
| `pbfm` | 从固定 commit 下载 4 个分片，合并为 1 个 ONNX |
| `twist`、`steadytray` | 各 1 个 `.pt`；下载完成不表示 ONNX 已就绪 |

```bash
python3 tools/assets/fetch_assets.py --list
python3 tools/assets/fetch_assets.py --policy bfm0 --dry-run
python3 tools/assets/fetch_assets.py --policy bfm0 --policy php
python3 tools/assets/fetch_assets.py --all
python3 tools/assets/fetch_assets.py --policy bfm0 --verify
```

`--all` 只下载已注册的公开文件，不代表所有 adapter 的 motion、传感器和 reference
资产齐全。Instinct 的准确公开文件、BeyondMimic/mjlab 等本地导出、UFO tracking latent
及其他缺少匹配下载地址的文件会保留为手动获取项。`--list` 会给出这些项的来源说明。
公开仓库中存在同名模型，也不能直接替换当前配置对应的模型。

默认缓存在仓库 `.cache/assets`，可用 `--cache /path/on/data-disk` 或
`ACTION_GO_ASSET_CACHE` 调整位置。已有文件匹配时跳过；Git LFS 指针可自动替换。
其他不匹配的已有文件需要显式 `--force`，新内容校验通过后才原子替换，下载或校验失败
会保留原文件。`--verify` 不访问网络，只检查选中的可下载文件。

下载行为不更改清单中的再分发限制；公开快照仍排除模型文件。

## 一个包的安装与预检

先只安装准备使用的包。以 BFM 为例：

```bash
python3 tools/assets/fetch_assets.py --policy bfm0
python3 tools/assets/check_model_assets.py --policy bfm0
./build/policy_spec_schema_check --spec actions/tracking/bfm0/config/generic_bfm_zero.yaml
./build/generic_policy_player --spec actions/tracking/bfm0/config/generic_bfm_zero.yaml --check-only
```

`--policy` 可以重复，例如 `--policy bfm0 --policy ufo`。
按包严格校验会要求该包清单中的全部文件，包括记录的 motion 和回归资产。
只有部分文件时，先使用 installed-only 核对现有文件，再对准备运行的 spec
做严格 schema 与 player 检查；installed-only 不能代替完整包校验。

PHP 同样使用统一下载器：

```bash
python3 tools/assets/fetch_assets.py --policy php
python3 tools/assets/check_model_assets.py --policy php
./build/policy_spec_schema_check --spec actions/perception/php/config/generic_php.yaml
./build/generic_sensor_policy_player --spec actions/perception/php/config/generic_php.yaml --check-only
```

SONIC 下载固定 HF revision 中与本地清单匹配的 original / v1.1 文件，
不随上游默认分支更新。
使用对应版本的 encoder、decoder 和 observation config，不混用 low-latency 版本。

## 代码版与本地开发检查

```bash
# 只检查已安装文件；报告缺失项，没有安装任何模型时明确提示。
python3 tools/assets/check_model_assets.py --installed-only

# 源码版允许 BYOW 文件缺失，同时核对任何存在的文件。
python3 tools/assets/check_model_assets.py --permit-missing --public-release

# 全量部署环境：要求所有清单文件存在并匹配。
python3 tools/assets/check_model_assets.py
```

资产校验不证明 observation、joint order 或动作效果正确。
继续执行包 README 中的 sensor/reference 检查、dry-run、仿真和有保护的真机测试。
