# 模型与运动资产获取

ActionGo 源码仓库保存 runtime、配置和资产清单，第三方模型优先从上游下载。
权重的公开下载地址不等于再分发许可；代码、模型和动作数据可能适用不同条款。
如果需要镜像上游权重，先核对模型文件本身的许可证，并保留许可证和来源。

## 当前本地清单

`MODEL_ASSETS.json` 已同步 2026-10-12 本地 `actions/` 中的 71 个模型、
运动、reference、机器人 XML 和标签文件。原版 SONIC 与 v1.1 分开记录。
清单里的校验值标识当前部署文件；`upstream_revision: unknown` 表示尚未确认
对应资产版本，不把同级代码仓库的 commit 当成模型版本。

本地没有找到 `actions/handoffguard/model/handoffguard_risk_tcn.onnx`，开发配置
引用的外部路径也不存在。它记录在 `external_assets`，只用于可选 Bridge 实验；
清单没有伪造其校验值或下载地址。Bridge 模型的获取和验收另行处理。

| 包 | 上游或获取方式 | 放置位置与限制 |
|---|---|---|
| BeyondMimic | [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)；当前 paoxiao 为本地 checkpoint，公开下载待补 | `actions/beyondmimic/model/`；`hlqk` 当前引用 `paoxiao.onnx`，旧 `hlqk.onnx` 单独保留 |
| BFM-Zero | [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) | `actions/bfm0/model/FBcprAuxModel.onnx`；固定 latent prompt 已随配置提供 |
| UFO | [UFO G1 权重](https://huggingface.co/xuewang/ufo-g1-policy) | `actions/ufo/model/FBcprAuxModel.onnx`；tracking 还需要 `config/ufo_tracking_zs7.npy` |
| Instinct | [Instinct](https://project-instinct.github.io/)；取得与旧 instinct_onboard 语义一致的模型对 | `actions/instinct/model/` 和 `stand_onboard/model/`；encoder 与 actor 必须配套 |
| Holosoma | [Holosoma](https://github.com/amazon-far/holosoma) | `actions/holosoma/model/ppo_g1_29dof_dancing.onnx` |
| Humanoid-GPT | [Humanoid-GPT](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT)；准确 checkpoint 版本待补 | `model/pns_wo_priv216.onnx`、`motions/` 和 `assets/` |
| mjlab | [MJLab](https://github.com/mujocolab/mjlab)；[包说明](../actions/mjlab/README.md) 记录本地训练 checkpoint | `actions/mjlab/model/`；当前没有这些本地导出的公开下载地址 |
| OmniXtreme | [OmniXtreme](https://github.com/Perkins729/OmniXtreme) | 三个 ONNX 和 `actions/omnixtreme/motions/`，包含完整 motion 与各 segment |
| PHP | [student-assets-v1](https://github.com/amazon-far/php_parkour/releases/tag/student-assets-v1) | `actions/php/model/`，两个文件的本地校验值与上游 release manifest 一致 |
| ProtoMotions | [ProtoMotions](https://github.com/NVlabs/ProtoMotions) | `model/unified_pipeline.onnx` 和 `motions/*.npz` |
| SONIC | [GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) | 原版在 `actions/sonic/model/`，v1.1 在 `model/sonic_v1_1/`；两版共用已记录的 V2 planner；准确资产 revision 待补 |
| SteadyTray | [SteadyTray](https://github.com/AllenHuangGit/SteadyTray) | `model/policy_9999.onnx`；标签源图与 printable PDF 也已记录 |
| TWIST | [TWIST](https://github.com/YanjieZe/TWIST) | `model/twist_general_motion_tracker.onnx` 和 `motions/*.pkl` |
| unitree_rl_mjlab | [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)；当前为本地导出 | `actions/unitree_rl_mjlab/model/policy.onnx`；公开 checkpoint 标识待补 |

这些来源说明项目关系，不保证上游当前默认下载就是清单对应的文件。
取不到相同版本时，需要先恢复训练和原部署语义，重新验证配置；不能仅按文件名替换。

## 一个包的安装与预检

先只安装准备使用的包。以 BFM 为例，从上游取得匹配模型并放到表中路径后：

```bash
python3 tools/assets/check_model_assets.py --policy bfm0
./build/policy_spec_schema_check --spec actions/bfm0/config/generic_bfm_zero.yaml
./build/generic_policy_player --spec actions/bfm0/config/generic_bfm_zero.yaml --check-only
```

`--policy` 可以重复，例如 `--policy bfm0 --policy ufo`。
按包严格校验会要求该包清单中的全部文件，包括记录的 motion 和回归资产。
只有部分文件时，先使用 installed-only 核对现有文件，再对准备运行的 spec
做严格 schema 与 player 检查；installed-only 不能代替完整包校验。

PHP 可使用上游公开下载器：

```bash
git clone --recurse-submodules https://github.com/amazon-far/php_parkour.git ../php_parkour
git -C ../php_parkour checkout 554edcc33393fbd8bd2a725a781ffb867f6c467a
python3 ../php_parkour/scripts/download_assets.py student --destination "$PWD/actions/php/model"
python3 tools/assets/check_model_assets.py --policy php
./build/policy_spec_schema_check --spec actions/php/config/generic_php.yaml
./build/generic_sensor_policy_player --spec actions/php/config/generic_php.yaml --check-only
```

SONIC 上游持续更新默认模型，目前没有确认本地 original / v1.1 对应的完整
HF revision，因此不提供把最新默认下载直接当作旧版或 v1.1 的命令。
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
