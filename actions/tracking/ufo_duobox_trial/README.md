# UFO DuoBox 临时真机验证包

动作 `ufo_duobox_050_ma_trial`，遥控器 **L1+A**。
这是用户选择的“自由对练 · Ma · 验证集”前约 10 秒片段，使用四卡训练的
`ufo_mjlab_4gpu_week` / 55,694,336 transitions 检查点。
独立目录、注册、按键、导出器及必要的通用 runtime 改动放在同一个提交中。

## 资产和来源

- `model/actor.onnx`：FP32 actor，包含本检查点的 learned observation normalizer，Git LFS 管理，约 122 MiB。
- `motions/duobox_050_ma_latents.npy`：499×256 FP32，50 Hz；使用同一检查点的 backward encoder 离线编码。
- `motions/duobox_050_ma_reference.npz`：500×36 reference qpos、29 关节名及 dt，root quaternion 为 wxyz；用于复现片段/仿真初始化。
- `motions/replay_fixture.json`：训练端 PyTorch 生成的 32 个非零 actor 样本，以及 16 帧 observation/history/action/target 对照。
- `manifest.json`：检查点、原始 motion、训练源文件、导出器和资产 SHA256。

原 motion 为 `duobox_g1_export__04_sparring_050_ma_free_liu_free__ma`，validation，
从第 0 帧开始。原预览含初始姿态及 499 个有效动作步；本包固定这一段，
不会播放完整 776 control-frame 轨迹或循环最终 latent。
训练全量 safetensors 含 critic/encoder/优化训练相关网络，约 3.2 GiB；部署只需 actor ONNX 和预编码动作。

用户明确要求将此临时验证资产上传至 ActionGo feature。
该授权不等于获得数据集/权重的通用再分发许可；全局资产清单保留
`blocked_pending_review`，不据此发布公共模型包。

## 训练语义与部署路径

训练源 UFO revision `0b89378de02f77712072d983bc43ec2a46455dc0`；
实际工作区关键源文件哈希另外记录，不能仅凭上游 revision 重建本地修改。

| 项目 | 对齐来源与部署值 |
| --- | --- |
| Robot/joints/default/gains | `boxing_bfm.runtime.build_env_config` 加载 `configs/robots/g1_29dof.yaml` 的 training overrides；29DoF 原生 Unitree 顺序，完整值在 `config/Robot.yaml` |
| Actor | `FBModel.act`：normalizer → actor mean；单输入 `actor_obs[1,721]`，输出 `action[1,29]`，opset 17 |
| State 64 | `HumanoidVerseMjlab._raw_actor_obs/get_observation`：q−default 29、dq 29、gravity 3、pelvis-local gyro×0.25 3；角度 rad，速度 rad/s |
| Last action 29 | 上一步 normalize×5 并 clip 至 ±5 的动作，初始全零 |
| History 372 | 四帧、field-major，按 actions/base_ang_vel/dof_pos/dof_vel/projected_gravity 排序，每字段最新在前；**previous_only**，读取后追加当前观测 |
| Latent 256 | `boxing_bfm.expert.encode_window`：t+1…t+8 的 state/privileged_state，经同一 normalizer/backward_map，平均后 project_z 至范数 16；每 20 ms 前进一步 |
| Action target | `_normalized_action/_build_robot`：default + `0.25 * effective effort_limit / kp * clip(5*actor_output,−5,5)`；gains/scale 从有效训练配置导出 |
| Limits | 关节范围来自相同 MJCF；力矩/速度/armature/friction 来自 `_g1_mjlab_mode15_actuator_params`。仿真使用 DC motor 速度相关饱和 |

使用现有 `single_actor_state/bfm_zero` 与 `generic_policy_player`。
通用 runtime 新增可选 `observation.history_timing`，原 BFM/UFO 默认 include_current，
P-BFM 继续 previous_only。非循环 latent 用尽后进入现有 completion/release barrier。
无效时序、缺失/错误维度/NaN latent 在 ready/接管前失败。

部署包含 5 步当前姿态 warmup、0.10 s activation blend、20 ms target interpolation；
硬件目标仍经过关节限幅及 CommandSafetyFilter。训练模拟器的目标本身没有此部署限幅。
这些启动/插值/限幅与原预览的 reference-state reset 有区别，不能把预览表现直接当作硬件表现。
结束后继续保持最后命令，supervisor 切 walkrun 并 release；正常发布 500 Hz，推理 50 Hz。
真机使用 `rt/user_lowcmd`，保留 IMU safety 和当前 supervisor 流程。

## 获取、构建与操作

```bash
git lfs pull --include='actions/tracking/ufo_duobox_trial/model/actor.onnx'
cmake -S . -B build
cmake --build build --target action_go_remote_test generic_policy_player policy_spec_schema_check bfm_policy_test -j
python3 tools/assets/check_model_assets.py --policy ufo_duobox_trial
python3 actions/tracking/ufo_duobox_trial/tests/check_trial.py
ACTION_GO_NET=lo tools/launch/start_dry_run.sh
```

在操作员现场确认实际网卡、保护及停止路径后，通过原 `tools/launch/start_safe.sh` 入口启动。
`Start` 启用外壳动作，站稳后 `L1+A` 播放一次；约 10 秒后切回 walkrun。
`R2+A` 停止当前动作并回 walkrun/睡眠外壳，`Select+Start` 切官方阻尼。
不要把 `lo` 当作真机网卡，也不要用裸 player 绕过 supervisor。

## 复现导出与仿真

在包含原 boxing/UFO 环境的机器中，从 boxing 根目录运行：

```bash
python /path/to/ActionGo/tools/export/export_ufo_duobox_trial.py \
  --boxing-root /path/to/boxing \
  --output /path/to/ActionGo/actions/tracking/ufo_duobox_trial
```

依赖原 UFO/boxing 源、PyTorch 2.7.1、NumPy 1.26.4、ONNX、MuJoCo、PyYAML 和检查点/缓存。
不需要在机器人上运行 backward encoder 或安装训练依赖。
CPU C++ 闭环回归需要 MuJoCo、unitree_sdk2py/CycloneDDS、NumPy/PyYAML：

```bash
python tools/sim/ufo_duobox_trial_smoke.py \
  --scene /path/to/UFO-main/humanoidverse/data/robots/g1_mjlab/g1_29dof.xml \
  --output /path/to/trial-output --initial default
```

仿真脚本固定使用 loopback、独立 DDS domain，输出 player 日志、events、rollout 和 result。
可用 `--initial reference`、`--stop-after 1`、`--drop-state-after 1` 检查不同路径。
它验证 player 与 completion/release barrier；实际官方 walkrun 服务只在真实 supervisor 中存在。

## 已完成的验证（2026-11-05，x86 CPU）

- 资产 SHA256、单包 schema/ONNX 初始化、32 非零输入 actor 数值及 16 帧 C++ history/action/target 回放通过，容差 `2e-4*(1+abs(reference))`。
- `ctest -R '^bfm_policy_test$'` 与负例检查通过：错误 history_timing、缺失 latent、错误 shape、NaN latent 均失败。
- 全量 schema 检查通过（allow-missing-assets）；51 specs，另有 61 个历史资产未安装，不能据此声称其他动作已验证。
- `start_dry_run.sh` 在 lo 上接收合成 Start/L1+A，110 个注册动作加载，选中本动作，ready/start/complete/release/Done 正常；不发布 LowCmd、不调用官方切换。
- 同一 C++ player 闭环参考初态：最低 root 0.573 m，最大倾角 17.67°，结束 root 0.739 m；默认站姿：最低 0.571 m，最大倾角 17.70°，结束 0.726 m。均完成有限播放，release 前额外保持 0.3 s。
- 1 s 手动 SIGTERM 正常退出；1 s 后断 LowState，于约 0.202 s stale age 被安全过滤拒绝并非零退出。

原 MJLab 预览：RMSE 0.1549 m，未摔倒，腕峰速 2.417 m/s（参考 3.323 m/s）。
此结果属于原训练环境，和上述 ActionGo C++ 仿真记录分开。
本机 CPU 通常约 5 ms 推理，但共享主机出现超过 20 ms 的尖峰；未完成目标板实时性或真机验证。
没有在物理机器人上运行本包。

## 真机验证后撤销

整个试验包只占一个提交，提交标题为
`Add reversible UFO DuoBox Ma hardware trial on L1+A`。
在包含该提交的 feature 上执行 `git revert <trial-commit>`，再 `git push origin feature`。
这会同时撤销模型引用/动作、按键、导出器、测试及 runtime 改动；保留其后的其他提交。
revert 不会从 Git/LFS 历史中清除已上传对象。本次尚未执行撤销。
