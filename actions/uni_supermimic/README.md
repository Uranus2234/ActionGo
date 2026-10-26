# Uni-SuperMimic

`runtime.variant: uni_supermimic` 使用 Uni-SuperMimic 训练仓库导出的参考运动条件策略，
由 `generic_motion_tracking_player` 运行，属于 `motion_conditioned_tracking` 模板。
动作名为 `uni_supermimic_tracking`，已通过 `config/motion_actions.yaml` 注册。

该 variant 与独立复现的 `supermimic` 分别注册，拥有自己的配置、机器人参数和模型目录。
两者共享已验证的 99 维观测、10 帧历史、730 维未来参考及动作解码实现。
模型归一化保留在各自的 ONNX 内，不可互换 checkpoint 或机器人参数。

## 输入输出

- 输入 `obs: float32[1,1720]`，先放从旧到新的 10×99 观测，再放 10×73 未来参考。
- 每帧观测：关节角相对默认角 29、关节速度 29、骨盆局部角速度 3、局部重力 3、上一原始动作 29、相对参考骨盆旋转 6。
- 输出 `actions: float32[1,29]`；`q_target = default_angles + calibration + action_scale * actions`。
- 控制频率 50 Hz；参考缓存使用共同的 `supermimic_csv_v1` 格式，共 739 列。

## 导出与运行

在 ActionGo 仓库根目录，用已有 Uni-SuperMimic Python 环境运行：

```bash
python tools/export/uni_supermimic_export_policy_onnx.py \
  --training-repo /path/to/Uni-SuperMimic \
  --checkpoint /path/to/ckpt_200000.pt \
  --motion /path/to/3_back_hand_flip.csv --source-fps 50 \
  --output-dir /path/to/package
```

目前导出入口限定本地 200000 轮的兼容 checkpoint；使用 Uni 自己的 Actor、MDP、MotionLib 和 MJCF。
导出包中的 `SuperMimic.yaml` 是 Uni 训练模型的参数，部署时放在 `config/UniSuperMimic.yaml`，
`policy.onnx` 放在 `model/`，`reference.csv` 放在 `motions/`。
仓库通过 Git LFS 附带 200000 轮导出的 `model/policy.onnx`，以及 `motions/` 中的
默认后空翻、paoxiao 和 wbst motive 参考缓存；每个参考动作附带对应的来源清单。
克隆仓库后运行 `git lfs install` 和 `git lfs pull` 下载实际模型和轨迹。

```bash
./build/policy_spec_schema_check --spec actions/uni_supermimic/config/generic_uni_supermimic.yaml
./build/generic_motion_tracking_player \
  --spec actions/uni_supermimic/config/generic_uni_supermimic.yaml --check-only
```

当前工作区的原生 MuJoCo 入口：

```bash
bash ../ActionGo-local/run_uni_supermimic_200000.sh
```

仿真桥接器 `tools/sim/uni_supermimic_mujoco.py` 使用此 variant，并通过独立回环 DDS domain
将 MuJoCo LowState 送往 C++ ONNX Runtime，接收其 LowCmd。空格暂停/继续、Backspace 手动重置、
G 切换参考；无跌倒自动暂停或重置，动作结束后保持参考末帧并继续推理。
该仿真覆盖通信超时为 5 秒，真机 spec 保留运行时默认的 0.2 秒，且启用 supervisor IMU safety。

`tools/sim/uni_supermimic_validate_trace.py` 用 Uni 训练 MDP/Actor 对比实际输入、历史、动作和 DDS 目标。
20 万轮模型的 `3_back_hand_flip` 已在原有共享实现上验证：393 步、7.86 秒，
关节 RMSE 0.167 rad，平均根位置误差 0.037 m；该结果不代表所有动作或真机性能。

`feature` 分支的新 variant 也已完成 12 秒、600 步的 DDS/MuJoCo 闭环与训练 Actor 对照，
参考结束后持续推理，观测最大误差 1.67e-6、动作最大误差 1.29e-5、DDS 目标误差为零。
运行记录位于本地 `ActionGo-local/runs/feature-uni-supermimic-200000-12s/`。

遥控器绑定由 `config/remote_actions.yaml` 决定：

| 按键 | 动作名 | 参考动作 | 源帧率 | 仿真时长 | 关节 RMSE | 平均根位置误差 |
| --- | --- | --- | --- | --- | --- | --- |
| →+A | `uni_supermimic_paoxiao` | `paoxiao` | 按 120 Hz 解释 | 27.68 秒 | 0.108 rad | 0.291 m |
| →+B | `uni_supermimic_wbst_motive` | `12_wbst_motive` | 50 Hz | 11.46 秒 | 0.108 rad | 0.054 m |

两者使用同一 Actor，分别加载 `config/generic_uni_supermimic_paoxiao.yaml` 与
`config/generic_uni_supermimic_wbst_motive.yaml`，对应缓存为 `motions/paoxiao.csv` 与
`motions/wbst_motive.csv`，来源清单与缓存一起保存在仓库中。
两段完整 ActionGo/DDS/MuJoCo 仿真共 1957 步，均通过训练 MDP/Actor 对照。
paoxiao 存在全局位置漂移；双视窗视频的相机分别跟随各自骨盆，位置指标不做平移对齐。

原生窗口播放这两段动作：

```bash
bash ../ActionGo-local/run_uni_supermimic_motion.sh paoxiao
bash ../ActionGo-local/run_uni_supermimic_motion.sh wbst_motive
```

按键配置位于 `feature` 分支，尚未部署到机器人；↓+A/B/X 仍用于 P-BFM，↓+Y 仍空闲。
此 variant 尚未经过真机验证。
