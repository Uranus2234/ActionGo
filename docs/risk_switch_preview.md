# BeyondMimic 风险切换预览

`hlqk` 的模型改为 `actions/proprio/beyondmimic/model/hlqk.onnx`；动作名和 `Left+X` 按键保持不变。

本机运行：

```bash
export HANDOFFGUARD_RISK_MODEL=/path/to/risk_mlp.onnx
export WBST_TRAINING_ENV=/path/to/matching/training/params/env.yaml
bash tools/launch/preview_risk_switch.sh paoxiao
bash tools/launch/preview_risk_switch.sh wbst
```

一次只启动一个仿真。终端按 `1`（不需要回车）请求切换；按 `v` 重新播放所选 BeyondMimic。
MuJoCo 窗口中也支持 `1` 和 `V`。关闭窗口或 Ctrl+C 结束。

## wbst 训练侧对齐

从 ActionGo 仓库根目录执行上述命令。风险 MLP 模型需另行提供；
`HANDOFFGUARD_RISK_MODEL` 指定预览使用的模型路径，不与 bridge.yaml 的 TCN 混用。
`WBST_TRAINING_ENV` 必须来自与部署 `wbst.onnx` 匹配的训练导出。
`wbst` 预览从指定的 `params/env.yaml` 提取精确 kp、kd、action scale、力矩上限和 armature。
移除通用 MuJoCo XML 额外的关节阻尼/干摩擦，使用仿真时间推进 50 Hz 参考帧。
仅预览的临时 spec 关闭目标插值、启动混合、初始 hold 和目标关节限幅，以匹配训练动作接口。
临时配置位于 `/tmp/actiongo-training-aligned-*`；真机配置不因此修改。
训练的物理步长为 0.005 s（4 倍降采样），当前 MuJoCo 保留 0.002 s（10 倍降采样）；
二者控制周期均为 0.02 s。MuJoCo 显式 PD 与 IsaacLab/PhysX 隐式执行器仍有差异，
因此这里是部署接口及关键动力学参数对齐，不宣称两个引擎数值完全等价。

50 Hz 读取 10 帧、每帧 93 维的风险输入：MuJoCo 关节位置 29、关节速度 29、
根部坐标系角速度 3、投影重力 3、当前控制器已下发的目标关节位置 29。
关节字段按模型的 MuJoCo 顺序排列，命令从 DDS 的 motor 顺序映射。
该观测顺序来自 `acb-final/tools/latent_bridge/final/current_risk_env.py`。
使用 2026-09-14 最新 MLP 的导出 ONNX；归一化和风险百分比计算已经包含在 ONNX 中。

两个预览入口均关闭倾角过大自动恢复（`--safety-disable`）。
请求后的 2 秒观察窗口内，连续 2 步风险严格低于 35% 才进入 `bfm_zero_stand`。
观察超时不停止原策略。BFM 沿用现有模型、latent 和控制配置；倾角严格低于 0.20 rad、
根部角速度模长严格低于 0.30 rad/s 连续 5 步后进入模拟 walkrun。
此仿真的 walkrun 是现有 `unitree_rl_mjlab` 策略，并非真机加密的官方服务。
仿真窗口和计步使用仿真时间；终端约每 0.1 秒墙钟时间输出当前风险、推理耗时和状态。

验证：paoxiao 与 wbst 各完成一次自动请求、BFM 恢复、walkrun 接管短测试。
这不是对所有动作时刻的成功保证。门控单测覆盖严格阈值、风险连续计数重置、
2 秒截止、第二秒内放行、超时重试和姿态连续计数重置。
