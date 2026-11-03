# SuperMimic

## Uni-SuperMimic：OmniXtreme 混合数据 200000 轮

在 `feature` 分支中，该策略已独立注册为 `uni_supermimic`，配置与动作入口位于
[`actions/tracking/uni_supermimic`](../uni_supermimic/README.md)。以下本地仿真脚本使用此新 variant。

另提供 Uni-SuperMimic 的独立导出与仿真入口，不使用下文独立复现的旧权重。
`tools/export/uni_supermimic_export_policy_onnx.py` 使用 Uni 训练仓库的 Actor、MDP、MotionLib 和 MJCF，
导出 `ckpt_200000.pt`。输入仍为 1720 维，输出为 29 维；机器人 PD、默认角、动作缩放来自 Uni 训练模型。

本地环境可直接运行：

```bash
bash /root/epfs/program/wbc/ActionGo-local/run_uni_supermimic_200000.sh
```

默认跟踪 50 Hz 的 `3_back_hand_flip`，动作长 7.86 秒，窗口用 VirtualGL GPU 渲染；
CPU 上的 ActionGo C++ ONNX Runtime 通过回环 DDS 完成策略推理。
空格暂停或继续，Backspace 手动重置，G 切换参考模型。
无跌倒检测、自动重置或自动暂停；参考结束后保持末帧并继续推理。

模型包位于 `ActionGo-local/packages/supermimic-omni-200000/3_back_hand_flip/`。
`manifest.json` 记录精确 checkpoint、动作来源和 SHA-256。
`tools/sim/uni_supermimic_validate_trace.py` 用训练 MDP/Actor 对照 C++ 实际执行记录，
`tools/sim/uni_supermimic_render_trace.py` 将实际 DDS 闭环轨迹渲染为 50 fps 实时对照视频。
两者的参数见 `--help`；仿真入口为 `tools/sim/uni_supermimic_mujoco.py`。

本地完整动作验证目录为 `ActionGo-local/runs/uni-supermimic-200000-3-backflip/`：
393 次推理、1572 个物理步，关节 RMSE 0.167 rad，平均根位置误差 0.037 m；
与训练代码逐步比较，观测最大绝对误差 1.67e-6，动作最大绝对误差 1.29e-5，DDS 目标误差为零。
CPU 推理中位耗时 1.16 ms、P95 2.13 ms。这些结果只覆盖此动作和当前仿真条件，尚未进行真机验证。

## 独立 SuperMimic-MuJoCo 适配

以下适配使用 SuperMimic-MuJoCo 独立复现的 G1 29 DoF、`upstream`、`legacy99` 策略，当前检查点为 `policy_098000.pt`。这是参考运动条件化策略；能否跟踪新的动作需要单独验证，不能视为任意动作都可执行。这里的检查点不是 Uni-SuperMimic 官方权重。

## 本地导出与打包

在已有 SuperMimic 环境中运行，不需要训练环境、世界模型或 critic 参与在线推理。工具只读取本地源码、检查点和 CSV，不下载模型，也不安装或修改依赖。

在 ActionGo 仓库根目录执行：

```bash
SUPERMIMIC_REPO=/path/to/SuperMimic-MuJoCo
SUPERMIMIC_PYTHON=/path/to/existing/supermimic/.venv/bin/python
SUPERMIMIC_MOTIONS=/path/to/local/motions
SUPERMIMIC_PACKAGE=/path/to/local/packages/supermimic/dance1_subject1

"$SUPERMIMIC_PYTHON" tools/export/supermimic_export_policy_onnx.py \
  --source-repo "$SUPERMIMIC_REPO" \
  --checkpoint "$SUPERMIMIC_REPO/checkpoints/g1_29dof/policy_098000.pt" \
  --motion "$SUPERMIMIC_MOTIONS" \
  --clip dance1_subject1 \
  --output-dir "$SUPERMIMIC_PACKAGE"
```

`--motion` 可直接指定一个 CSV，此时可以省略 `--clip`。目录内有多个 CSV 时必须指定 `--clip`，接受文件名、去掉扩展名的名称或按文件名排序后的零基索引。每个包对应一个 clip；不同 clip 请使用不同输出目录。源 CSV 必须是无表头的 36 列：根位置 XYZ、根四元数 XYZW、29 个 SDK 顺序的关节角。默认源帧率为 30 Hz，其他帧率通过 `--source-fps` 指定。

工具调用原始 `MotionLib` 和 `G1Kinematics`，按照原代码重采样到 50 Hz，导出四个文件：

| 文件 | 内容 |
| --- | --- |
| `policy.onnx` | CPU FP32 Actor，保留检查点的观测和 goal 归一化参数 |
| `reference.csv` | 无表头、逗号分隔，每行 739 个数，每行对应一个 20 ms 控制时刻 |
| `SuperMimic.yaml` | ActionGo 机器人参数，`policy.num_obs=1720`、`policy.num_actions=29` |
| `manifest.json` | 输入输出 schema、来源和文件 SHA-256、机器人 contract、参考动作初始状态、数值验证结果 |

权重和参考动作缓存应保留在仓库外的本地包目录。相同命令可重新生成并覆盖包中的这四个文件；导出与检查通过后才替换目标文件。

## 与运行时的接口约定

ONNX 只有一个输入 `obs`，形状 `[1,1720]`，一个输出 `actions`，形状 `[1,29]`，全部为 FP32。输入列索引采用零基、左闭右开区间：

| 输入范围 | 内容 |
| --- | --- |
| `[0,990)` | 10 帧 × 99 维观测，按从旧到新排列，每帧内部按下表排列 |
| `[990,1720)` | 10 帧 × 73 维未来参考，从当前参考帧的下一帧开始 |

| 每帧观测范围 | 内容 |
| --- | --- |
| `[0,29)` | `q - default_angles - calibration`，rad |
| `[29,58)` | 关节速度，rad/s，不额外缩放 |
| `[58,61)` | 骨盆局部角速度，rad/s |
| `[61,64)` | 骨盆坐标系下的单位重力方向 `R_robot^T [0,0,-1]` |
| `[64,93)` | 上次原始策略输出，尚未乘 `action_scale` |
| `[93,99)` | `R_robot^T R_reference` 的 6D 旋转表示 |

6D 旋转顺序为 `[R00,R01,R10,R11,R20,R21]`。重置时先将上一动作清零，再用第一帧观测重复填满历史。默认 `calibration=0`；使用骨盆姿态，不套用其他 variant 的腰关节到躯干姿态变换。归一化已经在 ONNX 内，运行时不要重复归一化。

`reference.csv` 每行 `[0,730)` 是原始 `MotionLib.goal()` 的结果，每个未来时刻依次为位置增量 3、旋转 6、局部线速度 3、局部角速度 3、关节角 29、乘以 0.1 的关节速度 29。位置增量与旋转采用当前参考帧的 heading 坐标系；速度采用对应未来参考帧的骨盆坐标系。未来帧超出动作长度时，索引截断到最后一帧。

每行 `[730,739)` 是**当前参考帧**的骨盆到世界旋转矩阵，按行展开，供本体观测的最后 6 维使用。读取后应按 3×3 矩阵重建。若运行时对初始航向进行对齐，应将同一个世界坐标航向变换应用于该参考矩阵；不要改变已在 heading 坐标系下构造的 goal。

运行时 reference 类型为 `supermimic_csv_v1`，使用 `observation.obs_frame_dim=99`、`observation.history_len=10`、`reference.goal_frames=10`、`reference.goal_dim=730`。初始航向对齐采用 `yaw_alignment: initial`，末尾处理为 `end_behavior: hold`，策略计时为 `policy_step_source: wall_clock`。配置和动作入口分别见 `config/generic_supermimic.yaml` 与 `supermimic.yaml`；本工具只生成模型包和机器人参数，不生成这些运行时配置。

`SuperMimic.yaml` 的 `robot.policy_joint_order` 为 SDK 顺序，`joint_map=[0,...,28]`。`default_angles`、`kp`、`kd`、`action_scale`、`torque_limits` 从检查点兼容的 robot contract 导出，关节上下限来自同一 MJCF。输出转换为：

```text
q_target = default_angles + calibration + action_scale * actions
```

当前 upstream contract 的动作没有裁剪；不要在 Actor 内增加 tanh、限幅或输出平滑。运行时的命令检查和底层力矩限制应独立执行。

## ActionGo MuJoCo 闭环

在已有 SuperMimic Python 环境中启动桥接器。Python 侧加载原始 MuJoCo 场景与隐式 PD 执行器，通过独立 DDS domain 的回环接口发布 LowState；实际 Actor 推理由 `generic_motion_tracking_player` 的 C++ ONNX Runtime 执行，LowCmd 返回仿真。不要把此仿真桥接器连接到机器人网卡。

```bash
"$SUPERMIMIC_PYTHON" tools/sim/supermimic_mujoco.py \
  --source-repo "$SUPERMIMIC_REPO" \
  --motion "$SUPERMIMIC_MOTIONS/dance1_subject1.csv" \
  --package "$SUPERMIMIC_PACKAGE" \
  --player /path/to/build/generic_motion_tracking_player \
  --sdk-root /path/to/unitree_sdk2_python \
  --python-deps /path/to/local/dds/python/packages \
  --gpu
```

桥接器依赖原环境的 MuJoCo、PyTorch、MJLab，以及额外的 Unitree Python SDK 和 `cyclonedds==0.10.2`。额外 Python 包可安装在独立目录并用 `--python-deps` 指定。`--gpu` 使用 VirtualGL，默认查找相邻的 `SuperMimic-MuJoCo-local/dependencies/virtualgl`，其他安装位置通过 `--vglrun` 指定。GPU 加速窗口渲染；当前单机器人 MuJoCo 物理和 C++ Actor 在 CPU 上执行。

窗口使用原生分辨率和 MuJoCo 交互：空格暂停／继续，Backspace 手动重置机器人、参考、上一动作和历史，G 切换旁置参考／重叠参考／隐藏。双击选择机器人身体，Ctrl 加鼠标右键拖动施加外力，Ctrl 加左键拖动施加外力矩。拖动身体属于扰动物理仿真，不是直接修改关节角。

加 `--headless --duration-sec 10 --no-realtime --trace --log-dir /path/to/new/run` 可以执行离屏测试。每次新运行使用新的日志目录。运行生成 `metrics.json`、实际仿真 spec 和 C++ 事件日志；`--trace` 额外记录逐步观测、动作和 `trace.npz`，用于下面的数值对照。日常播放不需要开启详细 trace。`--duration-sec` 按仿真时间计数。锁步协议每个新策略序号执行四个 5 ms 物理步，重复发布不重复步进。

输入 CSV 的源帧率由模型包中的 `provenance.motion.source_fps` 指定，原 MotionLib 按时间插值到 50 Hz，不改变动作时长。窗口的 `--fps` 仅控制显示刷新率。实时模式累加固定物理步的截止时间，短暂延迟后补齐物理步；`status.json` 与 `metrics.json` 的 `realtime_factor` 表示仿真时间／有效播放墙钟时间，排除启动等待和手动暂停，正常实时播放约为 1。

观测和动作对照原模型的命令如下。它使用 C++ 实际读取的状态和上一动作，重新调用原 `MDP.obs()`、`MotionLib.goal()` 和 FP32 Actor，并检查 DDS 状态、目标、历史和参考索引：

```bash
"$SUPERMIMIC_PYTHON" tools/sim/supermimic_validate_trace.py \
  --source-repo "$SUPERMIMIC_REPO" \
  --checkpoint "$SUPERMIMIC_REPO/checkpoints/g1_29dof/policy_098000.pt" \
  --motion "$SUPERMIMIC_MOTIONS/dance1_subject1.csv" \
  --package "$SUPERMIMIC_PACKAGE" \
  --log-dir /path/to/completed/run
```

## 真机接入边界

先在 MuJoCo 中验证 ActionGo 的观测、关节顺序、时序和 PD 控制闭环。策略运行频率为 50 Hz，底层物理步长与其他物理设置见 `manifest.json` 的 `robot_contract`。`initial_state` 给出同一 clip 的首帧姿态和速度，供仿真初始化使用。真实机器人不能直接瞬间设置为这些状态。

仿真应持续进行策略推理和物理步进，不因倒地自动重置或暂停。动作到末尾时保持最后一行参考并继续推理；空格只用于手动暂停，Backspace 手动重置参考帧、历史、上一动作和仿真状态。这些行为由运行时和仿真器实现，导出包本身不会控制暂停或重置。

**SuperMimic 的 ActionGo 适配尚未经过真机验证。** 真机接入应复用其他已验证 variant 的 LowState/LowCmd、模式切换、预启动 hold、激活过渡、命令检查和退出流程；其他 variant 经过真机验证，不等于这个新策略已经验证。

## 导出验证范围

工具检查原始 checkpoint 与本地机器人 contract 的兼容性，严格加载 Actor 和归一化参数，固定 `AMP_DTYPE=None`、CPU FP32、ONNX opset 17。它检查 ONNX 结构，并在首帧、中间帧、末帧参考状态及一个固定随机样本上对照 PyTorch 输出，容差为 `atol=1e-5, rtol=1e-4`；CSV 写回后必须保持 FP32 数据完全相同。

已有 `onnxruntime` 时使用 CPUExecutionProvider，否则使用 ONNX ReferenceEvaluator 并在 manifest 中记录。工具不会安装缺失依赖。ReferenceEvaluator 数值验证不能替代 ActionGo 的 ONNX Runtime 执行、MuJoCo 闭环或真机验证。原播放器使用 BF16；FP32 导出与 BF16 的轨迹差异仍需通过闭环评估确认。
