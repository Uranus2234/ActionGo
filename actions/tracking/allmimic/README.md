# AllMimic

独立 `runtime.variant: allmimic`，由 `generic_motion_tracking_player` 调用
`AllMimicController`，模型、参数和参考缓存均位于本目录。共享 DDS 发布、命令检查和
握手生命周期，观测、未来姿态旋转和残差目标由 AllMimic 自己的 controller 构建。

## 模型来源

当前包同时保留带 Omni 和不带 Omni 的两份模型，均来自
`20261026-wostate4096` 实验的 **159000** 轮存档：

| 按键 | 动作 | 模型 | 训练来源 |
| --- | --- | --- | --- |
| 未绑定 | `allmimic_paoxiao` | `model/policy.onnx` | `gpu1/ckpt_0159000.pt`，Omni 采样占 30%，seed 1 |
| 未绑定 | `allmimic_paoxiao_no_omni` | `model/policy_no_omni.onnx` | `gpu0/ckpt_0159000.pt`，无 Omni 动作，seed 0 |

两者均跟踪 paoxiao，部署参数和参考 CSV 经逐字节核对一致，直接共用
`config/AllMimic.yaml` 和 `motions/paoxiao.csv`。观测归一化随各自权重嵌入 ONNX。
当前两版 paoxiao 均去掉源动作前 1.12 秒的 T-pose 和放手过渡：50 Hz 下删除前 56 帧，
从原第 56 帧（零起始编号）开始，剩余 1329 帧，末帧时间为 26.56 秒。
原 0.5 秒处的肩 roll 仍约为 +77.5° / −82.0°；新首帧约为 +15.4° / −16.2°，
双臂已放下。裁剪起点经参考姿态的 MuJoCo 渲染核对。
参考 CSV、仿真 motion.npz 与两份 manifest 同步裁剪；训练库及 159000 轮权重不变。
`source_start_frame` 和 `trim_start_sec` 记录裁剪来源。wbst motive 不裁剪。
两个实验种子也不同，因此该对比不能将效果差异全部归因于 Omni 数据。
裁剪版参考状态的 PyTorch / ONNX Runtime 导出对照最大绝对误差分别为：
带 Omni 1.13e-6、无 Omni 1.43e-6。
用户提及的 158.7k 没有对应的 158700 存档；该实验每 1000 轮存档，使用最接近的
159000，不能当作精确的 158700 权重。
每段动作的 `motions/*.manifest.json` 记录 checkpoint SHA256、真实轮次、完整训练配置、
训练实现哈希、参考库及导出校验结果。模型与 CSV 使用仓库已有的 Git LFS 资产规则。

## 输入输出

- `obs: float32[1,1660]`：前 930 维为从旧到新的 10 帧历史，后 730 维为 10 帧未来目标。
- 单帧历史 93 维：29 个相对默认关节角、29 个关节速度、3 个骨盆局部角速度、3 个局部重力、29 个上次原始策略动作。初始历史重复首帧，上一动作初始化为零。
- 未来偏移为 1 到 10 个控制步，50 Hz 下覆盖 20–200 ms；单帧目标为位置 3、旋转 6、关节角 29、关节速度 29、参考局部线速度 3、参考局部角速度 3。
- 位置使用参考动作第一未来帧的 heading 坐标，z 为参考绝对高度；旋转使用当前 IMU 根姿态相对未来参考姿态，6D 顺序是第一列三项再第二列三项。初始 yaw 对齐后固定，不需要机器人全局位置或根线速度。
- `actions: float32[1,29]` 是原始策略动作，归一化参数嵌入 ONNX。腿部为 `q0 + action_scale * action`，上身为 `q_ref(t+1) + action_scale * action`。历史保留原始输出，不能存解码后的残差或关节目标。
- 参考 CSV 为 `allmimic_csv_v1`，每行 739 个数：730 个未来目标（其中旋转保存于参考世界坐标），再附当前参考根旋转矩阵 9 项。C++ 每步根据 IMU 和初始 yaw 对齐更新未来旋转，不能直接使用 Uni-SuperMimic 的参考缓存。

## 按键和交接

`config/remote_actions.yaml` 已取消 AllMimic 快捷键；→+A / →+B 改为四卡 92k mjlab combat
模型的 DuoBox 017 / A42 8 秒生成片段。AllMimic 动作和模型继续保留，可通过动作名调用。

这些真机入口均采用 `reference.end_behavior: stop`。先发布当前姿态 hold 并通知 ready，
supervisor 成功切入 UserCtrl 后通知 start；动作完成后保持最后安全命令并通知 complete，
supervisor 成功请求 walkrun 后重叠保持 0.30 秒，再通知 release 退出。失败时进入
`SwitchBackBlocked` 并保留 hold。握手与 RPC 返回不能替代真机物理控制权验证。
这些动作执行期间设置 `safety_enable: false`，与 OmniXtreme 的高动态动作一致。
原始 paoxiao 在约 9.08–9.48 秒倾斜超过 57.3°（裁剪后约 7.96–8.36 秒），
wbst 在约 6.46–7.02 秒越过该阈值，
最高约 177.6°；默认持续 60 ms 的倾倒判断会把正常动作误判为跌倒并进入阻尼。
此设置仅豁免当前动作的 IMU 倾倒判断；因此动作期间真实跌倒也不会按该阈值自动阻尼。
`R2+A` 回 walkrun、`Select+Start` 手动阻尼仍有效，结束回到 walkrun 后恢复普通倾倒判断。
命令有限值、LowState 超时、位置限位和 PD 力矩限制仍由 player 检查。

部署启用 `safety.preserve_pd_torque: true`，复用 Instinct 的 PD 力矩补偿过滤器。
训练策略可输出超出物理位置限位的虚拟 PD 目标，直接裁剪会改变控制力矩。
过滤器先按当前 q/dq 计算并限幅期望力矩，再限制发布的位置目标，通过 LowCmd 的
`tau` 补回差额；总力矩在当前反馈状态下与限幅后的训练 PD 一致。反馈延迟和执行器
动态仍需真机核实。历史继续记录原始 action，不能换成过滤后的目标或补偿力矩。

与现有策略对照的检查结果：

| 环节 | 对照结果 |
| --- | --- |
| 关节顺序、SDK 映射、PD、action scale | 与 Uni-SuperMimic 的 G1 29DoF 配置一致，未发现错位 |
| IMU | 使用 LowState 主 IMU 的骨盆姿态/角速度；不使用躯干 IMU 代替根姿态 |
| 观测与动作 | AllMimic 独立 1660 维观测、按列编码旋转、上身参考残差；不能复用 Uni 的 1720 维契约 |
| 频率 | 50 Hz 推理、500 Hz 命令发布；板端实际耗时需查看 runtime_metrics |
| 切入与切回 | 与 Uni 共用 ready/start、complete/release，0.10 秒切入混合、0.30 秒切回重叠 |
| 位置目标过滤 | AllMimic 开启上述 PD 力矩补偿，其余 generic variant 默认保持原行为 |
| 倾倒判断 | 与 OmniXtreme 的高动态动作一样按动作豁免，避免正常前倾/倒立触发阻尼 |

站姿启动的仿真已覆盖切入过程，但不能据此排除真机切入抖动：当前混合仅 0.10 秒，
`activation_blend_started.max_q_delta` 可检查当时姿态与策略目标的差距。尚未在机器人
板端检查推理时延、电机反馈延迟或物理控制权交接。

## 导出和预检

在包含 AllMimic、mjlab、PyTorch、MuJoCo、onnx、onnxruntime、numpy 和 PyYAML 的环境中：

```bash
python tools/export/allmimic_export_policy_onnx.py \
  --training-repo /path/to/allmimic \
  --checkpoint /path/to/ckpt_0159000.pt \
  --library /path/to/data/mixed/gpu1.npz \
  --clip user/paoxiao.csv --trim-start-sec 1.12 \
  --output-dir /path/to/export
```

`--trim-start-sec` 作用于本次导出的所有 `--clip`，且必须对齐 50 Hz 的整帧。
单独导出 wbst motive 时使用 `--clip omni/12_wbst_motive.csv` 并省略裁剪参数。

导出器严格检查任务编码、latent、历史和未来偏移，严格加载 Actor 和归一化权重，
使用 FP32 / ONNX opset 17。无 Omni 版导出时将 checkpoint 改为 `gpu0/ckpt_0159000.pt`、
library 改为 `data/mixed/gpu0.npz`，仅保留 `--clip user/paoxiao.csv`；输出的
`policy.onnx` 在部署包中命名为 `policy_no_omni.onnx`，来源记录在
`motions/paoxiao_no_omni.manifest.json`。
对首、中、末参考状态及固定随机输入进行 PyTorch/ORT 对照，
并检查 CSV 的 FP32 写回一致性。导出目录同时包含供本机闭环验证的 `scene.mjb` 与每段
`motion.npz`，部署 C++ player 只需要 ONNX、参数 YAML 和对应 CSV。

```bash
python3 tools/assets/check_model_assets.py --policy allmimic
for motion in paoxiao paoxiao_no_omni wbst_motive; do
  spec="actions/tracking/allmimic/config/generic_allmimic_${motion}.yaml"
  ./build/policy_spec_schema_check --spec "$spec" || break
  ./build/generic_motion_tracking_player --spec "$spec" --check-only || break
done
ACTION_GO_NET=enP8p1s0 tools/launch/start_safe.sh
```

## Dry run 延迟测量

在接收机器人 LowState 的机器上，编译更新后的 player，然后运行：

```bash
cmake --build build --target generic_motion_tracking_player -j4
python3 tools/runtime/allmimic_dry_run.py \
  --net enP8p1s0 --model both --duration-sec 10
```

将网卡名改为实际接收 LowState 的接口。脚本依次测试 →+A 的带 Omni 模型和 →+B 的
不带 Omni 模型，使用各自部署 spec，保持 50 Hz 推理、500 Hz 命令处理和默认 2 个
ONNX 线程。无需启动 remote supervisor 或切换 UserCtrl；该脚本不发 RPC，不创建
LowCmd publisher。Python 脚本仅依赖标准库。

测量链路为 LowState 回调入口 → 状态等待 → 观测构建 → ONNX → 动作解码 → 发布线程等待
→ 切入混合 / 安全过滤 / LowCmd 打包及 CRC。执行完命令生成后跳过 DDS Write，
因此这里的 E2E 不含传感器采样、回调前的 DDS / 网络传输、命令网络发送和电机响应。
`pipeline_ms` 从观测构建开始，不含输入状态年龄；`e2e_ms` 包含输入状态年龄。

每次推理只记录其命令的首次处理，避免把 500 Hz 重复处理计作独立样本。若发布线程
没来得及处理某次推理的目标，该次计入 `unmeasured_count`，不会伪造延迟样本。
默认剔除前 25 个推理步的预热样本，同时单独保留首步数据。输出每阶段的平均值、
P50/P95/P99/最大值、超过 20 ms 的次数和 deadline miss。
原始 JSONL、player 日志和 `latency.json` 位于
`outputs/allmimic_dry_run/<时间戳>/{omni,no-omni}/`，总表为 `summary.json`。

可用 `--model omni` / `--model no-omni` 单独测试；`--onnx-threads 1` 对照线程配置；
`--player /path/to/generic_motion_tracking_player` 指定构建目录；`--domain-id` 指定 DDS 域；
`--log-dir` 指定新的结果目录。
直接使用 C++ 时，增加 `--dry-run --dry-run-latency --event-log /path/to/events.jsonl`
和 `--duration-sec 10`。该延迟开关仅支持 AllMimic / SuperMimic，并要求 dry run，
普通 `--dry-run` 仍保持原行为。离线合成状态只用于短时功能测试，不代表 DDS E2E。

2026-10-27 本地测试使用 AMD EPYC 9374F、CPU ONNX Runtime、2 个线程，
以 500 Hz 经独立 DDS 域的 loopback 回放参考状态。各模型运行 10 秒，剔除前 25 步后
各有 475 个样本；最终测量如下，单位 ms：

| 模型 | ONNX 平均 | E2E 平均 | E2E P95 | E2E P99 | E2E 最大 |
| --- | --- | --- | --- | --- | --- |
| 带 Omni | 1.334 | 3.375 | 4.397 | 6.218 | 13.809 |
| 不带 Omni | 1.382 | 3.381 | 4.450 | 6.140 | 12.610 |

此次无漏测、无 E2E 超过 20 ms，`rt/user_lowcmd` 和 `rt/lowcmd` 均监测到零条命令。
前一次同环境测量曾出现约 28 / 43 ms 的 E2E 长尾，因此这些短时结果不能保证调度上界。
以上不是机器人板端或完整网络往返延迟；板端需使用同一脚本实测。

## MuJoCo 验证

实时可交互窗口复用原生 viewer，使用已导出包，不需要训练源码：

```bash
python tools/sim/uni_supermimic_mujoco.py --variant allmimic \
  --package /path/to/export --clip paoxiao \
  --player ./build/generic_motion_tracking_player \
  --sdk-root /path/to/unitree_sdk2_python --python-deps /path/to/dds/python/packages
```

`--clip 12_wbst_motive` 切到 wbst motive。空格暂停/继续，Backspace 手动重置，
G 切换旁边参考/重叠参考/关闭；默认 1 倍速，参考结束后持续推理末帧，无倒地自动暂停或重置。
支持 MuJoCo 原生鼠标选择与扰动。安装 VirtualGL 后可加 `--gpu --vglrun /path/to/vglrun`
进行 GPU 渲染；策略推理继续使用 C++ ONNX Runtime CPU。
增加 `--safety-profile hardware` 可在交互窗口启用真机的位置限位和 PD 力矩补偿；
默认 `simulation_parity` 用于对照训练。窗口仍使用逐步同步仿真和末帧持续推理，
完整真机配置的异步频率和结束握手请用下文的 deployment check 验证。

```bash
python tools/sim/allmimic_mujoco.py \
  --training-repo /path/to/allmimic --checkpoint /path/to/ckpt_0159000.pt \
  --package /path/to/export --clip paoxiao \
  --player ./build/generic_motion_tracking_player \
  --sdk-root /path/to/unitree_sdk2_python \
  --python-deps /path/to/dds/python/packages --log-dir /path/to/new/run
```

该入口为离屏闭环校验，用独立的非零 DDS domain 和 `lo`，逐策略序号执行 4×5 ms 物理步。
参考结束后持续推理，不因倒地重置或暂停。默认运行完整参考，也可用 `--steps` 延长验证末帧保持。
对 C++ 实际观测、原始动作、残差解码和 DDS 目标逐步调用训练 Task/Actor 对照，输出
`metrics.json`、`parity.json`、`trace.npz` 和 player 日志。此 variant 尚未经过真机验证。

以下为裁剪前参考的历史验证记录；裁剪后的结果见文末。159000 轮模型的本地完整闭环
结果如下，位置误差为世界坐标误差，未做平移对齐。
无噪声、无 push，原生 mjlab 模型在 MuJoCo CPU 中运行；这些结果不代表真机性能。

| 动作 | 控制步数 | 时长 | 关节 RMSE | 平均根位置误差 |
| --- | --- | --- | --- | --- |
| paoxiao | 1384 | 27.68 秒 | 0.1024 rad | 0.1005 m |
| wbst motive | 572 | 11.44 秒 | 0.1331 rad | 0.1628 m |

全部 1956 步通过训练 Task/Actor 对照；观测最大误差 1.91e-6、原始动作 9.60e-6、
残差解码 2.39e-7、DDS 目标误差为零。AllMimic 库中的 wbst motive 重采样为 573 帧，
与旧 Uni 缓存相差一帧，部署以本模型自己的缓存和来源清单为准。

另将 wbst motive 延长到 650 步（13 秒）验证末帧持续推理：全部通过逐步对照，
状态保持有限，没有自动重置或暂停。真机入口仍按上面的 stop 与握手机制结束。

### 真机配置的异步仿真检查

上述逐步对照采用仿真配置。以下入口直接读取本目录的真机 spec，在独立 DDS domain
与 `lo` 上以 500 Hz 异步发布命令，从站姿启动、初始 yaw 为 0.8 rad，执行完整动作，
收到 complete 后等待 0.30 秒再发送 release。它检查 player 和文件握手，不调用真机
UserCtrl / walkrun RPC，也不模拟 remote supervisor 的 IMU 判断。

```bash
python tools/sim/allmimic_deployment_check.py \
  --package /path/to/export --clip 12_wbst_motive \
  --player ./build/generic_motion_tracking_player \
  --log-dir /path/to/new/deployment-check
```

可用 `--pd explicit` 改为按 LowCmd 的 q/dq/kp/kd/tau 显式计算力矩，
或用 `--no-preserve-pd-torque` 复现原来直接裁剪位置目标的部署行为。
输出 `metrics.json`、`trace.npz` 和 player 日志；误差同时报告按固定 50 Hz 时间轴和
按 player 实际参考帧对齐的 `policy_phase_*`，避免把调度延迟全部算作姿态误差。
完整动作允许额外 10 秒完成交接；未收到 complete/release 或出现 safety_stop 时返回失败。
完整动作检查应有 `complete_seen: true`、`release_sent: true`、
`release_acknowledged: true`，目标关节角在限位内。

本机完整动作、站姿启动的对照结果（MuJoCo CPU，无噪声或外力）：

| 动作 / PD 模式 | 原关节 RMSE | 补偿后关节 RMSE | 原高度 RMSE | 补偿后高度 RMSE |
| --- | --- | --- | --- | --- |
| paoxiao / implicit | 0.1155 rad | 0.1086 rad | 0.0161 m | 0.0098 m |
| wbst motive / implicit | 0.1634 rad | 0.1346 rad | 0.0375 m | 0.0191 m |
| wbst motive / explicit | 0.1668 rad | 0.1372 rad | 0.0411 m | 0.0208 m |

三个完整动作均完成 complete/release 握手，发布位置目标均在限位内。wbst 自带倒立与
贴地段，不能仅凭最小根高度或最大倾角认定摔倒。这里的改进是部署链路的仿真结果，
不代表模型或真实电机已经过硬件验证。player 的 `runtime_metrics` 额外记录推理平均/
最大耗时及 deadline miss，板端应确认推理是否能持续满足 20 ms 周期。
最后一次 wbst 显式 PD 运行的本机平均推理约 1.27 ms，但出现一次约 40.8 ms 的峰值，
记录到 1 次 policy deadline miss；因此这些测试也不能证明板端或本机始终满足实时周期。
另一次 paoxiao 运行记录到 40 次 policy deadline miss，最大相邻推理间隔约 353 ms，
动作完成延迟约 0.8 秒，并使原先仅留 1 秒余量的测试器提前退出 release 等待。
测试器现已增加交接余量，但运行环境的调度延迟仍会影响跟踪，不能由延长等待解决。
修正等待后重跑完整 paoxiao，1385 次推理并完成交接；按策略进度对齐的关节 RMSE
为 0.1093 rad、高度 RMSE 为 0.0101 m，记录到 1 次 policy deadline miss。

无 Omni 的 159000 轮模型另行通过完整 paoxiao 检查：逐步同步验证 1384 步、27.68 秒，
关节 RMSE 0.1051 rad；训练 Task/Actor 对照最大误差为观测 9.54e-7、原始动作
1.38e-5、残差目标 2.38e-7，DDS 目标误差为零。站姿启动的真机过滤 / 显式 PD 异步
仿真完成 1385 次推理，complete/release 均成功并由 player 确认；关节目标在限位内，
按策略进度对齐的关节 RMSE 0.1127 rad、高度 RMSE 0.0096 m，记录到 1 次 policy
deadline miss。这些测试未包含真机。

### 历史验证：裁掉前 0.5 秒

带 Omni / 不带 Omni 的 paoxiao 均执行 1360 次推理；站姿启动、显式 PD、真机命令过滤
和异步 DDS 条件下，两个动作均完成 complete/release 握手并确认收到 release，
关节目标均在限位内。按策略进度对齐的关节 RMSE 分别为 0.1083 / 0.1157 rad，
高度 RMSE 为 0.0097 / 0.0095 m；各记录到 1 次 policy deadline miss。
这些结果只验证裁剪后的 MuJoCo 部署链路，尚未经过真机验证。

### 当前验证：裁掉前 1.12 秒

旧裁剪起点仍是双臂抬起姿态，因此两版参考统一改从原第 56 帧开始。
带 Omni / 不带 Omni 均通过站姿启动、显式 PD、真机命令过滤和异步 DDS 的完整仿真：
各完成 1329 次推理，complete/release 均成功并确认收到 release，关节目标均在限位内，
本次均未记录到 policy deadline miss。按策略进度对齐的关节 RMSE 分别为
0.1054 / 0.1103 rad，高度 RMSE 为 0.0096 / 0.0093 m。
新 CSV 与旧裁剪版再删除前 31 行逐字节相同，仿真 qpos/qvel 同步裁剪；
两份 ONNX 与原部署权重逐字节相同。这些结果尚未经过真机验证。
