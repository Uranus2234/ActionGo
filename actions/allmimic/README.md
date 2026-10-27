# AllMimic

独立 `runtime.variant: allmimic`，由 `generic_motion_tracking_player` 调用
`AllMimicController`，模型、参数和参考缓存均位于本目录。共享 DDS 发布、命令检查和
握手生命周期，观测、未来姿态旋转和残差目标由 AllMimic 自己的 controller 构建。

## 模型来源

当前包导出自本地 AllMimic 的 `20261026-wostate4096/gpu1/ckpt_0159000.pt`，实际轮次
**159000**，Omni 混合采样比例为 30%。用户提及的 158.7k 没有对应的 158700 存档；
该实验每 1000 轮存档，当前包使用最接近的 159000，不能当作精确的 158700 权重。
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

`config/remote_actions.yaml` 的 →+A / →+B 当前分别绑定 `allmimic_paoxiao` /
`allmimic_wbst_motive`。Uni-SuperMimic 的原动作和模型仍保留，可修改按键映射切回。

两个真机入口均采用 `reference.end_behavior: stop`。先发布当前姿态 hold 并通知 ready，
supervisor 成功切入 UserCtrl 后通知 start；动作完成后保持最后安全命令并通知 complete，
supervisor 成功请求 walkrun 后重叠保持 0.30 秒，再通知 release 退出。失败时进入
`SwitchBackBlocked` 并保留 hold。握手与 RPC 返回不能替代真机物理控制权验证。
IMU safety 默认开启，`R2+A` 回 walkrun，`Select+Start` 进入阻尼。

## 导出和预检

在包含 AllMimic、mjlab、PyTorch、MuJoCo、onnx、onnxruntime、numpy 和 PyYAML 的环境中：

```bash
python tools/export/allmimic_export_policy_onnx.py \
  --training-repo /path/to/allmimic \
  --checkpoint /path/to/ckpt_0159000.pt \
  --library /path/to/data/mixed/gpu1.npz \
  --clip user/paoxiao.csv --clip omni/12_wbst_motive.csv \
  --output-dir /path/to/export
```

导出器严格检查任务编码、latent、历史和未来偏移，严格加载 Actor 和归一化权重，
使用 FP32 / ONNX opset 17。对首、中、末参考状态及固定随机输入进行 PyTorch/ORT 对照，
并检查 CSV 的 FP32 写回一致性。导出目录同时包含供本机闭环验证的 `scene.mjb` 与每段
`motion.npz`，部署 C++ player 只需要 ONNX、参数 YAML 和对应 CSV。

```bash
python3 tools/assets/check_model_assets.py --policy allmimic
for motion in paoxiao wbst_motive; do
  spec="actions/allmimic/config/generic_allmimic_${motion}.yaml"
  ./build/policy_spec_schema_check --spec "$spec" || break
  ./build/generic_motion_tracking_player --spec "$spec" --check-only || break
done
ACTION_GO_NET=enP8p1s0 tools/launch/start_safe.sh
```

## MuJoCo 验证

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

159000 轮模型的本地完整闭环结果如下，位置误差为世界坐标误差，未做平移对齐。
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
