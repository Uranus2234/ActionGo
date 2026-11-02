# 四模型 154k 格斗动作部署

本包固定使用 2026-10-28 视频对比的四个 `ckpt_0154000.pt` 导出。
这里的“最新”指这批已看过的 154000 轮导出；训练继续产生的 checkpoint 不会自动替换本包。
四个 ONNX、六份按模型家族共享的 CSV 和部署配置使用普通 Git 文件。
本包无需 Git LFS 下载，也无需训练仓库、Python 或 CUDA 才能进行板载推理。

| 模型 | 051 / Ma（约 16 秒） | 功夫侧踢（约 6 秒） |
|---|---|---|
| SuperMimic-Reproduced，Omni 15% | R1+Y | R1+Up |
| SuperMimic-Reproduced，不带 Omni | 保留动作，未绑定 | 保留动作，未绑定 |
| AllMimic，Omni 15% | 保留动作，未绑定 | 保留动作，未绑定 |
| AllMimic，不带 Omni | 保留动作，未绑定 | 保留动作，未绑定 |

`R1+A` 使用同一份 SuperMimic-Reproduced Omni 15% 的 154k 模型播放 **wbst_motive**，
`R1+Down` 使用该模型播放 **yewendun**（`14_yewendun-kick`）。
参考直接取自该轮训练的 `omni15.npz`：`omni/12_wbst_motive.csv` 为 573 帧 / 50 Hz，
播放 11.46 秒；`omni/14_yewendun-kick.csv` 为 916 帧 / 50 Hz，播放 18.32 秒。
两段保持训练库的完整时间范围和速度，使用当前帧至未来第 9 帧的 SuperMimic-Reproduced
参考格式；不能替换为 Uni-SuperMimic / AllMimic 的同名参考缓存。

051 是 `duobox_g1_boxing.zip` 中的 `04_sparring_051_ma_free_liu_free.npz`、Ma 一侧，
使用视频所展示的 20261017 修正重定向版本。原始 481 帧 / 30 Hz，部署缓存 800 帧 / 50 Hz，
最后一帧时间 15.98 秒。该人体录制出现在训练集中；修正重定向不能算独立验证录制。
侧踢是训练集的 `kungfu_g1_export__kungfu_Side_kick__single`，原始 179 帧 / 30 Hz，
部署缓存 297 帧 / 50 Hz，最后一帧时间 5.92 秒。两段均完整保留，没有裁剪或改变时间比例。
首帧到末帧时间与有限控制周期总长相差一个 20 ms 周期；启动过渡和交接另计。

## 推理合同

两类策略都由 `build/generic_motion_tracking_player` 执行，SDK 关节顺序、29 个动作，
策略频率 50 Hz、命令发布 500 Hz、ONNX Runtime CPU 两线程。
`manifest.json` 记录 checkpoint、模型、参数、参考数据 SHA256 和当前按键组合。
共注册十个动作；SuperMimic 不带 Omni 的两个原动作仍可通过动作名调用。

| 项目 | SuperMimic-Reproduced | AllMimic |
|---|---|---|
| ActionGo variant | `supermimic` | `allmimic` |
| ONNX 输入 / 输出 | `obs[1,1720]` / `actions[1,29]` | `obs[1,1660]` / `actions[1,29]` |
| 原生网络输入 | 930 history + 686 goal | 930 history + 730 goal |
| 未来目标索引 | t 至 t+9 | t+1 至 t+10 |
| 动作解码 | 默认角 + action_scale × 输出 | 下肢同左；腰臂额外加入参考姿态相对默认角的残差 |

SuperMimic-Reproduced 的 ONNX 内含适配：将 10×99 的逐帧历史重排为原生 930 维逐项历史，
680 维未来目标拼接当前 IMU 相对参考的 6D 旋转，CSV 中剩余 50 个占位字段不参与原生网络。
不要将这个文件替换为上游不同结构的 SuperMimic 权重。
两个家族的 CSV 布局和未来偏移不同，不能互换。

## 同步、构建与检查

在 ActionGo 仓库目录运行：

```bash
git pull --ff-only origin feature
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DACTION_GO_CODE_ONLY=ON
cmake --build build --parallel 2
python3 tools/assets/check_model_assets.py --policy fighting154k
for spec in actions/fighting154k/config/generic_*.yaml; do
  ./build/policy_spec_schema_check --spec "$spec"
  ./build/generic_motion_tracking_player --spec "$spec" --check-only
done
```

仅做本机合成状态检查与交接文件测试，不连接 DDS、不发送 LowCmd：

```bash
python3 runtime/tests/fighting154k_deployment_test.py --player build/generic_motion_tracking_player
```

此测试需要 PyYAML。它会校验资产、八个按键配置和十个动作，执行真实 ONNX 推理，验证有限动作完成后
持续等待 release 文件，且整个 dry-run 无 LowCmd 发布。合成状态不能用来评价动作跟踪效果。

真机仍从统一 supervisor 启动，网卡示例为 `eth0`：

```bash
ACTION_GO_NET=eth0 tools/launch/start_safe.sh
```

先按 `Start` 或 `Left+B` ARM，再按表中的动作组合。正常完成后自动回官方 walkrun：

1. player 完成模型 / 参考 / LowState 初始化，发布当前姿态 hold，给出 ready。
2. supervisor 切 UserCtrl 成功后给出 start；策略对齐初始 yaw，经过 0.10 秒激活过渡。
3. 播完后 player 保持最后命令，给出 complete；supervisor 请求官方 walkrun。
4. 切换成功后继续重叠发布 0.30 秒，再给出 release，player 才退出。

walkrun 切换失败时继续等待交接，不将失败当作成功释放。
所有动作均启用既有 supervisor IMU safety；`R2+A` 中止并回 walkrun，`Select+Start` 进入阻尼。
运行中命令保留有限值、位置范围、LowState 新鲜度及训练力矩上限检查。
位置目标裁剪时通过前馈力矩补偿保持受限 PD 力矩，避免单独裁剪目标角改变训练控制律。

## 已有仿真证据

`validation.json` 收录八组已播放视频对应的完整 ActionGo C++ / DDS / MuJoCo 结果和原生网络对照，
并关联本包模型及 CSV 的哈希。十组均完成全段、状态有限且原生推理对照通过。
视频采用参考首帧姿态与速度初始化、5 ms 物理步长、显式受限 PD、`simulation_parity` 配置。
真机配置另外包含初始 yaw 对齐、激活过渡、位置过滤的力矩补偿和官方控制器交接；
这些新模型及动作尚无真机验收记录。
