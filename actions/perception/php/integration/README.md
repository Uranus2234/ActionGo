# ActionGo 驱动 PHP MuJoCo 仿真

此目录提供官方 PHP checkout 的伴随补丁，使 MuJoCo 接收 ActionGo C++ player 的推理命令。
先完成 [PHP 使用指南](../README.md) 中的模型安装和 C++ 编译。

## 准备仓库

补丁适用的 PHP 基础提交为 `554edcc33393fbd8bd2a725a781ffb867f6c467a`。
新安装时使用这个版本，并初始化其固定的 Holosoma 子模块：

```bash
# 在 ActionGo 根目录执行；PHP 仓库已按主指南 clone
cd ../php_parkour
git checkout 554edcc33393fbd8bd2a725a781ffb867f6c467a
git submodule update --init --recursive thirdparty/holosoma
git apply --check ../action_go/actions/perception/php/integration/php_parkour.patch
git apply ../action_go/actions/perception/php/integration/php_parkour.patch
```

已有本地修改时先检查兼容性；已应用补丁时无需重复应用。
目录名不是 `action_go` 时，将命令中的路径替换为实际 ActionGo 路径。

## 安装仿真环境

按 PHP 的 [官方 sim2sim 指南](https://github.com/amazon-far/php_parkour/blob/main/wbt_training/DEPLOY.md)
安装 Holosoma MuJoCo 和 inference 环境：

```bash
bash thirdparty/holosoma/scripts/setup_mujoco.sh
bash thirdparty/holosoma/scripts/setup_inference.sh
```

仿真解释器需要能导入 `mujoco`、`numpy`、`holosoma` 和 `unitree_sdk2py`，
录制需要 `ffmpeg`。在 PHP 根目录创建未纳入版本控制的 `.env.local`，
例如使用官方安装目录：

```bash
cat > .env.local <<'EOF'
PHP_SIM_PYTHON="$HOME/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python"
PHP_SIM_LIBRARY_PATH="$HOME/.holosoma_deps/miniconda3/envs/hsmujoco/lib"
EOF
```

若环境已有自己的安装位置，填写实际路径；已有 `.env.local` 时编辑对应变量。
补丁入口会直接读取这个文件。

仿真 walkrun 使用 ActionGo 的 `unitree_rl_mjlab` 模型，需准备真实 ONNX 文件
`../action_go/actions/proprio/unitree_rl_mjlab/model/policy.onnx`；若仓库文件为 Git LFS 指针，
先拉取对应模型。可先检查：

```bash
../action_go/build/generic_policy_player \
  --spec ../action_go/actions/proprio/unitree_rl_mjlab/config/generic_unitree_rl_mjlab.yaml --check-only
```

## 运行

在 PHP 根目录启动带画面的自动阶梯测试：

```bash
bash run_actiongo_sim.sh --actiongo-root ../action_go
```

无交互画面并保存结果：

```bash
bash run_actiongo_sim.sh --actiongo-root ../action_go --headless \
  --output "$PWD/outputs/php_stairs_acceptance"
```

停止可按 `Ctrl-C`。不要与官方 PHP 推理或另一个仿真同时运行，
以免争用 DDS topic 和 `depth_img_shm`。

默认入口自动执行切入 PHP、方向控制、站稳和切回 walkrun。
本补丁提供自动验收入口；官方 Python 的键盘操作见官方指南。
仿真通过标准为完整通过 12 个阶梯障碍、无跌倒、站稳并切回 walkrun。
开启吊架或绕过障碍不能作为通过证据。
结果写入输出目录，包括录像、运行日志和推理记录。

仿真使用 [SHM 配置](../config/generic_php_sim.yaml)，真机使用
[RealSense 配置](../config/generic_php.yaml)。共用模型和控制参数仍需配合实际传感器、
交接与电机测试验证。历史测量见 [部署记录](../../../docs/php_deployment_history.md)。
