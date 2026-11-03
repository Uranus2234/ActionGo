# ProtoMotions

ProtoMotions 当前动作：

```text
proto_walk
proto_dance
```

资源：

```text
actions/tracking/protomotions/model/unified_pipeline.onnx
actions/tracking/protomotions/config/unified_pipeline.yaml
actions/tracking/protomotions/motions/output_walk.50fps.npz
actions/tracking/protomotions/motions/dance1_subject2.50fps.npz
```

真机只加载已经转换好的 50Hz `.npz`，不在真机上读原始 `.motion` / `.pt`。

ProtoMotions 已迁移到 `motion_conditioned_tracking` 模板：ONNX actor 仍为单模型，
但运行时额外读取 motion command/reference cache。`proto_walk` / `proto_dance`
在 `protomotions_tracker.yaml` 中设置了 `max_frames: 1020`，Generic Motion
Tracking Runtime 会在 1020 个 policy/motion frame 后结束并切回官方控制。部署入口是
`generic_motion_tracking_player`，不再走 `generic_policy_player`。

## 模型检查

```bash
./build/generic_motion_tracking_player \
  --spec actions/tracking/protomotions/config/generic_protomotions.yaml \
  --trajectory actions/tracking/protomotions/motions/output_walk.50fps.npz \
  --check-only
```

```bash
./build/generic_motion_tracking_player \
  --spec actions/tracking/protomotions/config/generic_protomotions.yaml \
  --trajectory actions/tracking/protomotions/motions/dance1_subject2.50fps.npz \
  --check-only
```

## 真机执行

推荐 supervisor：

```bash
./build/action_go_remote_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --remote-map config/remote_actions.yaml
```

按键：

```text
Start    -> ARM
Right+X  -> proto_walk
Right+Y  -> proto_dance
```

## 单动作测试

```bash
./build/action_go_test \
  --net enP8p1s0 \
  --config config/motion_actions.yaml \
  --action proto_walk \
  --remote-control
```

## 仿真 viewer

```bash
python3 tools/sim/protomotions_unitree_mujoco_viewer.py \
  --motion actions/tracking/protomotions/motions/output_walk.50fps.npz
```
