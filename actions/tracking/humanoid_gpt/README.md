# Humanoid-GPT ActionGo Integration

This package runs the public Humanoid-GPT tracking ONNX:

- input: `obs[1,136]`
- output: `continuous_actions[1,29]`
- target: `default_q + continuous_actions * action_scale`

The first version is not configuration-only because the existing generic
motion-conditioned variants do not build Humanoid-GPT's reference-conditioned
136D observation. The reusable boundary is still small: model paths, IO names,
PD gains, default pose, action scale, reference source, UDP port, and stale
behavior are all YAML fields. Future Humanoid-GPT ONNX files with the same
contract should only need config/model replacement.

## Check

```bash
./build/generic_motion_tracking_player \
  --spec actions/tracking/humanoid_gpt/config/generic_humanoid_gpt.yaml \
  --input offline-npz \
  --trajectory actions/tracking/humanoid_gpt/motions/human_walking_50Hz_29dof.npz \
  --check-only
```

```bash
./build/generic_motion_tracking_player \
  --spec actions/tracking/humanoid_gpt/config/generic_humanoid_gpt.yaml \
  --input realtime \
  --udp-bind 127.0.0.1 \
  --udp-port 51234 \
  --check-only
```

Realtime mode expects Humanoid-GPT HMCP UDP packets carrying `qpos[36]`. The default bind is loopback; malformed, stale, replayed, or non-finite packets are rejected. Non-loopback reference transport is currently rejected until a separate authenticated protocol is implemented.
If the player is built without MuJoCo, offline converted `.npz` cache still
works, but realtime qpos-to-reference FK fails with a clear error.

## Remote Binding

This package intentionally does not change `config/remote_actions.yaml`.
Suggested manual bindings, after bench testing:

```yaml
bindings:
  R2+Up: humanoid_gpt_offline
  R2+Right: humanoid_gpt_realtime
```
