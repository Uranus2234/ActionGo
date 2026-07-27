# TWIST for action_go

TWIST 已迁移到 `motion_conditioned_tracking` 模板。`generic_motion_tracking_player`
运行 TWIST general motion tracker ONNX，并从 offline PKL/cache 或实时
UDP/ZMQ/Redis 输入读取 33 维 `mimic_obs` motion command。

## Actions

- `twist_offline_pkl`: offline PKL input. The action uses `actions/twist/motions/*.pkl`, converts it once to a 33-float `mimic_obs` CSV cache via `tools/runtime/twist_prepare_motion.py`, then runs policy inference in C++.
- `twist_nokov_realtime` / `twist_udp_realtime`: realtime UDP input, default loopback bind and UDP port `15110`; non-loopback reference transport is currently rejected until a separate authenticated protocol is implemented.
- `twist_zmq_realtime`: realtime ZMQ SUB input, default `tcp://127.0.0.1:15111` with the player binding.
- `twist_redis_realtime`: realtime Redis input, default `GET action_mimic_g1` from `127.0.0.1:6379`.

## Build

```bash
cd /path/to/action_go
cmake -S . -B build
cmake --build build --target generic_motion_tracking_player -j
```

## Re-export the ONNX model

```bash
cd /path/to/action_go
tools/export/twist_export_policy_onnx.py \
  --pt "$TWIST_ROOT/assets/twist_general_motion_tracker.pt" \
  --out actions/twist/model/twist_general_motion_tracker.onnx
```

## Offline PKL smoke check

```bash
cd /path/to/action_go
./build/generic_motion_tracking_player \
  --spec actions/twist/config/generic_twist.yaml \
  --input offline-pkl \
  --motion-pkl actions/twist/motions/125_SLP201.pkl \
  --motion-cache .cache/twist/twist_offline_pkl.mimic.csv \
  --check-only
```

## Realtime input protocol

All transports carry the same payload. Preferred payload:

```json
{"mimic_obs":[33 floats]}
```

A plain CSV line with exactly 33 floats is also accepted. The 33 floats are the TWIST G1 mimic observation:

```text
root_height,
roll, pitch, yaw,
local_root_vx, local_root_vy, local_root_vz,
local_root_yaw_rate,
25 G1 dof targets including left/right wrist-roll slots
```

The policy consumes the 31 non-wrist entries; indices 27 and 32 are forwarded to G1 wrist-roll motors 19 and 26.

### UDP

Player side:

```bash
./build/generic_motion_tracking_player --spec actions/twist/config/generic_twist.yaml \
  --input realtime --realtime-transport udp \
  --udp-bind 127.0.0.1 --udp-port 15110 --require-realtime --check-only
```

Sender example:

```bash
tools/remote/twist_nokov_udp_example.py --transport udp --host 127.0.0.1 --port 15110
```

### ZMQ

Player side defaults to a SUB socket that binds:

```bash
./build/generic_motion_tracking_player --spec actions/twist/config/generic_twist.yaml \
  --input realtime --realtime-transport zmq \
  --zmq-bind --zmq-endpoint tcp://127.0.0.1:15111 --require-realtime --check-only
```

Sender example:

```bash
tools/remote/twist_nokov_udp_example.py --transport zmq --zmq-endpoint tcp://127.0.0.1:15111
```

### Redis

Player side polls Redis with `GET`:

```bash
./build/generic_motion_tracking_player --spec actions/twist/config/generic_twist.yaml \
  --input realtime --realtime-transport redis \
  --redis-host 127.0.0.1 --redis-port 6379 --redis-key action_mimic_g1 \
  --require-realtime --check-only
```

Sender example:

```bash
tools/remote/twist_nokov_udp_example.py --transport redis --host 127.0.0.1 --port 6379 --redis-key action_mimic_g1
```

## Remote bindings

- `L1+Right` -> `twist_offline_pkl`
- `L2+Right` -> `twist_nokov_realtime` (UDP)

For ZMQ/Redis in the remote supervisor, bind your preferred combo to `twist_zmq_realtime` or `twist_redis_realtime` in `config/remote_actions.yaml`.
