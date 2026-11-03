# GentleHumanoid action_go payload

This action package wires `action_go` to the sibling repository:

```text
../gentle-humanoid/src/action_go_player.py
```

It does **not** add a new C++ player.  The released GentleHumanoid motion files
are pickle-backed `.npy` dictionaries, so the safest first integration is to
reuse the official Python policy/observation code and let `action_go` handle the
official-controller ↔ UserCtrl handoff.

## Quick checks

From `../gentle-humanoid`:

```bash
python3 src/action_go_player.py --check-only
```

From `action_go` on the robot environment:

```bash
./build/action_go_test --net enP8p1s0 --action gentle_walk --config config/motion_actions.yaml
```

Dry-run mode still initializes DDS/LowState/ONNX but does not publish LowCmd:

```bash
./build/action_go_test --net enP8p1s0 --action gentle_walk --config config/motion_actions.yaml --dry-run
```

## Actions

The action name maps to a motion in `../gentle-humanoid/config/tracking.yaml`:

- `gentle_default` → `default`
- `gentle_walk` → `walk`
- `gentle_hug` → `hug`
- `gentle_pick_ball` → `pick_ball(from video)`
- `gentle_drag_object` → `drag_object(from video)`
- `gentle_move_object` → `move_object(from video)`

`action_go` prestarts the Python process, waits for `--ready-file`, switches to
UserCtrl, touches `--start-file`, and then the player runs inference on
`rt/user_lowcmd`.
