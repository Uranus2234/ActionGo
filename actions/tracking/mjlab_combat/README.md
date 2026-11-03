# mjlab combat: four-GPU checkpoint 92000

Variant: `mjlab_combat`, template: `motion_conditioned_tracking`, player:
`generic_motion_tracking_player`.

| Remote binding | Registered action | Motion | Frames / duration |
| --- | --- | --- | --- |
| Right+A | `mjlab_combat_92000_duobox017_ma` | jab-cross 017 / Ma | 824 / 16.48s |
| Right+B | `mjlab_combat_92000_a42_8s` | A42 generated eight-second segment | 399 / 7.98s |

Right+B uses `generated_qpos36` from
`A42_30k_8s_segment_tracker_20261102.npz`, record
`motiondecode:03_swing_punch_00045`, starting at frame 320. The original 240 frames
at 30Hz are resampled to 399 frames at 50Hz with the existing G1 forward kinematics.
The four-GPU 92000 policy weights are unchanged.

These replace the old AllMimic paoxiao bindings. All other AllMimic remote bindings
are removed; its original actions remain callable by name.

The exported policy comes from:

```
combat-tracker/runs/g1_mixed_combat_wo_state_future40ms/
2026-11-01_12-11-21_mixed_future40ms_4gpu_8192_resume56000_200k/model_92000.pt
```

The deployable model, robot params and two full-length references are bundled under
this directory using ordinary Git files, without LFS. Both specs use package-relative
paths and the standard `build/generic_motion_tracking_player`; inference requires
neither the training repository nor Python/CUDA. `manifest.json` records source
checkpoint/model/params/motion SHA256 values, the policy contract and the corresponding
complete video-run validation. Simulation scenes and recordings stay local.

## Pull, rebuild and check

```bash
git pull --ff-only origin feature
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DACTION_GO_CODE_ONLY=ON
cmake --build build --parallel 2
python3 tools/assets/check_model_assets.py --policy mjlab_combat
for clip in 017 a42_8s; do
  ./build/policy_spec_schema_check --spec "actions/tracking/mjlab_combat/config/generic_mjlab_combat_${clip}.yaml"
  ./build/generic_motion_tracking_player --spec "actions/tracking/mjlab_combat/config/generic_mjlab_combat_${clip}.yaml" --check-only
done
python3 runtime/tests/mjlab_combat_deployment_test.py --player ./build/generic_motion_tracking_player
```

Rebuild is required for the new variant and the subsequent startup-heading fix.
The previous binary does not apply the new alignment even if the YAML is updated.
Replacing Right+B with A42 only changes motion data/configuration; an installation
that already includes the startup-heading fix does not need another rebuild.
Both actions retain the standard ready/start/complete/release handoff and return to
walkrun after the clip. Real-robot startup remains under investigation following a
no-motion first run and a distorted second run; local checks below are not hardware
acceptance. Do not use repeated unrestrained action starts to diagnose those symptoms.

## Reuse and input contract

The actor, normalizer, defaults, action scales and PD parameters come directly from
the mjlab task/checkpoint. The exporter uses the existing `MotionBank`,
`Phase2TrackingRunner.load_for_inference`, mjlab quaternion operations and the original
`build_hybrid_features`. These operations are captured into ONNX, so ActionGo does
not need to duplicate hybrid-reference coordinate math in C++ or import mjlab at runtime.
No policy weights are retrained or changed.

The actor still receives its original **259** features: current explicit reference
and proprioception (154), plus the existing hybrid reference (105). The offsets remain
**current and +40ms**, with no past frames. The deployment ONNX accepts **261 raw values**,
which it transforms into those same 259 actor features:

| Raw fields, in order | Width |
| --- | ---: |
| SDK joint positions, velocities | 29 + 29 |
| Pelvis world quaternion (wxyz), local gyro | 4 + 3 |
| Previous raw policy action | 29 |
| Current reference row | 83 |
| Reference at current frame + 2 | 83 |
| Future validity | 1 |

Each reference row contains joint positions (29), joint velocities (29), torso world
position (3), wxyz quaternion (4), world linear/angular velocity (3+3), and world
positions of left/right wrists followed by left/right ankles (12). References run at
50Hz. At clip boundaries, the original hybrid builder zeros invalid future kinematics
and retains the +0.04s offset with validity zero. With `--max-frames`, future lookup
and validity use the truncated playable boundary. Motions remain external to the ONNX.

ActionGo reuses its existing DDS control loop, ONNX runner, action decoder,
ready/start/handoff barriers, tracing and command safety. The new C++ controller only
loads reference rows, advances frames and packs raw state/reference values.
The decoded target is `default_joint_pos + action_scale * raw_action` for 29 SDK joints.

Both registered specs use `yaw_alignment: initial`. After the start barrier the
controller computes actual torso heading from the pelvis IMU and waist yaw/roll/pitch,
and rotates current/future references into that heading. Positions rotate around the
first reference torso position; world orientations and velocities rotate consistently.
Each reset starts from the unmodified reference and clears previous actions. Setting
`none` retains the recorded heading and is intended for controlled parity checks.
This does not align roll/pitch, change joint targets or make arbitrary initial poses
equivalent to the trained motion.

The registered action uses the normal hardware safety profile and a 0.1s activation
blend. The original closed-loop tests used reference-frame initialization and
`simulation_parity` settings, without activation blending or target clamps. The newer
checks also cover a default standing pose, heading changes and hardware command
filtering. Those DDS simulations remain lockstep and do not verify firmware control
ownership or real-time hardware response.

The player now prints heading/reset state, first policy-target delta, first active
LowCmd write, joint changes after 51 policy steps, and final inference/publication
counts. The first filtered delta can be zero because activation blending begins at
the held pose. A successful LowCmd write or UserCtrl RPC alone does not prove the
robot accepted or executed the commands; the original first-run no-motion symptom
is not yet explained.

## Run local ActionGo inference

From the ActionGo repository, using the combat-tracker environment:

```bash
export MUJOCO_GL=egl
export XDG_CACHE_HOME="$PWD/../combat-tracker/.cache"
../combat-tracker/.venv/bin/python tools/sim/mjlab_combat_mujoco.py \
  --training-repo ../combat-tracker \
  --checkpoint ../combat-tracker/runs/g1_mixed_combat_wo_state_future40ms/2026-11-01_12-11-21_mixed_future40ms_4gpu_8192_resume56000_200k/model_92000.pt \
  --package ../ActionGo-local/packages/mjlab-combat-4gpu-92000-v3 \
  --python-deps ../ActionGo-local/python313 \
  --log-dir ../ActionGo-local/runs/mjlab-combat-new-run \
  --steps 500 --video
```

Choose a fresh log directory. This launches the actual C++ player on loopback DDS in
a dedicated nonzero domain, steps native MuJoCo at the exported 5ms timestep with
20ms policy updates, validates the trace against the original PyTorch actor and
mjlab observation manager, and writes `preview.mp4`, `trace.npz`, `metrics.json` and
`parity.json`. It does not use or modify an ongoing training process.

The simulator reuses `DDSBridge` from the SuperMimic simulator and reference drawing
from the Uni-SuperMimic viewer. In this environment CycloneDDS 0.10.2 was built for
Python 3.13 into `ActionGo-local/python313`, with
`CFLAGS=-D_Py_IsFinalizing=Py_IsFinalizing` and the existing local CycloneDDS library.

## Another external motion, same ONNX

```bash
../combat-tracker/.venv/bin/python tools/export/mjlab_combat_prepare_motion.py \
  --training-repo ../combat-tracker \
  --package ../ActionGo-local/packages/mjlab-combat-4gpu-92000-v3 \
  --motion /absolute/path/to/standardized_g1_motion.npz \
  --output ../ActionGo-local/packages/mjlab-combat-new-motion
```

This only prepares a reference and simulation initialization. The model, params and
scene are symlinked to the base package; weights are never re-exported for each motion.
The original `MotionBank` validates NPZ fields, named SDK joint/body order, 50Hz FPS,
wxyz normalized quaternions and finite values; the preparer also checks the trained
robot model hash. Input must already be a retargeted, standardized G1 NPZ with
`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`,
`body_ang_vel_w`, `joint_names`, `body_names`, `fps`, `quaternion_order`, `model_sha256`.
No retargeting or augmentation occurs. Use the prepared package with the same simulator
command. For the registered ActionGo action, point `reference.path` to its CSV.

To export from the original checkpoint again, use `tools/export/mjlab_combat_export.py`
with `--training-repo`, `--checkpoint`, `--motion` and a fresh `--output` directory.
The export checks the policy contract and original observations/actions, including
invalid future frames, before writing its manifest. Simulator sensor bias is zero
during deployment/parity, rather than a random training encoder-bias sample.

## Verified locally

- Build: `generic_motion_tracking_player`, `policy_spec_schema_check`, `runtime_hardening_test`.
- Strict schema and C++ ONNX `--check-only` pass for the registered action.
- Controller tests cover dimensions, previous action, +2 frame lookup, final hold,
  future validity, reset, malformed CSV, invalid quaternions, torso-heading alignment,
  consistent current/future transforms and repeat resets without accumulated rotation.
- `fight1_subject2`: 500 policy steps / 10s, no resets; original mjlab observation
  maximum absolute error `2.39e-7`, C++ ONNX/PyTorch action error `5.73e-6`.
- Repaired DuoBox `01_drill_000_ma_jab_liu_defense__liu`: 370 steps / 7.4s including
  the clip end, same ONNX SHA256; observation error `1.93e-7`, action error `6.45e-6`.
- These are two deployment checks; they do not establish quality on every motion.
- The original 017/Ma and 034/Ma references matched the full 824/876-step video
  runs, without resets; C++ ONNX/PyTorch action errors are below `6.5e-6`.
- The current Right+B A42 reference passes all 399 steps / 7.98s from the default
  standing pose with initial heading alignment and hardware filtering: zero resets,
  joint RMSE `0.1885` rad, minimum pelvis height `0.7269` m, maximum tilt `21.14`
  degrees. Original mjlab observation error is `4.77e-7`; C++ ONNX/PyTorch action
  error is `7.04e-6`. This is a local simulation check, not a hardware test.
- Package tests verify hashes, remote bindings, portable paths, ONNX loading and
  ready/start/complete/release using synthetic state without publishing LowCmd.
- Startup-heading regression: original ONNX targets remain within `5.2e-7` rad when
  the same state is yaw-rotated by +90, 180 or -90 degrees with alignment enabled.
  Without alignment the target differences reach `0.786`, `2.475` and `2.234` rad.
- Standing-start checks with hardware filtering: 017 at +90 degrees runs 824 steps,
  joint RMSE `0.117` rad, minimum pelvis height `0.716` m, maximum tilt `15.41` degrees;
  034 at -90 degrees runs 876 steps, RMSE `0.152` rad, minimum height `0.649` m,
  maximum tilt `21.52` degrees. Neither run resets. Raw input/original observation
  parity passes, and C++ ONNX/PyTorch action errors stay below `7.8e-6`.

To reproduce a standing-start check, add `--initial-pose default --initial-yaw-deg 90
--yaw-alignment initial --hardware-safety` to the simulator command, select the
prepared 017 package and use `--steps 824` (034 uses 876). In hardware-filter mode
`parity.json` explicitly reports `dds_target_compared: false`: the filtered DDS target
must not be compared directly to the unfiltered policy target. MuJoCo still applies
the received filtered commands for the trajectory metrics.

Recordings and full results are in `ActionGo-local/runs/mjlab-combat-92000-fight1-v3`
and `ActionGo-local/runs/mjlab-combat-92000-duobox`.
The A42 recording and results are in `ActionGo-local/runs/mjlab-combat-92000-a42-8s`.
