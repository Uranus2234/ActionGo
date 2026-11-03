# New Policy Deployment Guide for Agents

Use this checklist for every new policy. It turns the repository's deployment experience into reviewable gates; it does not certify a policy as safe.

## 1. Recover the Policy Contract

Read `docs/policy_catalog.md`, `docs/runtime_architecture.md`, `docs/new_policy_onboarding.md`, the selected family's README, and the original training/export/deployment code before editing.

Record evidence for:

- Robot model, 29-DoF joint order, joint signs, default pose, coordinate frames, quaternion convention, and units.
- Observation field order, normalization, history length, update rates, timestamp rules, and missing/stale input behavior.
- ONNX input/output names and shapes, action meaning, scaling, default offsets, gains, and whether outputs are position, velocity, torque, or a tracking target.
- Reference, depth, object-pose, or realtime input formats; source, calibration, freshness gate, timeout, and safe response to loss.
- Checkpoint identity, exporter revision, model hash, dependencies, target CPU/Jetson provider, and artifact rights.

Tensor compatibility is not semantic compatibility. If the source code or required runtime semantics cannot be established, stop before registering a runnable action and document the missing evidence.

## 2. Choose the Existing Runtime

| Policy semantics | Contract/template | Player |
|---|---|---|
| Proprioceptive state, command, or fixed latent | `single_actor_state` | `generic_policy_player` |
| Reference embedded in the model with a timestep input | `embedded_mimic_time` | `generic_policy_player` |
| Cached motion or external realtime reference | `motion_conditioned_tracking` | `generic_motion_tracking_player` |
| Depth encoder plus actor | `depth_conditioned_actor` | `generic_sensor_policy_player` |
| Object pose plus proprioception | `object_conditioned_actor` | `generic_sensor_policy_player` |
| Reusable multi-model pipeline | `composite_motion_policy` | `generic_composite_policy_player` |

Extend the generic contract/runtime only when the new semantics cannot be expressed safely as a reusable template. Keep policy-specific C++ out of a one-off adapter when a spec or reusable module can describe it.

## 3. Add the Package

1. Place specs, parameters, action YAML, tests, and README under `actions/<family>/`.
2. Record every model/reference asset in `MODEL_ASSETS.json` with provenance, exact hash, source revision, redistribution status, and retrieval instructions. Use `tools/assets/fetch_assets.py` for pinned public sources. Do not commit private or uncleared weights/data.
3. Put observation/action semantics in the spec and ensure `PolicyContract` and `PolicyProtocolConstraint` agree with it.
4. Register actions through `config/motion_actions.yaml` includes. Use `rt/user_lowcmd` for hardware and the common command safety path. Keep simulation-only specs separate from hardware specs.
5. Add meaningful tests for tensor contract, decoding, reference/sensor failure, handoff, and stop behavior. Do not treat a fixture or zero input as a behavior test.

## 4. Preserve the Handoff and Failure Path

Use the supervisor lifecycle already implemented by `MotionActionController`: prestart the player, establish fresh state and a current-pose hold, wait for `ready_file`, transfer controller ownership, then release policy stepping through `start_file`. Cancel before ownership transfer if initialization or required input checks fail.

For finite actions, hold the final safe command through the completion/release barrier until the supervisor confirms switch-back. For continuous actions, define explicit stop, timeout, process-exit, and stale-input behavior. Route normal policy output through `CommandSafetyFilter`; document any narrowly scoped decoder behavior that must match the original controller, as MJLab does for its setpoint limits.

Never claim a fallback succeeded until the process and controller ownership have been checked. Use damping on unexpected exits where the existing runtime requires it. Do not add a new shell command path from remote input.

## 5. Validate in Order

Run targeted checks first and record the exact command, environment, asset hashes, and result:

```bash
python3 tools/assets/check_model_assets.py --policy <family>
./build/policy_spec_schema_check --spec actions/<family>/config/<spec>.yaml
./build/<generic_player> --spec actions/<family>/config/<spec>.yaml --check-only
ACTION_GO_NET=lo tools/launch/start_dry_run.sh
tools/launch/start_sim.sh --check <action-name>
```

Then compare observations, decoded targets, and timing against the original policy implementation. Run closed-loop MuJoCo/replay with the same C++ player and inference path used on hardware. Exercise start, stop, switch-back, reset, timeout, stale-state, sensor dropout, and process failure. Use the relevant package test suite and `ctest` before broadening the run.

`--check-only` checks initialization and ONNX I/O only. Supervisor dry-run checks configuration and lifecycle without normal `LowCmd` publication. A sim `--check` also avoids DDS and physics. None proves closed-loop stability or hardware timing.

Before a robot trial, validate on the target architecture and exact ORT/TensorRT provider. Measure cold and warm initialization, worst-case inference deadline, memory, and sensor rate on the target board. Do not assume an x86 result transfers to Jetson.

## 6. Stage Hardware Validation

An agent must stop before commanding a physical robot unless the user explicitly authorizes the hardware trial and a human operator is present. The operator must verify the network interface, damping path, controller mode, exclusion zone, physical support, and independent emergency stop.

Start through `tools/launch/start_safe.sh` with the verified robot interface. Begin from official walkrun or damping, use a low-risk/low-speed action, and test stop and switch-back before a dynamic action. For depth/object policies, require calibrated live input and prove that missing or stale data prevents ownership transfer or exits through the documented safe path. For realtime references, bind locally unless the stream has its own reviewed authentication and freshness protocol.

Record raw logs and the exact source/model hashes. Mark the catalog `real-robot verified` only for the tested policy, checkpoint, sensor configuration, target hardware, and bounded scenario; do not transfer evidence from another family or variant.

## Deployment Lessons Already Encoded

- AllMimic and Fighting154k use current-pose holds, explicit activation transitions, completion/release barriers, and switch-back overlap to avoid low-command gaps.
- Instinct, PHP, and P-BFM separate model checks from live depth readiness; camera failure must prevent handoff. SteadyTray additionally needs calibrated transforms into the policy frame.
- SONIC and TWIST treat realtime references as freshness-gated inputs. Loopback or supervisor-command authentication does not authenticate a separate motion stream.
- MJLab keeps simulator and hardware specs separate and preserves the original target semantics when public clamps would alter policy output.
- OmniXtreme's simulator friction/feedforward is a physics-model term and is not copied blindly to hardware.
- SuperMimic, Fighting154k, and SONIC document simulation evidence separately from robot evidence; preserve that distinction in every new report.

Update `docs/policy_catalog.md` with the category, runtime, dependencies, and evidence level. Keep `registered`, `initialized`, `simulation checked`, and `real-robot verified` as distinct states.
