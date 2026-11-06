# Runtime Architecture

ActionGo separates a policy's recovered semantics from reusable runtime machinery. A registered action is not considered deployable merely because its model has compatible tensor dimensions.

## Control Path

```text
trusted action YAML
  -> MotionActionController
  -> prestart player + current-pose hold
  -> ready_file barrier
  -> official controller ownership transfer
  -> start_file barrier
  -> policy inference
  -> CommandSafetyFilter
  -> rt/user_lowcmd
  -> switch back to walkrun/passive/last
```

`MotionActionController` owns process lifecycle and handoff. Generic players own policy semantics and inference. All normal policy commands pass through `CommandSafetyFilter`, which rejects NaN/Inf, stale or missing `LowState`, invalid dimensions, and unsafe gains, then applies joint limits. Exception paths attempt damping when a publisher is active.

Live depth, object-pose, and motion-teleoperation inputs use a two-policy-frame loss gate. The first invalid frame is tolerated from the last valid input where a cache exists; the second requests the existing supervisor handoff to WALKRUN and waits for its release before the policy exits. This applies to camera depth, torso IMU, tracked object pose, and realtime P-BFM, SONIC, TWIST, and Humanoid-GPT inputs. A neutral wireless joystick value carried in a fresh `LowState` remains a valid command; DDS does not expose a separate controller-link heartbeat.

The supervisor's finite current-pose prearm hold occurs before controller ownership transfer. It is a narrowly scoped protocol exception, not a policy-output path; its damping and gains are fixed and validated.

## Contracts and Players

`runtime/include/action_go/runtime/policy_contract.h` maps each template/variant to a deployment category, player, input source, and decoder. `policy_protocol.h` records G1 motor, observation/action, and template-specific dimensions. Specs must remain consistent with both layers.

| Template | Generic entry point | Typical source |
|---|---|---|
| `single_actor_state` | `generic_policy_player` | Robot state, command, latent prompt |
| `embedded_mimic_time` | `generic_policy_player` | Observation plus embedded time step |
| `motion_conditioned_tracking` | `generic_motion_tracking_player` | Cached or realtime motion reference |
| `depth_conditioned_actor` | `generic_sensor_policy_player` | Depth encoder and proprioception |
| `object_conditioned_actor` | `generic_sensor_policy_player` | Trusted object pose source |
| `composite_motion_policy` | `generic_composite_policy_player` | Multi-session motion pipeline |

Add policy-specific C++ only when existing templates cannot express a genuinely reusable source, graph, or decoder. Follow [new policy onboarding](new_policy_onboarding.md) to trace every semantic field to training/export/deployment evidence.

## Trust Boundaries

- `actions/*.yaml` and global config are trusted executable configuration. Their `command` fields can start local processes; do not accept YAML from an untrusted party.
- UDP control can select only an action already loaded by `MotionActionController`. It cannot provide a command, path, model, or shell fragment.
- Legacy UDP text is loopback-only. Remote/broadcast/multicast use signed JSON with timestamp, nonce replay, and bounded-queue checks.
- Models and motion assets are untrusted binary inputs. The schema/asset gates run before ownership transfer, but they cannot prove behavioral safety.
- Sensor and realtime reference streams require source-specific freshness and missing-data behavior.

## Validation Layers

1. `check_model_assets.py` covers every managed model/data extension and verifies hashes.
2. `policy_spec_schema_check` checks YAML, policy/protocol contracts, dimensions, assets, and installed ONNX I/O.
3. Player `--check-only` performs adapter-specific initialization without controller transfer.
4. Supervisor dry-run exercises configuration and child-process lifecycle without normal `LowCmd` publication.
5. Simulation/replay checks behavior before a physically protected robot test.

None of these layers makes ActionGo safety-certified. Each reduces a different class of avoidable deployment error.
