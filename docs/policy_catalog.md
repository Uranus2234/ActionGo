# Policy Catalog

This index groups the 94 actions registered by `config/motion_actions.yaml` by task and required input. Runtime implementation remains defined by each policy spec and `PolicyContract`; action names and commands remain defined by the YAML registry.

## Categories by Required Information

Group policies by deployment-facing inputs. A fixed motion clip embedded in a task-specific model stays with that task policy; a selectable/provided motion reference or latent belongs to tracking policies. Policies may appear in more than one group when they also require live perception.

### Proprio

These are task-specific deployments whose runtime primarily uses robot state and a fixed behavior or clip. They do not accept a reusable motion reference or latent as a deployment input:

- **[BeyondMimic](../actions/proprio/beyondmimic/README.md)** — `kick_right`, `hlqk`, `wbst`, `lie_down_beyondmimic`.
- **[MJLab](../actions/proprio/mjlab/README.md)** — `dance1_subject2_mjlab`, `reverse_vault_flat_mjlab`, `reverse_vault_terrain_mjlab`; task-specific actions grouped with BeyondMimic because they do not accept a reusable motion input.
- **[Holosoma](../actions/proprio/holosoma/README.md)** — `holosoma_dance`; task-specific policy with a fixed motion sequence embedded in ONNX.
- **[GentleHumanoid](../actions/proprio/gentle_humanoid/README.md)** — `gentle_default`, `gentle_drag_object`, `gentle_hug`, `gentle_move_object`, `gentle_pick_ball`, `gentle_walk`; separate Python runtime and task-specific integration.
- **[Unitree RL MJLab](../actions/proprio/unitree_rl_mjlab/README.md)** — `sim_walkrun`; simulation-only WALKRUN replacement.

### Tracking

Tracking policies receive selected motion information, either as an explicit reference or an implicit latent. The tensor interface can differ:

- **[AllMimic](../actions/tracking/allmimic/README.md) / [SuperMimic](../actions/tracking/supermimic/README.md) / [Uni-SuperMimic](../actions/tracking/uni_supermimic/README.md)** — `allmimic_paoxiao`, `allmimic_paoxiao_no_omni`, `allmimic_wbst_motive`, `supermimic_tracking`, `uni_supermimic_paoxiao`, `uni_supermimic_tracking`, `uni_supermimic_wbst_motive` (Uni-SuperMimic deferred by request).
- **Fighting154k** — `fighting154k_allmimic_no_omni_051_ma`, `fighting154k_allmimic_no_omni_side_kick`, `fighting154k_allmimic_omni_051_ma`, `fighting154k_allmimic_omni_side_kick`, `fighting154k_supermimic_no_omni_051_ma`, `fighting154k_supermimic_no_omni_side_kick`, `fighting154k_supermimic_omni_051_ma`, `fighting154k_supermimic_omni_side_kick`, `fighting154k_supermimic_omni_wbst_motive`, `fighting154k_supermimic_omni_yewendun`.
- **MJLab Combat** — `mjlab_combat_92000_a42_8s`, `mjlab_combat_92000_duobox017_ma`.
- **OmniXtreme** — `omni_motion`, `omni_motion_default`, `omni_seg01_0000_0850`, `omni_seg01_fuck_floor`, `omni_seg02_0850_1335`, `omni_seg02_back_hand_flip_roll`, `omni_seg03_1335_1728`, `omni_seg03_back_hand_flip`, `omni_seg04_1728_2193`, `omni_seg04_wbst_stand`, `omni_seg05_2193_2607`, `omni_seg05_side_roll_kick`, `omni_seg06_2607_3110`, `omni_seg06_front_roll`, `omni_seg07_3110_3592`, `omni_seg07_back_roll`, `omni_seg08_3592_3715`, `omni_seg08_none`, `omni_seg09_3715_4302`, `omni_seg09_back_hand_flip`, `omni_seg10_4302_4775`, `omni_seg10_back_hand_flip`, `omni_seg11_4775_5139`, `omni_seg11_side_flip_roll`, `omni_seg12_5139_5712`, `omni_seg12_wbst_motive`, `omni_seg13_5712_6122`, `omni_seg13_ceshoufan_double`, `omni_seg14_6122_7038`, `omni_seg14_yewendun_kick`, `omni_seg15_7038_7467`, `omni_seg15_front_paxing`.
- **ProtoMotions / Humanoid-GPT / SONIC / TWIST** — `proto_walk`, `proto_dance`, `humanoid_gpt_offline`, `humanoid_gpt_realtime`, `sonic_offline_squat`, `sonic_realtime_g1`, `sonic_pico_teleop`, `twist_offline_pkl`, `twist_nokov_realtime`, `twist_udp_realtime`, `twist_zmq_realtime`, `twist_redis_realtime`.
- **BFM-Zero / UFO** — `bfm_zero_stand`, `ufo_forward_slow`, `ufo_idle`, `ufo_move_left`, `ufo_spin_arms`, `ufo_stand`, `ufo_tracking_zs7`; fixed, prompted, or tracking latents carry implicit or explicit motion information.
- **P-BFM** — `pbfm_arms_up`, `pbfm_joystick`, `pbfm_loco`, `pbfm_tracking_dance`, `pbfm_pico_teleop`; motion/context latent, with live depth also required.

### Perception

These policies require live environment sensing beyond robot state. P-BFM is listed here as well because depth is a required control input:

- **Instinct** — `g1_parkour` (depth plus torso IMU).
- **PHP** — `php_locomotion` (depth-conditioned locomotion).
- **P-BFM** — `pbfm_arms_up`, `pbfm_joystick`, `pbfm_loco`, `pbfm_tracking_dance`, `pbfm_pico_teleop` (depth; PICO also requires a live reference).
- **SteadyTray** — `steadytray_balance`, `steadytray_zero_test` (AprilTag/object pose, or explicit test pose).

## Runtime Entry Points

| Input contract | Player | Current families |
|---|---|---|
| Robot state with fixed/latent behavior prompt | `generic_policy_player` | BFM-Zero, UFO |
| Robot state, command, or embedded reference | `generic_policy_player` | BeyondMimic, Holosoma, MJLab, Unitree RL MJLab |
| Embedded motion and timestep | `generic_policy_player` | BeyondMimic, Holosoma, MJLab dance |
| Offline or realtime motion reference | `generic_motion_tracking_player` | AllMimic, SuperMimic, Fighting154k, Uni-SuperMimic, MJLab Combat, ProtoMotions, Humanoid-GPT, SONIC, TWIST |
| Depth or object pose | `generic_sensor_policy_player` | Instinct, PHP, P-BFM, SteadyTray |
| Multi-model composite motion | `generic_composite_policy_player` | OmniXtreme |
| Separate Python/task adapter | Package-specific | GentleHumanoid |

See [runtime architecture](runtime_architecture.md), [onboarding](new_policy_onboarding.md), and the package READMEs for the recovered interface and deployment details.

## Startup Check Snapshot

Checked on 2026-11-03 with the local x86 sim2sim build and its private asset overlay. The registry lists 94 actions. The per-action sim `--check` path runs player `--check-only`; it does not connect DDS, step physics, verify full sensor behavior, or command a robot.

- **74 passed initialization checks** with the default local asset overlay and the selected sim Python environment. This confirms the configured player accepted its spec and installed assets in this local environment only.
- **0 failed:** `hlqk` now maps to `actions/proprio/beyondmimic/model/hlqk.onnx`, matching the registered action name and local asset.
- **20 deferred:** `g1_parkour`; six GentleHumanoid actions; `steadytray_balance` and `steadytray_zero_test` (no shared-scene object body/calibrated frame); `humanoid_gpt_realtime`; `pbfm_pico_teleop`; `sonic_pico_teleop` and `sonic_realtime_g1`; four realtime TWIST actions; and the three Uni-SuperMimic actions excluded from this audit.

`tools/launch/start_sim.sh` now places the selected sim Python environment on `PATH`, so the TWIST motion converter uses the available PyTorch environment. The local SuperMimic model and reference are linked into the private sim asset overlay; no model or motion data was copied into the source repository. The simulator's SteadyTray check was corrected to defer actions unless `--object-body` is supplied.

Passing `--check-only` is not evidence of stable motion or real-robot readiness. For the current readiness evidence and next checks, see [the README adapter table](../README.md#adapter-readiness) and each family README.
