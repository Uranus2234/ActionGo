<div align="center">

# ActionGo

**Run and switch heterogeneous whole-body policies on a Unitree G1 through one contract-validated runtime.**

[![CI](https://github.com/Uranus2234/ActionGo/actions/workflows/ci.yml/badge.svg)](https://github.com/Uranus2234/ActionGo/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Uranus2234/ActionGo?display_name=tag)](https://github.com/Uranus2234/ActionGo/releases)
![Platform](https://img.shields.io/badge/platform-Unitree%20G1-orange)

[中文说明](README.zh-CN.md) · [Policy onboarding](docs/new_policy_onboarding.md) · [Runtime architecture](docs/runtime_architecture.md)

</div>

ActionGo turns a policy package into a registered G1 action: it validates a YAML policy contract, prestarts the player, holds the current pose before transfer, switches controller ownership, filters normal in-process policy commands, and returns to official walkrun or damping when the action ends. Prearm, external payloads, controller transfer, and realtime reference transports are separate trust boundaries and are not a safety certification. It supports state, embedded-mimic, motion-conditioned, sensor-conditioned, object-conditioned, and composite ONNX policies without requiring a new C++ player for every model.

> [!CAUTION]
> ActionGo can send low-level commands to a real humanoid robot. It is research software, not a safety-certified product. A valid tensor shape does not prove that observation order, joint mapping, gains, action scaling, or reference frames are correct. Use physical support, a clear exclusion zone, an operator ready to enter damping, and the staged checks in this document. Never start with a high-dynamic action.

## Demo

A rights-cleared real-robot demo will be attached to a release rather than copied from private experiment or third-party media. The code release does not treat an unpublished video as evidence of safety or compatibility.

## What You Get

- Generic C++17 players selected by centralized policy and protocol contracts.
- File-coordinated `ready_file` / `start_file` handoff with current-pose hold before ownership transfer; this is not a transactional lease or safety certification.
- NaN/Inf, joint, gain, and stale-`LowState` checks before normal `LowCmd` publication.
- Schema, check-only, dry-run, failure-injection, and structured JSONL diagnostics.
- Remote-controller operation plus optional authenticated UDP broadcast/multicast control.

ActionGo is **not an automatic model adapter**. Before adding a policy, recover its semantics from the training, export, and original deployment code. Matching ONNX dimensions alone is insufficient and can be dangerous.

## Requirements

- Ubuntu 22.04 reference environment, GCC 11+, CMake 3.22+, and C++17.
- Unitree SDK2 at the revision listed in [DEPENDENCIES.md](DEPENDENCIES.md).
- Eigen3, yaml-cpp, zlib, and OpenSSL development packages.
- ONNX Runtime 1.22.0; install the pinned official archive with:

```bash
tools/deps/fetch_onnxruntime.sh
```

OpenCV is required to build `generic_sensor_policy_player`. RealSense, AprilTag, MuJoCo, ZeroMQ, Redis, and TensorRT are optional and only needed by their corresponding adapters. See [DEPENDENCIES.md](DEPENDENCIES.md) and [runtime troubleshooting](docs/runtime_troubleshooting.md).

## Models and Data

Policy weights, motion files, tags, and datasets are not distributed in the code-only release unless their asset-specific redistribution rights are verified. A source-code license does not automatically license a checkpoint or dataset.

Use [MODEL_ASSETS.json](MODEL_ASSETS.json) to identify the expected path, upstream source, and SHA256. `blocked_pending_review` means **bring your own weights/data**: obtain the asset from its rights holder and verify that your use complies with its terms. Then run:

```bash
python3 tools/assets/check_model_assets.py
```

ActionGo does not provide an uncontrolled downloader for assets with unclear terms.
The build generates a source-owned constant-output ONNX fixture for unit tests.
It is marked by design as check-only/simulation test material, is not registered
as a robot action, and must never be used to publish real-robot commands.

## Build and Preflight

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Use the two schema modes according to what is installed:

```bash
# Validate adapter semantics while BYOW assets are absent.
./build/policy_spec_schema_check --all --allow-missing-assets

# Inspect only specs whose assets are installed, including their real ONNX I/O.
./build/policy_spec_schema_check --all --installed-only

# Strict release/deployment check: every configured asset must exist and validate.
./build/policy_spec_schema_check --all
```

Run a package-specific check before simulation or robot use, for example:

```bash
./build/generic_policy_player \
  --spec actions/bfm0/config/generic_bfm_zero.yaml \
  --check-only
```

Then verify the supervisor without controller transfer or `LowCmd` publication:

```bash
ACTION_GO_NET=lo tools/launch/start_dry_run.sh
```

The mandatory progression is:

```text
asset hash -> schema -> package check-only -> dry-run -> simulation/replay
           -> physically protected low-risk robot test -> higher-risk action
```

## Real-Robot Start

Only proceed after the strict asset and schema gates pass, the action-specific sensor/reference checks pass, and the operator has verified the robot network interface and damping path.

```bash
ACTION_GO_NET=<robot-network-interface> tools/launch/start_safe.sh
```

The supervisor starts in `SHELL_IDLE`: it listens but does not take ownership. Confirm that the robot is already in official walkrun or damping before arming. `Start` arms with the primary recovery policy; `Left+B` uses the alternate recovery policy. `R2+A` stops the current action and returns to official walkrun. `Select+Start` forces official passive/damping.

### Default Remote Map

Arm first with `Start` or `Left+B`, then use an action binding. The checked-in defaults are:

| Buttons | Result |
|---|---|
| `Start` | ARM with `bfm_zero_stand`; enable actions and IMU safety |
| `Left+B` | ARM with alternate `ufo_stand` recovery backend |
| `R2+A` | Stop the current action, return to official walkrun, enter `SHELL_IDLE` |
| `Select+Start` | Stop the current action, enter official passive/damping, enter `SHELL_IDLE` |
| `R2+Down` | `kick_right` |
| `Left+X` | `hlqk` |
| `Left+A` | `ufo_tracking_zs7` |
| `F1+A` | `wbst` |
| `Up+Y` | `g1_parkour` |
| `Right+X` | `proto_walk` |
| `Right+Y` | `proto_dance` |
| `R1+F1` | `holosoma_dance` |
| `F1+X` | `omni_seg04_wbst_stand` |
| `F1+Y` | `omni_seg12_wbst_motive` |
| `F1+B` | `omni_seg11_side_flip_roll` |
| `L1+Right` | `twist_offline_pkl` |
| `L2+Right` | `twist_nokov_realtime` using local UDP reference input |
| `F2+X` | `sonic_offline_squat` |
| `F2+Y` | `sonic_pico_teleop` using the local PICO ZMQ bridge |
| `L2+Down` | `lie_down_beyondmimic`; finish in passive shell |

[`config/remote_actions.yaml`](config/remote_actions.yaml) is the source of truth. Review it before every robot session because local deployments may intentionally change or disable bindings.

Do not use `tools/launch/start_unsafe.sh` as a convenience shortcut. Some registered actions intentionally disable supervisor IMU interruption because an inversion, lie-down, or recovery would otherwise trigger it; those actions require stronger physical protection and an explicit operator review.

## Adapter Readiness

This table reports repository evidence, not marketing-level “support.” The project author has exercised ActionGo on a real G1, but the public release currently lacks independently reviewable raw logs/manifests for each adapter; individual rows remain `Not documented` unless their scope and evidence are archived.

| Adapter | Model/data acquisition | Extra hardware/source | Check-only | Simulation tooling | Real-robot verified | IMU safety default | Dynamic risk |
|---|---|---|---|---|---|---|---|
| BeyondMimic | BYOW; manifest source/hash | Embedded reference | Yes | No dedicated tool documented | Not documented | Mixed: `wbst` and lie-down off | High |
| UFO | BYOW; Hugging Face source in manifest | Latent prompt file | Yes | No dedicated tool documented | Not documented | Mixed: recovery stand off | Medium |
| BFM-Zero | BYOW; manifest source/hash | Latent prompt file | Yes | MuJoCo preview | Not documented | On | Medium |
| Instinct Parkour | BYOW; manifest source/hash | RealSense + torso IMU | Yes; camera checks separate | Dedicated viewer | Not documented | On | High |
| Holosoma | BYOW; manifest source/hash | Embedded reference | Yes | Official MuJoCo bridge notes | Not documented | On | High |
| ProtoMotions | BYOW; manifest source/hash | Motion cache | Yes | Dedicated viewer | Not documented | On | High |
| OmniXtreme | BYOW; manifest source/hash | Motion NPZ; TensorRT recommended | Yes | Dedicated viewer | Not documented | Off | Extreme |
| GentleHumanoid | External sibling repository and its assets | External Python runtime | External checker | No ActionGo-owned tool | Not documented | On via defaults | Medium–High |
| SteadyTray | BYOW; manifest source/hash | Camera + AprilTag or trusted pose input | Yes; camera checks separate | Zero-source smoke test only | Not documented | Off | High |
| unitree_rl_mjlab | BYOW/local export; provenance unresolved | Command input | Yes | ActionGo simulation bridge | Not documented | Off | Medium |
| TWIST | BYOW; manifest source/hash | PKL, mocap, UDP, ZMQ, or Redis source | Yes | No dedicated tool documented | Not documented | On | High |
| Humanoid-GPT | BYOW; exact release URL unresolved | Motion cache; optional MuJoCo FK | Yes | No dedicated viewer documented | Not documented | On | High |
| SONIC | BYOW; exact revision/terms unresolved | CSV or local ZMQ/VLA bridge (remote reference auth not implemented) | Yes | Upstream simulator bridge notes | Not documented | On | High |

Read the adapter README under `actions/<adapter>/` before use. Optional OpenCV support is accurately reflected at configure time: if OpenCV is absent, the sensor player target is disabled. GentleHumanoid is an external dependency rather than a bundled ActionGo runtime.

## Secure UDP Group Control

UDP control is disabled unless `--udp-listen` is passed and binds to `127.0.0.1` by default. Legacy text commands are accepted only on loopback. Any non-loopback, broadcast, or multicast listener requires a shared HMAC-SHA256 key file:

```bash
install -m 600 /dev/null /run/secrets/actiongo-udp.key
openssl rand -hex 32 > /run/secrets/actiongo-udp.key

./build/action_go_remote_test \
  --udp-listen --udp-bind 0.0.0.0 --udp-port 45678 \
  --udp-auth-key-file /run/secrets/actiongo-udp.key \
  --udp-auth-key-id lab-a
```

Send an authenticated command from the control machine:

```bash
python3 tools/remote/action_go_udp_send.py \
  --host <robot-ip> --action kick_right --start-after-ms 500 \
  --auth-key-file /run/secrets/actiongo-udp.key --auth-key-id lab-a
```

Signed messages include version, type, registered action, issue/execute timestamps, nonce, key ID, and signature. The receiver rejects stale or far-future messages, invalid signatures, nonce replay, unknown actions, and queue floods. Stop/damping commands receive queue priority. A shared key permits group control but does not identify individual senders; protect and rotate it outside this repository.

The supervisor command plane supports authenticated non-loopback UDP. Realtime TWIST and Humanoid-GPT reference inputs currently remain loopback-only; non-loopback reference transport is rejected until its separate authenticated protocol is implemented. YAML action `command` values are a trusted-code boundary and can launch processes. Only load reviewed local configuration. UDP clients can select an already registered action; they cannot upload commands, paths, YAML, or shell fragments.

## Developer Entry Points

- [Policy onboarding](docs/new_policy_onboarding.md): semantic reconnaissance and spec authoring.
- [Runtime architecture](docs/runtime_architecture.md): contracts, players, safety, and handoff boundaries.
- [Remote supervisor](docs/remote_supervisor.md): controls, authenticated UDP, and deployment behavior.
- [Simulation](docs/simulation.md): available simulator bridges.
- [Runtime troubleshooting](docs/runtime_troubleshooting.md): ONNX Runtime and TensorRT setup.
- [Contributing](CONTRIBUTING.md): safety and asset requirements for pull requests.

## Security, License, and Citation

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Apache-2.0 covers ActionGo-owned source code only; see [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md) for third-party boundaries. Models, data, robot firmware, and third-party projects retain their own terms.

Use [CITATION.cff](CITATION.cff) to cite the software release. A paper citation and experiment artifacts may be added in a later release; the `v0.1.0` software snapshot will remain immutable.
