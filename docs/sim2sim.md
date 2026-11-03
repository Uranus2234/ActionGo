# Unified x86 sim2sim

`tools/launch/start_sim.sh` runs ActionGo players in one persistent MuJoCo G1
scene. Unitree mjlab velocity control replaces the robot's official WALKRUN
service. The first integration target is flat ground and policy switching.

## Requirements

- An x86_64 ActionGo C++ build with `generic_policy_player` and the players used
  by the selected actions. Follow the main README for the CPU build.
- Python with MuJoCo 3.x, NumPy, PyYAML, Unitree SDK2 Python, and its compatible
  CycloneDDS Python package. Use the same Python environment for all imports.
- A local `unitree_mujoco` checkout containing
  `unitree_robots/g1/scene_29dof.xml` and its referenced robot assets.
- The mjlab WALKRUN ONNX export, its matching params, and the selected action's
  model and reference assets. The code-only repository does not bundle them.
- A desktop display for the viewer, or `--headless`. Pygame is needed only for
  `--use-joystick`; Python ONNX Runtime is needed only for optional risk-model
  inference or resetting to an embedded ONNX reference.

Configure paths once; these are examples to replace with local locations:

```bash
export ACTION_GO_SIM_PYTHON=/path/to/python
export UNITREE_SDK2PY_ROOT=/path/to/unitree_sdk2_python
export UNITREE_MUJOCO_ROOT=/path/to/unitree_mujoco
export ACTION_GO_BUILD_DIR=/path/to/ActionGo/build
export ACTION_GO_LOG_DIR=/path/to/sim2sim-runs

tools/launch/start_sim.sh --list-actions
```

Use [the asset downloader](model_assets.md) for assets with verified public
sources. The currently configured mjlab WALKRUN export still requires a
matching local export; no public download URL is asserted for it.

An existing private asset tree can be used without copying weights into the
source repository:

```bash
export ACTION_GO_ASSET_ROOT=/path/to/existing-ActionGo
tools/launch/start_sim.sh --check sim_walkrun ufo_idle kick_right holosoma_dance
```

The external tree must mirror `actions/<package>/...`. Each session creates a
resource overlay: repository files take priority; missing files and Git LFS
pointers can use external assets. YAML files are copied into the session and
other assets are linked. Existing source configuration is not rewritten.
Only supply trusted configuration and assets.

`--check` runs player model/spec acceptance without DDS or physics. No names
means all registered actions. Missing assets fail; deferred integrations are
reported as `SKIP`. `REGISTERED` in `--list-actions` describes configuration
availability, not proven stability.

## Run and switch

Start the viewer with mjlab WALKRUN:

```bash
tools/launch/start_sim.sh
```

Run a bounded sequence in the same scene:

```bash
tools/launch/start_sim.sh --headless \
  --sequence ufo_idle holosoma_dance \
  --sequence-action-sec 5 --sequence-walkrun-sec 3
```

The sequence is:

```text
WALKRUN -> ufo_idle -> WALKRUN -> holosoma_dance -> WALKRUN
```

Each action runs for at most the requested wall time, or ends earlier according
to its own player configuration. These short runs exercise switching; they do
not validate a complete motion clip. Before each next action and at the end,
the sequence requires a live WALKRUN player, root height at least 0.5 m, and
root tilt at most 30 degrees. Failure returns a nonzero exit code. Adjust the
acceptance bounds with `--sequence-walkrun-min-height` and
`--sequence-walkrun-max-tilt-deg`; the checks apply during WALKRUN, allowing
dynamic motions to leave an upright pose during their own phase.

Use normal realtime pacing for these checks. `--no-realtime` can change the
relationship between wall-clock policy inference and physics and is not an
equivalent stability test. Model loading and ready/start handshakes can briefly
pause physics; the simulator preserves the existing pose and velocity.

For an interactive action, choose it at launch:

```bash
tools/launch/start_sim.sh --auto-action holosoma_dance
```

| Viewer key | Effect |
|---|---|
| `W` | Stop the action and enter mjlab WALKRUN |
| `P` | Stop players and enter DAMPING |
| `V` | Restart the action selected by `--auto-action` |
| `R` | Explicitly reset the simulation and start WALKRUN |
| `Space` | Pause/resume physics |
| `S` / `B` | ARM the existing primary/alternate recovery controls |
| `T` / `G` | After ARM, start UFO tracking/stand |

Gamepad bindings remain in `config/remote_actions.yaml`; opt in with
`--use-joystick`. See [legacy simulation controls](simulation.md) for the full
recovery behavior. Ordinary action/WALKRUN/DAMPING switching does not reset
physics or load another XML. `R` and explicit reference-reset flags do reset it.

## Scene and control contract

Defaults come from `config/sim2sim.yaml`; command-line flags override them.
The native bridge uses loopback and DDS domain 42, with a 2 ms physics step.
Each independent simulator needs a separate domain. `--xml` selects a common
scene at startup. The unified profile requires 29 uniquely named G1 motor
joints, one unit-geared torque actuator per joint, and canonical Unitree DDS
motor ordering. Actuator array order is resolved by joint name. Position-servo
XMLs are rejected because they would ignore individual policy PD gains.

The bridge publishes `rt/lowstate` and applies exactly one command source:

| State | Command source |
|---|---|
| `WALKRUN` | mjlab player on `rt/lowcmd` |
| `USER_CTRL` | Selected ActionGo player on `rt/user_lowcmd` |
| `DAMPING` | Local passive damping |

Players use private ready/start files for each activation. When the player
supports complete/release handoff, it holds its final target until WALKRUN is
ready and the command gate has switched. Missing/stale commands fall back to
damping after 200 ms; nonfinite commands and negative gains are rejected.
Unexpected player failures fail a sequence and stop its processes.

The common scene retains one set of masses, contacts, joint limits and actuator
properties. Policy-specific gains and targets come from each player. The
simulator does not automatically replace the scene with each policy's training
physics; passing model checks does not establish compatible dynamics.

## Logs and current latency

The normal TTY display updates in place, with the policy name highlighted:

```text
-----------------------------------------------
  STATE    USER_CTRL
  POLICY   holosoma_dance    E2E  2.1 ms
-----------------------------------------------
```

Only the latest available E2E sample is shown. It measures local player
LowState receipt through LowCmd publication, including inference and publisher
waiting. It does not include subsequent bridge reception or MuJoCo torque
application. `--` means no sample; `STALE` means it is older than one second.
Redirected/non-TTY output prints state changes and one current sample per second.

Each run has a private session directory and a `latest` symlink:

- `session.json`: launch arguments, scene, DDS, build and asset paths, timestep and revision.
- `events.jsonl` / `events.log`: state changes, notices, sequence transitions,
  WALKRUN acceptance checks, and root pose/velocity/contact telemetry.
- `runtime.output.log`: detailed Python diagnostics.
- `actions/`: separate player stdout/stderr and structured runtime/timing logs.
- `resources/`: frozen YAML configuration and linked model/reference assets.
- `summary.json`: final exit code and success status.

The unified profile records physics telemetry every 0.5 seconds. Use
`--state-log-period-sec` to change it. `--console debug` exposes diagnostic
output directly; `--console quiet` suppresses the live panel. Native library
startup warnings can still appear. Existing C++ structured logs retain their
rotation settings; the Python session files have no automatic size rotation.
The existing `tools/logs/manage_logs.py` can list and prune session directories.

## Validation scope

Local tests use the stock flat G1 `scene_29dof.xml`, mjlab WALKRUN, UFO idle,
and Holosoma dance. Sequence logs record the same model instance and increasing
simulation time across transitions. BeyondMimic `kick_right` passes model
acceptance and has completed some sequences, but repeated tests also show falls
during the motion after entering from WALKRUN. Its physical stability in the
common scene remains unresolved; use explicit acceptance checks when testing
it. A short switching test does not validate all clips or entry poses.

Depth-conditioned PHP/P-BFM, object-conditioned SteadyTray, external realtime
references, Instinct, and GentleHumanoid need separate input/scene integration
and are deferred by this entry. UniSuperMimic is excluded for now. Other
registered offline actions can be checked and selected, but require individual
physical validation in the common scene. Missing models, reference files, and
offline conversion dependencies remain explicit check failures.
