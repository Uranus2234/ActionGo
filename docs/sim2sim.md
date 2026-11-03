# Unified x86 sim2sim

`tools/launch/start_sim.sh` runs ActionGo players in one persistent MuJoCo G1
scene. Unitree mjlab velocity control replaces the robot's official WALKRUN
service. Flat ground, generated terrain and simulated sensor inputs share the
same scene and player switching path.

## Shared inference and one-command startup

Simulation launches the same C++ player executables, policy specs, observation
builders, ONNX inference and command safety filters used on the real robot.
The Python simulator contains no replacement policy inference implementation.
It owns physics, DDS bridging and player process management.

After configuring the dependencies and asset paths, run
`tools/launch/start_sim.sh` with no arguments. It loads the common scene,
starts the mjlab WALKRUN player on `lo`, waits for readiness and enables its
commands. The robot starts in WALKRUN without ARM or a second terminal. Closing
the viewer or pressing Ctrl-C stops the managed player processes too.

The stock Unitree MuJoCo bridge accepts `rt/lowcmd` and provides no robot
WALKRUN/UserCtrl switching service. Changing the network interface alone is
therefore insufficient for the real robot's handoff protocol. The native
simulation bridge supplies the command gate between `rt/lowcmd` and
`rt/user_lowcmd`, while keeping the policy inference code shared. The current
Python simulation coordinator is separate from the real C++ supervisor.

## Requirements

- An x86_64 ActionGo C++ build with `generic_policy_player` and the players used
  by the selected actions. Follow the main README for the CPU build.
- Python with MuJoCo 3.2 or newer, NumPy, PyYAML, Unitree SDK2 Python, and its compatible
  CycloneDDS Python package. Use the same Python environment for all imports.
- PyZMQ for PHP/P-BFM shared-memory depth, and a C++ sensor player built with
  ZeroMQ support. RealSense and AprilTag libraries are unnecessary for simulated inputs.
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
| `Shift+A` / `B` / `X` / `Y` | Remote face buttons |
| Arrow keys | Remote D-pad |
| `R` + `1` / `L` + `1` | Remote `R1` / `L1` |
| `R` + `2` / `L` + `2` | Remote `R2` / `L2` |
| `F` + `1` / `F` + `2` | Remote `F1` / `F2` |
| `Shift+S` or `Enter` / `T` | Remote Start / Select |
| `W` / `A` / `S` / `D` | Walk forward / left / backward / right |
| `Q` / `E` | Turn left / right |
| `Backspace` | Explicitly reset the simulation and start WALKRUN |
| `Space` | Pause/resume physics |

The GLFW viewer accepts held and released keys, so remote combinations from
`config/remote_actions.yaml` work unchanged. For example, `R` + `1` + `B`
generates remote `R1+B`. Press `Enter` or `Shift+S` to arm the configured action bindings.
Unmodified `A` and `S` only drive the joystick; they never send remote A or
Start. For remote `R1+A`, hold `R+1+Shift+A`. Held movement keys continue
alongside remote combinations, so holding `A` while pressing `B+1+R` keeps
sidestepping and sends only remote `R1+B`. Release movement keys to stop.
Shift selects the remote function of A/S, while W/D/Q/E remain available.
Press letter F and digit 1 together for remote F1; the physical F1 function
key is not used. For example, remote `F1+A` is `F+1+Shift+A`, `R1+F1` is
`R+1+F+1`, and `L1+Right` is `L+1+Right`.
Keyboard input belongs to the MuJoCo window, not the launching terminal.
Click that window, press Enter to arm, then hold the desired combination.
The window overlay shows whether it has focus and whether actions are armed.
It also shows the last raw keyboard event, with full key/axis records in
`events.jsonl` under `keyboard_input`. Physics/DDS run independently of desktop
event polling. With the default `--render-backend auto`, the viewer first tries
GPU EGL rendering in a separate thread. The desktop window only presents the
completed image, avoiding NVIDIA GLX buffer-swap stalls on virtual desktops.
The startup message identifies the actual renderer. If GPU EGL is unavailable,
auto falls back to GLFW and records the reason; `--render-backend egl` requires
GPU EGL, while `--render-backend glfw` selects direct window rendering.
Scene snapshots are prepared under the physics state lock before rendering;
the renderer never reads the live physics data. The overlay shows the backend
and current FPS. Per-second `viewer_timing` records include measured presentation
FPS, GPU rendering/readback time, and desktop presentation time.
Short button presses are queued through the supervisor so a press and release
between rendered frames still reaches the action dispatcher. Movement axes
use the current held keys and stop immediately on release or loss of focus.
Use
`--use-joystick` to combine the same controls with a gamepad. See
[legacy simulation controls](simulation.md) for recovery behavior. Ordinary
action/WALKRUN/DAMPING switching does not reset physics or load another XML.
Backspace and explicit reference-reset flags do reset it.

Policy bindings, including the actual keyboard chord, are printed in the
terminal at startup and again when the selected policy starts. WALKRUN and
DAMPING shortcuts are shown alongside the policy list. Detailed player output
still goes to the session logs.

Session event files, console updates, and latency-file reads run on a logging
thread so slow storage cannot block the physics/DDS loop. Shutdown drains
pending records before closing the logs. If the WALKRUN player stops, the
simulation switches to DAMPING, disarms action bindings, and displays the
player's recorded failure reason. The window remains open; Backspace resets
the simulation and starts WALKRUN again. LowState freshness checks remain enabled.
The LowState stream continues during each policy's ready/start handshake,
including when WALKRUN restarts after an action completes.

The viewer defaults to 1920x1080 with 4x multisampling and a 4096-pixel shadow
map. Use `--viewer-width` and `--viewer-height` to select another window size;
resizing uses the native framebuffer resolution. Duplicate collision meshes
are hidden only in the rendered scene, preserving physics and sensor raycasts.
Ctrl+left drag applies a force at the selected robot body; Ctrl+right drag
applies rotation. Hold Shift for horizontal manipulation. Double-click selects
a body, normal left drag rotates the camera, and the wheel zooms. Release the
button or leave window focus to clear the applied force and held keys.

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

## Terrain and perception

Select terrain once at startup; switching policies preserves it:

```bash
tools/launch/start_sim.sh --terrain stairs
tools/launch/start_sim.sh --terrain ramp --terrain-height 0.3
tools/launch/start_sim.sh --terrain rough --terrain-seed 7
```

`--terrain-start` defaults to 1.8 m ahead of spawn, leaving a flat standing
area. `--terrain-height` means individual step rise for stairs (five levels),
total elevation for the ramp, and maximum height for rough terrain.
`--terrain-length` and `--terrain-width` default to 3 m and 2 m. Terrain has
collision geometry and is visible to the depth rays. Rough terrain is a seeded
heightfield; `--hfield-file` remains available for an external heightfield and
cannot be combined with generated terrain. Use the existing gamepad controls
with `--use-joystick` to approach the terrain; adding terrain does not command
the robot to traverse it.

PHP and P-BFM launch the same `generic_sensor_policy_player` as hardware. The
simulator raycasts optical-Z depth from the current robot pose, preprocesses it
and publishes the existing float32 shared-memory/ZeroMQ heartbeat protocol.
No OpenGL renderer is needed. Only the active sensor policy captures depth;
the default is 50 Hz, configurable with `--depth-hz`. The physics thread owns
capture, including while waiting for player readiness, so sensor reads do not
race physics updates. Capture costs appear in sensor logs; expensive scenes can
reduce realtime throughput and must be checked before interpreting behavior.

| Policy | Simulated input |
|---|---|
| P-BFM | 36x64 optical-Z, calibrated head mount, existing crop/blur/8-frame history, 8x18x32 float32 stream |
| PHP | 60x106 optical-Z, 58x87 bicubic antialias crop/resize/normalization, torso IMU on `rt/secondary_imu` |
| SteadyTray | Object body pose transformed into the configured policy frame, loopback UDP `x y z qw qx qy qz` |

Each activation gets a unique depth stream and a fresh history. Stream memory,
heartbeat sockets and player processes are cleaned up on switch or exit.
The real sensor YAML files stay intact; only the session copies select SHM.
Sensor C++ runtimes accept `--domain-id` (hardware default 0), matching the
simulator's domain 42. Missing/stale sensor inputs remain player errors and
cause the coordinator to enter DAMPING.

Test sensor switching with matching public assets fetched first:

```bash
tools/launch/start_sim.sh --headless --terrain stairs \
  --sequence php_locomotion pbfm_joystick \
  --sequence-action-sec 4 --sequence-walkrun-sec 3
```

PHP uses a 27-degree downward optical axis relative to the torso. Its default
translation uses the G1 head mount recorded for P-BFM; the complete official
PHP translation has not been established. Set `--php-camera-pos X Y Z` and
`--php-camera-pitch-deg DEG` to match the intended camera calibration before
evaluating terrain performance. Raycasts provide ideal depth, without camera
noise, occlusion artifacts from the real device, or transport delay.

SteadyTray requires an XML containing the actual object and a calibrated
`d435_link` policy frame, or its parent plus a calibrated local transform:

```bash
tools/launch/start_sim.sh --xml /path/to/shared_g1_with_tray.xml \
  --object-body ball --object-frame-body d435_link \
  --auto-action steadytray_balance
```

`--object-frame-pos X Y Z` and `--object-frame-quat W X Y Z` express the policy
frame relative to `--object-frame-body`. They are applied before transmitting
the object pose. The UDP receiver binds only to loopback in simulation;
`--object-port 0` chooses an available port. Missing/ambiguous bodies fail
explicitly. The stock G1 XML has no tray or object, so SteadyTray remains
deferred until `--object-body` is supplied. Its upstream tray dynamics have not
been reproduced here. This is a ground-truth pose bridge; it does not exercise
AprilTag detection or validate balancing in a substitute tray model.

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
- `sensors/`: per-activation frame JSONL, retained raw/processed depth or object
  pose NPZ snapshots, and a `<variant>.latest.npz` link. Frame records include
  capture time, simulation time, sequence, shape, valid fraction and value
  range; PHP also records torso quaternion, gyro and accelerometer. Snapshots
  default to one per second; use `--sensor-log-period-sec` to change it.
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

PHP/P-BFM have completed short WALKRUN switching sequences with terrain present.
These runs stay near the flat spawn and do not establish terrain traversal
performance. PHP preprocessing is compared against the C++ preprocessor; scene
tests check collision heights, depth changes, torso IMU and object frame transforms.
An injected depth heartbeat outage makes the C++ player report stale depth and
the coordinator enter DAMPING with a failed sequence. A rotated scene object
pose has also been checked through the C++ SteadyTray UDP camera-check path;
that check does not run the balance policy.
SteadyTray still requires a matching tray/object XML for physical validation.
External realtime references (including P-BFM Pico), Instinct, and GentleHumanoid
remain deferred. UniSuperMimic is excluded for now. Other
registered offline actions can be checked and selected, but require individual
physical validation in the common scene. Missing models, reference files, and
offline conversion dependencies remain explicit check failures.
