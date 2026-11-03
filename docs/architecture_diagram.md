# ActionGo Architecture Diagram

GitHub renders the Mermaid block below natively. To embed it in `README.md`,
copy the fenced block, or export an SVG with
`npx -y @mermaid-js/mermaid-cli -i docs/architecture_diagram.md -o docs/architecture.svg`.

```mermaid
flowchart TB
  subgraph OPERATOR["Operator"]
    RC["Unitree remote<br/>(button combos)"]
    UDP["UDP control plane<br/>loopback text / HMAC-signed JSON<br/>(select registered actions only)"]
  end

  subgraph SUP["Supervisor &nbsp;·&nbsp; action_go_remote_test"]
    SHELL["Shell state machine<br/>SHELL_IDLE → ARMED"]
    SAFE["IMU safety monitor<br/>tilt ≥ fall_angle for fall_hold → damping<br/>manual / auto recovery policy"]
    MAC["MotionActionController<br/>lifecycle · process-group supervision"]
  end

  CFG[("motion_actions.yaml + action YAML<br/>(trusted executable config)")]

  subgraph HANDOFF["File-coordinated handoff"]
    READY{{"ready_file"}}
    START{{"start_file"}}
    DONE{{"complete / release files<br/>(finite actions, switch-back overlap)"}}
  end

  subgraph PLAYER["Player child process &nbsp;·&nbsp; generic_(policy|motion_tracking|sensor|composite)_player"]
    SPEC[("policy spec YAML")]
    CONTRACT["PolicyContract + protocol constraints<br/>schema check · --check-only · --dry-run"]
    OBS["Observation assembly<br/>state / embedded-mimic time / motion ref /<br/>depth / object pose"]
    ONNX["OnnxRunner<br/>CPU · CUDA · TensorRT<br/>(finite + exact-shape checks)"]
    FILT["CommandSafetyFilter<br/>NaN/Inf · joint & gain limits ·<br/>stale-LowState gate"]
    PUB["AsyncCommandPublisher<br/>fixed-rate interpolation"]
  end

  subgraph REFS["Reference / sensor inputs"]
    OFFLINE[("offline motion cache<br/>CSV / NPZ / PKL")]
    RT["realtime refs (TWIST, Humanoid-GPT, PICO)<br/>UDP / ZMQ / Redis — loopback-only"]
    DEPTH["RealSense / shared-memory depth"]
  end

  subgraph ROBOT["Unitree G1"]
    OFFICIAL["Official controller<br/>walkrun / passive (damping)"]
    LS[["rt/lowstate (DDS)"]]
    LC[["rt/user_lowcmd (DDS)"]]
  end

  RC --> SHELL
  UDP --> SHELL
  LS --> SAFE
  SHELL --> MAC
  SAFE -- "fall / recovery interrupt" --> MAC
  CFG --> MAC

  MAC -- "1 · prestart player" --> PLAYER
  SPEC --> CONTRACT --> OBS
  PLAYER -- "2 · signals ready" --> READY --> MAC
  MAC -- "3 · current-pose hold + prearm<br/>SwitchToUserCtrl" --> OFFICIAL
  MAC -- "4 · signals start" --> START --> PLAYER

  LS --> OBS
  OFFLINE --> OBS
  RT --> OBS
  DEPTH --> OBS
  OBS --> ONNX --> FILT --> PUB --> LC

  PLAYER -- "5 · action complete" --> DONE --> MAC
  MAC -- "6 · SwitchToInternalCtrl<br/>walkrun / passive, or damping on fault" --> OFFICIAL
```

## Reading the diagram

- Everything inside **Supervisor** runs in one process; each **Player** is a
  separate child process in its own process group (`PDEATHSIG`-bound to the
  supervisor), so a supervisor crash cannot leave a payload publishing
  `LowCmd`.
- Steps 1–6 are the nominal lifecycle of one prestarted action. Non-prestart
  actions skip steps 1–2 and publish a prearm packet instead.
- Every normal policy command crosses **CommandSafetyFilter** before
  `rt/user_lowcmd`; the current-pose prearm hold is the only narrowly scoped
  exception and uses fixed validated gains.
- Trust boundaries: action YAML and spec YAML are trusted code; models,
  motion data, UDP/ZMQ/Redis bytes and depth frames are untrusted inputs and
  are validated or bounded before use.
