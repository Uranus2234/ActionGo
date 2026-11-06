# Optional BFM stand bridge

The stand bridge adapts the prepared-player and command-owner handoff from
[ReadyGo](https://github.com/vincent2069/action_go/tree/c552b66dc800ecb2891299b5d5d647cf4df494e6).
It does not load HandoffGuard or use risk scores to approve a switch.

## Enable

The feature is off by default. Enable it for the current launch:

```bash
tools/launch/start_sim.sh --stand-bridge
ACTION_GO_NET=<robot-interface> tools/launch/start_safe.sh --stand-bridge
ACTION_GO_NET=<robot-interface> tools/launch/start.sh start --stand-bridge
```

Use `--no-stand-bridge` to explicitly select the original switching behavior.
The local `ActionGo-local/run_sim2sim.sh` wrapper forwards these options too.
The BFM model and fixed latent are the existing `bfm_zero_stand` assets; no
weights are imported from ReadyGo. A risk-model option cannot be combined with
the stand bridge.

## Controls and scope

ARM with Start first (Enter in the MuJoCo window). Ordinary policy bindings
then request `current controller -> bfm_zero_stand -> requested policy`,
including requests made while another supported policy is running.

- `R2+A` requests a bridge to WALKRUN while a policy is running and the shell
  is armed. Simulation keyboard: `R+2+Shift+A`. This uses the existing shell
  WALKRUN binding and follows any configured override of that binding.
- Requests during a pending bridge are ignored; the pending target is not replaced.

Explicit switches use the bridge. Natural action completion retains the
action's existing completion/release and switch-back behavior. Emergency stop,
damping, process shutdown, sensor/reference input loss, lie-down, and existing
fall recovery retain their direct fallback paths. They do not wait for BFM.

The real supervisor requires supported prestart players and private command
routes. Unsupported targets or preflight-only adapters are rejected before
replacing the source. Simulation requires the native bridge and retains its
existing policy exclusions. This feature does not establish new policy
readiness or permit switching safely from arbitrary airborne/rotating poses.

## Handoff

1. Prestart BFM while the current controller continues running. Its hold
   commands go to a private DDS topic and do not drive the robot.
2. Require the ready signal and a command younger than 100 ms, then select
   BFM's command route before stopping the old player. Real hardware stays in
   UserCtrl during a policy-to-policy handoff.
3. Prestart the requested policy while BFM stands, or prepare WALKRUN after
   standing is confirmed. Preserve each player's observation, activation,
   sensor readiness, and command safety checks.
4. Require five distinct fresh IMU samples at the supervisor check cadence:
   tilt below 0.20 rad (about 11.46 degrees), angular speed below 0.30 rad/s,
   and sample age below 100 ms. Invalid or unstable input resets confirmation.
5. Select the fresh target route, or wait for WALKRUN acknowledgement before
   retiring BFM. The existing real WALKRUN overlap remains in effect.

A bridge has a 15-second deadline. A preparation timeout leaves the original
controller active. Failure after BFM takes ownership enters damping. Explicit
stop/reset discards prepared processes and prevents a delayed target start.
The source retains its configured fall monitoring during preparation. While
BFM owns the bridge, its get-up motion uses the standing confirmation and
deadline above; the normal source fall-angle trigger is suspended.
The terminal shows bridge requests, the current policy, and completion/failure;
the existing session logs retain the detailed lifecycle and player output.

## Validation

```bash
ctest --test-dir <build-dir> -R 'motion_action_controller_test|policy_command_relay_test|stand_bridge_test|simulation_recovery_test' --output-on-failure
tools/launch/start_sim.sh --headless --stand-bridge \
  --sequence ufo_idle --sequence-action-sec 2 --sequence-walkrun-sec 2
```

The sequence exercises WALKRUN -> BFM -> target -> BFM -> WALKRUN in one scene.
Closed-loop headless validation covered that sequence with `ufo_idle`, the
original path with the feature disabled, and direct switches from `ufo_idle`
through BFM to `ufo_stand` and back through BFM to WALKRUN. The DDS relay test
uses loopback and private test topics to check command ownership, stale-state
rejection, invalid-command rejection, and route release.
Unit tests and a simulation run do not constitute protected real-robot validation.
