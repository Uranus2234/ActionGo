# ActionGo Agent Instructions

Before adding or changing a policy, read:

1. `docs/agent_policy_deployment.md`
2. `docs/policy_catalog.md`
3. `docs/new_policy_onboarding.md`
4. The selected package's `actions/<family>/README.md`

Treat `config/motion_actions.yaml` and its included action files as the registry source of truth. Preserve existing user changes. Do not commit or push unless asked.

Never infer policy semantics from ONNX tensor shapes alone. Trace joint order, observation layout, normalization, timing, action decoding, controller gains, and reference/sensor behavior to the training and original deployment code. Keep models and motion assets out of Git unless their provenance and redistribution rights are recorded and cleared.

Keep real-robot commands behind the existing supervisor, handoff, and safety filters. Use `rt/user_lowcmd` for hardware; `rt/lowcmd` is for simulation. Do not bypass safety checks or run a policy on physical hardware as an unattended check. Stop at simulation and dry-run evidence until a human operator explicitly authorizes a protected hardware trial.

Report validation levels precisely: registration, `--check-only`, supervisor dry-run, closed-loop simulation, and protected real-robot validation are distinct evidence. A lower level never proves a higher one.
