## Change

Describe the behavior and why it belongs in a reusable runtime, adapter spec, or documentation layer.

## Safety and trust boundaries

- [ ] Normal policy `LowCmd` still passes through `CommandSafetyFilter`.
- [ ] LowState freshness, ready/start handoff, damping/fallback, and switch-back semantics are preserved or the behavioral change is documented and tested.
- [ ] No untrusted value reaches YAML `command`, a shell, path, or network-selected action.
- [ ] No unlicensed model, motion, data, label, binary, log, credential, or private path is included.

## Evidence

List exact commands and results. Do not claim real-robot verification without reviewable evidence.

```text
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/policy_spec_schema_check --all
python3 tools/assets/check_model_assets.py
```
