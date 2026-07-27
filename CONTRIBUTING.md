# Contributing to ActionGo

ActionGo can publish low-level commands to a real humanoid robot. Contributions
must preserve the shared safety filter, LowState freshness checks, and
ready/start handoff semantics.

By submitting a contribution, you agree that your contribution is provided
under the repository's Apache License 2.0 and that you have the right to submit
it. Do not include third-party code, weights, motions, datasets, labels, logs,
or media unless their provenance and redistribution permission are documented.

Action YAML is trusted executable configuration: `command` can start a local
process. Review command templates as code, use fixed registered actions for
network selection, and never interpolate untrusted paths or shell fragments.

## Development checks

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/policy_spec_schema_check --all
python3 tools/assets/check_model_assets.py
```

New policies must follow `docs/new_policy_onboarding.md`: recover semantics from
the training/export/deployment code, map the policy to `PolicyContract`, add a
complete generic spec, and provide a `--check-only` command. Tensor dimensions
alone are not sufficient evidence of compatibility.

Do not commit robot logs containing personal data, local credentials, private
network details, or model assets whose redistribution terms are unknown. Add
model provenance to `MODEL_ASSETS.json` and keep `redistribution` blocked until
the rights holder's terms have been verified.

## Pull requests

- Keep action names, YAML entry points, and remote bindings backward compatible.
- Add deterministic tests for lifecycle, protocol, or safety changes.
- Include the exact build and check-only commands used for validation.
- Do not claim real-robot success without raw Event V1 logs and a run manifest.
- Do not require a contributor to run on hardware unless they make a
  real-robot claim; deterministic check-only, simulation, replay, and failure
  tests are the preferred review evidence.
- Never stage unrelated experiment media, paper drafts, or local model variants.
