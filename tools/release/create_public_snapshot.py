#!/usr/bin/env python3
"""Create and validate a code-only ActionGo release snapshot without Git history."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ALWAYS_EXCLUDED = {
    "README_GITHUB_DRAFT.md",
    "EXPERIMENTS.md",
    "paper",
    "outputs",
    "experiments",
    "tools/experiments",
    "docs/papers",
    "AGENTS.md",
    "build",
    "build-debug",
    "CMakeFiles",
    "{handoff_complete_file}",
    "docs/GPT_PRO_REVIEW_PROMPT.md",
    "docs/open_source_plan.md",
    "tools/motion_prior",
    "runtime/tests/test_motion_prior.py",
}
EXCLUDED_PREFIXES = (
    "docs/actiongo_", "docs/preview_handoff_runtime.md",
    "docs/MOTION_PRIOR_RESIDUAL_BRIDGE_",
    "config/motion_prior_",
)
COMPILED_SUFFIXES = {".so", ".a", ".o", ".pyc"}
MANAGED_SUFFIXES = {
    ".onnx", ".onnx.data", ".npz", ".npy", ".pkl", ".pt", ".pth",
    ".engine", ".plan", ".csv", ".pdf", ".png", ".jpg", ".jpeg", ".svg",
    ".xml", ".mjcf", ".urdf", ".dae", ".stl", ".obj", ".bin",
}
EXCLUDED_DIRECTORY_PREFIXES = ("thirdparty/onnxruntime-",)


def pbfm_context(path: Path) -> bool:
    text = path.as_posix()
    return (text.startswith("actions/pbfm/motions/") and path.name in {"context.yaml", "initial_state.json"} or
            text.startswith("actions/pbfm/config/context_") and path.suffix == ".yaml" or
            text.startswith(("actions/pbfm/config/joystick_", "actions/pbfm/config/keyboard_")) and
            path.name in {"context.yaml", "idle.yaml", "turn_left.yaml", "turn_right.yaml"})


def tracked_paths(repo: Path) -> list[Path]:
    output = subprocess.check_output([
        "git", "-C", str(repo), "ls-files", "-z", "--cached", "--others",
        "--exclude-standard",
    ])
    return [Path(item) for item in output.decode("utf-8").split("\0") if item]


def manifest_assets(repo: Path) -> set[Path]:
    manifest = json.loads((repo / "MODEL_ASSETS.json").read_text(encoding="utf-8"))
    return {Path(path) for group in manifest["assets"] for path in group["paths"]}


def excluded(path: Path, assets: set[Path]) -> bool:
    if pbfm_context(path):
        return True
    path_text = path.as_posix()
    if (path in assets or path.suffix.lower() in COMPILED_SUFFIXES or
            any(path_text.lower().endswith(suffix) for suffix in MANAGED_SUFFIXES)):
        return True
    if path.parts and path.parts[0] in ALWAYS_EXCLUDED:
        return True
    return (path_text in ALWAYS_EXCLUDED or
            any(path_text.startswith(prefix + "/") for prefix in ALWAYS_EXCLUDED) or
            path_text.startswith(EXCLUDED_PREFIXES) or
            path_text.startswith(EXCLUDED_DIRECTORY_PREFIXES))


def run_checks(snapshot: Path) -> None:
    commands = [
        ["python3", "tools/assets/check_model_assets.py", "--permit-missing", "--public-release"],
        ["python3", "tools/release/public_snapshot_check.py", "--repo", "."],
    ]
    for command in commands:
        subprocess.run(command, cwd=snapshot, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {output}; pass --force to replace it")
        if output == repo or repo in output.parents or output in repo.parents:
            raise SystemExit("refusing to replace the source repository or an ancestor/descendant")
        shutil.rmtree(output)

    assets = manifest_assets(repo)
    with tempfile.TemporaryDirectory(prefix="actiongo-public-") as temp_dir:
        staging = Path(temp_dir) / "ActionGo"
        for relative in tracked_paths(repo):
            if excluded(relative, assets):
                continue
            source = repo / relative
            if not source.is_file():
                continue
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if not args.skip_checks:
            run_checks(staging)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), output)

    print(f"Public snapshot created at {output}")
    print("It contains no .git directory and must remain private until the patent release gate is cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
