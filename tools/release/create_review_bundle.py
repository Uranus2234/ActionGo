#!/usr/bin/env python3
"""Create a source-only archive for an external GPT Pro review."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def sanitize(text: str, repo: Path) -> str:
    replacements = {
        str(repo): "<LOCAL_REPO>",
        str(repo.parent): "<LOCAL_PARENT>",
        str(Path.home()): "<LOCAL_HOME>",
    }
    for original, replacement in sorted(replacements.items(), key=lambda item: -len(item[0])):
        text = text.replace(original, replacement)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="actiongo-review-") as temp_dir:
        snapshot = Path(temp_dir) / "ActionGo-review"
        subprocess.run([
            "python3", str(repo / "tools/release/create_public_snapshot.py"),
            "--repo", str(repo), "--output", str(snapshot),
        ], check=True)
        reports = snapshot / "review_materials"
        reports.mkdir()
        (reports / "GPT_PRO_REVIEW_PROMPT.md").write_text(
            (repo / "docs/GPT_PRO_REVIEW_PROMPT.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        tree = "\n".join(
            path.relative_to(snapshot).as_posix()
            for path in sorted(snapshot.rglob("*")) if path.is_file()
        ) + "\n"
        (reports / "SOURCE_TREE.txt").write_text(tree, encoding="utf-8")
        test = subprocess.run(
            ["ctest", "--test-dir", str(repo / "build"), "--output-on-failure"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (reports / "LOCAL_TEST_OUTPUT.txt").write_text(
            sanitize(test.stdout, repo), encoding="utf-8")
        checks = [
            [str(repo / "build/policy_spec_schema_check"), "--all", "--allow-missing-assets"],
            ["python3", "tools/assets/check_model_assets.py", "--permit-missing"],
            ["python3", "tools/assets/check_readme_links.py"],
        ]
        check_output = []
        for command in checks:
            result = subprocess.run(command, cwd=repo, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            check_output.append(sanitize(
                f"$ {' '.join(command)}\nexit={result.returncode}\n{result.stdout}", repo))
        (reports / "RELEASE_CHECK_OUTPUT.txt").write_text(
            "\n".join(check_output), encoding="utf-8")
        if shutil.which("cppcheck"):
            static = subprocess.run(
                ["cppcheck", "--enable=warning,performance,portability", "--std=c++17",
                 "--suppress=missingIncludeSystem", "--suppress=preprocessorErrorDirective",
                 "runtime/src", "runtime/apps",
                 "runtime/players", "runtime/tools"],
                cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            static_output = sanitize(static.stdout, repo)
        else:
            static_output = "cppcheck was not installed; no static-analysis result is claimed.\n"
        (reports / "CPPCHECK_OUTPUT.txt").write_text(static_output, encoding="utf-8")
        subprocess.run([
            "python3", str(snapshot / "tools/release/public_snapshot_check.py"),
            "--repo", str(snapshot),
        ], check=True)
        shutil.make_archive(str(output.with_suffix("")), "zip", snapshot.parent, snapshot.name)
    print(output.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
