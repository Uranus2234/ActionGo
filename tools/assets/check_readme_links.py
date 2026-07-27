#!/usr/bin/env python3
"""Check repository-relative Markdown links without making network requests."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote


LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    failures: list[str] = []
    checked = 0
    for markdown in sorted(repo.rglob("*.md")):
        if any(part in {".git", "build", "outputs", "thirdparty"} for part in markdown.relative_to(repo).parts):
            continue
        text = markdown.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            for raw in LINK.findall(line):
                target = raw.strip().split(maxsplit=1)[0].strip("<>")
                if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                local = unquote(target.split("#", 1)[0])
                if not local:
                    continue
                checked += 1
                resolved = (markdown.parent / local).resolve()
                if repo not in resolved.parents and resolved != repo:
                    failures.append(f"{markdown.relative_to(repo)}:{line_number}: escapes repository: {target}")
                elif not resolved.exists():
                    failures.append(f"{markdown.relative_to(repo)}:{line_number}: missing: {target}")
    for failure in failures:
        print(f"[FAIL] {failure}")
    if failures:
        return 1
    print(f"[OK] checked {checked} repository-relative Markdown links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
