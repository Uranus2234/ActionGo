#!/usr/bin/env python3
"""Fail when a proposed public snapshot contains release-blocked material."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FORBIDDEN_NAMES = {".git", "outputs", "experiments", "paper", "README_GITHUB_DRAFT.md"}
FORBIDDEN_SUFFIXES = {
    ".onnx", ".onnx.data", ".npz", ".npy", ".pkl", ".pt", ".pth",
    ".engine", ".plan", ".so", ".a", ".o", ".pyc", ".zip", ".tgz",
    ".tar", ".tar.gz", ".7z", ".png", ".jpg", ".jpeg", ".svg", ".xml",
    ".mjcf", ".urdf", ".dae", ".stl", ".obj", ".bin",
}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
]
# The Unitree service template deliberately uses /home/unitree/action_go. Other
# user-home paths usually reveal a development machine or stale private layout.
PRIVATE_PATH = re.compile(r"/(?:home|Users)/[^/\s]+/")

LARGE_CONTEXT_FILES = {
    "actions/pbfm/motions/lafan_dance/context.yaml",
    "actions/pbfm/motions/lafan_dance_56_52_98_72/context.yaml",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    repo = args.repo.resolve()
    failures: list[str] = []

    for path in repo.rglob("*"):
        relative = path.relative_to(repo)
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            failures.append(f"forbidden path: {relative}")
            continue
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            failures.append(f"forbidden binary/asset: {relative}")
            continue
        if path.stat().st_size > 5 * 1024 * 1024 and relative.as_posix() not in LARGE_CONTEXT_FILES:
            failures.append(f"unexpected file over 5 MiB: {relative}")
            continue
        with path.open("rb") as stream:
            head = stream.read(160)
        if head.startswith(b"\x7fELF") or head.startswith(b"MZ"):
            failures.append(f"forbidden executable binary: {relative}")
            continue
        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            failures.append(f"Git LFS pointer instead of source content: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                failures.append(f"possible secret in {relative}: {pattern.pattern}")
        path_scan_text = text.replace("/home/unitree/action_go", "/robot/action_go")
        if PRIVATE_PATH.search(path_scan_text):
            failures.append(f"machine-specific absolute path in {relative}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] public snapshot contains no blocked history, assets, secrets, or private paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
