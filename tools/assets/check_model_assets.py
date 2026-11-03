#!/usr/bin/env python3
"""Verify bundled model hashes and enforce public-release redistribution gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


MANAGED_SUFFIXES = {
    ".onnx", ".onnx.data", ".npz", ".npy", ".pkl", ".pt", ".pth",
    ".engine", ".plan", ".csv", ".pdf", ".png", ".jpg", ".jpeg", ".svg",
    ".xml", ".mjcf", ".urdf", ".dae", ".stl", ".obj", ".bin",
}


def tracked_managed_assets(repo: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        paths = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    else:
        # Source-only release snapshots intentionally contain no Git metadata.
        paths = [path.relative_to(repo).as_posix() for path in repo.rglob("*") if path.is_file()]
    managed = set()
    for relative in paths:
        lowered = relative.lower()
        if relative and (repo / relative).is_file() and any(lowered.endswith(suffix) for suffix in MANAGED_SUFFIXES):
            managed.add(relative)
    return managed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def asset_package(relative: str) -> str:
    parts = Path(relative).parts
    if len(parts) >= 4 and parts[0] == "actions" and parts[1] in {"proprio", "tracking", "perception"}:
        return parts[2]
    return parts[1] if len(parts) > 1 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, default=Path("MODEL_ASSETS.json"))
    parser.add_argument("--public-release", action="store_true",
                        help="fail if any bundled asset lacks verified redistribution permission")
    parser.add_argument("--permit-missing", action="store_true",
                        help="allow manifest-listed assets to be absent in a code-only export")
    parser.add_argument("--installed-only", action="store_true",
                        help="verify present assets and report absent ones without failing")
    parser.add_argument("--policy", action="append", default=[],
                        help="check one package under actions/<category>; may be repeated")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else repo / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known_policies = {asset_package(path) for group in manifest["assets"] for path in group["paths"]
                      if path.startswith("actions/")}
    unknown_policies = set(args.policy) - known_policies
    if unknown_policies:
        parser.error("unknown asset package(s): " + ", ".join(sorted(unknown_policies)))
    selected_policies = set(args.policy)
    permit_missing = args.permit_missing or args.installed_only
    results = []
    failures = 0
    declared_paths: set[str] = set()
    for group in manifest["assets"]:
        paths = group.get("paths", [])
        hashes = group.get("sha256", [])
        if len(paths) != len(hashes):
            raise SystemExit(f"paths/hash count mismatch for {group.get('source')}")
        release_allowed = group.get("redistribution") == "allowed"
        for relative, expected in zip(paths, hashes):
            if selected_policies and asset_package(relative) not in selected_policies:
                continue
            if relative in declared_paths:
                failures += 1
                results.append({"path": relative, "matches": False,
                                "error": "duplicate manifest path"})
                continue
            declared_paths.add(relative)
            path = repo / relative
            pointer = is_lfs_pointer(path)
            exists = path.is_file() and not pointer
            if args.public_release and exists and not release_allowed:
                failures += 1
            actual = sha256_file(path) if exists else ""
            matches = exists and actual == expected
            failures += int(not matches and not (permit_missing and not exists))
            results.append({
                "path": relative,
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": matches,
                "lfs_pointer": pointer,
                "source": group.get("source", ""),
                "redistribution": group.get("redistribution", ""),
                "public_release_allowed": release_allowed,
            })

    unmanaged = sorted(tracked_managed_assets(repo) - declared_paths)
    for relative in unmanaged:
        if selected_policies and asset_package(relative) not in selected_policies:
            continue
        failures += 1
        results.append({
            "path": relative,
            "exists": (repo / relative).is_file(),
            "expected_sha256": "",
            "actual_sha256": "",
            "matches": False,
            "source": "",
            "redistribution": "unmanaged",
            "public_release_allowed": False,
            "error": "tracked managed asset is missing from manifest",
        })
    if args.json:
        print(json.dumps({"assets": results, "failures": failures}, indent=2))
    else:
        for result in results:
            if result["matches"]:
                status = "OK"
            elif permit_missing and not result["exists"]:
                status = "MISSING-ALLOWED"
            else:
                status = "MISMATCH"
            release = result["redistribution"]
            print(f"[{status}] {result['path']} redistribution={release}")
            if not result["exists"]:
                print(f"  provide from: {result['source']}")
            if result.get("lfs_pointer"):
                print("  note: Git LFS pointer is not an installed asset")
            if result.get("error"):
                print(f"  error: {result['error']}")
        if args.public_release:
            print("Public-release mode requires redistribution=allowed for every asset actually bundled.")
        installed = sum(result.get("matches", False) for result in results)
        print(f"Checked {len(results)} asset(s): {installed} installed and matching, {failures} failure(s).")
        if args.installed_only and installed == 0:
            print("No installed assets were validated; this does not establish deployment readiness.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
