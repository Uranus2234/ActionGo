#!/usr/bin/env python3
"""Fetch pinned upstream assets from MODEL_ASSETS.json and verify every output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def asset_package(relative: str) -> str:
    parts = Path(relative).parts
    if len(parts) >= 4 and parts[0] == "actions" and parts[1] in {"proprio", "tracking", "perception"}:
        return parts[2]
    return parts[1] if len(parts) > 1 else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe asset destination: {relative}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"asset destination escapes repository: {relative}")
    return target


def download(url: str, output: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"expected HTTPS asset URL: {url}")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ActionGo-assets/1"})
            with urllib.request.urlopen(request, timeout=60) as response, output.open("wb") as stream:
                if not response.geturl().startswith("https://"):
                    raise ValueError("asset download redirected away from HTTPS")
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def cached_source(source: dict, cache: Path) -> Path:
    expected = source["sha256"]
    if not isinstance(expected, str) or len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected):
        raise ValueError("invalid source SHA256")
    target = cache / expected
    if target.is_file() and sha256_file(target) == expected:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fetch-", dir=cache) as temporary:
        staged = Path(temporary) / "asset"
        download(source["url"], staged)
        if sha256_file(staged) != expected:
            raise ValueError(f"SHA256 mismatch from {source['url']}")
        os.replace(staged, target)
    return target


def materialize(recipe: dict, expected: str, cache: Path, staged: Path) -> None:
    cached = cache / expected
    if cached.is_file() and sha256_file(cached) == expected:
        shutil.copyfile(cached, staged)
        return
    kind = recipe.get("kind", "file")
    if kind == "file":
        shutil.copyfile(cached_source({"url": recipe["url"], "sha256": expected}, cache), staged)
    elif kind == "concat":
        with staged.open("wb") as output:
            for index, url in enumerate(recipe["parts"]):
                part = staged.parent / f"part-{index}"
                download(url, part)
                with part.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
                part.unlink()
    elif kind == "tar":
        archive = cached_source(recipe["archive"], cache)
        with tarfile.open(archive, "r:*") as package:
            members = [member for member in package.getmembers() if member.name == recipe["member"]]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError(f"missing, duplicate or non-regular archive member: {recipe['member']}")
            with package.extractfile(members[0]) as stream, staged.open("wb") as output:
                shutil.copyfileobj(stream, output)
    else:
        raise ValueError(f"unsupported download recipe: {kind}")


def install_asset(root: Path, relative: str, expected: str, recipe: dict, cache: Path,
                  force: bool = False) -> None:
    target = checked_path(root, relative)
    if target.is_file() and sha256_file(target) == expected:
        print(f"[OK] {relative}")
        return
    if target.exists() and not force:
        pointer = target.is_file() and target.stat().st_size < 1024 and target.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/v1")
        if not pointer:
            raise ValueError(f"existing file differs: {relative}; use --force to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=target.parent) as temporary:
        staged = Path(temporary) / "asset"
        materialize(recipe, expected, cache, staged)
        if sha256_file(staged) != expected:
            raise ValueError(f"SHA256 mismatch for {relative}; existing destination preserved")
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="verified-", dir=cache, delete=False) as cached:
            cached_path = Path(cached.name)
            try:
                with staged.open("rb") as stream:
                    shutil.copyfileobj(stream, cached)
            except BaseException:
                cached_path.unlink(missing_ok=True)
                raise
        os.replace(cached_path, cache / expected)
        os.replace(staged, target)
    print(f"[FETCHED] {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("MODEL_ASSETS.json"))
    parser.add_argument("--cache", type=Path, help="default: <repo>/.cache/assets (or ACTION_GO_ASSET_CACHE)")
    parser.add_argument("--policy", action="append", default=[],
                        help="repeatable package name under actions/<category>")
    parser.add_argument("--all", action="store_true", help="fetch all registered downloadable assets")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="verify selected downloadable assets without network")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = {}
    for group in manifest["assets"]:
        if len(group["paths"]) != len(group["sha256"]):
            parser.error("manifest paths/sha256 lengths differ")
        downloads = group.get("downloads", {})
        if set(downloads) - set(group["paths"]):
            parser.error("download recipe has no corresponding asset hash")
        for relative, expected in zip(group["paths"], group["sha256"]):
            if relative in assets:
                parser.error(f"duplicate asset: {relative}")
            if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
                parser.error(f"invalid SHA256: {relative}")
            assets[relative] = (expected, downloads.get(relative), group)
    known = {asset_package(relative) for relative in assets}
    if set(args.policy) - known:
        parser.error("unknown policy: " + ", ".join(sorted(set(args.policy) - known)))
    if not (args.policy or args.all or args.list):
        parser.error("choose --policy NAME, --all or --list")
    cache = (args.cache or Path(os.environ.get("ACTION_GO_ASSET_CACHE", str(root / ".cache/assets")))).resolve()
    selected = [(relative, values) for relative, values in assets.items()
                if not args.policy or asset_package(relative) in args.policy]
    failures = 0
    available = 0
    for relative, (expected, recipe, group) in selected:
        if not recipe:
            if args.list or args.policy:
                print(f"[MANUAL] {relative}: {group.get('download_note', 'no verified public asset URL')} ({group['source']})")
            continue
        available += 1
        if args.list or args.dry_run:
            print(f"[DOWNLOAD] {relative}: {json.dumps(recipe, ensure_ascii=False)}")
            continue
        try:
            if args.verify:
                target = checked_path(root, relative)
                if not target.is_file() or sha256_file(target) != expected:
                    raise ValueError(f"missing or mismatched: {relative}")
                print(f"[OK] {relative}")
            else:
                install_asset(root, relative, expected, recipe, cache, args.force)
        except (OSError, ValueError, KeyError, tarfile.TarError) as error:
            failures += 1
            print(f"[FAIL] {relative}: {error}", file=sys.stderr)
    if not available and not args.list:
        print("No verified public downloads registered for this selection.", file=sys.stderr)
        return 1
    if not args.list:
        print(f"Selected {available} downloadable asset(s); {failures} failure(s). "
              "Unregistered motion/sensor assets still require manual acquisition.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
