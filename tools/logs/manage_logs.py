#!/usr/bin/env python3
"""Inspect, export and prune ActionGo runtime sessions."""

import argparse
import json
import os
import shutil
import tarfile
import time
from pathlib import Path


def sessions(root):
    found = []
    for directory in root.glob("*"):
        if directory.is_symlink() or not directory.is_dir():
            continue
        try:
            metadata = json.loads((directory / "session.json").read_text())
            if metadata.get("application") == "ActionGo":
                found.append((directory, metadata))
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda item: item[0].stat().st_mtime)


def active(metadata):
    try:
        process = int(metadata.get("pid", 0))
        if process <= 0:
            return False
        os.kill(process, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("ACTION_GO_LOG_DIR", "logs")))
    parser.add_argument("--run", default="latest")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--errors", action="store_true")
    parser.add_argument("--action", help="action filename prefix, such as 1_kick_right")
    parser.add_argument("--export", type=Path, metavar="ARCHIVE")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--keep-days", type=float, default=7)
    parser.add_argument("--max-bytes", type=int, default=1024 * 1024 * 1024)
    args = parser.parse_args()
    if args.keep_days < 0 or args.max_bytes < 0:
        parser.error("retention limits must be non-negative")
    found = sessions(args.root)
    if args.prune:
        sizes = {directory: sum(item.stat().st_size for item in directory.rglob("*")
                                if item.is_file() and not item.is_symlink()) for directory, _ in found}
        total = sum(sizes.values())
        for directory, metadata in found:
            if active(metadata):
                continue
            old = time.time() - directory.stat().st_mtime > args.keep_days * 86400
            if old or total > args.max_bytes:
                shutil.rmtree(directory)
                total -= sizes[directory]
                print(f"PRUNED {directory.name}")
        return 0
    if args.list:
        for directory, metadata in found:
            print(f"{directory.name} {'ACTIVE' if active(metadata) else 'STOPPED'}")
        return 0
    if not found:
        parser.error("no ActionGo sessions found")
    selected = found[-1][0] if args.run == "latest" else next(
        (directory for directory, _ in found if directory.name == args.run), None)
    if selected is None:
        parser.error("unknown run")
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        if args.export.resolve().is_relative_to(selected.resolve()):
            parser.error("export must be outside the session directory")
        with tarfile.open(args.export, "w:gz", dereference=False) as archive:
            archive.add(selected, arcname=selected.name)
        print(args.export)
        return 0
    records = []
    for path in selected.rglob("*"):
        if path.is_symlink() or not path.is_file() or not (
                path.name.endswith(".jsonl") or ".jsonl." in path.name and path.suffix[1:].isdigit()):
            continue
        if args.action and not path.name.startswith(args.action):
            continue
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
                if not args.errors or event.get("level") == "ERROR":
                    records.append(event)
            except ValueError:
                continue
    for event in sorted(records, key=lambda item: item.get("t_unix_ns", 0)):
        print(json.dumps(event, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
