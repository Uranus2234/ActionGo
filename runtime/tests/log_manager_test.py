import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


MANAGER = Path(__file__).resolve().parents[2] / "tools/logs/manage_logs.py"


class LogManagerTest(unittest.TestCase):
    def test_merge_filter_export_and_protect_active_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inactive = root / "inactive"
            running = root / "running"
            unrelated = root / "unrelated"
            for directory in (inactive, running, unrelated):
                directory.mkdir()
            (inactive / "session.json").write_text(json.dumps({"application": "ActionGo", "pid": 0}))
            (running / "session.json").write_text(json.dumps({"application": "ActionGo", "pid": os.getpid()}))
            (inactive / "events.jsonl").write_text(json.dumps(
                {"event": "later", "t_unix_ns": 2, "level": "ERROR"}) + "\npartial")
            (inactive / "child.jsonl").write_text(json.dumps(
                {"event": "earlier", "t_unix_ns": 1, "level": "INFO"}) + "\n")
            command = [sys.executable, str(MANAGER), "--root", str(root), "--run", "inactive"]
            merged = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual([json.loads(line)["event"] for line in merged.stdout.splitlines()],
                             ["earlier", "later"])
            errors = subprocess.run(command + ["--errors"], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(errors.stdout)["event"], "later")
            archive = root / "export.tar.gz"
            subprocess.run(command + ["--export", str(archive)], check=True, capture_output=True)
            with tarfile.open(archive) as package:
                self.assertIn("inactive/events.jsonl", package.getnames())
            rejected = subprocess.run(command + ["--export", str(inactive / "inside.tar.gz")],
                                      capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            subprocess.run(command + ["--prune", "--keep-days", "0", "--max-bytes", "0"],
                           check=True, capture_output=True)
            self.assertFalse(inactive.exists())
            self.assertTrue(running.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
