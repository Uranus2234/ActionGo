import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(__file__).resolve().parents[2] / "tools/assets/check_model_assets.py"


class ModelAssetsCheckTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="actiongo_assets_test_")
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.payload = b"fixture asset"
        self.primary = self.repo / "actions/tracking/primary/model/policy.onnx"
        self.primary.parent.mkdir(parents=True)
        self.primary.write_bytes(self.payload)
        self.secondary = self.repo / "actions/perception/secondary/model/policy.onnx"
        self.manifest = {
            "schema_version": 1,
            "assets": [
                {
                    "paths": [str(path.relative_to(self.repo))],
                    "sha256": [hashlib.sha256(self.payload).hexdigest()],
                    "source": "test fixture",
                    "redistribution": "blocked_pending_review",
                }
                for path in (self.primary, self.secondary)
            ],
        }
        (self.repo / "MODEL_ASSETS.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def check(self, *args):
        return subprocess.run(
            [sys.executable, str(CHECKER), "--repo", str(self.repo), "--json", *args],
            capture_output=True, text=True,
        )

    def test_package_selection_does_not_require_other_packages(self):
        result = self.check("--policy", "primary")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(len(report["assets"]), 1)
        self.assertTrue(report["assets"][0]["matches"])
        self.assertNotEqual(self.check().returncode, 0)

    def test_installed_only_checks_present_assets_and_reports_missing(self):
        result = self.check("--installed-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(len(report["assets"]), 2)
        self.assertEqual(sum(item["exists"] for item in report["assets"]), 1)
        self.primary.write_bytes(b"wrong model")
        self.assertNotEqual(self.check("--installed-only").returncode, 0)

    def test_unlisted_assets_and_unknown_packages_fail(self):
        extra = self.primary.parent / "extra.onnx"
        extra.write_bytes(self.payload)
        self.assertNotEqual(self.check("--policy", "primary").returncode, 0)
        self.assertNotEqual(self.check("--policy", "typo").returncode, 0)

    def test_public_release_rejects_bundled_blocked_weights(self):
        self.assertNotEqual(self.check("--permit-missing", "--public-release").returncode, 0)
        self.primary.unlink()
        self.assertEqual(self.check("--permit-missing", "--public-release").returncode, 0)

    def test_unstaged_asset_deletion_is_not_counted_as_bundled(self):
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        extra = self.primary.parent / "removed.onnx"
        extra.write_bytes(self.payload)
        subprocess.run(["git", "-C", str(self.repo), "add", str(extra)], check=True)
        extra.unlink()
        result = self.check("--policy", "primary")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
