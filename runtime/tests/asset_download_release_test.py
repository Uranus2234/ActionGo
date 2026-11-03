import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/assets"))
sys.path.insert(0, str(ROOT / "tools/release"))
import fetch_assets
import create_public_snapshot


def digest(data):
    return hashlib.sha256(data).hexdigest()


class AssetDownloadReleaseTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="actiongo_fetch_test_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache = self.root / "cache"
        self.relative = "actions/demo/model/policy.onnx"
        self.target = self.root / self.relative
        self.payload = b"verified model fixture"
        self.recipe = {"kind": "file", "url": "https://example.invalid/model"}

    def install(self, recipe=None, force=False):
        fetch_assets.install_asset(self.root, self.relative, digest(self.payload),
                                   recipe or self.recipe, self.cache, force)

    def response(self, data):
        response = io.BytesIO(data)
        response.geturl = lambda: self.recipe["url"]
        return response

    def test_verified_download_is_cached_and_installed_atomically(self):
        with patch.object(fetch_assets.urllib.request, "urlopen", return_value=self.response(self.payload)) as request:
            self.install()
            self.assertEqual(self.target.read_bytes(), self.payload)
            self.target.unlink()
            self.install()
            self.assertEqual(request.call_count, 1)

    def test_bad_download_preserves_destination_even_with_force(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"existing")
        with patch.object(fetch_assets.urllib.request, "urlopen", return_value=self.response(b"corrupt")):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.install(force=True)
        self.assertEqual(self.target.read_bytes(), b"existing")
        self.assertFalse((self.cache / digest(self.payload)).exists())

    def test_corrupt_cache_is_refetched(self):
        self.cache.mkdir()
        (self.cache / digest(self.payload)).write_bytes(b"bad cache")
        with patch.object(fetch_assets.urllib.request, "urlopen", return_value=self.response(self.payload)) as request:
            self.install()
        self.assertEqual(request.call_count, 1)
        self.assertEqual((self.cache / digest(self.payload)).read_bytes(), self.payload)

    def test_lfs_pointer_replaced_but_modified_asset_requires_force(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"user model")
        with patch.object(fetch_assets, "download") as download:
            with self.assertRaisesRegex(ValueError, "existing file differs"):
                self.install()
            download.assert_not_called()
        self.target.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        with patch.object(fetch_assets.urllib.request, "urlopen", return_value=self.response(self.payload)):
            self.install()
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_concat_order_and_final_checksum(self):
        parts = [self.payload[:4], self.payload[4:]]
        urls = ["https://example.invalid/part0", "https://example.invalid/part1"]
        def download(url, output):
            output.write_bytes(parts[urls.index(url)])
        with patch.object(fetch_assets, "download", side_effect=download):
            self.install({"kind": "concat", "parts": urls})
            self.target.unlink()
            (self.cache / digest(self.payload)).unlink()
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.install({"kind": "concat", "parts": urls[::-1]})
        self.assertFalse(self.target.exists())

    def archive(self, mode):
        output = self.root / "upstream.tar"
        with tarfile.open(output, "w") as package:
            info = tarfile.TarInfo("model.onnx")
            info.size = len(self.payload)
            if mode == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "../outside"
            package.addfile(info, io.BytesIO(self.payload) if info.isfile() else None)
            if mode == "duplicate":
                package.addfile(info, io.BytesIO(self.payload))
            other = tarfile.TarInfo("../outside")
            other.size = 4
            package.addfile(other, io.BytesIO(b"evil"))
        return output

    def test_tar_reads_exact_regular_member_without_extracting_paths(self):
        archive = self.archive("regular")
        recipe = {"kind": "tar", "member": "model.onnx", "archive": {}}
        with patch.object(fetch_assets, "cached_source", return_value=archive):
            self.install(recipe)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse((self.root.parent / "outside").exists())

    def test_tar_rejects_duplicate_or_symlink_member(self):
        recipe = {"kind": "tar", "member": "model.onnx", "archive": {}}
        for mode in ["duplicate", "symlink"]:
            with self.subTest(mode=mode):
                with patch.object(fetch_assets, "cached_source", return_value=self.archive(mode)):
                    with self.assertRaises(ValueError):
                        self.install(recipe)
                self.assertFalse(self.target.exists())

    def test_invalid_paths_sources_and_protocol_are_rejected(self):
        for relative in ["", "../outside", "/outside"]:
            with self.assertRaises(ValueError):
                fetch_assets.checked_path(self.root, relative)
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            fetch_assets.checked_path(self.root, "escape/outside")
        with self.assertRaises(ValueError):
            fetch_assets.cached_source({"sha256": "../bad", "url": "https://example.invalid"}, self.cache)
        with self.assertRaises(ValueError):
            fetch_assets.download("http://example.invalid", self.root / "output")

    def test_cli_dry_run_verify_and_unknown_policy(self):
        manifest = {"assets": [{"paths": [self.relative], "sha256": [digest(self.payload)],
                                "source": "fixture", "downloads": {self.relative: self.recipe}}]}
        (self.root / "MODEL_ASSETS.json").write_text(json.dumps(manifest))
        command = [sys.executable, str(ROOT / "tools/assets/fetch_assets.py"), "--repo", str(self.root)]
        dry = subprocess.run(command + ["--policy", "demo", "--dry-run"], capture_output=True)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertFalse(self.target.exists())
        missing = subprocess.run(command + ["--policy", "demo", "--verify"], capture_output=True)
        self.assertEqual(missing.returncode, 1)
        unknown = subprocess.run(command + ["--policy", "unknown"], capture_output=True)
        self.assertNotEqual(unknown.returncode, 0)

    def test_pbfm_derived_text_is_excluded_and_checker_rejects_it(self):
        for relative in ["actions/pbfm/motions/lafan_dance/context.yaml",
                         "actions/pbfm/motions/lafan_dance/initial_state.json",
                         "actions/pbfm/config/context_idle.yaml",
                         "actions/pbfm/config/joystick_walk/context.yaml"]:
            path = Path(relative)
            self.assertTrue(create_public_snapshot.excluded(path, set()))
            output = self.root / path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("fixture: data\n")
        self.assertFalse(create_public_snapshot.excluded(Path("actions/pbfm/config/PBFM.yaml"), set()))
        result = subprocess.run([sys.executable, str(ROOT / "tools/release/public_snapshot_check.py"),
                                 "--repo", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.count("upstream P-BFM context data:"), 4)

    def test_snapshot_checker_does_not_create_blocked_bytecode(self):
        release = self.root / "tools/release"
        release.mkdir(parents=True)
        for name in ["public_snapshot_check.py", "create_public_snapshot.py"]:
            (release / name).write_bytes((ROOT / "tools/release" / name).read_bytes())
        result = subprocess.run([sys.executable, str(release / "public_snapshot_check.py"),
                                 "--repo", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((release / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
