import argparse
import ast
import dataclasses
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/sim"))
from pbfm_depth import DepthHistory, MujocoDepthCamera, MOUNT_POSITION, preprocess_depth


class DepthTest(unittest.TestCase):
    def test_camera_housing_is_excluded_and_groups_restored(self):
        import mujoco

        vertices = " ".join(str(value) for vertex in (
            (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)) for value in vertex)
        mount = " ".join(str(value) for value in MOUNT_POSITION)
        model = mujoco.MjModel.from_xml_string(f"""
            <mujoco>
              <asset><mesh name="head_link" vertex="{vertices}" scale="0.03 0.03 0.03"/></asset>
              <worldbody>
                <geom name="floor" type="plane" size="10 10 0.1" group="0"/>
                <body name="torso_link" pos="0 0 0.4">
                  <geom name="camera_housing" type="mesh" mesh="head_link" pos="{mount}"
                        group="2" contype="0" conaffinity="0"/>
                </body>
              </worldbody>
            </mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera = MujocoDepthCamera(model)
        original_groups = model.geom_group.copy()
        depth = camera.capture(model, data)[18:, 16:48]
        self.assertEqual(len(camera.housing_geom_ids), 1)
        self.assertTrue(np.isfinite(depth).all())
        self.assertGreater(float(depth.min()), 0.1)
        np.testing.assert_array_equal(model.geom_group, original_groups)
        np.testing.assert_array_equal(camera.geom_ids.reshape(36, 64)[18:, 16:48], 0)

    def test_ring_ages_and_reset(self):
        history = DepthHistory()
        for step in range(50):
            stack = history.append(np.full((36, 64), step / 255 * 2.5, dtype=np.float32))
            for index, age in enumerate((35, 30, 25, 20, 15, 10, 5, 0)):
                expected = max(0, step - age) / 255
                np.testing.assert_allclose(stack[index], expected, atol=1e-7)

    def test_invalid_pixels_and_camera_zero(self):
        np.testing.assert_array_equal(preprocess_depth(np.full((36, 64), np.nan)), 255)
        np.testing.assert_array_equal(preprocess_depth(np.zeros((36, 64))), 0)
        np.testing.assert_array_equal(preprocess_depth(np.zeros((36, 64)), camera_zero_invalid=True), 255)
        with self.assertRaises(ValueError):
            preprocess_depth(np.zeros((18, 32)))


def upstream_parity(source):
    import torch

    path = source / "pbfm/perception/instinct_direct_depth.py"
    tree = ast.parse(path.read_text())
    names = {"RP1DirectDepthConfig", "depth_config_for_profile", "_gaussian_blur",
             "_sanitize_depth", "preprocess_rp1_depth"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in names or
             isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    namespace = {"__name__": __name__, "torch": torch, "F": torch.nn.functional,
                 "math": math, "dataclass": dataclasses.dataclass, "replace": dataclasses.replace,
                 "MAXCROP_PROFILE": "optical_z_maxcrop18_reset_v2", "DEPTH_CONVENTION": "optical_top_down_v1"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    config = namespace["depth_config_for_profile"]("optical_z_maxcrop18_reset_v2")
    generator = np.random.default_rng(441)
    cases = [generator.uniform(-0.5, 3.5, (36, 64)).astype(np.float32) for _ in range(32)]
    cases[0][18, 16:20] = [np.nan, np.inf, -np.inf, 0]
    for case in cases:
        reference = namespace["preprocess_rp1_depth"](
            torch.from_numpy(case[None]), config, enable_noise=False)[0].numpy()
        np.testing.assert_array_equal(preprocess_depth(case), reference)
    print("32 depth frames match upstream preprocessing exactly")


def cpp_parity(binary):
    generator = np.random.default_rng(441)
    cases = [generator.uniform(0, 3.5, (36, 64)).astype(np.float32) for _ in range(32)]
    cases.extend(np.full((36, 64), value, dtype=np.float32) for value in (0, np.nan, np.inf, 2.5))
    cases[0][18, 16:20] = [np.nan, np.inf, -np.inf, 0]
    with tempfile.TemporaryDirectory(prefix="pbfm-depth-parity-") as raw:
        folder = Path(raw)
        for case in cases:
            case.tofile(folder / "input.f32")
            subprocess.run([str(binary), "--profile", "pbfm", str(folder / "input.f32"),
                            str(folder / "output.f32")], check=True)
            actual = np.fromfile(folder / "output.f32", dtype=np.float32).reshape(18, 32)
            expected = preprocess_depth(case, camera_zero_invalid=True).astype(np.float32) / np.float32(255)
            np.testing.assert_array_equal(actual, expected)
    print("36 depth frames match C++ camera preprocessing exactly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--preprocessor", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DepthTest)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise SystemExit(1)
    if args.source:
        upstream_parity(args.source)
    if args.preprocessor:
        cpp_parity(args.preprocessor.resolve())
