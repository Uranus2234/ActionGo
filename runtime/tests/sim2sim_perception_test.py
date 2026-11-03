import argparse
import contextlib
import io
import json
import socket
from multiprocessing import shared_memory
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

try:
    import mujoco
    import numpy as np
    import yaml
except ImportError:
    print("SKIP: sim2sim perception tests require MuJoCo, NumPy and PyYAML")
    raise SystemExit(77)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/sim"))
from sim2sim_perception import DepthStream, PhpDepthCamera, SimPerception, object_pose, php_depth, torso_imu
from sim2sim_scene import add_terrain
from sim2sim_support import SimSession


SCENE = """<mujoco><worldbody>
  <geom name="floor" type="plane" size="10 10 .1"/>
  <body name="torso_link" pos="0 0 .8"/>
  <body name="d435_link" pos="1 2 3" quat=".7071067811865476 0 0 .7071067811865476"/>
  <body name="object" pos="1 3 3"/>
</worldbody></mujoco>"""


class PerceptionTest(unittest.TestCase):
    def test_generated_terrain_has_real_collision_and_depth(self):
        depths = {}
        for kind in ("flat", "stairs", "ramp", "rough"):
            spec = mujoco.MjSpec.from_string(SCENE)
            add_terrain(spec, kind, start=0.7, height=0.12)
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            depths[kind] = PhpDepthCamera(model, (0, 0, 0), 27).capture(model, data)
            geom = np.zeros(1, dtype=np.int32)
            distance = mujoco.mj_ray(model, data, np.array([1.9, 0, 2]), np.array([0., 0, -1]),
                                     None, True, -1, geom)
            self.assertLessEqual(distance, 2)
            if kind != "flat":
                self.assertLess(distance, 2)
                self.assertGreater(np.count_nonzero(np.nan_to_num(depths[kind]) != np.nan_to_num(depths["flat"])), 100)
                self.assertTrue(model.geom_contype[geom[0]])
            spawn = mujoco.mj_ray(model, data, np.array([0., 0, 2]), np.array([0., 0, -1]),
                                  None, True, -1, geom)
            self.assertAlmostEqual(spawn, 2)

    def test_rough_terrain_is_reproducible(self):
        fields = []
        for seed in (5, 5, 6):
            spec = mujoco.MjSpec.from_string(SCENE)
            add_terrain(spec, "rough", seed=seed)
            fields.append(spec.compile().hfield_data.copy())
        np.testing.assert_array_equal(fields[0], fields[1])
        self.assertGreater(np.max(abs(fields[0] - fields[2])), 0.05)

    def test_object_pose_uses_calibrated_frame_and_rotation(self):
        model = mujoco.MjModel.from_xml_string(SCENE)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        pose = object_pose(model, data, 3, 2, (0, 0, 0), (1, 0, 0, 0))
        np.testing.assert_allclose(pose[:3], [1, 0, 0], atol=1e-15)
        np.testing.assert_allclose(pose[3:], [2 ** -.5, 0, 0, -2 ** -.5], atol=1e-15)
        offset = object_pose(model, data, 3, 2, (1, 0, 0), (1, 0, 0, 0))
        np.testing.assert_allclose(offset[:3], 0, atol=1e-15)

    def test_php_camera_reports_optical_z(self):
        model = mujoco.MjModel.from_xml_string(SCENE)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        depth = PhpDepthCamera(model, (0, 0, 0), 90).capture(model, data)
        np.testing.assert_allclose(depth, 0.8, atol=1e-6)
        processed = php_depth(depth)
        np.testing.assert_allclose(processed, (0.8 - 0.3) / 2.7 - 0.5, atol=1e-6)
        self.assertEqual(processed.shape, (58, 87))
        self.assertTrue(np.isfinite(php_depth(np.full((60, 106), np.nan))).all())

    def test_shared_memory_heartbeat_and_cleanup(self):
        try:
            import zmq
        except ImportError:
            self.skipTest("pyzmq is not installed")

        stream = DepthStream((58, 87), "php")
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.connect("ipc://" + str(stream.endpoint))
        reader = shared_memory.SharedMemory(name=stream.name)
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                stream.write(np.full((58, 87), 0.25, dtype=np.float32))
                if subscriber.poll(50):
                    break
            else:
                self.fail("no simulated depth heartbeat")
            heartbeat = subscriber.recv_json()
            self.assertEqual(heartbeat["state"], "OK")
            self.assertGreater(heartbeat["seq"], 0)
            np.testing.assert_array_equal(np.ndarray((58, 87), dtype=np.float32, buffer=reader.buf), 0.25)
        finally:
            reader.close()
            subscriber.close(linger=0)
            context.term()
            stream.close()
        self.assertFalse(stream.endpoint.exists())
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=stream.name)

    def test_torso_imu_matches_mujoco_sensor_and_is_independent_of_root(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <body><freejoint/><geom size=".1"/>
            <body name="torso_link" pos="0 0 .3"><joint axis="0 1 0"/>
              <geom size=".1"/><site name="imu"/></body>
          </body></worldbody><sensor><accelerometer site="imu"/><gyro site="imu"/>
          <framequat objtype="site" objname="imu"/></sensor></mujoco>""")
        data = mujoco.MjData(model)
        data.qpos[7] = 0.4
        data.qvel[6] = 0.7
        mujoco.mj_forward(model, data)
        quaternion, gyro, acceleration = torso_imu(model, data, 2)
        np.testing.assert_allclose(acceleration, data.sensordata[:3], atol=1e-12)
        np.testing.assert_allclose(gyro, data.sensordata[3:6], atol=1e-12)
        np.testing.assert_allclose(quaternion, data.sensordata[6:], atol=1e-12)
        self.assertGreater(abs(gyro[1]), 0.5)
        self.assertGreater(abs(quaternion[2]), 0.1)

    def test_sensor_activation_logs_frames_and_cleans_up_on_reuse(self):
        try:
            import zmq
        except ImportError:
            self.skipTest("pyzmq is not installed")
        model = mujoco.MjModel.from_xml_string(SCENE)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with SimSession(Path(directory), "quiet") as session:
                spec_path = Path(directory) / "spec.yaml"
                spec_path.write_text("runtime:\n  variant: pbfm\nsensor:\n  depth_source: realsense\n")
                args = SimpleNamespace(session=session, php_camera_pos=(0, 0, 0), php_camera_pitch_deg=27,
                                       object_frame_pos=(0, 0, 0), object_frame_quat=(1, 0, 0, 0),
                                       object_body="", object_frame_body="d435_link", depth_hz=50,
                                       sensor_log_period_sec=1, execution_provider="cpu")
                perception = SimPerception(model, data, args)
                try:
                    perception.activate(SimpleNamespace(params=str(spec_path)), "generic_sensor_policy_player")
                    first_name = perception.stream.name
                    latest = session.directory / "sensors/pbfm.latest.npz"
                    self.assertEqual(np.load(latest)["processed"].shape, (8, 18, 32))
                    frame_log = perception.activation_directory / "frames.jsonl"
                    self.assertEqual(json.loads(frame_log.read_text())["sequence"], 1)
                    perception.activate(SimpleNamespace(params=str(spec_path)), "generic_sensor_policy_player")
                    self.assertNotEqual(first_name, perception.stream.name)
                    with self.assertRaises(FileNotFoundError):
                        shared_memory.SharedMemory(name=first_name)
                finally:
                    perception.close()
            events = [json.loads(line) for line in (session.directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(event["event"] == "sensor_frame" for event in events), 2)

    def test_object_stream_transmits_updated_scene_pose(self):
        model = mujoco.MjModel.from_xml_string(SCENE)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with SimSession(Path(directory), "quiet") as session, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
                receiver.bind(("127.0.0.1", 0))
                receiver.settimeout(1)
                path = Path(directory) / "spec.yaml"
                path.write_text("runtime:\n  variant: steadytray\n")
                args = SimpleNamespace(session=session, php_camera_pos=(0, 0, 0), php_camera_pitch_deg=27,
                                       object_frame_pos=(0, 0, 0), object_frame_quat=(1, 0, 0, 0),
                                       object_body="object", object_frame_body="d435_link",
                                       object_port=receiver.getsockname()[1], depth_hz=50,
                                       sensor_log_period_sec=1)
                perception = SimPerception(model, data, args)
                try:
                    command = perception.activate(SimpleNamespace(params=str(path)), "generic_sensor_policy_player")
                    self.assertIn("--udp-bind 127.0.0.1", command)
                    np.testing.assert_allclose(np.fromstring(receiver.recv(1024).decode(), sep=" ")[:3], [1, 0, 0], atol=1e-8)
                    model.body_pos[3, 1] += 0.2
                    mujoco.mj_forward(model, data)
                    perception.next_frame = 0
                    perception.tick()
                    np.testing.assert_allclose(np.fromstring(receiver.recv(1024).decode(), sep=" ")[:3], [1.2, 0, 0], atol=1e-8)
                finally:
                    perception.close()


def cpp_parity(binary):
    generator = np.random.default_rng(51)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for _ in range(16):
            raw = generator.uniform(0.1, 4, (60, 106)).astype(np.float32)
            raw.tofile(directory / "raw.f32")
            subprocess.run([str(binary), str(directory / "raw.f32"), str(directory / "out.f32")], check=True)
            expected = np.fromfile(directory / "out.f32", dtype=np.float32).reshape(58, 87)
            np.testing.assert_allclose(php_depth(raw), expected, atol=1e-6, rtol=1e-6)
    print("16 PHP frames match C++ preprocessing within 1e-6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessor", type=Path)
    args = parser.parse_args()
    if not unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(PerceptionTest)).wasSuccessful():
        raise SystemExit(1)
    if args.preprocessor:
        cpp_parity(args.preprocessor.resolve())
