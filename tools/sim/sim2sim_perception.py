"""Physics-thread sensor producers for the real C++ sensor policy players."""

from multiprocessing import shared_memory
from pathlib import Path
import json
import socket
import time
import uuid

import mujoco
import numpy as np
import yaml

from pbfm_depth import DepthHistory, MOUNT_POSITION, MujocoDepthCamera, OPTICAL_QUAT_XYZW, PROFILE


def body_id(model, name):
    matches = [index for index in range(1, model.nbody)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or "").split("/")[-1]
               in {name, "robot_" + name}]
    if len(matches) != 1:
        raise ValueError("scene requires exactly one body: " + name)
    return matches[0]


def php_depth(optical_z):
    raw = np.asarray(optical_z, dtype=np.float32)
    if raw.shape != (60, 106):
        raise ValueError("PHP optical-Z input must be [60,106] meters")
    raw = np.clip(np.where(np.isfinite(raw) & (raw > 0), raw, 3.0), 0.3, 3.0)[2:, 4:102]
    scale = np.float32(98 / 87)
    output = np.empty((58, 87), dtype=np.float32)
    for column in range(87):
        center = np.float32(column + 0.5) * scale
        begin = max(0, int(center - 2 * scale + 0.5))
        end = min(98, int(center + 2 * scale + 0.5))
        distance = np.abs((np.arange(begin, end, dtype=np.float32) + np.float32(0.5) - center) / scale)
        weights = np.where(distance < 1, ((1.5 * distance - 2.5) * distance) * distance + 1,
                           np.where(distance < 2, ((-0.5 * distance + 2.5) * distance - 4) * distance + 2, 0))
        weights /= weights.sum()
        value = np.zeros(58, dtype=np.float32)
        for source, weight in zip(range(begin, end), weights):
            value += raw[:, source] * weight
        value = np.minimum(value, 3)
        output[:, column] = (np.where(value < 0.15, 3, value) - np.float32(0.3)) / np.float32(2.7) - np.float32(0.5)
    return output


class PhpDepthCamera:
    def __init__(self, model, position, pitch):
        self.body_id = body_id(model, "torso_link")
        self.position = np.asarray(position, dtype=float)
        angle = np.deg2rad(pitch)
        pitch_rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                                   [-np.sin(angle), 0, np.cos(angle)]])
        self.rotation = pitch_rotation @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        rows, columns = np.indices((60, 106), dtype=float)
        focal = 53 / np.tan(np.deg2rad(89.5 / 2))
        self.rays = np.stack([(columns - 53) / focal, (rows - 30) / focal, np.ones_like(rows)], axis=-1).reshape(-1, 3)
        self.rays /= np.linalg.norm(self.rays, axis=1, keepdims=True)
        self.geom_ids = np.empty(len(self.rays), dtype=np.int32)
        self.distances = np.empty(len(self.rays), dtype=float)
        self.groups = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)

    def capture(self, model, data):
        torso_rotation = data.xmat[self.body_id].reshape(3, 3)
        position = data.xpos[self.body_id] + torso_rotation @ self.position
        directions = np.ascontiguousarray(self.rays @ (torso_rotation @ self.rotation).T).ravel()
        mujoco.mj_multiRay(model, data, position, directions, self.groups, True, self.body_id,
                          self.geom_ids, self.distances, None, len(self.rays), 3 / self.rays[:, 2].min())
        depth = self.distances * self.rays[:, 2]
        depth[self.distances < 0] = np.nan
        return depth.reshape(60, 106).astype(np.float32)


class DepthStream:
    def __init__(self, shape, profile):
        import zmq

        self.name = "actiongo_sim_" + uuid.uuid4().hex
        self.endpoint = Path("/tmp") / (self.name + ".heartbeat")
        self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=int(np.prod(shape)) * 4)
        self.array = np.ndarray(shape, dtype=np.float32, buffer=self.shm.buf)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        try:
            self.socket.bind("ipc://" + str(self.endpoint))
        except Exception:
            self.close()
            raise
        self.sequence = 0
        self.profile = profile

    def write(self, depth):
        if depth.shape != self.array.shape or not np.isfinite(depth).all():
            raise ValueError("invalid simulated depth frame")
        np.copyto(self.array, depth)
        self.sequence += 1
        self.socket.send_json(dict(seq=self.sequence, state="OK", t_capture=time.time(), profile=self.profile))

    def close(self):
        self.socket.close(linger=0)
        self.context.term()
        self.shm.close()
        self.shm.unlink()
        self.endpoint.unlink(missing_ok=True)


def object_pose(model, data, object_body, frame_body, frame_position, frame_quaternion):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(frame_quaternion, dtype=float))
    frame_rotation = data.xmat[frame_body].reshape(3, 3) @ rotation.reshape(3, 3)
    frame_origin = data.xpos[frame_body] + data.xmat[frame_body].reshape(3, 3) @ np.asarray(frame_position)
    position = frame_rotation.T @ (data.xpos[object_body] - frame_origin)
    orientation = frame_rotation.T @ data.xmat[object_body].reshape(3, 3)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, orientation.ravel())
    return np.concatenate((position, quaternion))


def torso_imu(model, data, torso_body):
    velocity = np.empty(6)
    acceleration = np.empty(6)
    mujoco.mj_rnePostConstraint(model, data)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, torso_body, velocity, 1)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, torso_body, acceleration, 1)
    return data.xquat[torso_body].copy(), velocity[:3], acceleration[3:]


class SimPerception:
    def __init__(self, model, data, args):
        self.model, self.data, self.args = model, data, args
        self.session = args.session
        self.stream = None
        self.camera = None
        self.object_socket = None
        self.frame_log = None
        self.variant = ""
        self.imu_publisher = None
        self.next_frame = 0
        self.next_snapshot = 0
        self.sequence = 0
        self.directory = self.session.directory / "sensors"
        self.directory.mkdir()
        if not np.isfinite([*args.php_camera_pos, args.php_camera_pitch_deg,
                            *args.object_frame_pos, *args.object_frame_quat]).all():
            raise ValueError("sensor transforms must be finite")
        if not np.isclose(np.linalg.norm(args.object_frame_quat), 1.0, atol=1e-5):
            raise ValueError("object frame quaternion must be unit length (wxyz)")
        if not np.isfinite(args.depth_hz) or not 1 <= args.depth_hz <= 60:
            raise ValueError("--depth-hz must be in [1,60]")
        if not np.isfinite(args.sensor_log_period_sec) or args.sensor_log_period_sec <= 0:
            raise ValueError("--sensor-log-period-sec must be finite and positive")

    def activate(self, action, command, warmup=True):
        self.deactivate()
        if "generic_sensor_policy_player" not in command:
            return command
        spec_path = Path(action.params)
        spec = yaml.safe_load(spec_path.read_text())
        variant = spec.get("runtime", {}).get("variant", "")
        if variant in {"pbfm", "php"}:
            self.camera = (MujocoDepthCamera(self.model) if variant == "pbfm" else
                           PhpDepthCamera(self.model, self.args.php_camera_pos, self.args.php_camera_pitch_deg))
            self.history = DepthHistory() if variant == "pbfm" else None
            self.stream = DepthStream((8, 18, 32) if variant == "pbfm" else (58, 87),
                                      PROFILE if variant == "pbfm" else "php_bicubic_antialias")
            sensor = spec.setdefault("sensor", {})
            sensor.update(depth_source="shm", shm_name=self.stream.name, require_depth_camera=False)
            if variant == "php":
                sensor.update(require_torso_imu=True, disable_torso_imu=False)
                self._start_imu()
            else:
                command += " --ep " + self.args.execution_provider
            spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
        elif variant == "steadytray":
            if not self.args.object_body:
                raise ValueError("SteadyTray requires --object-body and a calibrated policy frame in the shared XML")
            self.object_body = body_id(self.model, self.args.object_body)
            self.frame_body = body_id(self.model, self.args.object_frame_body)
            self.object_port = self.args.object_port
            if not self.object_port:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    self.object_port = reservation.getsockname()[1]
            self.object_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            command += f" --object-source udp --udp-port {self.object_port} --udp-bind 127.0.0.1 --require-tag"
        else:
            raise ValueError("unsupported simulation sensor variant: " + variant)
        self.variant = variant
        self.activation_directory = self.directory / (variant + "_" + uuid.uuid4().hex[:12])
        self.activation_directory.mkdir()
        self.frame_log = (self.activation_directory / "frames.jsonl").open("w", buffering=1)
        self.next_frame = (0.0 if warmup else
                           time.monotonic() + 1 / self.args.depth_hz)
        self.next_snapshot = 0
        self.sequence = 0
        self.session.event("sensors_activated", variant=variant, source="mujoco_ground_truth",
                           depth_shm=self.stream.name if self.stream else None,
                           depth_hz=self.args.depth_hz, output_directory=str(self.activation_directory),
                           camera_position=(list(self.args.php_camera_pos) if variant == "php" else
                                            list(MOUNT_POSITION) if variant == "pbfm" else None),
                           php_pitch_deg=self.args.php_camera_pitch_deg if variant == "php" else None,
                           pbfm_optical_quat_xyzw=list(OPTICAL_QUAT_XYZW) if variant == "pbfm" else None,
                           object_body=self.args.object_body, object_frame_body=self.args.object_frame_body,
                           object_port=self.object_port if variant == "steadytray" else None,
                           object_frame_pos=list(self.args.object_frame_pos), object_frame_quat=list(self.args.object_frame_quat))
        if warmup:
            self.tick()
        return command

    def _start_imu(self):
        if self.imu_publisher is not None:
            return
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__IMUState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_

        self.torso_body = body_id(self.model, "torso_link")
        self.imu = unitree_hg_msg_dds__IMUState_()
        self.imu_publisher = ChannelPublisher("rt/secondary_imu", IMUState_)
        self.imu_publisher.Init()

    def tick(self):
        if not self.variant:
            return
        if self.variant == "php":
            quaternion, gyroscope, accelerometer = torso_imu(self.model, self.data, self.torso_body)
            self.imu.quaternion = quaternion.tolist()
            self.imu.gyroscope = gyroscope.tolist()
            self.imu.accelerometer = accelerometer.tolist()
            self.imu_publisher.Write(self.imu)
        now = time.monotonic()
        if now < self.next_frame:
            return
        self.next_frame = now + 1 / self.args.depth_hz
        started = time.perf_counter()
        if self.stream:
            raw = self.camera.capture(self.model, self.data)
            processed = self.history.append(raw) if self.history else php_depth(raw)
            self.stream.write(processed)
        else:
            processed = object_pose(self.model, self.data, self.object_body, self.frame_body,
                                    self.args.object_frame_pos, self.args.object_frame_quat)
            raw = processed
            self.object_socket.sendto((" ".join(format(float(value), ".9g") for value in processed)).encode(),
                                      ("127.0.0.1", self.object_port))
        capture_ms = (time.perf_counter() - started) * 1000
        self.sequence += 1
        record = dict(variant=self.variant, sequence=self.sequence, t_capture_steady_ns=time.monotonic_ns(),
                      sim_time=float(self.data.time), shape=list(processed.shape), capture_ms=capture_ms,
                      valid_fraction=float(np.isfinite(raw).mean()), minimum=float(processed.min()),
                      maximum=float(processed.max()))
        if self.variant == "php":
            record.update(torso_quaternion=quaternion.tolist(), torso_gyroscope=gyroscope.tolist(),
                          torso_accelerometer=accelerometer.tolist())
        elif self.variant == "steadytray":
            record.update(object_pose=processed.tolist())
        self.frame_log.write(json.dumps(record) + "\n")
        if now >= self.next_snapshot:
            self.next_snapshot = now + self.args.sensor_log_period_sec
            snapshot = self.activation_directory / f"{self.sequence:08d}.npz"
            np.savez_compressed(snapshot, raw=raw, processed=processed,
                                sim_time=float(self.data.time), sequence=self.sequence,
                                torso_quaternion=self.data.xquat[body_id(self.model, "torso_link")])
            latest = self.directory / (self.variant + ".latest.npz")
            latest.unlink(missing_ok=True)
            latest.symlink_to(snapshot.relative_to(self.directory))
            self.session.event("sensor_frame", snapshot=str(snapshot), **record)

    def deactivate(self):
        if self.variant:
            self.session.event("sensors_deactivated", variant=self.variant, sequence=self.sequence)
        self.variant = ""
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.object_socket:
            self.object_socket.close()
            self.object_socket = None
        if self.frame_log:
            self.frame_log.close()
            self.frame_log = None

    def close(self):
        self.deactivate()
        if self.imu_publisher:
            self.imu_publisher.Close()
            self.imu_publisher = None
