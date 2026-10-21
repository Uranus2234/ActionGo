from collections import deque
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


class PoseResampler:
    def __init__(self, dt=0.02, max_gap=0.2):
        self.dt = dt
        self.max_gap = max_gap
        self.previous = None
        self.next_time = None

    def push(self, timestamp, pose):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (36,) or not np.isfinite(pose).all():
            raise ValueError("G1 reference must be finite root XYZ + WXYZ + 29 joint angles")
        if not np.isclose(np.linalg.norm(pose[3:7]), 1, atol=1e-3):
            raise ValueError("G1 reference root quaternion must be unit WXYZ")
        if not np.isfinite(timestamp):
            raise ValueError("Reference timestamp must be finite")
        if self.previous is None:
            self.previous = (timestamp, pose.copy())
            self.next_time = timestamp + self.dt
            return [(timestamp, pose.copy())]
        previous_time, previous_pose = self.previous
        if timestamp <= previous_time:
            return []
        if timestamp - previous_time > self.max_gap:
            raise RuntimeError("Pico reference is stale; restart and recalibrate")
        rotations = Rotation.from_quat(np.array([previous_pose[3:7], pose[3:7]])[:, [1, 2, 3, 0]])
        slerp = Slerp([previous_time, timestamp], rotations)
        result = []
        while self.next_time <= timestamp + 1e-9:
            sample_time = min(self.next_time, timestamp)
            alpha = (sample_time - previous_time) / (timestamp - previous_time)
            interpolated = previous_pose + alpha * (pose - previous_pose)
            interpolated[3:7] = slerp(sample_time).as_quat()[[3, 0, 1, 2]]
            result.append((self.next_time, interpolated))
            self.next_time += self.dt
        self.previous = (timestamp, pose.copy())
        return result


class ReferenceEncoder:
    dt = 0.02
    window_size = 27
    reference_index = 9
    lookahead_frames = 8
    delay_sec = 0.34

    def __init__(self, source):
        source = Path(source)
        package = source / "model/public_demo"
        self.config = json.loads((package / "config.json").read_text())
        if self.config["checkpointStep"] != 364822528 or self.config["controlDt"] != self.dt:
            raise ValueError("Expected public P-BFM checkpoint 364822528 at 50 Hz")
        self.model = mujoco.MjModel.from_xml_path(str(source / "pbfm/data/robots/g1_mjlab/g1_29dof.xml"))
        self.data = mujoco.MjData(self.model)
        self.joint_names = self.config["jointNames"]
        self.addresses = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.defaults = np.asarray(self.config["defaultJointPosition"])
        self.links = ["pelvis", *["torso_link" if name == "waist_pitch_joint" else
                                name.removesuffix("_joint") + "_link" for name in self.joint_names]]
        self.body_ids = [self.model.body(name).id for name in self.links]
        self.torso = self.links.index("torso_link")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.encoder = ort.InferenceSession(str(package / "backward_encoder.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        signatures = {value.name: value.shape[-1] for value in self.encoder.get_inputs()}
        if signatures != {"state": 64, "privileged_state": 463}:
            raise ValueError(f"Unexpected public backward encoder inputs: {signatures}")
        if [(value.name, value.shape[-1]) for value in self.encoder.get_outputs()] != [("z", 256)]:
            raise ValueError("Unexpected public backward encoder output")
        self.frames = deque(maxlen=self.window_size)
        self.origin = None
        self.heading_inverse = None

    def push(self, timestamp, pose):
        pose = np.asarray(pose, dtype=np.float64).copy()
        if pose.shape != (36,) or not np.isfinite(pose).all():
            raise ValueError("Invalid retargeted G1 reference")
        root_rotation = Rotation.from_quat(pose[3:7][[1, 2, 3, 0]])
        if self.origin is None:
            self.origin = np.array([pose[0], pose[1], 0.0])
            matrix = root_rotation.as_matrix()
            self.heading_inverse = Rotation.from_euler("z", -np.arctan2(matrix[1, 0], matrix[0, 0]))
        pose[:3] = self.heading_inverse.apply(pose[:3] - self.origin)
        pose[3:7] = (self.heading_inverse * root_rotation).as_quat()[[3, 0, 1, 2]]
        if self.frames and not np.isclose(timestamp - self.frames[-1][0], self.dt, atol=1e-5):
            raise ValueError("P-BFM reference must be resampled to 50 Hz")
        self.data.qpos[:7] = pose[:7]
        self.data.qpos[self.addresses] = pose[7:]
        mujoco.mj_forward(self.model, self.data)
        positions = self.data.xpos[self.body_ids].copy()
        rotations = self.data.xmat[self.body_ids].reshape(-1, 3, 3).copy()
        positions = np.concatenate([positions, [positions[self.torso] + rotations[self.torso] @ [0, 0, 0.35]]])
        rotations = np.concatenate([rotations, [rotations[self.torso]]])
        self.frames.append((timestamp, pose, positions, rotations))
        if len(self.frames) < self.window_size:
            return None
        state, privileged, velocity = self.features()
        first = self.reference_index + 1
        selected = slice(first, first + self.lookahead_frames)
        embeddings = self.encoder.run(["z"], {"state": state[selected],
                                              "privileged_state": privileged[selected]})[0]
        latent = embeddings.mean(axis=0)
        norm = np.linalg.norm(latent)
        if embeddings.shape != (8, 256) or not np.isfinite(latent).all() or norm < 1e-8:
            raise ValueError("Invalid P-BFM backward latent")
        latent *= 16 / norm
        frame = self.reference_index
        rotation = self.frames[frame][3][0]
        heading = np.arctan2(rotation[1, 0], rotation[0, 0])
        return {"z": latent.tolist(), "heading_relative_xy": [float(np.cos(heading)), float(np.sin(heading))],
                "heading_valid": True, "velocity_command": [*velocity[frame, 0, :2].tolist(), 1.0],
                "reference_time_sec": float(self.frames[frame][0]), "reference_delay_sec": self.delay_sec}

    def features(self):
        poses = np.array([frame[1] for frame in self.frames])
        positions = np.array([frame[2] for frame in self.frames])
        rotations = np.array([frame[3] for frame in self.frames])
        count = len(poses)
        velocity = np.gradient(positions, self.dt, axis=0)
        velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest", truncate=4)
        angular_velocity = Rotation.from_matrix(
            (rotations[1:] @ rotations[:-1].swapaxes(-1, -2)).reshape(-1, 3, 3)
        ).as_rotvec().reshape(count - 1, 31, 3) / self.dt
        angular_velocity = np.concatenate([angular_velocity, angular_velocity[-1:]])
        angular_velocity = gaussian_filter1d(angular_velocity, 2, axis=0, mode="nearest", truncate=4)
        joint_velocity = np.diff(poses[:, 7:], axis=0) / self.dt
        joint_velocity = np.concatenate([joint_velocity, joint_velocity[-1:]])
        yaw = np.arctan2(rotations[:, 0, 1, 0], rotations[:, 0, 0, 0])
        inverse_heading = Rotation.from_euler("z", -yaw).as_matrix()
        relative_positions = np.einsum("nij,nkj->nki", inverse_heading, positions[:, 1:] - positions[:, :1])
        local_rotations = inverse_heading[:, None] @ rotations
        local_velocity = np.einsum("nij,nkj->nki", inverse_heading, velocity)
        local_angular = np.einsum("nij,nkj->nki", inverse_heading, angular_velocity)
        state = np.concatenate([poses[:, 7:] - self.defaults, joint_velocity, -rotations[:, 0, 2, :],
                                np.einsum("nji,nj->ni", rotations[:, 0], angular_velocity[:, 0]) *
                                self.config["angularVelocityScale"]], axis=1).astype(np.float32)
        privileged = np.concatenate([poses[:, 2:3], relative_positions.reshape(count, -1),
                                     local_rotations[:, :, :, [0, 2]].transpose(0, 1, 3, 2).reshape(count, -1),
                                     local_velocity.reshape(count, -1), local_angular.reshape(count, -1)],
                                    axis=1).astype(np.float32)
        if state.shape != (count, 64) or privileged.shape != (count, 463):
            raise ValueError("P-BFM reference feature layout mismatch")
        return state, privileged, velocity
