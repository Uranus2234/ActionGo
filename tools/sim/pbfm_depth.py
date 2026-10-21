from collections import deque
from multiprocessing import shared_memory
import time

import numpy as np


PROFILE = "optical_z_maxcrop18_reset_v2"
SAMPLED_AGES = (35, 30, 25, 20, 15, 10, 5, 0)
MOUNT_POSITION = (0.0487988662332928, 0.01, 0.4378029937970051)
OPTICAL_QUAT_XYZW = (-0.6579550475696607, 0.6623183568544073,
                    -0.25121840449386046, 0.25558171377860706)


def preprocess_depth(optical_z, *, camera_zero_invalid=False):
    depth = np.asarray(optical_z, dtype=np.float32)
    if depth.shape != (36, 64):
        raise ValueError("P-BFM optical-Z input must be [36,64] meters")
    valid = np.isfinite(depth) & (depth >= 0)
    if camera_zero_invalid:
        valid &= depth > 0
    depth = np.clip(np.where(valid, depth, np.float32(2.5)), 0, 2.5)[18:, 16:48]
    gaussian = np.exp(-np.arange(-1, 2, dtype=np.float32) ** 2 / 2)
    gaussian /= gaussian.sum()
    kernel = gaussian[:, None] * gaussian[None, :]
    padded = np.pad(depth, 1, mode="reflect")
    blurred = np.zeros_like(depth)
    for row in range(3):
        for column in range(3):
            blurred += kernel[row, column] * padded[row:row + 18, column:column + 32]
    return np.rint(np.clip(blurred, 0, 2.5) / np.float32(2.5) * np.float32(255)).astype(np.uint8)


class DepthHistory:
    def __init__(self):
        self.frames = deque(maxlen=37)

    def append(self, optical_z, *, camera_zero_invalid=False):
        frame = preprocess_depth(optical_z, camera_zero_invalid=camera_zero_invalid)
        if not self.frames:
            self.frames.extend(frame.copy() for _ in range(37))
        else:
            self.frames.append(frame)
        return np.stack([self.frames[-1 - age] for age in SAMPLED_AGES]).astype(np.float32) / np.float32(255)


class DepthPublisher:
    def __init__(self, name):
        import zmq

        self.name = name
        self.shm = shared_memory.SharedMemory(name=name, create=True, size=8 * 18 * 32 * 4)
        self.array = np.ndarray((8, 18, 32), dtype=np.float32, buffer=self.shm.buf)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("ipc:///tmp/" + name + ".heartbeat")
        self.sequence = 0
        self.history = DepthHistory()

    def write(self, optical_z):
        np.copyto(self.array, self.history.append(optical_z))
        self.sequence += 1
        self.socket.send_json({"seq": self.sequence, "state": "OK", "t_capture": time.time(),
                               "profile": PROFILE})

    def close(self):
        self.socket.send_json({"seq": self.sequence, "state": "STOPPED"})
        self.socket.close(linger=0)
        self.context.term()
        self.shm.close()
        self.shm.unlink()


class MujocoDepthCamera:
    def __init__(self, model):
        import mujoco

        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.body_id < 0:
            raise ValueError("P-BFM camera requires torso_link")
        housing_mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "head_link")
        self.housing_geom_ids = np.flatnonzero(
            (model.geom_bodyid == self.body_id) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
            & (model.geom_dataid == housing_mesh)) if housing_mesh >= 0 else np.empty(0, dtype=np.int32)
        rows, columns = np.indices((36, 64), dtype=np.float64)
        focal_x = 32 / np.tan(np.deg2rad(89.51) / 2)
        focal_y = 18 / np.tan(np.deg2rad(58.29) / 2)
        rays = np.stack([(columns + 0.5 - 32) / focal_x,
                         (rows + 0.5 - 18) / focal_y, np.ones_like(rows)], axis=-1)
        self.rays = rays.reshape(-1, 3)
        self.rays /= np.linalg.norm(self.rays, axis=1, keepdims=True)
        rotation = np.empty(9, dtype=np.float64)
        quat = np.asarray(OPTICAL_QUAT_XYZW)[[3, 0, 1, 2]]
        mujoco.mju_quat2Mat(rotation, quat)
        self.optical_rotation = rotation.reshape(3, 3)
        self.geom_ids = np.empty(36 * 64, dtype=np.int32)
        self.distances = np.empty(36 * 64, dtype=np.float64)
        self.geom_groups = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.uint8)
        self.cutoff = 2.5 / self.rays[:, 2].min() + 1e-4

    def capture(self, model, data):
        import mujoco

        torso_rotation = data.xmat[self.body_id].reshape(3, 3)
        position = data.xpos[self.body_id] + torso_rotation @ np.asarray(MOUNT_POSITION)
        rotation = torso_rotation @ self.optical_rotation
        directions = np.ascontiguousarray(self.rays @ rotation.T).reshape(-1)
        housing_groups = model.geom_group[self.housing_geom_ids].copy()
        try:
            model.geom_group[self.housing_geom_ids] = 4
            mujoco.mj_multiRay(model, data, position, directions, self.geom_groups, True, -1,
                              self.geom_ids, self.distances, None, len(self.distances), self.cutoff)
        finally:
            model.geom_group[self.housing_geom_ids] = housing_groups
        optical_z = self.distances * self.rays[:, 2]
        optical_z[self.distances < 0] = np.nan
        return optical_z.reshape(36, 64).astype(np.float32)
