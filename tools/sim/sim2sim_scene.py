"""Terrain additions compiled once into the shared simulation scene."""

import mujoco
import numpy as np


def add_terrain(spec, kind, *, start=1.8, height=0.12, length=3.0, width=2.0, seed=0):
    values = np.asarray([start, height, length, width], dtype=float)
    if not np.isfinite(values).all() or start < 0.5 or min(height, length, width) <= 0:
        raise ValueError("terrain requires finite start >= 0.5 and positive height/length/width")
    metadata = dict(kind=kind, start=start, height=height, length=length, width=width, seed=seed)
    common = dict(group=0, contype=1, conaffinity=1, condim=3,
                  friction=(1.0, 0.005, 0.0001), rgba=(0.45, 0.55, 0.5, 1.0))
    if kind == "flat":
        return metadata
    if kind == "stairs":
        count = 5
        depth = length / (2 * count + 1)
        for index in range(2 * count + 1):
            top = height * min(index + 1, count, 2 * count + 1 - index)
            spec.worldbody.add_geom(name=f"sim_step_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=(start + (index + 0.5) * depth, 0, top / 2),
                                    size=(depth / 2, width / 2, top / 2), **common)
    elif kind == "ramp":
        vertices = [(start, -width / 2, -0.05), (start, width / 2, -0.05),
                    (start + length, -width / 2, -0.05), (start + length, width / 2, -0.05),
                    (start, -width / 2, 0), (start, width / 2, 0),
                    (start + length, -width / 2, height), (start + length, width / 2, height)]
        spec.add_mesh(name="sim_ramp_mesh", uservert=np.asarray(vertices).ravel())
        spec.worldbody.add_geom(name="sim_ramp", type=mujoco.mjtGeom.mjGEOM_MESH,
                                meshname="sim_ramp_mesh", **common)
        spec.worldbody.add_geom(name="sim_ramp_landing", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=(start + length + 0.5, 0, height / 2),
                                size=(0.5, width / 2, height / 2), **common)
    elif kind == "rough":
        rng = np.random.default_rng(seed)
        heights = rng.uniform(0, 1, (33, 65))
        for _ in range(3):
            padded = np.pad(heights, 1, mode="edge")
            heights = sum(padded[row:row + 33, column:column + 65]
                          for row in range(3) for column in range(3)) / 9
        heights -= heights.min()
        heights /= heights.max()
        heights *= np.minimum(np.linspace(0, 8, 65), np.linspace(8, 0, 65)).clip(0, 1)
        heights[[0, -1], :] = 0
        spec.add_hfield(name="sim_rough_field", nrow=33, ncol=65,
                        size=(length / 2, width / 2, height, 0.1), userdata=heights.ravel())
        spec.worldbody.add_geom(name="sim_rough", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                                hfieldname="sim_rough_field", pos=(start + length / 2, 0, 0), **common)
    else:
        raise ValueError("unsupported terrain: " + kind)
    return metadata
