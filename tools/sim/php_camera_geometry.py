import numpy as np


def estimate_lower_plane(record):
    depth = (np.asarray(record["depth"]).reshape(58, 87) + 0.5) * 2.7 + 0.3
    rows, columns = np.indices(depth.shape)
    focal = 53 / np.tan(np.deg2rad(89.5 / 2))
    ray_x = ((columns + 0.5) * 98 / 87 - 0.5 + 4 - 53) / focal
    ray_y = (rows + 2 - 30) / focal
    mask = (rows >= 35) & ((columns < 22) | (columns >= 65)) & (depth > 0.35) & (depth < 2.9)
    features = np.stack([ray_x[mask], ray_y[mask], np.ones(mask.sum())], axis=1)
    if len(features) < 500:
        return None
    inverse_depth = 1 / depth[mask]
    plane = np.linalg.lstsq(features, inverse_depth, rcond=None)[0]
    residual = np.sqrt(np.mean((features @ plane - inverse_depth) ** 2))
    if residual > 0.01:
        return None
    quaternion = np.asarray(record["quat"])
    quaternion /= np.linalg.norm(quaternion)
    torso_pitch = np.rad2deg(np.arcsin(np.clip(
        2 * (quaternion[0] * quaternion[2] - quaternion[3] * quaternion[1]), -1, 1
    )))
    optical_pitch = np.rad2deg(np.arctan2(plane[2], plane[1]))
    return {
        "optical_pitch_below_horizontal_deg": float(optical_pitch),
        "torso_pitch_deg": float(torso_pitch),
        "approx_camera_pitch_relative_to_torso_deg": float(optical_pitch - torso_pitch),
        "plane_distance_m": float(1 / np.linalg.norm(plane)),
        "inverse_depth_rmse_per_m": float(residual),
    }
