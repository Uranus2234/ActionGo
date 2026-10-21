import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/teleop"))
from pbfm_reference import PoseResampler, ReferenceEncoder


def check_encoder(source):
    encoder = ReferenceEncoder(source)
    frames = []
    contexts = []
    for index in range(80):
        timestamp = index * 0.02
        pose = np.concatenate([[0.4 * timestamp, 0.1 * timestamp, 0.793 + 0.01 * np.sin(timestamp * 3)],
                               Rotation.from_euler("xyz", [0.05 * np.sin(timestamp), 0.03, 0.3 * timestamp])
                               .as_quat()[[3, 0, 1, 2]], encoder.defaults.copy()])
        pose[7 + encoder.joint_names.index("left_elbow_joint")] = 0.5 + 0.2 * np.sin(timestamp * 4)
        context = encoder.push(timestamp, pose)
        frames.append(encoder.frames[-1])
        if context:
            contexts.append(context)
    revision = "671fbafe71a5f58bab77eeb5d969051fd2d283f4"
    asset = "interactive/assets/motion-import-uImOWE3z.js"
    official = subprocess.check_output(["git", "-C", str(source), "show", f"{revision}:{asset}"], text=True)
    official = re.sub(r"^import.*?;", "", official)
    official = re.sub(r"export\{[^}]+\};?", "", official)
    prelude = """
import fs from 'node:fs';
const $ = value => Math.atan2(2*(value[0]*value[3]+value[1]*value[2]),1-2*(value[2]**2+value[3]**2));
const O = value => N(_(value),[0,0,-1]);
const k = value => value.map(item => item/Math.hypot(...value));
"""
    program = prelude + official + """
const input = JSON.parse(fs.readFileSync(0,'utf8'));
const result = R(input.frames,{positions:new Float32Array(input.positions),rotations:new Float32Array(input.rotations)},input.config);
console.log(JSON.stringify(Object.fromEntries(Object.entries(result).map(([key,value])=>[key,Array.from(value)]))));
"""
    payload = {"config": encoder.config,
               "frames": [{"rootPosition": frame[1][:3].tolist(), "rootQuaternionWxyz": frame[1][3:7].tolist(),
                           "jointPosition": frame[1][7:].tolist()} for frame in frames],
               "positions": np.array([frame[2] for frame in frames]).reshape(-1).tolist(),
               "rotations": Rotation.from_matrix(np.array([frame[3] for frame in frames]).reshape(-1, 3, 3))
               .as_quat()[:, [3, 0, 1, 2]].reshape(-1).tolist()}
    result = subprocess.run(["node", "--input-type=module", "-e", program], input=json.dumps(payload),
                            capture_output=True, text=True, check=True)
    expected = json.loads(result.stdout)
    state, privileged, _ = encoder.features()
    offset = len(frames) - encoder.window_size
    first = encoder.reference_index + 1
    last = first + encoder.lookahead_frames
    selected = slice(first, last)
    expected_state = np.asarray(expected["state"], dtype=np.float32).reshape(-1, 64)[offset + first:offset + last]
    expected_privileged = np.asarray(expected["privileged"], dtype=np.float32).reshape(-1, 463)[offset + first:offset + last]
    np.testing.assert_allclose(state[selected], expected_state, atol=8e-6)
    np.testing.assert_allclose(privileged[selected], expected_privileged, atol=8e-6)
    latent = encoder.encoder.run(["z"], {"state": expected_state, "privileged_state": expected_privileged})[0].mean(axis=0)
    latent *= 16 / np.linalg.norm(latent)
    np.testing.assert_allclose(contexts[-1]["z"], latent, atol=2e-5)
    expected_velocity = np.asarray(expected["linear"]).reshape(-1, 31, 3)[offset + encoder.reference_index, 0, :2]
    np.testing.assert_allclose(contexts[-1]["velocity_command"][:2], expected_velocity, atol=8e-6)
    assert len(contexts) == 80 - encoder.window_size + 1
    assert np.isclose(frames[-1][0] - contexts[-1]["reference_time_sec"], 0.34)
    print("P-BFM online features and future-eight latent match official browser backwardFeatures")


def check_resampler():
    resampler = PoseResampler()
    samples = []
    for index in range(91):
        timestamp = index / 90
        pose = np.concatenate([[timestamp, 0, 0.8], Rotation.from_euler("z", timestamp).as_quat()[[3, 0, 1, 2]],
                               np.full(29, timestamp)])
        samples.extend(resampler.push(timestamp, pose))
    assert len(samples) == 51
    for timestamp, pose in samples:
        np.testing.assert_allclose(pose[[0, 7]], [timestamp, timestamp], atol=1e-8)
        np.testing.assert_allclose(pose[3:7], Rotation.from_euler("z", timestamp).as_quat()[[3, 0, 1, 2]], atol=1e-8)
    assert resampler.push(1, samples[-1][1]) == []
    try:
        resampler.push(1.5, samples[-1][1])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Resampler bridged a stale source gap")
    print("Pico 90 -> 50 Hz resampling, quaternion interpolation and stale gap checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    check_resampler()
    check_encoder(args.source.resolve())
