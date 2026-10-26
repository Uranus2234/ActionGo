#!/usr/bin/env python3
"""Render actual ActionGo DDS states beside their reference at the recorded simulation speed."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from uni_supermimic_mujoco import load_training, set_motion_state

    manifest = json.loads((args.package / "manifest.json").read_text())
    mujoco, torch, mat_to_quat, motion, model, data, indices = load_training(
        args.training_repo.resolve(), args.motion.resolve(), manifest["provenance"]["motion"]["source_fps"])
    torch.set_num_threads(2)
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 720
    for geom in indices["robot_geoms"]:
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            model.geom_group[geom] = 3
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.4, 135, -15
    font = ImageFont.truetype("DejaVuSans.ttf", 19)
    output = args.log_dir / "actiongo-comparison-50fps.mp4"
    labels = ("ActionGo C++ | ckpt_200000 | MuJoCo", f"Reference | {motion.names[0]}")
    with np.load(args.log_dir / "trace.npz") as trace:
        if len(np.unique(trace["reset_index"])) != 1:
            raise ValueError("Render one reset episode at a time")
        np.testing.assert_array_equal(trace["tick_ms"], np.arange(len(trace["tick_ms"])) * 20)
        with mujoco.Renderer(model, height=720, width=640) as renderer, imageio.get_writer(
            output, fps=50, codec="libx264", quality=8, macro_block_size=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2"],
        ) as writer:
            for row in range(len(trace["q"])):
                frame = int(trace["reference_frame"][row])
                panels = []
                for panel, label in enumerate(labels):
                    if panel == 0:
                        root_q = indices["root_q"]
                        data.qpos[:] = model.qpos0
                        data.qpos[root_q:root_q + 3] = trace["root_p"][row]
                        data.qpos[root_q + 3:root_q + 7] = trace["root_quat"][row]
                        data.qpos[indices["q"]] = trace["q"][row]
                        mujoco.mj_forward(model, data)
                    else:
                        set_motion_state(mujoco, torch, mat_to_quat, model, data, indices, motion, frame)
                    camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.75]
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    image = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 640, 78), fill=(18, 23, 32))
                    draw.text((12, 10), label, fill="white", font=font)
                    draw.text((12, 43), f"{motion.names[0]} | {trace['tick_ms'][row]/1000:.2f}s | 1x | 50 fps",
                              fill=(125, 215, 255), font=font)
                    panels.append(np.asarray(image))
                combined = np.concatenate(panels, axis=1)
                writer.append_data(combined)
                if row == len(trace["q"]) // 2:
                    Image.fromarray(combined).save(args.log_dir / "preview.jpg")
    print(output)


if __name__ == "__main__":
    main()
