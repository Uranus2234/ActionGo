from pathlib import Path
import subprocess
import tempfile

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parents[1]


def main():
    spec = PACKAGE / "config/generic_ufo_duobox_050_ma.yaml"
    player = ROOT / "build/generic_policy_player"
    checker = ROOT / "build/policy_spec_schema_check"
    subprocess.run([str(checker), "--spec", str(spec)], check=True)
    subprocess.run([str(player), "--spec", str(spec), "--check-only"], check=True)
    subprocess.run([str(ROOT / "build/bfm_policy_test"), str(PACKAGE / "motions/replay_fixture.json")], check=True)
    base = yaml.safe_load(spec.read_text())
    base["robot"]["params"] = str(PACKAGE / "config/Robot.yaml")
    base["model"]["path"] = str(PACKAGE / "model/actor.onnx")
    base["observation"]["latent_sequence_file"] = str(PACKAGE / "motions/duobox_050_ma_latents.npy")
    with tempfile.TemporaryDirectory(prefix="actiongo-ufo-trial-") as directory:
        directory = Path(directory)
        invalid = directory / "invalid.yaml"
        for case in ("timing", "missing", "shape", "nonfinite"):
            config = yaml.safe_load(yaml.safe_dump(base))
            if case == "timing":
                config["observation"]["history_timing"] = "invalid"
            else:
                latent = directory / f"{case}.npy"
                if case == "shape":
                    np.save(latent, np.zeros((1, 255), dtype=np.float32))
                elif case == "nonfinite":
                    np.save(latent, np.full((1, 256), np.nan, dtype=np.float32))
                config["observation"]["latent_sequence_file"] = str(latent)
            invalid.write_text(yaml.safe_dump(config))
            result = subprocess.run([str(player), "--spec", str(invalid), "--check-only"],
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode != 0, f"{case} was accepted"
            expected = {"timing": "history_timing", "missing": "missing.npy",
                        "shape": "shape", "nonfinite": "nonfinite"}[case]
            assert expected in result.stdout + result.stderr, result.stdout + result.stderr
            print(f"rejected before handoff: {case}")
    print("UFO DuoBox trial checks passed")


if __name__ == "__main__":
    main()
