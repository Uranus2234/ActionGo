"""Export the latest causal HandoffGuard critic to a fixed-shape ONNX score model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


class RiskPercent(torch.nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = payload["config"]
        state = payload["state_dict"]
        self.register_buffer("mean", state["mean"])
        self.register_buffer("scale", state["scale"])
        widths = config["hidden_widths"]
        architecture = config.get("architecture", "mlp")
        self.architecture = architecture
        self.reference = float(config["risk_label"]["reference_cost"])
        if architecture == "tcn":
            self.temporal = torch.nn.Sequential(
                torch.nn.Conv1d(93, 128, 3), torch.nn.GELU(),
                torch.nn.Conv1d(128, 128, 3, dilation=2), torch.nn.GELU(),
                torch.nn.Conv1d(128, 128, 3, dilation=4), torch.nn.GELU())
            previous = 128 * 10 + 74
        else:
            previous = 930 + 74
        layers = []
        for width in widths:
            layers.extend((torch.nn.Linear(previous, width), torch.nn.ELU()))
            previous = width
        layers.append(torch.nn.Linear(previous, 5))
        self.network = torch.nn.Sequential(*layers)
        self.load_state_dict(state)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        normalized = (history - self.mean) / self.scale
        if self.architecture == "tcn":
            padding = lambda x, amount: torch.nn.functional.pad(x, (amount, 0))
            x = padding(normalized.transpose(1, 2), 2)
            x = torch.nn.functional.gelu(self.temporal[0](x))
            x = torch.nn.functional.gelu(self.temporal[2](padding(x, 4)))
            x = torch.nn.functional.gelu(self.temporal[4](padding(x, 8)))
            x = x.flatten(1)
        else:
            x = normalized.flatten(1)
        x = torch.cat((x, torch.zeros(x.shape[0], 74, device=x.device)), dim=1)
        raw = self.network(x)
        return (torch.nn.functional.softplus(raw[:, :1]) / self.reference).clamp(0, 1) * 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = RiskPercent(args.checkpoint).eval()
    sample = torch.zeros(1, 10, 93)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, args.output, input_names=["history"],
                      output_names=["risk_percent"], opset_version=17,
                      dynamic_axes={"history": {0: "batch"}, "risk_percent": {0: "batch"}})
    args.output.with_suffix(".json").write_text(json.dumps({"status": "EXPORTED", "input_shape": [1, 10, 93],
                                                               "output_shape": [1, 1], "architecture": model.architecture}, indent=2) + "\n")


if __name__ == "__main__":
    main()
