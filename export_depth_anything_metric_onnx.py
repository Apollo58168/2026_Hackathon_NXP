#!/usr/bin/env python3
"""Export Depth Anything V2 Metric Small to a static, eIQ-ready ONNX model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from depth_anything_metric import DepthAnythingMetricSmall


DEFAULT_CHECKPOINT = Path("models/depth_anything_v2_metric_hypersim_vits.pth")
DEFAULT_OUTPUT = Path("models/depth_anything_v2_metric_hypersim_vits_518.onnx")
INPUT_NAME = "image"
OUTPUT_NAME = "depth_meters"
INPUT_SIZE = 518


class OnnxMetricWrapper(torch.nn.Module):
    """Expose the metric model with one preprocessed NCHW float32 input."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def write_contract(output_path: Path) -> Path:
    contract_path = output_path.with_suffix(".eiq.json")
    payload = {
        "model": output_path.name,
        "format": "onnx",
        "opset": 17,
        "input": {
            "name": INPUT_NAME,
            "dtype": "float32",
            "shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "layout": "NCHW",
            "colour_order": "RGB",
            "preprocess": {
                "resize": [INPUT_SIZE, INPUT_SIZE],
                "scale": "pixel / 255.0",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "formula": "(RGB / 255.0 - mean) / std",
            },
        },
        "output": {
            "name": OUTPUT_NAME,
            "dtype": "float32",
            "shape": [1, INPUT_SIZE, INPUT_SIZE],
            "unit": "estimated metres",
            "near_is_positive": False,
        },
        "source": "Depth Anything V2 Metric Hypersim Small",
        "note": "Monocular metric estimates are not a calibrated depth sensor. Use representative preprocessed .npy inputs for eIQ post-training quantization.",
    }
    contract_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return contract_path


def export(checkpoint: Path, output_path: Path) -> tuple[Path, Path, float, float]:
    estimator = DepthAnythingMetricSmall(checkpoint)
    model = OnnxMetricWrapper(estimator.model.cpu()).eval()
    example = torch.linspace(-2.0, 2.0, INPUT_SIZE * INPUT_SIZE * 3, dtype=torch.float32).reshape(
        1, 3, INPUT_SIZE, INPUT_SIZE
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        output_path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    model_proto = onnx.load(output_path)
    onnx.checker.check_model(model_proto)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    numpy_input = example.numpy()
    with torch.inference_mode():
        expected = model(example).numpy()
    actual = session.run([OUTPUT_NAME], {INPUT_NAME: numpy_input})[0]
    max_abs_error = float(np.max(np.abs(expected - actual)))
    mean_abs_error = float(np.mean(np.abs(expected - actual)))
    if max_abs_error > 1e-3:
        raise RuntimeError(f"ONNX Runtime parity check failed: max_abs_error={max_abs_error}")
    return output_path, write_contract(output_path), max_abs_error, mean_abs_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Depth Anything V2 Metric Small to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output, contract, max_abs_error, mean_abs_error = export(args.checkpoint, args.output)
    print(
        f"ONNX export OK: {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)\n"
        f"contract: {contract}\n"
        f"ONNX Runtime parity: max_abs_error={max_abs_error:.3e}, mean_abs_error={mean_abs_error:.3e}"
    )


if __name__ == "__main__":
    main()
