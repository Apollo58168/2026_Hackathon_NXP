"""Depth Anything V2 Metric Small adapter.

The PyPI Depth Anything V2 package exposes the relative-depth head.  The
official Metric Small checkpoint uses the same ViT-S/DPT backbone but replaces
the final ReLU with sigmoid and scales it to the indoor model's 20 metre range.
This adapter makes that small, checkpoint-compatible difference explicit.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2


METRIC_SMALL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}
INDOOR_MAX_DEPTH_METERS = 20.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(bgr_frame: np.ndarray, *, input_size: int = 518) -> np.ndarray:
    """Prepare one OpenCV BGR frame for the exported Metric Small ONNX model."""
    if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
        raise ValueError(f"expected a BGR frame with three channels, got {bgr_frame.shape}")
    image = cv2.resize(bgr_frame, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)


class MetricDepthAnythingV2(DepthAnythingV2):
    """Official Metric V2 head over the package's shared backbone/DPT code."""

    def __init__(self, *, max_depth: float = INDOOR_MAX_DEPTH_METERS) -> None:
        super().__init__(**METRIC_SMALL_CONFIG)
        relative_layers = list(self.depth_head.scratch.output_conv2.children())
        # Official metric head: conv -> ReLU -> conv -> Sigmoid.
        self.depth_head.scratch.output_conv2 = torch.nn.Sequential(
            *relative_layers[:-2], torch.nn.Sigmoid()
        )
        self.max_depth = float(max_depth)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # The relative implementation applies ReLU after the head.  It is an
        # identity for sigmoid output; multiplying here yields metric metres.
        return super().forward(image) * self.max_depth


class DepthAnythingMetricSmall:
    """Load and run the official indoor Metric Small checkpoint locally."""

    input_size = 518
    near_is_positive = False  # Metric depth grows as a surface moves farther away.
    metric_depth = True
    backend = "PyTorch CPU"

    def __init__(self, checkpoint: Path, *, input_size: int = 518) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Depth Anything Metric Small checkpoint not found: {checkpoint}")
        self.input_size = int(input_size)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.backend = f"PyTorch {self.device.type.upper()}"
        self.model = MetricDepthAnythingV2()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def preprocess_bgr(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Match the exported ONNX input contract: RGB, NCHW, 518 square."""
        return preprocess_bgr(bgr_frame, input_size=self.input_size)

    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return a square metric-depth map in estimated metres."""
        tensor = torch.from_numpy(self.preprocess_bgr(bgr_frame)).to(self.device)
        with torch.inference_mode():
            depth = self.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise RuntimeError("Depth Anything returned invalid metric depth")
        return depth
