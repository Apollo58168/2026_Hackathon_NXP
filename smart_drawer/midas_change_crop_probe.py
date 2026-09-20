"""Small depth helpers used by the standalone A/B inventory probe."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .depth_denoise import normalize_relative_depth, spatial_filter


def _first_model(*paths: Path) -> Path:
    return next((path for path in paths if path.is_file()), paths[0])


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIDAS_MODEL = _first_model(
    Path("/root/midas_2_1_small_int8_vela.tflite"),
    Path("/opt/gopoint-apps/downloads/midas_v2_1_small_quant_vela.tflite"),
    ROOT / "models/midas_v2_1_small_quant_vela.tflite",
)
DEFAULT_DEPTH_ANYTHING_MODEL = ROOT / "models/depth_anything_v2_metric_hypersim_vits.pth"


@dataclass(frozen=True)
class Thresholds:
    stable_median: float = 0.015
    stable_ratio: float = 0.08
    motion_median: float = 0.030
    motion_ratio: float = 0.015
    change_depth: float = 0.040
    min_change_area: int = 80
    morphology_size: int = 5
    padding: float = 0.25
    noise_multiplier: float = 1.5
    noise_warmup_frames: int = 8
    change_noise_multiplier: float = 4.0
    hysteresis_low_ratio: float = 0.50


@dataclass(frozen=True)
class Snapshot:
    frame: np.ndarray
    depth: np.ndarray
    noise_p95: float
    noise_map: Optional[np.ndarray] = None


def stability_metrics(
    current: np.ndarray,
    previous: np.ndarray,
    threshold: float,
    mask: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    difference = np.abs(current - previous)
    if mask is not None:
        difference = difference[mask]
    if difference.size == 0:
        raise ValueError("depth comparison mask has no pixels")
    return float(np.median(difference)), float(np.mean(difference > threshold))


def is_stable(
    current: np.ndarray,
    previous: np.ndarray,
    thresholds: Thresholds,
    *,
    median_limit: Optional[float] = None,
    pixel_limit: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> bool:
    median_limit = thresholds.stable_median if median_limit is None else median_limit
    pixel_limit = thresholds.stable_median if pixel_limit is None else pixel_limit
    median_delta, ratio = stability_metrics(current, previous, pixel_limit, mask)
    return median_delta <= median_limit and ratio <= thresholds.stable_ratio


def has_motion(
    current: np.ndarray,
    previous: np.ndarray,
    thresholds: Thresholds,
    mask: Optional[np.ndarray] = None,
) -> bool:
    median_delta, ratio = stability_metrics(current, previous, thresholds.motion_median, mask)
    return median_delta >= thresholds.motion_median or ratio >= thresholds.motion_ratio


def make_snapshot(frames: list[np.ndarray], depths: list[np.ndarray]) -> Snapshot:
    if len(frames) != len(depths) or not depths:
        raise ValueError("snapshot needs matching RGB and depth frames")
    stack = np.stack(depths).astype(np.float32)
    median_depth = np.median(stack, axis=0).astype(np.float32)
    noise_map = (1.4826 * np.median(np.abs(stack - median_depth), axis=0)).astype(np.float32)
    return Snapshot(
        frame=frames[len(frames) // 2].copy(),
        depth=median_depth,
        noise_p95=float(np.percentile(noise_map, 95)),
        noise_map=noise_map,
    )


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("depth contains no finite values")
    low, high = np.percentile(finite, (2, 98))
    normalized = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


class MidasTFLite:
    """CPU LiteRT adapter for the standalone probe's float MiDaS model."""

    def __init__(self, model_path: Path, delegate_path: Optional[str] = None) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"MiDaS model not found: {model_path}")
        if delegate_path and Path(delegate_path).is_file():
            import tflite_runtime.interpreter as tflite

            delegate = tflite.load_delegate(delegate_path)
            self.interpreter = tflite.Interpreter(
                model_path=str(model_path), experimental_delegates=[delegate]
            )
            self.backend = "Ethos-U"
        else:
            from ai_edge_litert.interpreter import Interpreter

            self.interpreter = Interpreter(model_path=str(model_path))
            self.backend = "LiteRT CPU"
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        shape = tuple(int(value) for value in self.input["shape"])
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3 or self.input["dtype"] is not np.float32:
            raise ValueError(f"expected float32 1xHxWx3 MiDaS input, got {shape} {self.input['dtype']}")
        self.height, self.width = shape[1], shape[2]
        self.relative_depth = True
        self.name = f"MiDaS TFLite ({self.backend})"

    def infer(self, frame: np.ndarray) -> np.ndarray:
        image = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.interpreter.set_tensor(self.input["index"], image[None])
        self.interpreter.invoke()
        depth = np.asarray(self.interpreter.get_tensor(self.output["index"]), dtype=np.float32).squeeze()
        if depth.shape != (self.height, self.width) or not np.isfinite(depth).all():
            raise RuntimeError(f"invalid MiDaS output: {depth.shape}")
        return depth


class DepthAnythingMetric:
    """Optional lazy adapter retained for PC experiments."""

    def __init__(self, model_path: Path) -> None:
        from depth_anything_metric import DepthAnythingMetricSmall

        self.estimator = DepthAnythingMetricSmall(model_path)
        self.width = self.height = self.estimator.input_size
        self.relative_depth = False
        self.name = f"Depth Anything V2 Metric Small ({self.estimator.backend})"

    def infer(self, frame: np.ndarray) -> np.ndarray:
        return self.estimator.infer(frame)
