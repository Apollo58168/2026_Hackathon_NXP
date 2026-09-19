#!/usr/bin/env python3
"""Depth-only preview for local development and i.MX93 bring-up.

This program intentionally has no YOLO, database, voice, or semantic-catalog
dependency.  The default backend is Depth Anything V2 Metric Small for indoor
scenes; its output is estimated metres, not a depth-sensor measurement.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from depth_anything_metric import DepthAnythingMetricSmall


DEFAULT_MODEL = Path("models/depth_anything_v2_metric_hypersim_vits.pth")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def model_input_shape(details: dict[str, Any]) -> tuple[int, int]:
    """Validate the expected NHWC RGB model input and return height, width."""
    shape = tuple(int(value) for value in details["shape"])
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
        raise ValueError(f"expected a 1xHxWx3 model input, got {shape}")
    if details["dtype"] is not np.float32:
        raise ValueError(f"expected a float32 MiDaS input, got {details['dtype']}")
    return shape[1], shape[2]


def make_interpreter(model_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load Metric Small by default; retain TFLite as an explicit fallback."""
    if model_path.suffix.lower() in {".pth", ".pt"}:
        estimator = DepthAnythingMetricSmall(model_path)
        details: dict[str, Any] = {
            "shape": np.array([1, estimator.input_size, estimator.input_size, 3]),
            "dtype": np.float32,
            "metric_depth": True,
            "near_is_positive": estimator.near_is_positive,
            "backend": estimator.backend,
        }
        return estimator, details, {}
    if not model_path.is_file():
        raise FileNotFoundError(f"depth model not found: {model_path}")
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    model_input_shape(input_details)
    input_details["metric_depth"] = False
    input_details["near_is_positive"] = True
    input_details["backend"] = "LiteRT XNNPACK"
    return interpreter, input_details, output_details


def preprocess_bgr(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert an OpenCV BGR frame to the MiDaS v2.1 Small input tensor."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected a BGR image with three channels, got {frame.shape}")
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return normalized[None, ...].astype(np.float32)


def infer_depth(
    interpreter: Interpreter,
    input_details: dict[str, Any],
    output_details: dict[str, Any],
    frame: np.ndarray,
) -> np.ndarray:
    if isinstance(interpreter, DepthAnythingMetricSmall):
        return interpreter.infer(frame)
    height, width = model_input_shape(input_details)
    tensor = preprocess_bgr(frame, height, width)
    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    depth = np.asarray(interpreter.get_tensor(output_details["index"]), dtype=np.float32).squeeze()
    if depth.shape != (height, width) or not np.isfinite(depth).all():
        raise RuntimeError(f"invalid MiDaS depth output: shape={depth.shape}")
    return depth


def colorize_depth(
    depth: np.ndarray,
    output_size: tuple[int, int],
    *,
    near_is_positive: bool = False,
) -> np.ndarray:
    """Return a MAGMA visualisation, with warm colours representing near depth."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise RuntimeError("MiDaS returned no finite depth values")
    low, high = np.percentile(finite, (2, 98))
    normalized = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    if not near_is_positive:
        normalized = 1.0 - normalized
    visual = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.resize(visual, output_size, interpolation=cv2.INTER_CUBIC)


def self_test(model_path: Path) -> None:
    """Run one real inference without requiring camera access or a display."""
    interpreter, input_details, output_details = make_interpreter(model_path)
    height, width = 480, 640
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    frame = np.repeat(gradient[None, :], height, axis=0)
    frame = np.dstack((frame, frame, frame))
    started = time.perf_counter()
    depth = infer_depth(interpreter, input_details, output_details, frame)
    elapsed_ms = (time.perf_counter() - started) * 1000
    preview = colorize_depth(depth, (width, height), near_is_positive=bool(input_details["near_is_positive"]))
    assert preview.shape == (height, width, 3)
    print(
        "self-test OK "
        f"input={tuple(input_details['shape'])} output={tuple(depth.shape)} "
        f"depth_range=({depth.min():.5f}, {depth.max():.5f}) latency_ms={elapsed_ms:.1f} "
        f"backend={input_details['backend']}"
    )


def run_camera(args: argparse.Namespace) -> None:
    interpreter, input_details, output_details = make_interpreter(args.model)
    backend = cv2.CAP_AVFOUNDATION if args.avfoundation else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera}; use --camera to choose another device")

    print(f"model: {args.model}; backend: {input_details['backend']}")
    print(f"camera: {args.camera}; press q or Esc to stop")
    frame_count = 0
    started = time.perf_counter()
    try:
        while args.max_frames == 0 or frame_count < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame capture failed")
            inference_started = time.perf_counter()
            depth = infer_depth(interpreter, input_details, output_details, frame)
            latency_ms = (time.perf_counter() - inference_started) * 1000
            frame_count += 1
            if not args.no_display:
                depth_view = colorize_depth(
                    depth,
                    (frame.shape[1], frame.shape[0]),
                    near_is_positive=bool(input_details["near_is_positive"]),
                )
                combined = np.hstack((frame, depth_view))
                fps = frame_count / max(time.perf_counter() - started, 1e-6)
                cv2.putText(
                    combined,
                    f"Metric depth | {latency_ms:.1f} ms | {fps:.1f} FPS",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Smart Drawer: RGB | relative depth", combined)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    print(f"processed_frames={frame_count} average_fps={frame_count / max(elapsed, 1e-6):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth Anything V2 Metric Small camera preview")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to Metric Small .pth checkpoint")
    parser.add_argument("--self-test", action="store_true", help="run one model inference without a camera")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until q/Esc; useful for headless tests")
    parser.add_argument("--no-display", action="store_true", help="capture and infer without opening a window")
    parser.add_argument("--avfoundation", action="store_true", help="force macOS AVFoundation capture backend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.model)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
