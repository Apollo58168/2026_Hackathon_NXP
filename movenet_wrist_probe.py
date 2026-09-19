#!/usr/bin/env python3
"""Live MoveNet wrist-turn probe for candidate object-placement locations.

The probe is intentionally RGB-only.  It establishes whether SinglePose
MoveNet can observe a wrist reliably from the overhead drawer camera before
that signal is allowed to gate depth-change detection.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


DEFAULT_MODEL = Path("models/movenet_singlepose_lightning_int8.tflite")
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
ARM_CONNECTIONS = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10))
WRISTS = {"left": 9, "right": 10}


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class TurnEvent:
    side: str
    x: float
    y: float
    angle_degrees: float
    confidence: float
    timestamp: float


class MoveNetLightning:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"MoveNet model not found: {model_path}")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        shape = tuple(int(value) for value in self.input["shape"])
        if shape != (1, 192, 192, 3) or self.input["dtype"] is not np.uint8:
            raise ValueError(f"expected uint8 MoveNet input [1,192,192,3], got {shape} {self.input['dtype']}")
        output_shape = tuple(int(value) for value in self.output["shape"])
        if output_shape != (1, 1, 17, 3):
            raise ValueError(f"expected MoveNet output [1,1,17,3], got {output_shape}")
        self.size = 192

    def infer(self, frame: np.ndarray) -> list[Keypoint]:
        height, width = frame.shape[:2]
        scale = min(self.size / width, self.size / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pad_x = (self.size - resized_width) // 2
        pad_y = (self.size - resized_height) // 2
        tensor = np.zeros((1, self.size, self.size, 3), dtype=np.uint8)
        tensor[0, pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        self.interpreter.set_tensor(self.input["index"], tensor)
        self.interpreter.invoke()
        output = np.asarray(self.interpreter.get_tensor(self.output["index"]), dtype=np.float32)[0, 0]
        keypoints = []
        for normalized_y, normalized_x, confidence in output:
            x = (float(normalized_x) * self.size - pad_x) / scale
            y = (float(normalized_y) * self.size - pad_y) / scale
            keypoints.append(Keypoint(x, y, float(confidence)))
        return keypoints


class WristTurnDetector:
    """Detect a 2-D wrist reversal after enough travel on both sides."""

    def __init__(
        self,
        side: str,
        *,
        window: int,
        minimum_leg: float,
        minimum_angle: float,
        smoothing: float,
        cooldown_seconds: float,
    ) -> None:
        self.side = side
        self.window = window
        self.minimum_leg = minimum_leg
        self.minimum_angle = minimum_angle
        self.smoothing = smoothing
        self.cooldown_seconds = cooldown_seconds
        self.history: Deque[tuple[float, np.ndarray, float]] = deque(maxlen=window * 2 + 1)
        self.trail: Deque[tuple[int, int]] = deque(maxlen=80)
        self.smoothed: Optional[np.ndarray] = None
        self.last_event_time = -float("inf")

    def reset(self) -> None:
        self.history.clear()
        self.trail.clear()
        self.smoothed = None
        self.last_event_time = -float("inf")

    def add(self, timestamp: float, keypoint: Keypoint, confidence_threshold: float) -> Optional[TurnEvent]:
        if keypoint.confidence < confidence_threshold:
            self.history.clear()
            self.smoothed = None
            return None
        point = np.array([keypoint.x, keypoint.y], dtype=np.float32)
        self.smoothed = point if self.smoothed is None else self.smoothing * point + (1.0 - self.smoothing) * self.smoothed
        self.history.append((timestamp, self.smoothed.copy(), keypoint.confidence))
        self.trail.append((round(float(self.smoothed[0])), round(float(self.smoothed[1]))))
        if len(self.history) < self.history.maxlen or timestamp - self.last_event_time < self.cooldown_seconds:
            return None
        observations = list(self.history)
        centre_time, centre, centre_confidence = observations[self.window]
        before = np.mean([item[1] for item in observations[: self.window]], axis=0)
        after = np.mean([item[1] for item in observations[self.window + 1 :]], axis=0)
        incoming = centre - before
        outgoing = after - centre
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length < self.minimum_leg or outgoing_length < self.minimum_leg:
            return None
        cosine = float(np.dot(incoming, outgoing) / max(incoming_length * outgoing_length, 1e-6))
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if angle < self.minimum_angle:
            return None
        self.last_event_time = timestamp
        return TurnEvent(self.side, float(centre[0]), float(centre[1]), angle, centre_confidence, centre_time)


def draw_pose(image: np.ndarray, keypoints: list[Keypoint], confidence: float) -> None:
    for first, second in ARM_CONNECTIONS:
        a, b = keypoints[first], keypoints[second]
        if a.confidence >= confidence and b.confidence >= confidence:
            cv2.line(image, (round(a.x), round(a.y)), (round(b.x), round(b.y)), (0, 220, 0), 2, cv2.LINE_AA)
    for side, index in WRISTS.items():
        point = keypoints[index]
        if point.confidence >= confidence:
            colour = (255, 160, 0) if side == "left" else (0, 160, 255)
            cv2.circle(image, (round(point.x), round(point.y)), 7, colour, cv2.FILLED, cv2.LINE_AA)
            cv2.putText(image, f"{side} wrist {point.confidence:.2f}", (round(point.x) + 9, round(point.y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def draw_trail(image: np.ndarray, detector: WristTurnDetector, colour: tuple[int, int, int]) -> None:
    points = list(detector.trail)
    for first, second in zip(points, points[1:]):
        cv2.line(image, first, second, colour, 2, cv2.LINE_AA)


def draw_event(image: np.ndarray, event: TurnEvent, radius: int) -> None:
    centre = (round(event.x), round(event.y))
    cv2.circle(image, centre, radius, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(image, centre, (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.putText(
        image,
        f"TURN ROI: {event.side} wrist, {event.angle_degrees:.0f} deg",
        (max(8, centre[0] - radius), max(24, centre[1] - radius - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def self_test() -> None:
    detector = WristTurnDetector("right", window=3, minimum_leg=8, minimum_angle=120, smoothing=1.0, cooldown_seconds=0)
    path = [(10, 30), (20, 30), (30, 30), (40, 30), (30, 30), (20, 30), (10, 30)]
    event = None
    for index, (x, y) in enumerate(path):
        event = detector.add(index / 10, Keypoint(x, y, 0.9), 0.3) or event
    assert event is not None and event.side == "right" and event.angle_degrees > 170
    assert abs(event.x - 40) < 1e-6 and abs(event.y - 30) < 1e-6
    print("movenet_wrist_probe: self-test OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoveNet wrist trajectory and turn-point probe")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--avfoundation", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--turn-window", type=int, default=4, help="frames before/after the candidate turn")
    parser.add_argument("--minimum-leg", type=float, default=18.0, help="minimum pixels travelled on both sides of a turn")
    parser.add_argument("--minimum-angle", type=float, default=125.0)
    parser.add_argument("--smoothing", type=float, default=0.40)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--roi-radius", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("captures/movenet-wrist-probe"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.turn_window < 2 or args.minimum_leg <= 0 or not 90 <= args.minimum_angle <= 180:
        raise ValueError("turn window/leg must be positive and angle must be in [90, 180]")
    if not 0 < args.smoothing <= 1 or not 0 <= args.confidence <= 1:
        raise ValueError("smoothing must be in (0,1] and confidence in [0,1]")
    model = MoveNetLightning(args.model)
    detectors = {
        side: WristTurnDetector(
            side,
            window=args.turn_window,
            minimum_leg=args.minimum_leg,
            minimum_angle=args.minimum_angle,
            smoothing=args.smoothing,
            cooldown_seconds=args.cooldown,
        )
        for side in WRISTS
    }
    backend = cv2.CAP_AVFOUNDATION if args.avfoundation else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera}")
    latest_event: Optional[TurnEvent] = None
    latest_view: Optional[np.ndarray] = None
    frames = 0
    window_name = "MoveNet wrist turn probe"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("MoveNet wrist probe: move one hand into the drawer, pause/turn, then retract; r resets, s saves, q quits")
    try:
        while args.max_frames == 0 or frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame capture failed")
            started = time.perf_counter()
            keypoints = model.infer(frame)
            latency_ms = (time.perf_counter() - started) * 1000
            now = time.monotonic()
            for side, keypoint_index in WRISTS.items():
                event = detectors[side].add(now, keypoints[keypoint_index], args.confidence)
                if event is not None:
                    latest_event = event
                    print(f"turn side={event.side} x={event.x:.1f} y={event.y:.1f} angle={event.angle_degrees:.1f} confidence={event.confidence:.2f}")
            view = frame.copy()
            draw_pose(view, keypoints, args.confidence)
            draw_trail(view, detectors["left"], (255, 160, 0))
            draw_trail(view, detectors["right"], (0, 160, 255))
            if latest_event is not None:
                draw_event(view, latest_event, args.roi_radius)
            status = (
                f"MoveNet {latency_ms:.1f} ms | "
                f"L wrist={keypoints[9].confidence:.2f} R wrist={keypoints[10].confidence:.2f} | "
                "move in -> turn/pause -> retract"
            )
            cv2.rectangle(view, (0, 0), (view.shape[1], 34), (20, 20, 20), cv2.FILLED)
            cv2.putText(view, status, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            latest_view = view
            frames += 1
            if args.no_display:
                continue
            cv2.imshow(window_name, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                for detector in detectors.values():
                    detector.reset()
                latest_event = None
            if key == ord("s") and latest_view is not None:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                path = args.output_dir / f"wrist-turn-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(str(path), latest_view)
                print(f"saved {path}")
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
