#!/usr/bin/env python3
"""Use depth only to time A/B captures, then compare YOLOv8 inventories.

The operator manually requests Snapshot A.  Depth inference then watches the
full camera frame for motion and waits for the scene to become stable before
capturing Snapshot B.  After A is ready, YOLOv8m samples RGB frames in a
background worker until the first depth motion; objects present in at least
80% of those frames form the before multiset.  B uses several stable RGB
frames, and the multiset difference determines whether one object was put in
or taken out.

This is deliberately an isolated validation probe.  It does not mutate the
SmartDrawer database or depend on a hand-selected drawer ROI.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Sequence

import cv2
import numpy as np

from midas_change_crop_probe import (
    DEFAULT_DEPTH_ANYTHING_MODEL,
    DEFAULT_MIDAS_MODEL,
    DepthAnythingMetric,
    MidasTFLite,
    Snapshot,
    Thresholds,
    colorize_depth,
    has_motion,
    is_stable,
    make_snapshot,
    normalize_relative_depth,
    spatial_filter,
    stability_metrics,
)


DEFAULT_YOLO_MODEL = Path("models/yolov8m.pt")
FEEDBACK_HEIGHT = 44
HEADER_HEIGHT = 76
BUTTONS = {
    "capture_a": (12, 8, 154, 38),
    "reset": (178, 8, 110, 38),
    "save": (300, 8, 110, 38),
}
PHASE_FEEDBACK = {
    "ready_for_a": ("STEP 1 - Keep the scene still, then click Capture A", (0, 180, 255)),
    "capture_a": ("CAPTURING A - Keep hands out while stable frames are collected", (0, 180, 255)),
    "collect_before": ("SNAPSHOT A READY - Live YOLO sampling; put in OR take out one item", (44, 170, 44)),
    "capture_b": ("MOTION SEEN - Remove your hand; waiting for stable Snapshot B", (0, 180, 255)),
    "analyze_b": ("SNAPSHOT B CAPTURED - YOLOv8m is comparing inventories", (0, 180, 255)),
    "complete": ("COMPARISON COMPLETE - Review the result, then Save or Reset", (180, 100, 40)),
    "error": ("COMPARISON FAILED - Review the message, then Reset", (40, 50, 210)),
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    votes: int = 1


@dataclass(frozen=True)
class CapturedBatch:
    snapshot: Snapshot
    frames: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SnapshotAnalysis:
    batch: CapturedBatch
    detections: tuple[Detection, ...]
    annotated: np.ndarray
    inference_ms: float
    frames_analyzed: int

    @property
    def inventory(self) -> Counter[str]:
        return Counter(detection.name for detection in self.detections)


@dataclass(frozen=True)
class InventoryDelta:
    action: str
    item: Optional[str]
    added: dict[str, int]
    removed: dict[str, int]
    message: str


@dataclass
class _DetectionTrack:
    class_id: int
    name: str
    detections: list[Detection]
    last_frame: int


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    return intersection / max(first_area + second_area - intersection, 1e-9)


def consensus_detections(
    frames: Sequence[Sequence[Detection]],
    vote_ratio: float,
    match_iou: float,
) -> tuple[Detection, ...]:
    """Associate same-class boxes across stable frames and retain voted tracks."""
    if not frames:
        return ()
    # Tiny epsilon prevents floating-point roundoff turning an exact 80% into
    # one extra required frame (for example, 12.0000000002 -> 13).
    required_votes = max(1, math.ceil(len(frames) * vote_ratio - 1e-9))
    tracks: list[_DetectionTrack] = []
    for frame_index, detections in enumerate(frames):
        # Higher-confidence objects claim tracks first when duplicate classes exist.
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            candidates = [
                (bbox_iou(track.detections[-1].bbox, detection.bbox), track)
                for track in tracks
                if track.class_id == detection.class_id and track.last_frame != frame_index
            ]
            overlap, track = max(candidates, default=(0.0, None), key=lambda item: item[0])
            if track is None or overlap < match_iou:
                tracks.append(_DetectionTrack(detection.class_id, detection.name, [detection], frame_index))
            else:
                track.detections.append(detection)
                track.last_frame = frame_index

    consensus: list[Detection] = []
    for track in tracks:
        if len(track.detections) < required_votes:
            continue
        boxes = np.asarray([item.bbox for item in track.detections], dtype=np.float32)
        consensus.append(
            Detection(
                class_id=track.class_id,
                name=track.name,
                confidence=float(np.median([item.confidence for item in track.detections])),
                bbox=tuple(float(value) for value in np.median(boxes, axis=0)),
                votes=len(track.detections),
            )
        )
    return tuple(sorted(consensus, key=lambda item: (item.class_id, item.bbox[0], item.bbox[1])))


def compare_inventories(before: Counter[str], after: Counter[str]) -> InventoryDelta:
    added_counter = after - before
    removed_counter = before - after
    added = dict(sorted(added_counter.items()))
    removed = dict(sorted(removed_counter.items()))
    added_count = sum(added.values())
    removed_count = sum(removed.values())
    if added_count == 1 and removed_count == 0:
        item = next(iter(added))
        return InventoryDelta("put_in", item, added, removed, f"PUT IN: {item}")
    if removed_count == 1 and added_count == 0:
        item = next(iter(removed))
        return InventoryDelta("take_out", item, added, removed, f"TAKE OUT: {item}")
    if added_count == 0 and removed_count == 0:
        return InventoryDelta("no_change", None, added, removed, "NO RELIABLE OBJECT CHANGE")
    return InventoryDelta(
        "ambiguous",
        None,
        added,
        removed,
        f"AMBIGUOUS: added={format_inventory(added_counter)} removed={format_inventory(removed_counter)}",
    )


def format_inventory(inventory: Counter[str] | dict[str, int]) -> str:
    if not inventory:
        return "(none)"
    return ", ".join(f"{name} x{count}" for name, count in sorted(inventory.items()))


def select_evenly(frames: Sequence[np.ndarray], count: int) -> list[np.ndarray]:
    if count >= len(frames):
        return [frame.copy() for frame in frames]
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int)
    return [frames[int(index)].copy() for index in indices]


def detection_color(class_id: int) -> tuple[int, int, int]:
    class_id = int(class_id)
    return (
        int((37 * class_id + 80) % 220 + 35),
        int((17 * class_id + 130) % 220 + 35),
        int((29 * class_id + 40) % 220 + 35),
    )


def annotate(frame: np.ndarray, detections: Sequence[Detection], total_frames: int) -> np.ndarray:
    result = frame.copy()
    for detection in detections:
        x0, y0, x1, y1 = (int(round(value)) for value in detection.bbox)
        colour = detection_color(detection.class_id)
        cv2.rectangle(result, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
        label = f"{detection.name} {detection.confidence:.2f} [{detection.votes}/{total_frames}]"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_top = max(0, y0 - label_height - 8)
        cv2.rectangle(result, (x0, label_top), (x0 + label_width + 6, label_top + label_height + 7), colour, cv2.FILLED)
        cv2.putText(result, label, (x0 + 3, label_top + label_height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)
    return result


class YoloV8Detector:
    def __init__(self, model_path: Path, confidence: float, nms_iou: float, image_size: int, device: str) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.image_size = image_size
        self.device = device
        self.name = model_path.name

    def detect(self, frames: Sequence[np.ndarray]) -> tuple[list[list[Detection]], float]:
        if not frames:
            raise ValueError("YOLO needs at least one RGB frame")
        started = time.perf_counter()
        results = self.model.predict(
            source=list(frames),
            conf=self.confidence,
            iou=self.nms_iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        all_detections: list[list[Detection]] = []
        for result in results:
            frame_detections: list[Detection] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    names = result.names
                    name = str(names[class_id] if isinstance(names, dict) else names[class_id])
                    frame_detections.append(
                        Detection(int(class_id), name, float(confidence), tuple(float(value) for value in box))
                    )
            all_detections.append(frame_detections)
        return all_detections, elapsed_ms


class YoloABProbe:
    """Manual A, live pre-motion YOLO consensus, stable B, inventory diff."""

    def __init__(
        self,
        stable_frames: int,
        thresholds: Thresholds,
        yolo_vote_frames: int,
        before_vote_ratio: float,
        after_vote_ratio: float,
        match_iou: float,
    ) -> None:
        if stable_frames < 3:
            raise ValueError("stable_frames must be at least 3")
        if not 1 <= yolo_vote_frames <= stable_frames:
            raise ValueError("yolo_vote_frames must be between 1 and stable_frames")
        if not 0 < before_vote_ratio <= 1 or not 0 < after_vote_ratio <= 1 or not 0 <= match_iou <= 1:
            raise ValueError("vote ratios must be in (0, 1] and match_iou in [0, 1]")
        self.required = stable_frames
        self.thresholds = thresholds
        self.yolo_vote_frames = yolo_vote_frames
        self.before_vote_ratio = before_vote_ratio
        self.after_vote_ratio = after_vote_ratio
        self.match_iou = match_iou
        self.capture_id = 0
        self.phase = "ready_for_a"
        self.message = "Click Capture A while the full scene is still."
        self.before_batch: Optional[CapturedBatch] = None
        self.after_batch: Optional[CapturedBatch] = None
        self.before: Optional[SnapshotAnalysis] = None
        self.after: Optional[SnapshotAnalysis] = None
        self.delta: Optional[InventoryDelta] = None
        self.before_observations: list[list[Detection]] = []
        self.before_inference_ms = 0.0
        self.frames: Deque[np.ndarray] = deque(maxlen=stable_frames)
        self.depths: Deque[np.ndarray] = deque(maxlen=stable_frames)
        self.previous: Optional[np.ndarray] = None
        self.noise_medians: Deque[float] = deque(maxlen=thresholds.noise_warmup_frames)
        self.noise_pixel_p95: Deque[float] = deque(maxlen=thresholds.noise_warmup_frames)
        self.last_median_delta = 0.0
        self.last_ratio = 0.0
        self.effective_median_limit = thresholds.stable_median
        self.effective_pixel_limit = thresholds.stable_median

    def reset(self) -> None:
        self.capture_id += 1
        self.phase = "ready_for_a"
        self.message = "Click Capture A while the full scene is still."
        self.before_batch = self.after_batch = None
        self.before = self.after = None
        self.delta = None
        self.before_observations.clear()
        self.before_inference_ms = 0.0
        self.frames.clear()
        self.depths.clear()
        self.noise_medians.clear()
        self.noise_pixel_p95.clear()
        self.last_median_delta = self.last_ratio = 0.0
        self.effective_median_limit = self.thresholds.stable_median
        self.effective_pixel_limit = self.thresholds.stable_median

    def request_before(self) -> None:
        if self.phase in {"ready_for_a", "complete", "error"}:
            self.reset()
            self.phase = "capture_a"
            self.message = f"Capturing stable A: 0/{self.required}. Keep hands out."

    def _observe_noise(self, depth: np.ndarray) -> None:
        if self.previous is None:
            return
        difference = np.abs(depth - self.previous)
        self.noise_medians.append(float(np.median(difference)))
        self.noise_pixel_p95.append(float(np.percentile(difference, 95)))
        if len(self.noise_medians) >= self.thresholds.noise_warmup_frames:
            self.effective_median_limit = max(
                self.thresholds.stable_median,
                float(np.percentile(self.noise_medians, 90)) * self.thresholds.noise_multiplier,
            )
            self.effective_pixel_limit = max(
                self.thresholds.stable_median,
                float(np.percentile(self.noise_pixel_p95, 90)) * self.thresholds.noise_multiplier,
            )

    def _capture_if_stable(self, frame: np.ndarray, depth: np.ndarray) -> Optional[CapturedBatch]:
        if self.phase == "capture_a":
            self._observe_noise(depth)
        if self.previous is None:
            self.frames.clear()
            self.depths.clear()
            return None
        self.last_median_delta, self.last_ratio = stability_metrics(depth, self.previous, self.effective_pixel_limit)
        if not is_stable(
            depth,
            self.previous,
            self.thresholds,
            median_limit=self.effective_median_limit,
            pixel_limit=self.effective_pixel_limit,
        ):
            self.frames.clear()
            self.depths.clear()
            return None
        self.frames.append(frame.copy())
        self.depths.append(depth.copy())
        label = "A" if self.phase == "capture_a" else "B"
        self.message = f"Capturing stable {label}: {len(self.depths)}/{self.required}."
        if len(self.depths) != self.required:
            return None
        frames = tuple(self.frames)
        snapshot = make_snapshot(list(frames), list(self.depths))
        self.frames.clear()
        self.depths.clear()
        return CapturedBatch(snapshot, frames)

    def process(self, frame: np.ndarray, depth: np.ndarray) -> None:
        if self.phase == "capture_a":
            batch = self._capture_if_stable(frame, depth)
            if batch is not None:
                self.before_batch = batch
                self.phase = "collect_before"
                self.message = "Snapshot A ready. Starting live YOLO before-set sampling..."
        elif self.phase == "collect_before" and self.previous is not None:
            motion_threshold = max(self.thresholds.motion_median, self.effective_pixel_limit * 2.5)
            motion_thresholds = Thresholds(
                stable_median=self.thresholds.stable_median,
                stable_ratio=self.thresholds.stable_ratio,
                motion_median=motion_threshold,
                motion_ratio=self.thresholds.motion_ratio,
                change_depth=self.thresholds.change_depth,
                min_change_area=self.thresholds.min_change_area,
                morphology_size=self.thresholds.morphology_size,
                padding=self.thresholds.padding,
                noise_multiplier=self.thresholds.noise_multiplier,
                noise_warmup_frames=self.thresholds.noise_warmup_frames,
            )
            if has_motion(depth, self.previous, motion_thresholds):
                self.phase = "capture_b"
                self.frames.clear()
                self.depths.clear()
                self.message = (
                    f"Motion seen after {len(self.before_observations)} YOLO frames. "
                    f"Remove your hand; waiting for stable B (0/{self.required})."
                )
        elif self.phase == "capture_b":
            batch = self._capture_if_stable(frame, depth)
            if batch is not None:
                self.after_batch = batch
                self.phase = "analyze_b"
                self.message = "Snapshot B captured. Running YOLOv8m consensus..."
        self.previous = depth.copy()

    def observe_before(self, detector: YoloV8Detector, frame: np.ndarray) -> None:
        """Add one pre-motion YOLO observation and refresh the provisional set."""
        if self.phase != "collect_before":
            return
        frame_detections, inference_ms = detector.detect([frame])
        self.record_before_observation(frame_detections[0], inference_ms)

    def record_before_observation(self, detections: list[Detection], inference_ms: float) -> None:
        """Accept one completed worker result captured before depth motion."""
        if self.phase not in {"collect_before", "capture_b"} or self.before_batch is None:
            return
        self.before_observations.append(detections)
        self.before_inference_ms += inference_ms
        self._refresh_before_analysis()
        assert self.before is not None
        if self.phase == "collect_before":
            self.message = (
                f"Snapshot A ready: live YOLO {len(self.before_observations)} frames, "
                f">={self.before_vote_ratio:.0%} set: {format_inventory(self.before.inventory)}. "
                "Put in or take out ONE item."
            )

    def _refresh_before_analysis(self) -> None:
        if not self.before_observations:
            return
        assert self.before_batch is not None
        detections = consensus_detections(
            self.before_observations,
            self.before_vote_ratio,
            self.match_iou,
        )
        self.before = SnapshotAnalysis(
            batch=self.before_batch,
            detections=detections,
            annotated=annotate(self.before_batch.snapshot.frame, detections, len(self.before_observations)),
            inference_ms=self.before_inference_ms,
            frames_analyzed=len(self.before_observations),
        )

    def finalize_before(self) -> None:
        """Freeze the 80%-presence before set when depth motion begins."""
        if self.before_observations:
            self._refresh_before_analysis()
            return
        self.phase = "error"
        self.message = "Depth motion started before any YOLO before frame completed. Reset and wait for YOLO sampling."

    def analyze_pending(self, detector: YoloV8Detector) -> None:
        if self.phase != "analyze_b":
            return
        batch = self.after_batch
        assert batch is not None
        selected = select_evenly(batch.frames, self.yolo_vote_frames)
        frame_detections, inference_ms = detector.detect(selected)
        detections = consensus_detections(frame_detections, self.after_vote_ratio, self.match_iou)
        analysis = SnapshotAnalysis(
            batch=batch,
            detections=detections,
            annotated=annotate(batch.snapshot.frame, detections, len(selected)),
            inference_ms=inference_ms,
            frames_analyzed=len(selected),
        )
        self.after = analysis
        assert self.before is not None
        self.delta = compare_inventories(self.before.inventory, analysis.inventory)
        self.phase = "complete"
        self.message = self.delta.message


def inside(point: tuple[int, int], rectangle: tuple[int, int, int, int]) -> bool:
    x, y = point
    left, top, width, height = rectangle
    return left <= x < left + width and top <= y < top + height


def draw_button(image: np.ndarray, label: str, rectangle: tuple[int, int, int, int], enabled: bool = True) -> None:
    x, y, width, height = rectangle
    cv2.rectangle(image, (x, y), (x + width, y + height), (54, 128, 54) if enabled else (70, 70, 70), cv2.FILLED)
    cv2.putText(image, label, (x + 9, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def label_row(left: str, right: str, width: int) -> np.ndarray:
    row = np.zeros((24, width * 2, 3), dtype=np.uint8)
    cv2.putText(row, left, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(row, right, (width + 12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return row


def render(frame: np.ndarray, depth: np.ndarray, probe: YoloABProbe, depth_ms: float) -> np.ndarray:
    height, width = frame.shape[:2]
    instruction, colour = PHASE_FEEDBACK[probe.phase]
    feedback = np.zeros((FEEDBACK_HEIGHT, width * 2, 3), dtype=np.uint8)
    cv2.rectangle(feedback, (0, 0), (width * 2, FEEDBACK_HEIGHT), colour, cv2.FILLED)
    cv2.putText(feedback, instruction, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (15, 15, 15), 2, cv2.LINE_AA)

    header = np.zeros((HEADER_HEIGHT, width * 2, 3), dtype=np.uint8)
    for name, rectangle in BUTTONS.items():
        enabled = name != "save" or probe.delta is not None
        draw_button(header, {"capture_a": "Capture A", "reset": "Reset", "save": "Save"}[name], rectangle, enabled)
    cv2.putText(header, probe.message[:125], (430, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        header,
        f"state={probe.phase}  depth={depth_ms:.0f}ms  depth delta={probe.last_median_delta:.4f}/{probe.effective_median_limit:.4f}  noisy={probe.last_ratio:.1%}/{probe.thresholds.stable_ratio:.1%}",
        (12, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    live_depth = fit_panel(colorize_depth(depth), width, height)
    before_view = np.zeros_like(frame) if probe.before is None else fit_panel(probe.before.annotated, width, height)
    if probe.after is not None:
        after_view = fit_panel(probe.after.annotated, width, height)
    elif probe.after_batch is not None:
        after_view = fit_panel(probe.after_batch.snapshot.frame, width, height)
    else:
        after_view = np.zeros_like(frame)

    footer = np.zeros((104, width * 2, 3), dtype=np.uint8)
    before_inventory = Counter() if probe.before is None else probe.before.inventory
    after_inventory = Counter() if probe.after is None else probe.after.inventory
    lines = [
        f"A inventory: {format_inventory(before_inventory)}",
        f"B inventory: {format_inventory(after_inventory)}",
        "Result: pending" if probe.delta is None else f"Result: {probe.delta.message}",
    ]
    if probe.before is not None:
        lines[0] += f"  ({probe.before.inference_ms:.0f} ms / {probe.before.frames_analyzed} frames)"
    if probe.after is not None:
        lines[1] += f"  ({probe.after.inference_ms:.0f} ms / {probe.after.frames_analyzed} frames)"
    for index, line in enumerate(lines):
        cv2.putText(footer, line[:180], (12, 27 + index * 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (225, 225, 225), 1, cv2.LINE_AA)

    return np.vstack(
        (
            feedback,
            header,
            label_row("Live RGB", "Live depth (capture timing only)", width),
            np.hstack((frame, live_depth)),
            label_row("Snapshot A - YOLOv8m consensus", "Snapshot B - YOLOv8m consensus", width),
            np.hstack((before_view, after_view)),
            footer,
        )
    )


def detection_dict(detection: Detection) -> dict[str, object]:
    data = asdict(detection)
    data["bbox"] = list(detection.bbox)
    return data


def save_result(output_dir: Path, probe: YoloABProbe) -> Optional[Path]:
    if probe.before is None or probe.after is None or probe.delta is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir()
    cv2.imwrite(str(run_dir / "rgb_a.png"), probe.before.batch.snapshot.frame)
    cv2.imwrite(str(run_dir / "rgb_b.png"), probe.after.batch.snapshot.frame)
    cv2.imwrite(str(run_dir / "yolo_a.png"), probe.before.annotated)
    cv2.imwrite(str(run_dir / "yolo_b.png"), probe.after.annotated)
    cv2.imwrite(str(run_dir / "depth_a.png"), colorize_depth(probe.before.batch.snapshot.depth))
    cv2.imwrite(str(run_dir / "depth_b.png"), colorize_depth(probe.after.batch.snapshot.depth))
    np.save(run_dir / "depth_a.npy", probe.before.batch.snapshot.depth)
    np.save(run_dir / "depth_b.npy", probe.after.batch.snapshot.depth)
    payload = {
        "snapshot_a": {
            "inventory": dict(sorted(probe.before.inventory.items())),
            "detections": [detection_dict(item) for item in probe.before.detections],
            "inference_ms": probe.before.inference_ms,
            "frames_analyzed": probe.before.frames_analyzed,
        },
        "snapshot_b": {
            "inventory": dict(sorted(probe.after.inventory.items())),
            "detections": [detection_dict(item) for item in probe.after.detections],
            "inference_ms": probe.after.inference_ms,
            "frames_analyzed": probe.after.frames_analyzed,
        },
        "difference": asdict(probe.delta),
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run_dir


def self_test() -> None:
    def item(class_id: int, name: str, box: tuple[float, float, float, float], confidence: float = 0.9) -> Detection:
        return Detection(class_id, name, confidence, box)

    # Ultralytics class IDs originate as NumPy scalars.  OpenCV 5 rejects
    # NumPy integers in the rectangle colour tuple unless they are normalized.
    annotated = annotate(
        np.zeros((40, 60, 3), dtype=np.uint8),
        [Detection(np.int64(41), "cup", 0.9, (5.0, 5.0, 25.0, 30.0))],  # type: ignore[arg-type]
        1,
    )
    assert annotated.shape == (40, 60, 3) and annotated.any()

    frames = [
        [item(41, "cup", (10, 10, 30, 30)), item(39, "bottle", (60, 10, 80, 50))],
        [item(41, "cup", (11, 10, 31, 30)), item(39, "bottle", (61, 10, 81, 50))],
        [item(41, "cup", (10, 11, 30, 31)), item(0, "person", (0, 0, 5, 5))],
    ]
    consensus = consensus_detections(frames, vote_ratio=2 / 3, match_iou=0.4)
    assert Counter(result.name for result in consensus) == Counter({"cup": 1, "bottle": 1})
    assert all(result.votes >= 2 for result in consensus)
    eighty_percent_frames = [
        [item(41, "cup", (10, 10, 30, 30)), item(39, "bottle", (60, 10, 80, 50))],
        [item(41, "cup", (10, 10, 30, 30)), item(39, "bottle", (60, 10, 80, 50))],
        [item(41, "cup", (10, 10, 30, 30)), item(39, "bottle", (60, 10, 80, 50))],
        [item(41, "cup", (10, 10, 30, 30))],
        [],
    ]
    eighty_percent = consensus_detections(eighty_percent_frames, vote_ratio=0.80, match_iou=0.4)
    assert Counter(result.name for result in eighty_percent) == Counter({"cup": 1})
    assert eighty_percent[0].votes == 4
    assert compare_inventories(Counter({"cup": 1}), Counter({"cup": 2})).action == "put_in"
    removed = compare_inventories(Counter({"cup": 1, "bottle": 1}), Counter({"cup": 1}))
    assert removed.action == "take_out" and removed.item == "bottle"
    assert compare_inventories(Counter({"cup": 1}), Counter({"cup": 1})).action == "no_change"
    assert compare_inventories(Counter({"cup": 1}), Counter({"bottle": 1})).action == "ambiguous"

    thresholds = Thresholds()
    probe = YoloABProbe(3, thresholds, 3, 0.80, 2 / 3, 0.4)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    depth_a = np.full((16, 16), 0.5, dtype=np.float32)
    depth_b = depth_a.copy()
    depth_b[4:12, 4:12] += 0.20
    probe.process(frame, depth_a)  # live preview establishes the predecessor
    probe.request_before()
    for _ in range(3):
        probe.process(frame, depth_a)
    assert probe.phase == "collect_before" and probe.before_batch is not None

    class FakeDetector:
        def __init__(self) -> None:
            self.after_mode = False

        def detect(self, selected: Sequence[np.ndarray]) -> tuple[list[list[Detection]], float]:
            objects = [item(41, "cup", (10, 10, 30, 30))]
            if self.after_mode:
                objects.append(item(39, "bottle", (35, 10, 50, 35)))
            return [objects.copy() for _ in selected], 1.0

    fake_detector = FakeDetector()
    for _ in range(5):
        probe.observe_before(fake_detector, frame)  # type: ignore[arg-type]
    assert probe.phase == "collect_before" and probe.before is not None
    assert probe.before.frames_analyzed == 5 and probe.before.inventory == Counter({"cup": 1})
    probe.process(frame, depth_b)
    assert probe.phase == "capture_b"
    probe.finalize_before()
    for _ in range(3):
        probe.process(frame, depth_b)
    assert probe.phase == "analyze_b" and probe.after_batch is not None
    fake_detector.after_mode = True
    probe.analyze_pending(fake_detector)  # type: ignore[arg-type]
    assert probe.phase == "complete" and probe.delta is not None
    assert probe.delta.action == "put_in" and probe.delta.item == "bottle"
    print("yolo_ab_inventory_probe: self-test OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth-timed A/B capture with YOLOv8m inventory difference")
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-confidence", type=float, default=0.35)
    parser.add_argument("--yolo-nms-iou", type=float, default=0.70)
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--yolo-device", default="cpu", help="Ultralytics device, e.g. cpu or mps")
    parser.add_argument("--yolo-vote-frames", type=int, default=3)
    parser.add_argument(
        "--before-vote-ratio",
        type=float,
        default=0.80,
        help="minimum fraction of live pre-motion frames required in the before set",
    )
    parser.add_argument("--yolo-vote-ratio", type=float, default=2 / 3, help="stable Snapshot B vote ratio")
    parser.add_argument("--yolo-match-iou", type=float, default=0.40)
    parser.add_argument(
        "--depth-model",
        choices=("midas", "depth-anything-metric"),
        default="depth-anything-metric",
    )
    parser.add_argument("--depth-model-path", type=Path, help="override the selected depth checkpoint")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--avfoundation", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until q/Esc")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--stable-median", type=float, default=Thresholds.stable_median)
    parser.add_argument("--stable-ratio", type=float, default=Thresholds.stable_ratio)
    parser.add_argument("--motion-median", type=float, default=Thresholds.motion_median)
    parser.add_argument("--motion-ratio", type=float, default=Thresholds.motion_ratio)
    parser.add_argument("--noise-multiplier", type=float, default=Thresholds.noise_multiplier)
    parser.add_argument("--noise-warmup-frames", type=int, default=Thresholds.noise_warmup_frames)
    parser.add_argument("--bilateral-diameter", type=int, default=5)
    parser.add_argument("--bilateral-sigma", type=float, default=0.08)
    parser.add_argument("--output-dir", type=Path, default=Path("captures/yolo-ab-probe"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def make_depth_model(args: argparse.Namespace) -> MidasTFLite | DepthAnythingMetric:
    if args.depth_model == "midas":
        return MidasTFLite(args.depth_model_path or DEFAULT_MIDAS_MODEL)
    return DepthAnythingMetric(args.depth_model_path or DEFAULT_DEPTH_ANYTHING_MODEL)


def run(args: argparse.Namespace) -> None:
    if not 0 < args.yolo_confidence <= 1 or not 0 < args.yolo_nms_iou <= 1:
        raise ValueError("YOLO confidence and NMS IoU must be in (0, 1]")
    if args.noise_warmup_frames < 1 or args.noise_multiplier < 1:
        raise ValueError("noise warmup must be positive and multiplier at least 1")
    depth_model = make_depth_model(args)
    detector = YoloV8Detector(
        args.yolo_model,
        args.yolo_confidence,
        args.yolo_nms_iou,
        args.yolo_image_size,
        args.yolo_device,
    )
    thresholds = Thresholds(
        stable_median=args.stable_median,
        stable_ratio=args.stable_ratio,
        motion_median=args.motion_median,
        motion_ratio=args.motion_ratio,
        noise_multiplier=args.noise_multiplier,
        noise_warmup_frames=args.noise_warmup_frames,
    )
    probe = YoloABProbe(
        args.stable_frames,
        thresholds,
        args.yolo_vote_frames,
        args.before_vote_ratio,
        args.yolo_vote_ratio,
        args.yolo_match_iou,
    )

    backend = cv2.CAP_AVFOUNDATION if args.avfoundation else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera}")

    window = "Depth-timed A/B inventory - YOLOv8m"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def save() -> None:
        saved = save_result(args.output_dir, probe)
        if saved is not None:
            probe.message = f"Saved: {saved}"

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = (x, y - FEEDBACK_HEIGHT)
        if inside(point, BUTTONS["capture_a"]):
            probe.request_before()
        elif inside(point, BUTTONS["reset"]):
            probe.reset()
        elif inside(point, BUTTONS["save"]):
            save()

    if not args.no_display:
        cv2.setMouseCallback(window, mouse)
    print(
        f"depth={depth_model.name} input={depth_model.width}x{depth_model.height}; "
        f"detector={detector.name} device={args.yolo_device}; click Capture A, then move exactly one item"
    )
    frames_processed = 0
    before_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="before-yolo")
    before_future: Optional[Future[tuple[int, list[list[Detection]], float]]] = None

    def detect_before(frame: np.ndarray, capture_id: int) -> tuple[int, list[list[Detection]], float]:
        frame_detections, inference_ms = detector.detect([frame])
        return capture_id, frame_detections, inference_ms

    def accept_before_result(*, wait: bool) -> None:
        nonlocal before_future
        if before_future is None or (not wait and not before_future.done()):
            return
        capture_id, frame_detections, inference_ms = before_future.result()
        if capture_id == probe.capture_id:
            probe.record_before_observation(frame_detections[0], inference_ms)
        before_future = None

    try:
        while args.max_frames == 0 or frames_processed < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame capture failed")
            try:
                accept_before_result(wait=False)
            except Exception as error:
                probe.phase = "error"
                probe.message = f"Live YOLO sampling failed: {error}"
            started = time.perf_counter()
            depth = depth_model.infer(frame)
            if depth_model.relative_depth:
                depth = normalize_relative_depth(depth)
            depth = spatial_filter(depth, args.bilateral_diameter, args.bilateral_sigma)
            depth_ms = (time.perf_counter() - started) * 1000.0
            phase_before_depth = probe.phase
            probe.process(frame, depth)
            if phase_before_depth == "collect_before" and probe.phase == "capture_b":
                try:
                    # This future owns an earlier, pre-motion frame.  Finish
                    # it before freezing the set, but never submit this first
                    # motion frame to YOLO.
                    accept_before_result(wait=True)
                    probe.finalize_before()
                except Exception as error:
                    probe.phase = "error"
                    probe.message = f"Could not finalize the before set: {error}"
            frames_processed += 1

            key = -1
            if not args.no_display:
                cv2.imshow(window, render(frame, depth, probe, depth_ms))
                key = cv2.waitKey(1) & 0xFF
            try:
                if probe.phase == "collect_before":
                    # Keep at most one YOLO job in flight.  Depth therefore
                    # continues polling without waiting for CPU inference.
                    if before_future is None:
                        before_future = before_executor.submit(detect_before, frame.copy(), probe.capture_id)
                elif probe.phase == "analyze_b":
                    probe.analyze_pending(detector)
            except Exception as error:
                probe.phase = "error"
                probe.message = f"YOLO analysis failed: {error}"
            if args.no_display:
                continue
            if key in (27, ord("q")):
                break
            if key == ord("a"):
                probe.request_before()
            elif key == ord("r"):
                probe.reset()
            elif key == ord("s"):
                save()
    finally:
        before_executor.shutdown(wait=True, cancel_futures=True)
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
