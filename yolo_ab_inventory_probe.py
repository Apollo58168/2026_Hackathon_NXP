#!/usr/bin/env python3
"""Auto-open/close drawer probe using depth timing and COCO detection.

A large depth motion means the drawer opened. After stable Snapshot A, item
motion triggers stable Snapshot B and a YOLOv8m inventory comparison. The
confirmed PUT/TAKE is saved, then the next motion closes and resets the drawer.

Confirmed PUT/TAKE results are persisted in the probe SQLite inventory by
VL53L0X layer; no hand-selected drawer ROI is required.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Sequence

import cv2
import numpy as np

# The board probe lives beside the shared detector adapter in /root; the repo
# probe lives in 2026_Hackathon_NXP beside it.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from vl53l0x_layers import DistanceSensor, LayerProfiles, make_profiles

from midas_change_crop_probe import (
    DEFAULT_DEPTH_ANYTHING_MODEL,
    DEFAULT_MIDAS_MODEL,
    DepthAnythingMetric,
    MidasTFLite,
    Snapshot,
    Thresholds,
    colorize_depth,
    is_stable,
    make_snapshot,
    normalize_relative_depth,
    spatial_filter,
    stability_metrics,
)


def first_existing(*paths: Path) -> Path:
    return next((path for path in paths if path.is_file()), paths[0])


DEFAULT_DETECTOR_MODEL = first_existing(
    Path("/opt/gopoint-tui/downloads/yolov8m_640_int8_vela5.tflite"),
    SCRIPT_DIR / "yolov8m_640_int8_vela5.tflite",
)
DEFAULT_DETECTOR_LABELS = first_existing(
    Path("/opt/gopoint-apps/downloads/coco_labels_list.txt"),
    SCRIPT_DIR / "models/coco_labels_list.txt",
    SCRIPT_DIR.parent / "models/coco_labels_list.txt",
)
DEFAULT_DETECTOR_PRIORS = first_existing(
    Path("/opt/gopoint-apps/downloads/box_priors.txt"),
    SCRIPT_DIR / "models/box_priors.txt",
    SCRIPT_DIR.parent / "models/box_priors.txt",
)
DEFAULT_ENABLED_CONFIG = first_existing(
    Path("/root/enabled_classes.json"),
    SCRIPT_DIR / "config/enabled_classes.json",
    SCRIPT_DIR.parent / "config/enabled_classes.json",
)
FEEDBACK_HEIGHT = 44
HEADER_HEIGHT = 76
BUTTONS = {
    "capture_a": (12, 8, 154, 38),
    "reset": (178, 8, 110, 38),
    "save": (300, 8, 110, 38),
}
PHASE_FEEDBACK = {
    "ready_for_a": ("WAITING - Open the drawer to auto-capture A", (0, 180, 255)),
    "capture_a": ("DRAWER OPENED - Waiting for stable Snapshot A", (0, 180, 255)),
    "collect_before": ("SNAPSHOT A READY - Move one item", (44, 170, 44)),
    "capture_b": ("ITEM CHANGE SEEN - Waiting for stable Snapshot B", (0, 180, 255)),
    "analyze_b": ("SNAPSHOT B READY - YOLOv8m deciding PUT or TAKE", (0, 180, 255)),
    "wait_close": ("RESULT SAVED - Close the drawer", (44, 170, 44)),
    "closing": ("DRAWER CLOSING - Waiting for stable close", (0, 180, 255)),
    "complete": ("READY FOR NEXT OPEN", (180, 100, 40)),
    "error": ("PROBE FAILED - Review the message, then Reset", (40, 50, 210)),
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


class CocoDetector:
    """Use YOLOv8m TFLite on RGB camera frames."""

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        priors_path: Path,
        confidence: float,
        nms_iou: float,
        enabled_labels: Sequence[str],
        delegate_path: Optional[str],
    ) -> None:
        from yolov8_tflite_detector import YoloV8TFLiteDetector

        delegate = delegate_path if delegate_path and Path(delegate_path).is_file() else None
        self.model = YoloV8TFLiteDetector(
            str(model_path),
            delegate_path=delegate,
            confidence=confidence,
            nms_iou=nms_iou,
            enabled_labels=enabled_labels,
        )
        self.name = model_path.name
        self.backend = self.model.backend
        self.nms_iou = nms_iou

    def diagnostic_text(self) -> str:
        input_shape = tuple(int(value) for value in self.model.input["shape"])
        output_shape = tuple(int(value) for value in self.model.output["shape"])
        quantization = (
            "affine INT8" if np.issubdtype(self.model.input["dtype"], np.integer) else "Float32"
        )
        return (
            f"input={input_shape} {self.model.input['dtype']} q={self.model.input['quantization']}; "
            f"output={output_shape} {self.model.output['dtype']} q={self.model.output['quantization']}; "
            f"preprocess=RGB letterbox + one /255 + {quantization}; "
            f"layout={self.model.input_layout}; head=xywh + 80 scores"
        )

    def detect(self, frames: Sequence[np.ndarray]) -> tuple[list[list[Detection]], float]:
        if not frames:
            raise ValueError("COCO detector needs at least one RGB frame")
        started = time.perf_counter()
        all_detections: list[list[Detection]] = []
        for frame in frames:
            # OpenCV capture frames are BGR; the shared object detector
            # contract expects RGB camera bytes.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections: list[Detection] = []
            for item in self.model.detect(rgb):
                x, y, width, height = item.box
                class_id, label = canonical_inventory_class(int(item.class_id), str(item.label))
                detections.append(
                    Detection(
                        class_id,
                        label,
                        float(item.confidence),
                        (float(x), float(y), float(x + width), float(y + height)),
                    )
                )
            all_detections.append(deduplicate_canonical_detections(detections, self.nms_iou))
        return all_detections, (time.perf_counter() - started) * 1000.0


def canonical_inventory_class(class_id: int, label: str) -> tuple[int, str]:
    return (65, "remote") if label == "keyboard" else (class_id, label)


def deduplicate_canonical_detections(detections: Sequence[Detection], nms_iou: float) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if not any(
            detection.class_id == previous.class_id
            and bbox_iou(detection.bbox, previous.bbox) >= nms_iou
            for previous in kept
        ):
            kept.append(detection)
    return kept


def load_enabled_labels(path: Path) -> frozenset[str]:
    from ssdlite_detector import DEFAULT_ENABLED_LABELS

    if not path.is_file():
        return DEFAULT_ENABLED_LABELS
    data = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(item["canonical_label"] for item in data["classes"])


def selected_labels(args: argparse.Namespace) -> frozenset[str]:
    if args.all_classes:
        from ssdlite_detector import COCO_LABELS

        return frozenset(COCO_LABELS)
    return load_enabled_labels(args.enabled_config)


class CocoABProbe:
    """Automatic four-stage A/B inventory transaction state."""

    def __init__(
        self,
        stable_frames: int,
        thresholds: Thresholds,
        detector_vote_frames: int,
        before_vote_ratio: float,
        after_vote_ratio: float,
        match_iou: float,
        layer_profiles: Optional[LayerProfiles] = None,
    ) -> None:
        if stable_frames < 3:
            raise ValueError("stable_frames must be at least 3")
        if not 1 <= detector_vote_frames <= stable_frames:
            raise ValueError("detector_vote_frames must be between 1 and stable_frames")
        if not 0 < before_vote_ratio <= 1 or not 0 < after_vote_ratio <= 1 or not 0 <= match_iou <= 1:
            raise ValueError("vote ratios must be in (0, 1] and match_iou in [0, 1]")
        self.required = stable_frames
        self.thresholds = thresholds
        self.detector_vote_frames = detector_vote_frames
        self.before_vote_ratio = before_vote_ratio
        self.after_vote_ratio = after_vote_ratio
        self.match_iou = match_iou
        self.layer_profiles = layer_profiles
        self.capture_id = 0
        self.phase = "ready_for_a"
        self.message = "Open the drawer; stable Snapshot A will be captured automatically."
        self.before_batch: Optional[CapturedBatch] = None
        self.after_batch: Optional[CapturedBatch] = None
        self.before: Optional[SnapshotAnalysis] = None
        self.after: Optional[SnapshotAnalysis] = None
        self.delta: Optional[InventoryDelta] = None
        self.before_observations: list[list[Detection]] = []
        self.before_inference_ms = 0.0
        self.distance_observations: list[float] = []
        self.last_distance_mm: Optional[float] = None
        self.active_layer: Optional[int] = None
        self.frames: Deque[np.ndarray] = deque(maxlen=stable_frames)
        self.depths: Deque[np.ndarray] = deque(maxlen=stable_frames)
        self.previous: Optional[np.ndarray] = None
        self.noise_medians: Deque[float] = deque(maxlen=thresholds.noise_warmup_frames)
        self.noise_pixel_p95: Deque[float] = deque(maxlen=thresholds.noise_warmup_frames)
        self.last_median_delta = 0.0
        self.last_ratio = 0.0
        self.last_motion_median = 0.0
        self.last_motion_ratio = 0.0
        self.effective_median_limit = thresholds.stable_median
        self.effective_pixel_limit = thresholds.stable_median

    def reset(self) -> None:
        self.capture_id += 1
        self.phase = "ready_for_a"
        self.message = "Open the drawer; waiting for the next stable Snapshot A."
        self.before_batch = self.after_batch = None
        self.before = self.after = None
        self.delta = None
        self.before_observations.clear()
        self.before_inference_ms = 0.0
        self.distance_observations.clear()
        self.last_distance_mm = None
        self.active_layer = None
        self.frames.clear()
        self.depths.clear()
        self.noise_medians.clear()
        self.noise_pixel_p95.clear()
        self.last_median_delta = self.last_ratio = 0.0
        self.last_motion_median = self.last_motion_ratio = 0.0
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

    @property
    def effective_motion_limit(self) -> float:
        return self.thresholds.motion_median

    def _motion_detected(self) -> bool:
        return (
            self.last_motion_median > self.effective_motion_limit
            or self.last_motion_ratio >= self.thresholds.motion_ratio
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
        label = {"capture_a": "A", "capture_b": "B", "closing": "close"}[self.phase]
        self.message = f"Capturing stable {label}: {len(self.depths)}/{self.required}."
        if len(self.depths) != self.required:
            return None
        frames = tuple(self.frames)
        snapshot = make_snapshot(list(frames), list(self.depths))
        self.frames.clear()
        self.depths.clear()
        return CapturedBatch(snapshot, frames)

    def process(self, frame: np.ndarray, depth: np.ndarray) -> None:
        if self.previous is not None:
            self.last_motion_median, self.last_motion_ratio = stability_metrics(
                depth, self.previous, self.effective_motion_limit
            )
        if self.phase == "ready_for_a":
            # Camera auto-exposure and the depth model are noisy just after
            # startup/reset. Arm opening detection only after a stable baseline.
            if len(self.noise_medians) < self.thresholds.noise_warmup_frames:
                self._observe_noise(depth)
                count = len(self.noise_medians)
                self.message = (
                    "Baseline ready. Open the drawer."
                    if count >= self.thresholds.noise_warmup_frames
                    else f"Stabilizing closed-drawer baseline: {count}/{self.thresholds.noise_warmup_frames}. Do not open yet."
                )
            else:
                # Check before updating noise so opening motion cannot inflate
                # its own adaptive threshold.
                opened = self._motion_detected()
                if opened:
                    self.phase = "capture_a"
                    self.frames.clear()
                    self.depths.clear()
                    self.message = f"Drawer opening detected; waiting for stable A (0/{self.required})."
                else:
                    self._observe_noise(depth)
        elif self.phase == "capture_a":
            batch = self._capture_if_stable(frame, depth)
            if batch is not None:
                self.before_batch = batch
                self.phase = "collect_before"
                self.message = "Snapshot A ready. Starting live COCO detector sampling..."
        elif self.phase == "collect_before" and self._motion_detected():
            self.phase = "capture_b"
            self.frames.clear()
            self.depths.clear()
            self.message = (
                f"Item motion detected after {len(self.before_observations)} detector frames. "
                f"Remove your hand; waiting for stable B (0/{self.required})."
            )
        elif self.phase == "capture_b":
            batch = self._capture_if_stable(frame, depth)
            if batch is not None:
                self.after_batch = batch
                self.phase = "analyze_b"
                self.message = "Snapshot B captured. Comparing YOLOv8m inventories..."
        elif self.phase == "wait_close" and self._motion_detected():
            self.phase = "closing"
            self.frames.clear()
            self.depths.clear()
            self.message = f"Drawer closing detected; waiting for stable close (0/{self.required})."
        elif self.phase == "closing":
            if self._capture_if_stable(frame, depth) is not None:
                self.reset()
                self.message = "Drawer closed and reset. Waiting for the next opening motion."
        self.previous = depth.copy()

    def observe_before(self, detector: CocoDetector, frame: np.ndarray) -> None:
        """Add one pre-motion detector observation and refresh the provisional set."""
        if self.phase != "collect_before":
            return
        frame_detections, inference_ms = detector.detect([frame])
        self.record_before_observation(frame_detections[0], inference_ms)

    def record_before_observation(
        self,
        detections: list[Detection],
        inference_ms: float,
        distance_mm: Optional[float] = None,
    ) -> None:
        """Accept one completed detector result and its simultaneous ToF range."""
        if self.phase not in {"collect_before", "capture_b"} or self.before_batch is None:
            return
        self.before_observations.append(detections)
        self.before_inference_ms += inference_ms
        if distance_mm is not None:
            self.last_distance_mm = distance_mm
            if self.active_layer is None and self.layer_profiles is not None:
                self.distance_observations.append(distance_mm)
                self.active_layer = self.layer_profiles.match(float(np.median(self.distance_observations)))
        self._refresh_before_analysis()
        assert self.before is not None
        if self.phase == "collect_before":
            self.message = (
                f"Snapshot A ready: live detector {len(self.before_observations)} frames, "
                f">={self.before_vote_ratio:.0%} set: {format_inventory(self.before.inventory)}. "
                f"Layer: {self.active_layer or 'unknown'}."
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
        self.message = "Depth motion started before any detector frame completed. Reset and wait for detector sampling."

    def analyze_pending(self, detector: CocoDetector) -> None:
        if self.phase != "analyze_b":
            return
        batch = self.after_batch
        assert batch is not None
        selected = select_evenly(batch.frames, self.detector_vote_frames)
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
        self.phase = "wait_close"
        self.message = f"{self.delta.message} | layer {self.active_layer or 'unknown'}"


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


def render(
    frame: np.ndarray,
    depth: np.ndarray,
    probe: CocoABProbe,
    depth_ms: float,
    inventory_by_layer: dict[int, Counter[str]],
) -> np.ndarray:
    height, width = frame.shape[:2]
    instruction, colour = PHASE_FEEDBACK[probe.phase]
    feedback = np.zeros((FEEDBACK_HEIGHT, width * 2, 3), dtype=np.uint8)
    cv2.rectangle(feedback, (0, 0), (width * 2, FEEDBACK_HEIGHT), colour, cv2.FILLED)
    cv2.putText(feedback, instruction, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (15, 15, 15), 2, cv2.LINE_AA)

    header = np.zeros((HEADER_HEIGHT, width * 2, 3), dtype=np.uint8)
    for name, rectangle in BUTTONS.items():
        enabled = name != "capture_a" and (name != "save" or probe.delta is not None)
        draw_button(header, {"capture_a": "Auto A", "reset": "Reset", "save": "Save"}[name], rectangle, enabled)
    cv2.putText(header, probe.message[:125], (430, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        header,
        f"state={probe.phase}  depth={depth_ms:.0f}ms  ToF={probe.last_distance_mm or 0:.0f}mm  layer={probe.active_layer or '?'}  depth delta={probe.last_median_delta:.4f}/{probe.effective_median_limit:.4f}",
        (12, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    live_depth = fit_panel(colorize_depth(depth), width, height)
    motion_now = (
        probe.last_motion_median > probe.effective_motion_limit
        or probe.last_motion_ratio >= probe.thresholds.motion_ratio
    )
    metric_lines = (
        f"Depth median: {probe.last_motion_median:.4f} / {probe.effective_motion_limit:.4f}",
        f"Changed pixels: {probe.last_motion_ratio * 100:.1f}% / {probe.thresholds.motion_ratio * 100:.1f}%",
    )
    text_colour = (80, 80, 255) if motion_now else (255, 255, 255)
    cv2.rectangle(live_depth, (max(0, width - 330), 8), (width - 8, 66), (20, 20, 20), cv2.FILLED)
    for index, text in enumerate(metric_lines):
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(
            live_depth,
            text,
            (max(8, width - text_width - 16), 29 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            text_colour,
            1,
            cv2.LINE_AA,
        )
    before_view = np.zeros_like(frame) if probe.before is None else fit_panel(probe.before.annotated, width, height)
    after_view = np.zeros_like(frame) if probe.after is None else fit_panel(probe.after.annotated, width, height)
    if probe.after is None:
        cv2.putText(after_view, "Waiting for item change", (max(12, width // 5), height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180, 180, 180), 2, cv2.LINE_AA)

    footer = np.zeros((154, width * 2, 3), dtype=np.uint8)
    before_inventory = Counter() if probe.before is None else probe.before.inventory
    for layer, left in ((1, 6), (2, width + 6)):
        border = (80, 220, 120) if probe.active_layer == layer else (105, 105, 105)
        cv2.rectangle(footer, (left, 6), (left + width - 12, 91), border, 4 if probe.active_layer == layer else 2)
        cv2.putText(footer, f"LAYER {layer}", (left + 14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, border, 2, cv2.LINE_AA)
        inventory_text = format_inventory(inventory_by_layer[layer])
        cv2.putText(footer, inventory_text, (left + 14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2, cv2.LINE_AA)
    status = "Detector active" if probe.phase == "collect_before" else probe.message
    cv2.putText(
        footer,
        f"Snapshot A: {format_inventory(before_inventory)} | active layer: {probe.active_layer or 'unknown'}",
        (12, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(footer, status[:180], (12, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (225, 225, 225), 1, cv2.LINE_AA)

    return np.vstack(
        (
            feedback,
            header,
            label_row("Live RGB", "Live depth (capture timing only)", width),
            np.hstack((frame, live_depth)),
            label_row("Snapshot A - before", "Snapshot B - after", width),
            np.hstack((before_view, after_view)),
            footer,
        )
    )


def detection_dict(detection: Detection) -> dict[str, object]:
    data = asdict(detection)
    data["bbox"] = list(detection.bbox)
    return data


def save_result(output_dir: Path, probe: CocoABProbe) -> Optional[Path]:
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


def open_inventory_storage(path: Path, profiles: LayerProfiles):
    from core import LayerCalibration
    from ssdlite_detector import COCO_LABELS
    from storage import Storage

    storage = Storage(path)
    model_catalog = {class_id: label for class_id, label in enumerate(COCO_LABELS)}
    storage.seed_catalog(model_catalog.items())
    database_catalog = {
        int(row["class_id"]): str(row["canonical_label"])
        for row in storage.catalog_rows()
        if int(row["enabled"])
    }
    if database_catalog != model_catalog:
        storage.close()
        raise RuntimeError("database item catalog does not match the YOLOv8 COCO model")
    for layer, baseline in profiles.distances_mm.items():
        storage.save_calibration(
            LayerCalibration(
                layer_no=layer,
                bottom_depth_baseline=baseline,
                drawer_mask=((True,),),
                interior_mask=((True,),),
                open_threshold=profiles.tolerance_mm,
                close_threshold=profiles.tolerance_mm,
            )
        )
    storage.check_integrity()
    return storage


def load_database_inventory(storage) -> dict[int, Counter[str]]:
    inventory = {1: Counter(), 2: Counter()}
    for row in storage.inventory_rows():
        layer = int(row["layer_no"])
        if layer in inventory:
            inventory[layer][str(row["canonical_label"])] = int(row["quantity"])
    return inventory


def commit_probe_transaction(storage, probe: CocoABProbe) -> bool:
    from core import Candidate, Crop

    if probe.delta is None or probe.delta.action not in {"put_in", "take_out"}:
        return False
    if probe.active_layer not in {1, 2} or probe.delta.item is None:
        raise RuntimeError("cannot save transaction without a known drawer layer and item")
    source = probe.after if probe.delta.action == "put_in" else probe.before
    assert source is not None
    matches = [item for item in source.detections if item.name == probe.delta.item]
    if not matches:
        raise RuntimeError(f"no YOLO detection found for {probe.delta.item}")
    detection = max(matches, key=lambda item: item.confidence)
    x0, y0, x1, y1 = detection.bbox
    magnitude = float(np.mean(np.abs(probe.after.batch.snapshot.depth - probe.before.batch.snapshot.depth)))
    action = "put" if probe.delta.action == "put_in" else "take"
    storage.commit_candidate(
        Candidate(
            layer_no=probe.active_layer,
            action=action,
            class_id=detection.class_id,
            canonical_label=detection.name,
            confidence=detection.confidence,
            signed_depth_change=magnitude if action == "put" else -magnitude,
            crop=Crop(round(x0), round(y0), max(1, round(max(x1 - x0, y1 - y0))), "snapshot-ab"),
        )
    )
    probe.message = f"{action.upper()} {detection.name} {'->' if action == 'put' else '<-'} LAYER {probe.active_layer} | saved"
    return True


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
    assert canonical_inventory_class(66, "keyboard") == (65, "remote")
    assert canonical_inventory_class(64, "mouse") == (64, "mouse")
    aliases = deduplicate_canonical_detections(
        [item(65, "remote", (10, 10, 30, 30), 0.9), item(65, "remote", (11, 10, 31, 30), 0.8)],
        0.4,
    )
    assert len(aliases) == 1

    thresholds = Thresholds()
    profiles = LayerProfiles({1: 120.0, 2: 320.0}, 60.0)
    assert profiles.match(125.0) == 1 and profiles.match(300.0) == 2 and profiles.match(220.0) is None
    probe = CocoABProbe(3, thresholds, 3, 0.80, 2 / 3, 0.4, profiles)
    probe.last_motion_median = thresholds.motion_median + 0.001
    assert probe._motion_detected()
    probe.last_motion_median = 0.0
    probe.last_motion_ratio = thresholds.motion_ratio
    assert probe._motion_detected()  # Either metric independently triggers motion.
    probe.last_motion_ratio = 0.0
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    depth_a = np.full((16, 16), 0.5, dtype=np.float32)
    depth_b = depth_a.copy()
    depth_b[4:12, 4:12] += 0.20
    for _ in range(thresholds.noise_warmup_frames + 1):
        probe.process(frame, depth_a)
    assert probe.phase == "ready_for_a"  # Baseline warm-up must not auto-capture.
    depth_open = depth_a + 0.20
    probe.process(frame, depth_open)  # large motion opens the drawer
    assert probe.phase == "capture_a"
    for _ in range(3):
        probe.process(frame, depth_open)
    assert probe.phase == "collect_before" and probe.before_batch is not None

    class FakeDetector:
        def detect(self, selected: Sequence[np.ndarray]) -> tuple[list[list[Detection]], float]:
            return [[item(41, "cup", (10, 10, 30, 30))] for _ in selected], 1.0

    class AfterDetector:
        def detect(self, selected: Sequence[np.ndarray]) -> tuple[list[list[Detection]], float]:
            return [
                [item(41, "cup", (10, 10, 30, 30)), item(65, "remote", (32, 10, 52, 30))]
                for _ in selected
            ], 1.0

    fake_detector = FakeDetector()
    for distance_mm in (122.0, 318.0, 318.0, 318.0, 318.0):
        detections, inference_ms = fake_detector.detect([frame])
        probe.record_before_observation(detections[0], inference_ms, distance_mm)
    assert probe.phase == "collect_before" and probe.before is not None
    assert probe.before.frames_analyzed == 5 and probe.before.inventory == Counter({"cup": 1})
    assert probe.active_layer == 1  # The first YOLO sample locks the drawer layer.
    probe.process(frame, depth_b)
    assert probe.phase == "capture_b"
    probe.finalize_before()
    for _ in range(3):
        probe.process(frame, depth_b)
    assert probe.phase == "analyze_b"
    probe.analyze_pending(AfterDetector())
    assert probe.phase == "wait_close" and probe.delta is not None
    assert probe.delta.action == "put_in" and probe.delta.item == "remote"
    storage = open_inventory_storage(Path(":memory:"), profiles)
    try:
        assert commit_probe_transaction(storage, probe)
        assert storage.inventory(1, 65) == 1
    finally:
        storage.close()
    probe.process(frame, depth_a)
    assert probe.phase == "closing"
    for _ in range(3):
        probe.process(frame, depth_a)
    assert probe.phase == "ready_for_a" and probe.before is None
    print("yolo_ab_inventory_probe: self-test OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth-timed A/B capture with the board COCO detector")
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--detector-labels", type=Path, default=DEFAULT_DETECTOR_LABELS)
    parser.add_argument("--detector-priors", type=Path, default=DEFAULT_DETECTOR_PRIORS)
    parser.add_argument("--detector-delegate", default="/usr/lib/libethosu_delegate.so")
    parser.add_argument("--enabled-config", type=Path, default=DEFAULT_ENABLED_CONFIG)
    parser.add_argument("--all-classes", action="store_true", help="show all COCO labels instead of the 20 demo classes")
    parser.add_argument("--detector-confidence", type=float, default=0.50)
    parser.add_argument("--detector-nms-iou", type=float, default=0.50)
    parser.add_argument("--detector-vote-frames", type=int, default=3)
    parser.add_argument("--detector-only", action="store_true", help="test detector without depth or A/B capture")
    parser.add_argument("--detector-window", type=int, default=5, help="recent frames used by detector-only consensus")
    parser.add_argument(
        "--before-vote-ratio",
        type=float,
        default=0.80,
        help="minimum fraction of live pre-motion frames required in the before set",
    )
    parser.add_argument("--detector-vote-ratio", type=float, default=2 / 3, help="detector-only temporal consensus vote ratio")
    parser.add_argument("--detector-match-iou", type=float, default=0.40)
    parser.add_argument(
        "--depth-model",
        choices=("midas", "depth-anything-metric"),
        default="midas",
    )
    parser.add_argument("--depth-model-path", type=Path, help="override the selected depth checkpoint")
    parser.add_argument("--depth-delegate", default="/usr/lib/libethosu_delegate.so")
    parser.add_argument("--camera", default="/dev/video2", help="device path or numeric camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--avfoundation", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until q/Esc")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--stable-frames", type=int, default=10)
    parser.add_argument("--stable-median", type=float, default=0.03)
    parser.add_argument("--stable-ratio", type=float, default=Thresholds.stable_ratio)
    parser.add_argument("--motion-median", type=float, default=0.10)
    parser.add_argument("--motion-ratio", type=float, default=1)
    parser.add_argument("--noise-multiplier", type=float, default=Thresholds.noise_multiplier)
    parser.add_argument("--noise-warmup-frames", type=int, default=20)
    parser.add_argument("--bilateral-diameter", type=int, default=5)
    parser.add_argument("--bilateral-sigma", type=float, default=0.08)
    parser.add_argument("--output-dir", type=Path, default=Path("captures/yolo-ab-probe"))
    parser.add_argument("--database", type=Path, default=Path("/root/yolo-ab-probe/inventory.db"))
    parser.add_argument("--vl53-device", default="/dev/i2c-0")
    parser.add_argument("--vl53-address", type=lambda value: int(value, 0), default=0x29)
    parser.add_argument("--vl53-calibration", type=Path, default=Path("/root/vl53l0x_layers.json"))
    parser.add_argument("--vl53-samples", type=int, default=15)
    parser.add_argument("--calibrate-vl53", action="store_true", help="record layer 1 and 2 distances, then exit")
    parser.add_argument("--no-vl53", action="store_true", help="run without layer ranging")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def open_layer_sensor(args: argparse.Namespace) -> tuple[Optional[DistanceSensor], Optional[LayerProfiles]]:
    if args.no_vl53:
        return None, None
    if not args.vl53_calibration.is_file():
        raise FileNotFoundError(
            f"VL53 calibration not found: {args.vl53_calibration}; run with --calibrate-vl53 first"
        )
    return (
        DistanceSensor(args.vl53_device, args.vl53_address),
        LayerProfiles.load(args.vl53_calibration),
    )


def make_depth_model(args: argparse.Namespace) -> MidasTFLite | DepthAnythingMetric:
    if args.depth_model == "midas":
        return MidasTFLite(args.depth_model_path or DEFAULT_MIDAS_MODEL, args.depth_delegate or None)
    return DepthAnythingMetric(args.depth_model_path or DEFAULT_DEPTH_ANYTHING_MODEL)


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    camera = int(args.camera) if str(args.camera).isdigit() else args.camera
    if args.avfoundation:
        backend = cv2.CAP_AVFOUNDATION
    elif isinstance(camera, str) and camera.startswith("/dev/video"):
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY
    capture = cv2.VideoCapture(camera, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open camera {args.camera}")
    return capture


def run_vl53_calibration(args: argparse.Namespace) -> None:
    if args.no_display:
        raise ValueError("VL53 HDMI calibration requires the display")
    if args.vl53_samples < 3:
        raise ValueError("VL53 calibration needs at least 3 samples")

    sensor = DistanceSensor(args.vl53_device, args.vl53_address)
    capture = open_camera(args)
    window = "VL53L0X two-layer initialization"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state = "ready"
    samples: Deque[float] = deque(maxlen=args.vl53_samples)
    readings: dict[int, float] = {}
    noise: list[float] = []
    message = "Press START INIT"
    clicked = False
    done_at: Optional[float] = None
    button_rect = [20, max(0, args.height - 68), 220, 48]

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN and inside((x, y), tuple(button_rect)):
            clicked = True

    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame capture failed")
            distance_mm = None
            try:
                distance_mm = float(sensor.read_mm())
            except RuntimeError:
                pass
            if state in {"layer1", "layer2"} and distance_mm is not None:
                samples.append(distance_mm)

            height, width = frame.shape[:2]
            button_rect[:] = (20, height - 68, 220, 48)
            button = tuple(button_rect)
            ready = len(samples) == args.vl53_samples
            if state == "ready":
                stage = "STAGE: READY"
                next_step = "NEXT: click START INIT, then open only layer 1"
                button_label, button_enabled = "START INIT", True
            elif state == "layer1":
                stage = "STAGE: RECORD LAYER 1"
                next_step = "NEXT: keep layer 1 still, then click NEXT"
                button_label, button_enabled = "NEXT", ready
            elif state == "layer2":
                stage = "STAGE: RECORD LAYER 2"
                next_step = "NEXT: keep layer 2 still, then click FINISH INIT"
                button_label, button_enabled = "FINISH INIT", ready
            else:
                stage = "STAGE: INITIALIZATION COMPLETE"
                next_step = "NEXT: start the normal drawer probe"
                button_label, button_enabled = "SAVED", False

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), cv2.FILLED)
            cv2.putText(overlay, stage, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, next_step, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(overlay, message[:90], (18, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 180), 1, cv2.LINE_AA)
            range_text = "VL53: OUT OF RANGE" if distance_mm is None else f"VL53: {distance_mm:.0f} mm"
            (range_width, _), _ = cv2.getTextSize(range_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
            cv2.putText(
                overlay,
                range_text,
                (max(18, width - range_width - 18), 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (40, 40, 255) if distance_mm is None else (40, 255, 40),
                2,
                cv2.LINE_AA,
            )
            if state in {"layer1", "layer2"}:
                cv2.putText(
                    overlay,
                    f"stable samples: {len(samples)}/{args.vl53_samples}",
                    (width - 245, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
            x, y, button_width, button_height = button
            cv2.rectangle(
                overlay,
                (x, y),
                (x + button_width, y + button_height),
                (40, 150, 40) if button_enabled else (80, 80, 80),
                cv2.FILLED,
            )
            cv2.putText(overlay, button_label, (x + 16, y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if clicked:
                clicked = False
                if state == "ready":
                    state = "layer1"
                    samples.clear()
                    message = "Open only layer 1 and hold it still"
                elif state in {"layer1", "layer2"} and ready:
                    values = np.asarray(samples, dtype=np.float32)
                    median = float(np.median(values))
                    mad = float(np.median(np.abs(values - median)))
                    layer = 1 if state == "layer1" else 2
                    readings[layer] = median
                    noise.append(mad)
                    print(f"layer {layer}: {median:.0f} mm (MAD {mad:.1f} mm)")
                    if state == "layer1":
                        state = "layer2"
                        samples.clear()
                        message = "Layer 1 saved. Close it, open only layer 2, and hold still"
                    else:
                        try:
                            profiles = make_profiles(readings, noise)
                            profiles.save(args.vl53_calibration)
                            state = "done"
                            message = f"Saved: {args.vl53_calibration}"
                            done_at = time.monotonic()
                        except RuntimeError as error:
                            state = "ready"
                            samples.clear()
                            readings.clear()
                            noise.clear()
                            message = f"Calibration rejected: {error}"
            if done_at is not None and time.monotonic() - done_at >= 2.0:
                break
    finally:
        capture.release()
        sensor.close()
        cv2.destroyAllWindows()


def run_detector_only(args: argparse.Namespace) -> None:
    if not 0 < args.detector_confidence <= 1 or not 0 <= args.detector_nms_iou <= 1:
        raise ValueError("detector confidence must be in (0, 1] and NMS IoU in [0, 1]")
    if args.detector_window < 1:
        raise ValueError("detector window must be positive")
    detector = CocoDetector(
        args.detector_model,
        args.detector_labels,
        args.detector_priors,
        args.detector_confidence,
        args.detector_nms_iou,
        selected_labels(args),
        args.detector_delegate or None,
    )
    layer_sensor, layer_profiles = open_layer_sensor(args)
    capture = open_camera(args)
    history: Deque[list[Detection]] = deque(maxlen=args.detector_window)
    window = "YOLOv8m detector only"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(
        f"detector={detector.name} backend={detector.backend}; "
        f"confidence={args.detector_confidence:.2f}; "
        "press q/Esc to stop"
    )
    print(detector.diagnostic_text())
    frames_processed = 0
    last_signature: Optional[tuple[tuple[str, int], ...]] = None
    try:
        while args.max_frames == 0 or frames_processed < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame capture failed")
            frame_detections, inference_ms = detector.detect([frame])
            distance_mm = None
            if layer_sensor is not None:
                try:
                    distance_mm = float(layer_sensor.read_mm())
                except RuntimeError:
                    pass
            layer = layer_profiles.match(distance_mm) if layer_profiles is not None and distance_mm is not None else None
            history.append(frame_detections[0])
            stable = consensus_detections(
                list(history), args.detector_vote_ratio, args.detector_match_iou
            )
            view = annotate(frame, stable, len(history))
            signature = tuple((item.name, item.votes) for item in stable)
            cv2.putText(
                view,
                f"NPU {inference_ms:.1f} ms | layer {layer or '?'} ({distance_mm or 0:.0f} mm) | stable: {format_inventory(Counter(item.name for item in stable))}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA,
            )
            if args.no_display:
                if signature != last_signature:
                    print(
                        f"frame={frames_processed + 1} raw={len(frame_detections[0])} "
                        f"stable={format_inventory(Counter(item.name for item in stable))} "
                        f"latency_ms={inference_ms:.1f} distance_mm={distance_mm} layer={layer}"
                    )
            else:
                cv2.imshow(window, view)
            last_signature = signature
            frames_processed += 1
            if not args.no_display and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        if layer_sensor is not None:
            layer_sensor.close()
        if not args.no_display:
            cv2.destroyAllWindows()


def run(args: argparse.Namespace) -> None:
    if not 0 < args.detector_confidence <= 1 or not 0 <= args.detector_nms_iou <= 1:
        raise ValueError("detector confidence must be in (0, 1] and NMS IoU in [0, 1]")
    if args.noise_warmup_frames < 1 or args.noise_multiplier < 1:
        raise ValueError("noise warmup must be positive and multiplier at least 1")
    depth_model = make_depth_model(args)
    detector = CocoDetector(
        args.detector_model,
        args.detector_labels,
        args.detector_priors,
        args.detector_confidence,
        args.detector_nms_iou,
        selected_labels(args),
        args.detector_delegate or None,
    )
    layer_sensor, layer_profiles = open_layer_sensor(args)
    if layer_profiles is None:
        raise RuntimeError("the four-stage inventory workflow requires VL53 layer calibration")
    storage = open_inventory_storage(args.database, layer_profiles)
    inventory_by_layer = load_database_inventory(storage)
    thresholds = Thresholds(
        stable_median=args.stable_median,
        stable_ratio=args.stable_ratio,
        motion_median=args.motion_median,
        motion_ratio=args.motion_ratio,
        noise_multiplier=args.noise_multiplier,
        noise_warmup_frames=args.noise_warmup_frames,
    )
    probe = CocoABProbe(
        args.stable_frames,
        thresholds,
        args.detector_vote_frames,
        args.before_vote_ratio,
        args.detector_vote_ratio,
        args.detector_match_iou,
        layer_profiles,
    )

    capture = open_camera(args)

    window = "Depth-timed A/B inventory - COCO detector"
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
            probe.message = "Open the drawer to trigger automatic Snapshot A."
        elif inside(point, BUTTONS["reset"]):
            probe.reset()
        elif inside(point, BUTTONS["save"]):
            save()

    if not args.no_display:
        cv2.setMouseCallback(window, mouse)
    print(
        f"depth={depth_model.name} input={depth_model.width}x{depth_model.height}; "
        f"detector={detector.name} backend={detector.backend}; open drawer for auto A, close drawer to reset"
    )
    print(detector.diagnostic_text())
    frames_processed = 0
    before_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="before-detector")
    before_future: Optional[Future[tuple[int, list[list[Detection]], float, Optional[float]]]] = None

    def detect_before(
        frame: np.ndarray, capture_id: int
    ) -> tuple[int, list[list[Detection]], float, Optional[float]]:
        frame_detections, inference_ms = detector.detect([frame])
        distance_mm = None
        if layer_sensor is not None:
            try:
                distance_mm, _ = layer_sensor.median_mm(samples=5, delay_s=0.01)
            except RuntimeError:
                pass
        return capture_id, frame_detections, inference_ms, distance_mm

    def accept_before_result(*, wait: bool) -> None:
        nonlocal before_future
        if before_future is None or (not wait and not before_future.done()):
            return
        capture_id, frame_detections, inference_ms, distance_mm = before_future.result()
        if capture_id == probe.capture_id:
            probe.record_before_observation(frame_detections[0], inference_ms, distance_mm)
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
                probe.message = f"Live detector sampling failed: {error}"
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
                    # motion frame to the detector.
                    accept_before_result(wait=True)
                    probe.finalize_before()
                except Exception as error:
                    probe.phase = "error"
                    probe.message = f"Could not finalize the before set: {error}"
            frames_processed += 1

            key = -1
            if not args.no_display:
                cv2.imshow(window, render(frame, depth, probe, depth_ms, inventory_by_layer))
                key = cv2.waitKey(1) & 0xFF
            try:
                if probe.phase == "collect_before":
                    # Keep at most one detector job in flight.  Depth therefore
                    # continues polling without waiting for CPU inference.
                    if before_future is None:
                        before_future = before_executor.submit(detect_before, frame.copy(), probe.capture_id)
                elif probe.phase == "analyze_b":
                    probe.analyze_pending(detector)
                    if probe.active_layer is None and layer_sensor is not None:
                        distance_mm, _ = layer_sensor.median_mm(samples=5, delay_s=0.01)
                        probe.last_distance_mm = distance_mm
                        probe.active_layer = layer_profiles.match(distance_mm)
                    commit_probe_transaction(storage, probe)
                    inventory_by_layer = load_database_inventory(storage)
            except Exception as error:
                probe.phase = "error"
                probe.message = f"Detector analysis failed: {error}"
            if args.no_display:
                continue
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                probe.reset()
            elif key == ord("s"):
                save()
    finally:
        before_executor.shutdown(wait=True, cancel_futures=True)
        capture.release()
        if layer_sensor is not None:
            layer_sensor.close()
        storage.close()
        if not args.no_display:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.calibrate_vl53:
        run_vl53_calibration(args)
    elif args.self_test:
        self_test()
    elif args.detector_only:
        run_detector_only(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
