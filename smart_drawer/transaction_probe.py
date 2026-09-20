"""MiDaS A/B depth transaction with denoising and voted crop detection.

Depth still decides the changed crop and put/take direction.  The detector is
only run on that crop; stable RGB frames are voted to avoid one bad detector
frame deciding the inventory update.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import replace
from typing import Any, Optional, Sequence

import numpy as np

from .core import RecoverableRejection, affine_align, build_change_result, stable_snapshot
from .depth_denoise import denoise_relative_depth


def _xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x, y, x + max(0.0, width), y + max(0.0, height)


def _detection_box(detection: Any) -> tuple[float, float, float, float]:
    if hasattr(detection, "box"):
        return _xyxy(detection.box)
    return tuple(float(value) for value in detection.bbox)


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    first_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    second_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return intersection / max(first_area + second_area - intersection, 1e-9)


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """IoU for x, y, width, height boxes."""
    return _box_iou(_xyxy(first), _xyxy(second))


def select_evenly(frames: Sequence[Any], count: int) -> list[Any]:
    if count <= 0 or not frames:
        return []
    if count >= len(frames):
        return [frame.copy() for frame in frames]
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int)
    return [frames[int(index)].copy() for index in indices]


def consensus_detections(
    frames: Sequence[Sequence[Any]],
    vote_ratio: float = 2 / 3,
    match_iou: float = 0.40,
) -> tuple[Any, ...]:
    """Keep same-class boxes present in enough stable frames.

    This is the portable part of ``yolo_ab_inventory_probe.py``.  It accepts
    the board SSDLite objects as well as YOLO-like objects with class_id, box,
    confidence and label attributes.
    """
    if not frames:
        return ()
    if not 0.0 < vote_ratio <= 1.0 or not 0.0 <= match_iou <= 1.0:
        raise ValueError("vote_ratio must be in (0, 1] and match_iou in [0, 1]")
    required_votes = max(1, math.ceil(len(frames) * vote_ratio - 1e-9))
    tracks: list[dict[str, Any]] = []
    for frame_index, detections in enumerate(frames):
        for detection in sorted(detections, key=lambda item: float(item.confidence), reverse=True):
            candidates = [
                (_box_iou(_detection_box(track["detections"][-1]), _detection_box(detection)), track)
                for track in tracks
                if int(track["class_id"]) == int(detection.class_id)
                and track["last_frame"] != frame_index
            ]
            overlap, track = max(candidates, default=(0.0, None), key=lambda item: item[0])
            if track is None or overlap < match_iou:
                tracks.append({
                    "class_id": int(detection.class_id),
                    "detections": [detection],
                    "last_frame": frame_index,
                })
            else:
                track["detections"].append(detection)
                track["last_frame"] = frame_index

    result: list[Any] = []
    for track in tracks:
        detections = track["detections"]
        if len(detections) < required_votes:
            continue
        sample = detections[0]
        boxes = np.asarray([_detection_box(item) for item in detections], dtype=np.float32)
        merged_xyxy = tuple(float(value) for value in np.median(boxes, axis=0))
        if hasattr(sample, "box"):
            x0, y0, x1, y1 = merged_xyxy
            merged_box = (x0, y0, x1 - x0, y1 - y0)
            fields = {"box": merged_box}
        else:
            fields = {"bbox": merged_xyxy}
        if hasattr(sample, "votes"):
            fields["votes"] = len(detections)
        result.append(replace(
            sample,
            confidence=float(np.median([float(item.confidence) for item in detections])),
            **fields,
        ))
    return tuple(sorted(
        result,
        key=lambda item: (
            int(item.class_id),
            (item.box if hasattr(item, "box") else item.bbox)[0],
            (item.box if hasattr(item, "box") else item.bbox)[1],
        ),
    ))


class DepthTransactionProbe:
    """Capture stable A/B depth, then run voted detection on one changed crop."""

    def __init__(
        self,
        roi: tuple[int, int, int, int],
        detector: Any,
        *,
        near_is_positive: bool = True,
        motion_threshold: float = 0.04,
        stability_threshold: float = 0.015,
        stability_ratio: float = 0.08,
        motion_ratio: float = 0.02,
        object_change_threshold: float = 0.10,
        direction_threshold: float = 0.05,
        min_change_area: int = 4,
        crop_padding: float = 0.25,
        bilateral_diameter: int = 5,
        bilateral_sigma: float = 0.08,
        noise_multiplier: float = 1.5,
        noise_warmup_frames: int = 8,
        change_noise_multiplier: float = 4.0,
        detector_frames: int = 3,
        detector_vote_ratio: float = 2 / 3,
        detector_match_iou: float = 0.40,
    ) -> None:
        self.x, self.y, self.width, self.height = roi
        if self.x < 0 or self.y < 0 or self.width < 2 or self.height < 2:
            raise ValueError("invalid drawer ROI")
        if noise_warmup_frames < 1 or noise_multiplier < 1.0 or change_noise_multiplier < 1.0:
            raise ValueError("invalid depth noise configuration")
        if detector_frames < 1 or not 0.0 < detector_vote_ratio <= 1.0:
            raise ValueError("invalid detector vote configuration")
        self.detector = detector
        self.near_is_positive = near_is_positive
        # ponytail: these remain calibration knobs; adaptive noise only raises
        # them, so a real object cannot be hidden by a too-quiet warm-up.
        self.motion_threshold = float(motion_threshold)
        self.stability_threshold = float(stability_threshold)
        self.stability_ratio = float(stability_ratio)
        self.motion_ratio = float(motion_ratio)
        self.object_change_threshold = float(object_change_threshold)
        self.direction_threshold = float(direction_threshold)
        self.min_change_area = int(min_change_area)
        self.crop_padding = float(crop_padding)
        self.bilateral_diameter = int(bilateral_diameter)
        self.bilateral_sigma = float(bilateral_sigma)
        self.noise_multiplier = float(noise_multiplier)
        self.change_noise_multiplier = float(change_noise_multiplier)
        self.detector_frames = int(detector_frames)
        self.detector_vote_ratio = float(detector_vote_ratio)
        self.detector_match_iou = float(detector_match_iou)
        self.state = "capturing_a"
        self.previous_depth: Optional[np.ndarray] = None
        self.stable_depth: list[np.ndarray] = []
        self.stable_rgb: list[np.ndarray] = []
        self.snapshot_a = None
        self.snapshot_b = None
        self.snapshot_a_frames: tuple[np.ndarray, ...] = ()
        self.snapshot_b_frames: tuple[np.ndarray, ...] = ()
        self.snapshot_a_noise: Optional[np.ndarray] = None
        self.snapshot_b_noise: Optional[np.ndarray] = None
        self.started = time.monotonic()
        self.result: Optional[dict[str, Any]] = None
        self.noise_medians: deque[float] = deque(maxlen=noise_warmup_frames)
        self.noise_pixel_p95: deque[float] = deque(maxlen=noise_warmup_frames)
        self.effective_stability_threshold = self.stability_threshold
        self.effective_pixel_threshold = self.stability_threshold
        self.last_median_delta = 0.0
        self.last_changed_ratio = 0.0

    def _roi(self, matrix: Any) -> Any:
        crop = matrix[self.y:self.y + self.height, self.x:self.x + self.width]
        if crop.shape[0] != self.height or crop.shape[1] != self.width:
            raise ValueError("drawer ROI is outside the 256x256 analysis frame")
        return crop

    def _observe_noise(self, current: np.ndarray, previous: Optional[np.ndarray]) -> None:
        if previous is None:
            return
        difference = np.abs(current - previous)
        self.noise_medians.append(float(np.median(difference)))
        self.noise_pixel_p95.append(float(np.percentile(difference, 95)))
        if len(self.noise_medians) < self.noise_medians.maxlen:
            return
        self.effective_stability_threshold = max(
            self.stability_threshold,
            float(np.percentile(self.noise_medians, 90)) * self.noise_multiplier,
        )
        self.effective_pixel_threshold = max(
            self.stability_threshold,
            float(np.percentile(self.noise_pixel_p95, 90)) * self.noise_multiplier,
        )

    def _metrics(self, current: np.ndarray, previous: Optional[np.ndarray], pixel_threshold: float) -> tuple[float, float]:
        if previous is None:
            return float("inf"), 1.0
        difference = np.abs(current - previous)
        median_delta = float(np.median(difference))
        changed_ratio = float(np.mean(difference > pixel_threshold))
        self.last_median_delta = median_delta
        self.last_changed_ratio = changed_ratio
        return median_delta, changed_ratio

    def _stable(self, current: np.ndarray, previous: Optional[np.ndarray]) -> bool:
        median_delta, changed_ratio = self._metrics(current, previous, self.effective_pixel_threshold)
        return median_delta <= self.effective_stability_threshold and changed_ratio <= self.stability_ratio

    def _motion(self, current: np.ndarray, previous: Optional[np.ndarray]) -> bool:
        pixel_threshold = max(self.motion_threshold, self.effective_pixel_threshold * 2.5)
        median_delta, changed_ratio = self._metrics(current, previous, pixel_threshold)
        return median_delta >= max(self.motion_threshold, self.effective_stability_threshold * 2.5) or changed_ratio >= self.motion_ratio

    def _reset_window(self) -> None:
        self.stable_depth.clear()
        self.stable_rgb.clear()

    def _take_stable_snapshot(self) -> tuple[Any, tuple[np.ndarray, ...], np.ndarray]:
        depths = [frame.copy() for frame in self.stable_depth[-3:]]
        frames = tuple(frame.copy() for frame in self.stable_rgb[-3:])
        stack = np.stack(depths).astype(np.float32)
        median_depth = np.median(stack, axis=0).astype(np.float32)
        noise = (1.4826 * np.median(np.abs(stack - median_depth), axis=0)).astype(np.float32)
        return stable_snapshot(frames, depths), frames, noise

    def feed(self, depth: Any, rgb: Any) -> Optional[dict[str, Any]]:
        """Feed one MiDaS depth and matching 256x256 RGB frame."""
        if self.state == "done":
            return self.result
        normalized = denoise_relative_depth(
            np.asarray(depth, dtype=np.float32), self.bilateral_diameter, self.bilateral_sigma
        )
        depth_roi = self._roi(normalized)
        rgb_roi = self._roi(rgb)
        previous = self.previous_depth
        self.previous_depth = depth_roi.copy()

        if self.state == "capturing_a":
            self._observe_noise(depth_roi, previous)
            if self._stable(depth_roi, previous):
                self.stable_depth.append(depth_roi.copy())
                self.stable_rgb.append(rgb_roi.copy())
            else:
                self._reset_window()
            if len(self.stable_depth) >= 3:
                self.snapshot_a, self.snapshot_a_frames, self.snapshot_a_noise = self._take_stable_snapshot()
                self._reset_window()
                self.state = "waiting_item_motion"
                print("probe: snapshot A captured", flush=True)
            return None

        if self.state == "waiting_item_motion":
            if self._motion(depth_roi, previous):
                self._reset_window()
                self.state = "capturing_b"
                print("probe: item motion detected; waiting for snapshot B", flush=True)
            return None

        if self.state != "capturing_b":
            return self.result

        if self._stable(depth_roi, previous):
            self.stable_depth.append(depth_roi.copy())
            self.stable_rgb.append(rgb_roi.copy())
        else:
            self._reset_window()
        if len(self.stable_depth) < 3:
            return None

        self.snapshot_b, self.snapshot_b_frames, self.snapshot_b_noise = self._take_stable_snapshot()
        self._reset_window()
        self.state = "done"
        return self._detect_changed_crop()

    def _detect_changed_crop(self) -> dict[str, Any]:
        try:
            border = max(1, min(self.width, self.height) // 10)
            background = tuple(
                tuple(
                    x < border or y < border or x >= self.width - border or y >= self.height - border
                    for x in range(self.width)
                )
                for y in range(self.height)
            )
            aligned_b = affine_align(self.snapshot_b.depth, self.snapshot_a.depth, background)
            aligned_b_snapshot = type(self.snapshot_b)(
                rgb=self.snapshot_b.rgb,
                depth=aligned_b,
                frame_ids=self.snapshot_b.frame_ids,
            )
            noise_a = self.snapshot_a_noise
            noise_b = self.snapshot_b_noise
            noise_p95 = 0.0
            if noise_a is not None and noise_b is not None:
                noise_p95 = float(np.percentile(np.sqrt(noise_a * noise_a + noise_b * noise_b), 95))
            change = build_change_result(
                self.snapshot_a,
                aligned_b_snapshot,
                tuple(tuple(True for _ in range(self.width)) for _ in range(self.height)),
                near_is_positive=self.near_is_positive,
                object_change_threshold=max(self.object_change_threshold, noise_p95 * self.change_noise_multiplier),
                direction_threshold=self.direction_threshold,
                min_change_area=self.min_change_area,
                component_merge_gap=1,
                morphology_radius=1,
                crop_padding=self.crop_padding,
            )
        except RecoverableRejection as error:
            self.result = {"ok": False, "reason": str(error)}
            print("probe: rejected change: %s" % error, flush=True)
            return self.result

        if self.detector is None:
            self.result = {"ok": False, "reason": "detector_unavailable"}
            return self.result
        source_frames = self.snapshot_b_frames if change.action == "put" else self.snapshot_a_frames
        selected_frames = select_evenly(source_frames, min(self.detector_frames, len(source_frames)))
        if not selected_frames:
            self.result = {"ok": False, "reason": "detector_frames_unavailable"}
            return self.result

        observations: list[list[Any]] = []
        started = time.monotonic()
        try:
            for frame in selected_frames:
                crop = frame[change.crop.y:change.crop.y + change.crop.size,
                             change.crop.x:change.crop.x + change.crop.size]
                if crop.shape[:2] != (change.crop.size, change.crop.size):
                    raise ValueError("changed crop is outside the RGB snapshot")
                observations.append(self.detector.detect(crop))
            detections = consensus_detections(
                observations, self.detector_vote_ratio, self.detector_match_iou
            )
        except Exception as error:
            self.result = {"ok": False, "reason": "detector_failed:%s" % error}
            print("probe: detector failed: %s" % error, flush=True)
            return self.result
        latency_ms = (time.monotonic() - started) * 1000.0
        self.result = {
            "ok": True,
            "action": change.action,
            "signed_change": change.signed_change,
            "crop": change.crop,
            "detections": list(detections),
            "detector_calls": len(selected_frames),
            "detector_latency_ms": latency_ms,
            "detector_frames": len(selected_frames),
        }
        backend = getattr(self.detector, "backend", "unknown")
        print(
            "probe: action=%s signed_depth=%.4f crop=%s detector_backend=%s "
            "latency_ms=%.1f calls=%d detections=%s"
            % (change.action, change.signed_change, change.crop, backend, latency_ms,
               len(selected_frames), detections),
            flush=True,
        )
        return self.result


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Detection:
        class_id: int
        label: str
        confidence: float
        box: tuple[float, float, float, float]
        votes: int = 1

    first = _Detection(41, "cup", 0.9, (10, 10, 20, 20))
    second = _Detection(41, "cup", 0.8, (11, 10, 20, 20))
    assert len(consensus_detections([[first], [second], []], 2 / 3, 0.4)) == 1
    print("transaction_probe: self-test OK")
