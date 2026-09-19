"""Hardware adapter for the two-stable-depth changed-crop transaction probe.

It intentionally stops before layer calibration/SQLite commit.  Those require
real drawer baselines.  The probe proves the important edge-only join:
MiDaS depth A/B -> signed changed crop -> one GoPoint detector call.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from core import RecoverableRejection, affine_align, build_change_result, stable_snapshot


class DepthTransactionProbe:
    """Capture Snapshot A/B from one fixed drawer ROI and detect once."""

    def __init__(
        self,
        roi: tuple[int, int, int, int],
        detector: Any,
        *,
        near_is_positive: bool = True,
        motion_threshold: float = 0.04,
        stability_threshold: float = 0.015,
        object_change_threshold: float = 0.10,
        direction_threshold: float = 0.05,
        min_change_area: int = 4,
        crop_padding: float = 0.25,
    ) -> None:
        self.x, self.y, self.width, self.height = roi
        if self.x < 0 or self.y < 0 or self.width < 2 or self.height < 2:
            raise ValueError("invalid drawer ROI")
        self.detector = detector
        self.near_is_positive = near_is_positive
        # ponytail: fixed thresholds are a hardware calibration knob; replace
        # them with Gate-0 measurements before demo acceptance.
        self.motion_threshold = motion_threshold
        self.stability_threshold = stability_threshold
        self.object_change_threshold = object_change_threshold
        self.direction_threshold = direction_threshold
        self.min_change_area = min_change_area
        self.crop_padding = crop_padding
        self.state = "capturing_a"
        self.previous_depth: Optional[Any] = None
        self.stable_depth: list[Any] = []
        self.stable_rgb: list[Any] = []
        self.snapshot_a = None
        self.snapshot_b = None
        self.started = time.monotonic()
        self.result: Optional[dict[str, Any]] = None

    def _roi(self, matrix: Any) -> Any:
        crop = matrix[self.y:self.y + self.height, self.x:self.x + self.width]
        if crop.shape[0] != self.height or crop.shape[1] != self.width:
            raise ValueError("drawer ROI is outside the 256x256 analysis frame")
        return crop

    @staticmethod
    def _stable(current: Any, previous: Optional[Any], threshold: float) -> bool:
        if previous is None:
            return False
        return float(abs(current - previous).mean()) <= threshold

    def _reset_window(self) -> None:
        self.stable_depth.clear()
        self.stable_rgb.clear()

    def _take_stable_snapshot(self) -> Any:
        return stable_snapshot(self.stable_rgb[-3:], self.stable_depth[-3:])

    def feed(self, depth: Any, rgb: Any) -> Optional[dict[str, Any]]:
        """Feed one MiDaS depth and matching 256x256 RGB frame.

        Return a result only after B is stable.  Subsequent frames are ignored,
        matching the MVP rule that post-B changes cannot create another YOLO
        call.
        """
        if self.state == "done":
            return self.result
        import numpy as np

        # MiDaS is relative depth.  Normalize every frame before comparing it;
        # affine alignment below then removes the remaining fixed-background drift.
        low, high = np.percentile(depth, (5, 95))
        normalized = np.clip((depth.astype(np.float32) - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        depth_roi = self._roi(normalized)
        rgb_roi = self._roi(rgb)
        previous = self.previous_depth
        self.previous_depth = depth_roi.copy()

        if self.state == "capturing_a":
            if self._stable(depth_roi, previous, self.stability_threshold):
                self.stable_depth.append(depth_roi.copy())
                self.stable_rgb.append(rgb_roi.copy())
            else:
                self._reset_window()
            if len(self.stable_depth) >= 3:
                self.snapshot_a = self._take_stable_snapshot()
                self._reset_window()
                self.state = "waiting_item_motion"
                print("probe: snapshot A captured", flush=True)
            return None

        if self.state == "waiting_item_motion":
            if self.snapshot_a is not None:
                import numpy as np

                baseline = np.asarray(self.snapshot_a.depth, dtype=np.float32)
                if float(np.abs(depth_roi - baseline).mean()) > self.motion_threshold:
                    self._reset_window()
                    self.state = "capturing_b"
                    print("probe: item motion detected; waiting for snapshot B", flush=True)
            return None

        if self.state != "capturing_b":
            return self.result

        if self._stable(depth_roi, previous, self.stability_threshold):
            self.stable_depth.append(depth_roi.copy())
            self.stable_rgb.append(rgb_roi.copy())
        else:
            self._reset_window()
        if len(self.stable_depth) < 3:
            return None

        self.snapshot_b = self._take_stable_snapshot()
        self._reset_window()
        self.state = "done"
        return self._detect_changed_crop()

    def _detect_changed_crop(self) -> dict[str, Any]:
        import numpy as np

        try:
            # Use a border around the selected ROI as fixed cabinet/background
            # pixels for the MiDaS affine alignment required by DESIGN.md.
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
            interior = tuple(tuple(True for _ in range(self.width)) for _ in range(self.height))
            change = build_change_result(
                self.snapshot_a,
                aligned_b_snapshot,
                interior,
                near_is_positive=self.near_is_positive,
                object_change_threshold=self.object_change_threshold,
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

        source = self.snapshot_b if change.action == "put" else self.snapshot_a
        crop = source.rgb[change.crop.y:change.crop.y + change.crop.size,
                          change.crop.x:change.crop.x + change.crop.size]
        started = time.monotonic()
        detections = self.detector.detect(crop)
        latency_ms = (time.monotonic() - started) * 1000.0
        self.result = {
            "ok": True,
            "action": change.action,
            "signed_change": change.signed_change,
            "crop": change.crop,
            "detections": detections,
            "detector_calls": self.detector.calls,
            "detector_latency_ms": latency_ms,
        }
        print(
            "probe: action=%s signed_depth=%.4f crop=%s detector_backend=%s "
            "latency_ms=%.1f calls=%d detections=%s"
            % (
                change.action,
                change.signed_change,
                change.crop,
                self.detector.backend,
                latency_ms,
                self.detector.calls,
                detections,
            ),
            flush=True,
        )
        return self.result
