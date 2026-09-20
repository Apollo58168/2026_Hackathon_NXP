"""Two-layer, depth-only initialization for the i.MX93 camera adapter.

The adapter first records the closed cabinet.  The user then starts layer 1
and layer 2 individually; each layer waits for drawer motion, requires three stable depth frames, derives its ROI from the closed/open difference, and
requires the drawer to be closed before the next button can proceed.  No fixed
pixel ROI is required.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from .core import LayerCalibration, robust_bottom_stat
from .depth_denoise import denoise_relative_depth


class TwoLayerCalibrator:
    """Automatic two-layer initializer driven by successive MiDaS frames."""

    def __init__(
        self,
        *,
        stable_frame_count: int = 3,
        near_is_positive: bool = True,
        bilateral_diameter: int = 5,
        bilateral_sigma: float = 0.08,
    ) -> None:
        self.np = __import__("numpy")
        self.cv2 = __import__("cv2")
        self.stable_frame_count = int(stable_frame_count)
        if self.stable_frame_count < 3:
            raise ValueError("stable_frame_count must be at least 3")
        self.near_is_positive = near_is_positive
        self.bilateral_diameter = int(bilateral_diameter)
        self.bilateral_sigma = float(bilateral_sigma)
        self.state = "idle"
        self.message = "INITIALIZATION"
        self.closed_depth: Optional[Any] = None
        self.calibrations: list[LayerCalibration] = []
        self._pending: list[tuple[float, LayerCalibration]] = []
        self._previous: Optional[Any] = None
        self._close_since: Optional[float] = None
        self._stable_frames: list[Any] = []
        self._closed_frames: list[Any] = []
        self.stability_threshold = 0.015
        self.open_threshold = 0.04
        self.close_threshold = 0.025
        self.change_threshold = 0.10
        self.noise = 0.0

    @property
    def active(self) -> bool:
        return self.state in (
            "capturing_closed",
            "waiting_open_1",
            "capturing_layer_1",
            "waiting_open_2",
            "capturing_layer_2",
        )

    @property
    def completed(self) -> bool:
        return self.state == "complete"

    def start(self) -> None:
        self.state = "capturing_closed"
        self.message = "INITIALIZING: keep both drawers closed"
        self.closed_depth = None
        self.calibrations = []
        self._pending = []
        self._previous = None
        self._close_since = None
        self._stable_frames = []
        self._closed_frames = []

    def is_closed(self, depth: Any) -> bool:
        if self.closed_depth is None:
            return False
        return self._distance_from_closed(self._normalize(depth)) <= self.close_threshold

    def begin_layer(self, layer: int, depth: Optional[Any] = None) -> bool:
        expected_state = "ready_layer_%d" % layer
        if self.state != expected_state:
            self.message = "WAITING FOR %s" % self.message
            return False
        if depth is not None and not self.is_closed(depth):
            self.message = "CLOSE LAYER %d BEFORE STARTING LAYER %d" % (layer - 1, layer)
            return False
        self.state = "waiting_open_%d" % layer
        self._previous = None
        self._stable_frames.clear()
        self.message = "LAYER %d INIT: open layer %d" % (layer, layer)
        return True

    def finish(self, depth: Optional[Any] = None) -> bool:
        if self.state != "ready_finish":
            self.message = "WAITING FOR LAYER 2 DEPTH"
            return False
        if depth is not None and not self.is_closed(depth):
            self.message = "CLOSE LAYER 2 BEFORE FINISH INIT"
            return False
        self._finish_all()
        return True

    def _normalize(self, depth: Any) -> Any:
        return denoise_relative_depth(depth, self.bilateral_diameter, self.bilateral_sigma)

    def _align(self, source: Any) -> Any:
        """Robustly align source depth to the closed background."""
        reference = self.closed_depth
        if reference is None:
            return source
        x = source.reshape(-1).astype(self.np.float64)
        y = reference.reshape(-1).astype(self.np.float64)
        keep = self.np.isfinite(x) & self.np.isfinite(y)
        scale, offset = 1.0, 0.0
        for _ in range(3):
            if int(keep.sum()) < 2:
                break
            design = self.np.column_stack((x[keep], self.np.ones(int(keep.sum()))))
            scale, offset = self.np.linalg.lstsq(design, y[keep], rcond=None)[0]
            scale = float(self.np.clip(scale, 0.5, 2.0))
            offset = float(offset)
            residual = self.np.abs(y - (scale * x + offset))
            valid = residual[keep]
            if not len(valid):
                break
            cutoff = float(self.np.percentile(valid, 70))
            keep = keep & (residual <= max(cutoff, 0.01))
        return (scale * source + offset).astype(self.np.float32)

    def _distance_from_closed(self, depth: Any) -> float:
        aligned = self._align(depth)
        return float(self.np.mean(self.np.abs(aligned - self.closed_depth)))

    @staticmethod
    def _groups(values: Any, gap: int = 4) -> list[tuple[int, int]]:
        if len(values) == 0:
            return []
        groups = []
        start = previous = int(values[0])
        for value in values[1:]:
            value = int(value)
            if value - previous > gap + 1:
                groups.append((start, previous))
                start = value
            previous = value
        groups.append((start, previous))
        return groups

    def _derive_calibration(self, depth: Any, provisional_layer: int) -> tuple[float, LayerCalibration]:
        aligned = self._align(depth)
        difference = self.np.abs(aligned - self.closed_depth)
        raw_mask = (difference > self.change_threshold).astype(self.np.uint8) * 255
        kernel = self.np.ones((5, 5), self.np.uint8)
        mask = self.cv2.morphologyEx(raw_mask, self.cv2.MORPH_OPEN, kernel)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)
        row_score = (mask > 0).mean(axis=1)
        active_rows = self.np.flatnonzero(row_score > 0.02)
        if len(active_rows) == 0:
            active_rows = self.np.flatnonzero(mask.any(axis=1))
        groups = self._groups(active_rows)
        if not groups:
            raise ValueError("drawer ROI could not be separated from closed depth")
        y0, y1 = max(groups, key=lambda item: float(row_score[item[0]:item[1] + 1].sum()))
        region = mask[y0:y1 + 1] > 0
        column_score = region.mean(axis=0)
        active_columns = self.np.flatnonzero(column_score > 0.02)
        if len(active_columns) == 0:
            active_columns = self.np.flatnonzero(region.any(axis=0))
        if len(active_columns) == 0:
            raise ValueError("drawer ROI has no changed pixels")
        x0, x1 = int(active_columns[0]), int(active_columns[-1])
        height, width = difference.shape
        pad_x = max(2, int(round((x1 - x0 + 1) * 0.10)))
        pad_y = max(2, int(round((y1 - y0 + 1) * 0.10)))
        x0, x1 = max(0, x0 - pad_x), min(width - 1, x1 + pad_x)
        y0, y1 = max(0, y0 - pad_y), min(height - 1, y1 + pad_y)
        interior = tuple(
            tuple(y0 <= y <= y1 and x0 <= x <= x1 for x in range(width))
            for y in range(height)
        )
        drawer_mask = interior
        bottom = robust_bottom_stat(
            depth,
            interior,
            farthest_is_larger=not self.near_is_positive,
        )
        calibration = LayerCalibration(
            layer_no=provisional_layer,
            bottom_depth_baseline=bottom,
            drawer_mask=drawer_mask,
            interior_mask=interior,
            open_threshold=self.open_threshold,
            close_threshold=self.close_threshold,
        )
        return (y0 + y1) / 2.0, calibration

    def _finish_closed_baseline(self) -> None:
        stack = self.np.stack(self._closed_frames)
        self.closed_depth = self.np.median(stack, axis=0).astype(self.np.float32)
        noise_values = [float(self.np.mean(self.np.abs(frame - self.closed_depth))) for frame in stack]
        self.noise = max(float(self.np.percentile(noise_values, 90)), 0.001)
        self.stability_threshold = max(0.015, self.noise * 4.0)
        self.open_threshold = max(0.04, self.noise * 8.0)
        self.close_threshold = max(0.025, self.noise * 5.0)
        self.change_threshold = max(0.10, self.noise * 10.0)
        self._closed_frames.clear()
        self._stable_frames.clear()
        self._previous = None
        self.state = "ready_layer_1"
        self.message = "CLOSED BASELINE READY: press LAYER 1 INIT"

    def _finish_layer(self) -> None:
        if len(self._stable_frames) < 3:
            raise ValueError("not enough stable depth frames")
        depth = self.np.median(self.np.stack(self._stable_frames[-15:]), axis=0).astype(self.np.float32)
        provisional = len(self._pending) + 1
        center_y, calibration = self._derive_calibration(depth, provisional)
        self._pending.append((center_y, calibration))
        self._stable_frames.clear()
        self._close_since = None
        self._previous = depth
        self.state = "ready_layer_2" if provisional == 1 else "ready_finish"
        if provisional == 1:
            self.message = "LAYER 1 CAPTURED: close it, then press LAYER 2 INIT"
        else:
            self.message = "LAYER 2 CAPTURED: close it, then press FINISH INIT"

    def _finish_all(self) -> None:
        ordered = sorted(self._pending, key=lambda item: item[0])
        self.calibrations = [replace(calibration, layer_no=index + 1)
                             for index, (_center, calibration) in enumerate(ordered)]
        if len(self.calibrations) != 2:
            raise ValueError("expected exactly two drawer layers")
        self.state = "complete"
        self.message = "INITIALIZATION COMPLETE: 2 LAYERS READY"

    def feed(self, depth: Any) -> None:
        if not self.active:
            return
        try:
            current = self._normalize(depth)
            previous = self._previous
            self._previous = current.copy()
            if self.state == "capturing_closed":
                if previous is None or float(self.np.mean(self.np.abs(current - previous))) <= self.stability_threshold:
                    self._closed_frames.append(current.copy())
                else:
                    self._closed_frames.clear()
                if len(self._closed_frames) >= 15:
                    self._finish_closed_baseline()
                return

            if self.state in ("waiting_open_1", "waiting_open_2"):
                if self._distance_from_closed(current) > self.open_threshold:
                    layer = 1 if self.state.endswith("1") else 2
                    self.state = "capturing_layer_%d" % layer
                    self.message = "LAYER %d OPEN: keep it still for 3 stable frames" % layer
                    self._stable_frames = [current.copy()]
                return

            if self.state in ("capturing_layer_1", "capturing_layer_2"):
                if previous is None or float(self.np.mean(self.np.abs(current - previous))) <= self.stability_threshold:
                    self._stable_frames.append(current.copy())
                    if len(self._stable_frames) >= self.stable_frame_count:
                        self._finish_layer()
                else:
                    self._stable_frames.clear()
                return

        except Exception as error:
            self.state = "error"
            self.message = "INITIALIZATION ERROR: %s" % error
