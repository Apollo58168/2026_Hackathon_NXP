#!/usr/bin/env python3
"""Mouse-operated drawer calibration and depth-change detection.

This is the depth-only vertical slice.  It deliberately stops after finding a
single, signed changed region: it does not load YOLO, write inventory, or use
voice/semantic services.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from core import (
    DrawerState,
    DrawerStateMachine,
    LayerCalibration,
    RecoverableRejection,
    affine_align,
    build_change_result,
    find_changed_components,
    robust_bottom_stat,
    stable_snapshot,
)
from depth_demo import DEFAULT_MODEL, colorize_depth, infer_depth, make_interpreter


@dataclass(frozen=True)
class Thresholds:
    # MiDaS is relative and normalized per frame.  These defaults tolerate
    # ordinary camera/inference jitter while remaining below motion_median.
    stable_median: float = 0.030
    stable_ratio: float = 0.08
    motion_median: float = 0.050
    motion_ratio: float = 0.02
    change_depth: float = 0.060
    direction: float = 0.030
    layer_tolerance: float = 0.080
    min_change_area: int = 16


@dataclass
class StableWindow:
    required: int
    rgb: list[np.ndarray] = field(default_factory=list)
    depth: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.rgb.clear()
        self.depth.clear()

    def append(self, frame: np.ndarray, depth: np.ndarray) -> bool:
        self.rgb.append(frame.copy())
        self.depth.append(depth.copy())
        if len(self.depth) > self.required:
            self.rgb.pop(0)
            self.depth.pop(0)
        return len(self.depth) == self.required

    @property
    def count(self) -> int:
        """Number of consecutive stable frames collected for the current capture."""
        return len(self.depth)

    def snapshot(self):
        return stable_snapshot(self.rgb, self.depth)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Robust per-frame MiDaS normalization, as required before alignment."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise RuntimeError("MiDaS returned no finite depth values")
    low, high = np.percentile(finite, (5, 95))
    return np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0).astype(np.float32)


def mask_metrics(current: np.ndarray, previous: np.ndarray, mask: np.ndarray, threshold: float) -> tuple[float, float]:
    values = np.abs(current[mask] - previous[mask])
    if values.size == 0:
        raise ValueError("mask contains no pixels")
    return float(np.median(values)), float(np.mean(values > threshold))


def automatic_drawer_mask(
    open_depth: np.ndarray,
    closed_depth: np.ndarray,
    thresholds: Thresholds,
) -> np.ndarray:
    """Derive a drawer interior mask from its stable open-vs-closed depth change.

    This deliberately uses a connected region, rather than a single depth pixel:
    MiDaS emits relative depth and a one-pixel extreme is not a dependable layer
    signature.  The resulting mask replaces operator-drawn selection for both
    layer calibration and later item-change detection.
    """
    # ``stable_snapshot`` intentionally returns immutable tuple matrices for
    # the state-machine core; convert here at the OpenCV/Numpy boundary.
    open_depth = np.asarray(open_depth, dtype=np.float32)
    closed_depth = np.asarray(closed_depth, dtype=np.float32)
    if open_depth.ndim != 2 or closed_depth.shape != open_depth.shape:
        raise ValueError("open and closed depth must be same-sized 2-D matrices")
    full_frame = tuple(tuple(True for _ in range(open_depth.shape[1])) for _ in range(open_depth.shape[0]))
    try:
        mask = find_changed_components(
            open_depth - closed_depth,
            full_frame,
            threshold=thresholds.motion_median,
            min_area=thresholds.min_change_area,
            # A drawer edge can be fragmented in monocular depth.  Merge nearby
            # fragments before requiring one unambiguous drawer region.
            merge_gap=12,
            morphology_radius=1,
        )
    except RecoverableRejection as error:
        raise RecoverableRejection(f"automatic_drawer_region_failed:{error}") from error
    return np.asarray(mask, dtype=bool)


class DrawerDepthRuntime:
    """Connect live metric or relative depth frames to the safe state-machine core."""

    def __init__(
        self,
        depth_shape: tuple[int, int],
        thresholds: Thresholds,
        *,
        metric_depth: bool = False,
        near_is_positive: bool = True,
    ) -> None:
        self.depth_shape = depth_shape
        self.thresholds = thresholds
        self.metric_depth = metric_depth
        self.near_is_positive = near_is_positive
        self.machine = DrawerStateMachine()
        self.phase = "startup"
        self.closed_depth: Optional[np.ndarray] = None
        self.background_mask: Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.window = StableWindow(3)
        self.closed_window = StableWindow(15)
        self.result: Optional[Any] = None
        self.message = "Click Initialize with every drawer closed. Drawer regions are found automatically."
        self.close_stable_frames = 0
        self.close_diagnostic: Optional[str] = None

    @property
    def calibrations(self) -> list[LayerCalibration]:
        return self.machine.calibrations

    def capture_progress(self) -> Optional[str]:
        """Human-visible progress for a currently running stable-frame capture."""
        if self.phase == "capture_closed":
            return f"CAPTURING CLOSED BASELINE  {self.closed_window.count}/{self.closed_window.required} stable frames"
        if self.phase == "capture_layer":
            layer_no = self.machine.current_layer or len(self.calibrations) + 1
            return f"CAPTURING LAYER {layer_no}  {self.window.count}/{self.window.required} stable frames"
        return None

    def inference_status(self) -> Optional[str]:
        """The visible state of the repeatable open/close inference loop."""
        if self.phase == "ready":
            return "INFERENCE MODE: READY — open one calibrated drawer"
        if self.phase == "opening":
            return "INFERENCE MODE: identifying the open drawer"
        if self.phase == "wait_change":
            return "INFERENCE MODE: drawer identified — waiting for item change or close"
        if self.phase == "item_motion":
            return "INFERENCE MODE: analysing depth change"
        if self.phase == "wait_close":
            detail = f"  {self.close_diagnostic}" if self.close_diagnostic else ""
            return f"INFERENCE MODE: close drawer to reset  {self.close_stable_frames}/3 stable frames{detail}"
        return None

    def begin_initialization(self) -> None:
        if self.phase != "startup":
            return
        self.machine.choose_clear()
        self.phase = "capture_closed"
        self.closed_window.reset()
        self.last_depth = None
        self.message = "Hold all drawers closed: capturing 15 stable depth frames."

    def request_layer_capture(self) -> None:
        if self.machine.state != DrawerState.INITIALIZING_WAIT_OPEN:
            self.message = "Close the current drawer and click Next before capturing another layer."
            return
        layer_no = len(self.calibrations) + 1
        self.machine.init_open(layer_no)
        self.phase = "capture_layer"
        self.window.reset()
        self.message = f"Layer {layer_no}: waiting for 3 stable frames. Keep hands out of the drawer."

    def next_layer(self) -> None:
        if self.machine.state != DrawerState.INITIALIZING_WAIT_CLOSE:
            self.message = "Capture a layer before moving on."
            return
        self.machine.init_close()
        self.phase = "initializing"
        self.message = f"Close the drawer. Then open layer {len(self.calibrations) + 1} and click Capture."

    def finish_initialization(self) -> None:
        if self.machine.state != DrawerState.INITIALIZING_WAIT_OPEN:
            self.message = "Close the last drawer and click Next before finishing."
            return
        try:
            self.machine.finish_initialization()
        except RecoverableRejection as error:
            self.message = str(error)
            return
        union = np.zeros(self.depth_shape, dtype=bool)
        for calibration in self.calibrations:
            union |= np.asarray(calibration.interior_mask, dtype=bool)
        self.background_mask = ~union
        if int(self.background_mask.sum()) < 2:
            self.machine.fatal_error("background_mask_empty")
            self.phase = "error"
            self.message = "Automatic drawer regions leave no fixed cabinet background; restart with a wider cabinet view."
            return
        self.phase = "ready"
        self.last_depth = self.closed_depth
        self.message = "INFERENCE READY: open one calibrated drawer and wait for it to settle."

    def _stable(self, current: np.ndarray, previous: np.ndarray, mask: np.ndarray) -> bool:
        median_delta, ratio = mask_metrics(current, previous, mask, self.thresholds.stable_median)
        return median_delta <= self.thresholds.stable_median and ratio <= self.thresholds.stable_ratio

    def _motion(self, current: np.ndarray, previous: np.ndarray, mask: np.ndarray) -> bool:
        median_delta, ratio = mask_metrics(current, previous, mask, self.thresholds.motion_median)
        return median_delta >= self.thresholds.motion_median or ratio >= self.thresholds.motion_ratio

    def _matches_closed(self, current: np.ndarray, mask: np.ndarray) -> bool:
        assert self.closed_depth is not None
        median_delta, ratio = mask_metrics(current, self.closed_depth, mask, self.thresholds.stable_median)
        return median_delta <= self.thresholds.stable_median and ratio <= self.thresholds.stable_ratio

    def _closed_reset_ready(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        mask: np.ndarray,
    ) -> bool:
        """Return whether a stable frame has returned from open depth to closed.

        Relative-depth predictions can drift after frame normalization even
        after background alignment. In addition to the strict closed-baseline test,
        accept a stable frame that is substantially closer to the closed
        baseline than the recorded open snapshot.  This cannot accept a drawer
        left open: that frame remains closest to Snapshot A.
        """
        assert self.closed_depth is not None
        close_delta, close_ratio = mask_metrics(current, self.closed_depth, mask, self.thresholds.stable_median)
        stable = self._stable(current, previous, mask)
        strict_match = close_delta <= self.thresholds.stable_median and close_ratio <= self.thresholds.stable_ratio
        open_delta: Optional[float] = None
        if self.machine.snapshot_a is not None:
            open_delta, _ = mask_metrics(
                current,
                np.asarray(self.machine.snapshot_a.depth, dtype=np.float32),
                mask,
                self.thresholds.stable_median,
            )
        relative_match = open_delta is not None and open_delta > 1e-6 and close_delta <= open_delta * 0.65
        self.close_diagnostic = (
            f"closed Δ={close_delta:.3f} ({'strict' if strict_match else 'relative' if relative_match else 'open'})"
            + (f", open Δ={open_delta:.3f}" if open_delta is not None else "")
        )
        return stable and (strict_match or relative_match)

    def _capture_stable(self, frame: np.ndarray, depth: np.ndarray, stable: bool) -> Optional[Any]:
        if not stable:
            self.window.reset()
            return None
        if self.window.append(frame, depth):
            snapshot = self.window.snapshot()
            self.window.reset()
            return snapshot
        return None

    def process(self, frame: np.ndarray, raw_depth: np.ndarray) -> None:
        depth = np.asarray(raw_depth, dtype=np.float32) if self.metric_depth else normalize_depth(raw_depth)
        if self.phase == "startup" or self.phase == "error":
            return
        if self.phase == "capture_closed":
            if self.last_depth is None or self._stable(depth, self.last_depth, np.ones(self.depth_shape, dtype=bool)):
                if self.closed_window.append(frame, depth):
                    self.closed_depth = np.asarray(self.closed_window.snapshot().depth, dtype=np.float32)
                    self.closed_window.reset()
                    self.machine.begin_initialization()
                    self.phase = "initializing"
                    self.message = "Open layer 1, then click Capture. Its drawer region is found automatically."
            else:
                self.closed_window.reset()
            self.last_depth = depth
            return
        if self.phase == "capture_layer":
            full_frame = np.ones(self.depth_shape, dtype=bool)
            stable = self.last_depth is not None and self._stable(depth, self.last_depth, full_frame)
            snapshot = self._capture_stable(frame, depth, stable)
            if snapshot is not None:
                assert self.closed_depth is not None
                try:
                    mask = automatic_drawer_mask(snapshot.depth, self.closed_depth, self.thresholds)
                    baseline = robust_bottom_stat(
                        snapshot.depth,
                        mask,
                        farthest_is_larger=not self.near_is_positive,
                    )
                except RecoverableRejection as error:
                    self.message = f"Layer capture rejected: {error}. Keep this drawer open; it will retry automatically."
                    self.last_depth = depth
                    return
                calibration = LayerCalibration(
                    layer_no=len(self.calibrations) + 1,
                    bottom_depth_baseline=baseline,
                    drawer_mask=tuple(tuple(bool(value) for value in row) for row in mask),
                    interior_mask=tuple(tuple(bool(value) for value in row) for row in mask),
                    open_threshold=self.thresholds.motion_median,
                    close_threshold=self.thresholds.stable_median,
                )
                self.machine.init_stable(calibration)
                self.phase = "initializing"
                self.message = f"Layer {calibration.layer_no} recorded automatically. Close it, then click Next."
            self.last_depth = depth
            return
        # Between the closed baseline and each layer capture, the operator is
        # deliberately opening/closing a drawer.  Calibration is incomplete at
        # this point by design, so keep previewing instead of treating it as a
        # runtime calibration failure.
        if self.phase == "initializing":
            return
        if self.closed_depth is None or self.background_mask is None:
            self.phase = "error"
            self.machine.fatal_error("missing_calibration")
            self.message = "Calibration data is incomplete. Restart initialization."
            return
        # Metric Small already estimates metres.  Do not fit it to a closed
        # background: a visually uniform cabinet can make the affine fit
        # collapse valid open-drawer depth to a constant. Relative models keep
        # the original background-based affine alignment path.
        aligned = (
            depth
            if self.metric_depth
            else np.asarray(affine_align(depth, self.closed_depth, self.background_mask), dtype=np.float32)
        )
        previous = self.last_depth
        self.last_depth = aligned
        if previous is None:
            return
        drawer_mask = np.zeros(self.depth_shape, dtype=bool)
        for calibration in self.calibrations:
            drawer_mask |= np.asarray(calibration.drawer_mask, dtype=bool)
        if self.machine.state == DrawerState.READY and self.phase == "ready":
            if self._motion(aligned, previous, drawer_mask):
                self.phase = "opening"
                self.window.reset()
                self.message = "Drawer motion detected; waiting for 3 stable frames."
            return
        if self.phase == "opening":
            snapshot = self._capture_stable(frame, aligned, self._stable(aligned, previous, drawer_mask))
            if snapshot is None:
                return
            try:
                ranked = sorted(
                    (
                        abs(
                            robust_bottom_stat(
                                snapshot.depth,
                                calibration.interior_mask,
                                farthest_is_larger=not self.near_is_positive,
                            )
                            - calibration.bottom_depth_baseline
                        ),
                        calibration.layer_no,
                    )
                    for calibration in self.calibrations
                )
                if ranked[0][0] > self.thresholds.layer_tolerance or (
                    len(ranked) > 1 and ranked[0][0] == ranked[1][0]
                ):
                    raise RecoverableRejection("drawer_unknown")
                layer_no = ranked[0][1]
                self.machine.start_drawer(layer_no)
                self.machine.snapshot_a_stable(snapshot, layer_no)
                self.phase = "wait_change"
                self.message = f"Layer {layer_no} open. Put in or take out one item, then wait."
            except RecoverableRejection as error:
                self.machine.start_drawer(0)
                self.phase = "wait_close"
                self.message = f"Drawer rejected: {error}. Close all drawers."
            return
        if self.machine.state == DrawerState.WAIT_ITEM_CHANGE and self.phase == "wait_change":
            active = next(item for item in self.calibrations if item.layer_no == self.machine.current_layer)
            active_mask = np.asarray(active.interior_mask, dtype=bool)
            if self._motion(aligned, previous, active_mask):
                self.machine.item_change_motion()
                self.phase = "item_motion"
                self.window.reset()
                self.message = "Item motion detected; waiting for 3 stable frames."
            return
        if self.phase == "item_motion":
            active = next(item for item in self.calibrations if item.layer_no == self.machine.current_layer)
            active_mask = np.asarray(active.interior_mask, dtype=bool)
            snapshot = self._capture_stable(frame, aligned, self._stable(aligned, previous, active_mask))
            if snapshot is None:
                return
            # Closing a drawer without moving an item is a valid replay action,
            # not an item change. Return directly to the inference loop.
            if self._matches_closed(np.asarray(snapshot.depth, dtype=np.float32), active_mask):
                self.machine.reject("drawer_closed_without_item_change")
                self.machine.drawer_closed()
                self.phase = "ready"
                self.result = None
                self.close_diagnostic = None
                self.message = "INFERENCE READY: drawer closed; open a calibrated drawer to replay."
                return
            self.machine.snapshot_b_stable(snapshot)
            try:
                assert self.machine.snapshot_a is not None
                self.result = build_change_result(
                    self.machine.snapshot_a,
                    snapshot,
                    active_mask,
                    near_is_positive=self.near_is_positive,
                    object_change_threshold=self.thresholds.change_depth,
                    direction_threshold=self.thresholds.direction,
                    min_change_area=self.thresholds.min_change_area,
                    component_merge_gap=2,
                    morphology_radius=1,
                )
                self.message = f"Change detected: {self.result.action}; close drawer to continue."
                self.machine.reject("depth_change_detected_no_detector")
            except RecoverableRejection as error:
                self.result = None
                self.machine.reject(str(error))
                self.message = f"Change rejected: {error}; close drawer to continue."
            self.phase = "wait_close"
            self.close_stable_frames = 0
            return
        if self.machine.state == DrawerState.WAIT_DRAWER_CLOSE:
            active_mask = drawer_mask if self.machine.current_layer is None else np.asarray(
                next(item for item in self.calibrations if item.layer_no == self.machine.current_layer).interior_mask,
                dtype=bool,
            )
            if self._closed_reset_ready(aligned, previous, active_mask):
                self.close_stable_frames += 1
                if self.close_stable_frames >= 3:
                    self.machine.drawer_closed()
                    self.phase = "ready"
                    self.result = None
                    self.close_diagnostic = None
                    self.message = "INFERENCE READY: drawer closed; open a calibrated drawer to replay."
            else:
                self.close_stable_frames = 0


BUTTONS = {
    "initialize": (10, 10, 155, 38),
    "capture": (175, 10, 145, 38),
    "next": (330, 10, 110, 38),
    "finish": (450, 10, 110, 38),
    "restart": (570, 10, 110, 38),
}


def inside(point: tuple[int, int], rectangle: tuple[int, int, int, int]) -> bool:
    x, y = point
    left, top, width, height = rectangle
    return left <= x < left + width and top <= y < top + height


def draw_button(image: np.ndarray, name: str, mode: str) -> None:
    x, y, width, height = BUTTONS[name]
    colour = {
        "enabled": (54, 128, 54),
        "capturing": (0, 180, 255),
        "disabled": (70, 70, 70),
    }[mode]
    cv2.rectangle(image, (x, y), (x + width, y + height), colour, cv2.FILLED)
    cv2.putText(image, name.title(), (x + 8, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def button_mode(name: str, runtime: DrawerDepthRuntime) -> str:
    if name == "restart":
        return "enabled"
    if name == "initialize":
        return "enabled" if runtime.phase == "startup" else "disabled"
    if name == "capture":
        if runtime.phase == "capture_layer":
            return "capturing"
        return "enabled" if runtime.machine.state == DrawerState.INITIALIZING_WAIT_OPEN else "disabled"
    if name == "next":
        return "enabled" if runtime.machine.state == DrawerState.INITIALIZING_WAIT_CLOSE else "disabled"
    if name == "finish":
        return "enabled" if runtime.machine.state == DrawerState.INITIALIZING_WAIT_OPEN and runtime.calibrations else "disabled"
    raise ValueError(f"unknown button: {name}")


def render(frame: np.ndarray, depth: np.ndarray, runtime: DrawerDepthRuntime, fps: float, latency_ms: float) -> np.ndarray:
    view = frame.copy()
    height, width = frame.shape[:2]
    depth_height, depth_width = runtime.depth_shape
    scale_x, scale_y = width / depth_width, height / depth_height
    for calibration in runtime.calibrations:
        mask = np.asarray(calibration.interior_mask, dtype=bool)
        ys, xs = np.where(mask)
        if xs.size:
            cv2.rectangle(view, (int(xs.min() * scale_x), int(ys.min() * scale_y)), (int((xs.max() + 1) * scale_x), int((ys.max() + 1) * scale_y)), (0, 220, 0), 2)
            cv2.putText(view, f"L{calibration.layer_no}", (int(xs.min() * scale_x), max(60, int(ys.min() * scale_y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    if runtime.result is not None:
        component = runtime.result.component
        cv2.rectangle(view, (int(component.x0 * scale_x), int(component.y0 * scale_y)), (int((component.x1 + 1) * scale_x), int((component.y1 + 1) * scale_y)), (0, 0, 255), 2)
    progress = runtime.capture_progress()
    banner = progress or runtime.inference_status()
    if banner is not None:
        colour = (0, 180, 255) if progress is not None else (54, 128, 54)
        cv2.rectangle(view, (0, 0), (width, 42), colour, cv2.FILLED)
        cv2.putText(view, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    depth_view = colorize_depth(depth, (width, height), near_is_positive=runtime.near_is_positive)
    control = np.zeros((58, width * 2, 3), dtype=np.uint8)
    state = f"{runtime.machine.state.value} | {latency_ms:.0f} ms | {fps:.1f} FPS"
    cv2.putText(control, state, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    for name in BUTTONS:
        draw_button(control, name, button_mode(name, runtime))
    status = np.zeros((45, width * 2, 3), dtype=np.uint8)
    cv2.putText(status, runtime.message[:160], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((status, np.hstack((view, depth_view)), control))


def run(args: argparse.Namespace) -> None:
    interpreter, input_details, output_details = make_interpreter(args.model)
    model_shape = tuple(int(value) for value in input_details["shape"])
    runtime = DrawerDepthRuntime(
        (model_shape[1], model_shape[2]),
        Thresholds(
            stable_median=args.stable_median,
            stable_ratio=args.stable_ratio,
            motion_median=args.motion_median,
            motion_ratio=args.motion_ratio,
            change_depth=args.change_depth,
            direction=args.direction,
            layer_tolerance=args.layer_tolerance,
            min_change_area=args.min_change_area,
        ),
        metric_depth=bool(input_details["metric_depth"]),
        near_is_positive=bool(input_details["near_is_positive"]),
    )
    backend = cv2.CAP_AVFOUNDATION if args.avfoundation else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera}")
    window_name = "Smart Drawer: RGB | Metric Depth"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    frame_size: Optional[tuple[int, int]] = None

    def mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if frame_size is None:
            return
        _, frame_height = frame_size
        control_y = y - 45 - frame_height
        if event == cv2.EVENT_LBUTTONDOWN and y >= 45 + frame_height:
            point = (x, control_y)
            if inside(point, BUTTONS["restart"]):
                runtime.__init__(
                    runtime.depth_shape,
                    runtime.thresholds,
                    metric_depth=runtime.metric_depth,
                    near_is_positive=runtime.near_is_positive,
                )
            elif inside(point, BUTTONS["initialize"]):
                runtime.begin_initialization()
            elif inside(point, BUTTONS["capture"]):
                runtime.request_layer_capture()
            elif inside(point, BUTTONS["next"]):
                runtime.next_layer()
            elif inside(point, BUTTONS["finish"]):
                runtime.finish_initialization()

    if not args.no_display:
        cv2.setMouseCallback(window_name, mouse)
    frames, started = 0, time.perf_counter()
    try:
        while args.max_frames == 0 or frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame capture failed")
            frame_size = (frame.shape[1], frame.shape[0])
            inference_started = time.perf_counter()
            raw_depth = infer_depth(interpreter, input_details, output_details, frame)
            latency_ms = (time.perf_counter() - inference_started) * 1000
            runtime.process(frame, raw_depth)
            frames += 1
            if args.no_display:
                continue
            cv2.imshow(window_name, render(frame, raw_depth, runtime, frames / max(time.perf_counter() - started, 1e-6), latency_ms))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    print(f"processed_frames={frames} average_fps={frames / max(elapsed, 1e-6):.2f} state={runtime.machine.state.value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mouse-operated, depth-only Smart Drawer runtime")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--avfoundation", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until q/Esc")
    parser.add_argument("--no-display", action="store_true", help="run capture/inference without the interactive window")
    parser.add_argument("--stable-median", type=float, default=Thresholds.stable_median)
    parser.add_argument("--stable-ratio", type=float, default=Thresholds.stable_ratio)
    parser.add_argument("--motion-median", type=float, default=Thresholds.motion_median)
    parser.add_argument("--motion-ratio", type=float, default=Thresholds.motion_ratio)
    parser.add_argument("--change-depth", type=float, default=Thresholds.change_depth)
    parser.add_argument("--direction", type=float, default=Thresholds.direction)
    parser.add_argument("--layer-tolerance", type=float, default=Thresholds.layer_tolerance)
    parser.add_argument("--min-change-area", type=int, default=Thresholds.min_change_area)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
