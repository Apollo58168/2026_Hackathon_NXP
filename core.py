"""Pure SmartDrawer transaction logic used by the PC simulator and board app.

This module deliberately has no camera, TFLite, GUI, or cloud dependency.  The
real adapters can feed it the same stable snapshots and detector results.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt
from statistics import median
from typing import Any, Iterable, Mapping, Optional, Sequence


class DrawerState(str, Enum):
    STARTUP_CHOICE = "startup_choice"
    RESUME_VALIDATION = "resume_validation"
    NEEDS_INITIALIZATION = "needs_initialization"
    INITIALIZING_WAIT_OPEN = "initializing_wait_open"
    INITIALIZING_OPEN_STABLE = "initializing_open_stable"
    INITIALIZING_WAIT_CLOSE = "initializing_wait_close"
    READY = "ready"
    FIRST_CHANGE_MOTION = "first_change_motion"
    SNAPSHOT_A_STABLE = "snapshot_a_stable"
    WAIT_ITEM_CHANGE = "wait_item_change"
    SECOND_CHANGE_MOTION = "second_change_motion"
    SNAPSHOT_B_STABLE = "snapshot_b_stable"
    DETECTING_CHANGED_CROP = "detecting_changed_crop"
    CANDIDATE_CONFIRMED = "candidate_confirmed"
    WAIT_DRAWER_CLOSE = "wait_drawer_close"
    COMMITTING = "committing"
    ERROR = "error"


class SmartDrawerError(Exception):
    """Base class for expected, safe-to-handle SmartDrawer errors."""


class TransitionError(SmartDrawerError):
    pass


class RecoverableRejection(SmartDrawerError):
    pass


class FatalDependencyError(SmartDrawerError):
    pass


NumberMatrix = Sequence[Sequence[float]]
BoolMatrix = Sequence[Sequence[bool]]


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    size: int
    source: str


@dataclass(frozen=True)
class Snapshot:
    rgb: Any
    depth: tuple[tuple[float, ...], ...]
    frame_ids: tuple[Any, ...]


@dataclass(frozen=True)
class Component:
    pixels: frozenset[tuple[int, int]]
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def area(self) -> int:
        return len(self.pixels)


@dataclass(frozen=True)
class ChangeResult:
    delta: tuple[tuple[float, ...], ...]
    mask: tuple[tuple[bool, ...], ...]
    component: Component
    signed_change: float
    action: str
    crop: Crop


@dataclass(frozen=True)
class Detection:
    class_id: int
    canonical_label: str
    confidence: float


@dataclass(frozen=True)
class Candidate:
    layer_no: int
    action: str
    class_id: int
    canonical_label: str
    confidence: float
    signed_depth_change: float
    crop: Crop
    yolo_calls: int = 1


@dataclass(frozen=True)
class LayerCalibration:
    layer_no: int
    bottom_depth_baseline: float
    drawer_mask: tuple[tuple[bool, ...], ...]
    interior_mask: tuple[tuple[bool, ...], ...]
    open_threshold: float
    close_threshold: float


def _shape(matrix: Sequence[Sequence[Any]]) -> tuple[int, int]:
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    if not height or not width or any(len(row) != width for row in matrix):
        raise ValueError("matrix must be non-empty and rectangular")
    return height, width


def _same_shape(a: Sequence[Sequence[Any]], b: Sequence[Sequence[Any]]) -> None:
    if _shape(a) != _shape(b):
        raise ValueError("matrices must have the same shape")


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= q <= 100:
        raise ValueError("percentile must be in [0, 100]")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def matrix_median(frames: Sequence[NumberMatrix]) -> tuple[tuple[float, ...], ...]:
    if not frames:
        raise ValueError("at least one frame is required")
    height, width = _shape(frames[0])
    if any(_shape(frame) != (height, width) for frame in frames):
        raise ValueError("all frames must have the same shape")
    return tuple(
        tuple(
            float(median(float(frame[y][x]) for frame in frames))
            for x in range(width)
        )
        for y in range(height)
    )


def stable_snapshot(
    rgb_frames: Sequence[Any], depth_frames: Sequence[NumberMatrix]
) -> Snapshot:
    """Build a snapshot only after a three-frame stable window.

    The middle RGB frame is paired with the pixel-wise median depth, matching
    the design contract while keeping the adapter responsible for frame I/O.
    """
    if len(rgb_frames) != len(depth_frames) or len(depth_frames) < 3:
        raise ValueError("a stable snapshot needs at least three paired frames")
    depth = matrix_median(depth_frames)
    middle = len(depth_frames) // 2
    frame_ids = tuple(getattr(frame, "frame_id", index) for index, frame in enumerate(rgb_frames))
    return Snapshot(rgb=rgb_frames[middle], depth=depth, frame_ids=frame_ids)


def _selected_values(matrix: NumberMatrix, mask: Optional[BoolMatrix]) -> list[float]:
    height, width = _shape(matrix)
    if mask is not None:
        _same_shape(matrix, mask)
    return [
        float(matrix[y][x])
        for y in range(height)
        for x in range(width)
        if mask is None or bool(mask[y][x])
    ]


def robust_bottom_stat(
    depth: NumberMatrix,
    mask: Optional[BoolMatrix] = None,
    *,
    farthest_is_larger: bool = True,
    farthest_percentile: float = 90.0,
) -> float:
    """Return a robust statistic from the farthest percentile band.

    A band median is intentional: a single extreme pixel must not select a
    drawer layer.  ``farthest_is_larger`` is frozen by model calibration.
    """
    values = _selected_values(depth, mask)
    if not values:
        raise ValueError("bottom mask contains no depth values")
    cutoff = percentile(values, farthest_percentile if farthest_is_larger else 100 - farthest_percentile)
    if farthest_is_larger:
        band = [value for value in values if value >= cutoff]
    else:
        band = [value for value in values if value <= cutoff]
    return float(median(band))


def affine_align(
    source: NumberMatrix,
    reference: NumberMatrix,
    background_mask: BoolMatrix,
) -> tuple[tuple[float, ...], ...]:
    """Align source depth to reference using fixed cabinet background pixels."""
    _same_shape(source, reference)
    _same_shape(source, background_mask)
    pairs = [
        (float(source[y][x]), float(reference[y][x]))
        for y in range(len(source))
        for x in range(len(source[0]))
        if background_mask[y][x]
    ]
    if len(pairs) < 2:
        raise RecoverableRejection("depth_alignment_failed")
    source_mean = sum(pair[0] for pair in pairs) / len(pairs)
    reference_mean = sum(pair[1] for pair in pairs) / len(pairs)
    variance = sum((pair[0] - source_mean) ** 2 for pair in pairs)
    if variance < 1e-12:
        scale = 1.0
    else:
        covariance = sum((s - source_mean) * (r - reference_mean) for s, r in pairs)
        scale = covariance / variance
    offset = reference_mean - scale * source_mean
    return tuple(
        tuple(scale * float(source[y][x]) + offset for x in range(len(source[0])))
        for y in range(len(source))
    )


def subtract_depth(b: NumberMatrix, a: NumberMatrix) -> tuple[tuple[float, ...], ...]:
    _same_shape(a, b)
    return tuple(
        tuple(float(b[y][x]) - float(a[y][x]) for x in range(len(a[0])))
        for y in range(len(a))
    )


def _neighbours(y: int, x: int, height: int, width: int) -> Iterable[tuple[int, int]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dx or dy):
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                yield ny, nx


def _erode(mask: BoolMatrix, radius: int) -> tuple[tuple[bool, ...], ...]:
    height, width = _shape(mask)
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            keep = True
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < height and 0 <= nx < width and mask[ny][nx]):
                        keep = False
                        break
                if not keep:
                    break
            row.append(keep)
        result.append(tuple(row))
    return tuple(result)


def _dilate(mask: BoolMatrix, radius: int) -> tuple[tuple[bool, ...], ...]:
    height, width = _shape(mask)
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            keep = any(
                0 <= y + dy < height
                and 0 <= x + dx < width
                and mask[y + dy][x + dx]
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
                if abs(dx) + abs(dy) <= radius
            )
            row.append(keep)
        result.append(tuple(row))
    return tuple(result)


def clean_mask(mask: BoolMatrix, morphology_radius: int = 1) -> tuple[tuple[bool, ...], ...]:
    """Remove speckle and close small gaps using dependency-free morphology."""
    _shape(mask)
    if morphology_radius <= 0:
        return tuple(tuple(bool(value) for value in row) for row in mask)
    opened = _dilate(_erode(mask, morphology_radius), morphology_radius)
    return _erode(_dilate(opened, morphology_radius), morphology_radius)


def _components(mask: BoolMatrix) -> list[Component]:
    height, width = _shape(mask)
    remaining = {(y, x) for y in range(height) for x in range(width) if mask[y][x]}
    components: list[Component] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        pixels = {seed}
        while stack:
            y, x = stack.pop()
            for neighbour in _neighbours(y, x, height, width):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    pixels.add(neighbour)
                    stack.append(neighbour)
        ys = [point[0] for point in pixels]
        xs = [point[1] for point in pixels]
        components.append(Component(frozenset(pixels), min(xs), min(ys), max(xs), max(ys)))
    return components


def _bbox_gap(a: Component, b: Component) -> int:
    horizontal = max(a.x0 - b.x1 - 1, b.x0 - a.x1 - 1, 0)
    vertical = max(a.y0 - b.y1 - 1, b.y0 - a.y1 - 1, 0)
    return max(horizontal, vertical)


def _merge_components(components: list[Component], gap: int) -> list[Component]:
    changed = True
    while changed:
        changed = False
        for index, first in enumerate(components):
            for other_index in range(index + 1, len(components)):
                second = components[other_index]
                if _bbox_gap(first, second) <= gap:
                    pixels = first.pixels | second.pixels
                    ys = [point[0] for point in pixels]
                    xs = [point[1] for point in pixels]
                    merged = Component(frozenset(pixels), min(xs), min(ys), max(xs), max(ys))
                    components = [
                        component
                        for position, component in enumerate(components)
                        if position not in (index, other_index)
                    ]
                    components.append(merged)
                    changed = True
                    break
            if changed:
                break
    return components


def find_changed_components(
    delta: NumberMatrix,
    interior_mask: BoolMatrix,
    *,
    threshold: float,
    min_area: int = 1,
    merge_gap: int = 0,
    morphology_radius: int = 0,
) -> tuple[tuple[bool, ...], ...]:
    _same_shape(delta, interior_mask)
    raw = tuple(
        tuple(abs(float(delta[y][x])) > threshold and bool(interior_mask[y][x]) for x in range(len(delta[0])))
        for y in range(len(delta))
    )
    cleaned = clean_mask(raw, morphology_radius)
    components = [component for component in _components(cleaned) if component.area >= min_area]
    components = _merge_components(components, merge_gap)
    if len(components) != 1:
        reason = "ambiguous_change" if len(components) > 1 else "change_too_small"
        raise RecoverableRejection(reason)
    component_mask = [[False] * len(cleaned[0]) for _ in range(len(cleaned))]
    for y, x in components[0].pixels:
        component_mask[y][x] = True
    return tuple(tuple(row) for row in component_mask)


def unique_component(mask: BoolMatrix) -> Component:
    components = _components(mask)
    if len(components) != 1:
        raise RecoverableRejection("ambiguous_change")
    return components[0]


def square_crop(
    component: Component,
    image_height: int,
    image_width: int,
    *,
    source: str,
    padding: float = 0.25,
) -> Crop:
    if image_height <= 0 or image_width <= 0 or padding < 0:
        raise ValueError("invalid image dimensions or padding")
    box_width = component.x1 - component.x0 + 1
    box_height = component.y1 - component.y0 + 1
    side = max(box_width, box_height) * (1.0 + 2.0 * padding)
    side = min(max(1, int(round(side))), min(image_height, image_width))
    center_x = (component.x0 + component.x1) / 2.0
    center_y = (component.y0 + component.y1) / 2.0
    x = int(round(center_x - side / 2.0))
    y = int(round(center_y - side / 2.0))
    x = min(max(0, x), image_width - side)
    y = min(max(0, y), image_height - side)
    return Crop(x=x, y=y, size=side, source=source)


def signed_action(
    delta: NumberMatrix,
    change_mask: BoolMatrix,
    *,
    near_is_positive: bool,
    direction_threshold: float,
) -> tuple[str, float]:
    values = [
        float(delta[y][x])
        for y in range(len(delta))
        for x in range(len(delta[0]))
        if change_mask[y][x]
    ]
    if not values:
        raise RecoverableRejection("change_too_small")
    signed = float(median(values))
    effective = signed if near_is_positive else -signed
    if effective > direction_threshold:
        return "put", signed
    if effective < -direction_threshold:
        return "take", signed
    raise RecoverableRejection("unknown_action")


def build_change_result(
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
    interior_mask: BoolMatrix,
    *,
    near_is_positive: bool,
    object_change_threshold: float,
    direction_threshold: float,
    min_change_area: int = 1,
    component_merge_gap: int = 0,
    morphology_radius: int = 0,
    crop_padding: float = 0.25,
) -> ChangeResult:
    delta = subtract_depth(snapshot_b.depth, snapshot_a.depth)
    mask = find_changed_components(
        delta,
        interior_mask,
        threshold=object_change_threshold,
        min_area=min_change_area,
        merge_gap=component_merge_gap,
        morphology_radius=morphology_radius,
    )
    action, signed = signed_action(
        delta,
        mask,
        near_is_positive=near_is_positive,
        direction_threshold=direction_threshold,
    )
    component = unique_component(mask)
    height, width = _shape(delta)
    crop = square_crop(
        component,
        height,
        width,
        source="rgb_b" if action == "put" else "rgb_a",
        padding=crop_padding,
    )
    return ChangeResult(delta=delta, mask=mask, component=component, signed_change=signed, action=action, crop=crop)


def match_layer(
    bottom_depth: float,
    baselines: Mapping[int, float],
    tolerance: float,
) -> int:
    if tolerance < 0 or not baselines:
        raise ValueError("invalid layer matching configuration")
    ranked = sorted((abs(float(bottom_depth) - float(value)), int(layer)) for layer, value in baselines.items())
    if ranked[0][0] > tolerance or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        raise RecoverableRejection("drawer_unknown")
    return ranked[0][1]


def select_unique_detection(
    detections: Sequence[Detection],
    enabled_class_ids: set[int],
    *,
    confidence_threshold: float,
) -> Detection:
    usable = [
        detection
        for detection in detections
        if detection.class_id in enabled_class_ids and detection.confidence >= confidence_threshold
    ]
    if not usable:
        raise RecoverableRejection("unsupported_or_occluded")
    if len(usable) != 1:
        raise RecoverableRejection("multiple_items")
    return usable[0]


class DrawerStateMachine:
    """Small explicit state machine; all unsafe paths fail before DB writes."""

    def __init__(self, max_layers: int = 6) -> None:
        self.max_layers = max_layers
        self.state = DrawerState.STARTUP_CHOICE
        self.calibrations: list[LayerCalibration] = []
        self.current_layer: Optional[int] = None
        self.snapshot_a: Optional[Snapshot] = None
        self.snapshot_b: Optional[Snapshot] = None
        self.candidate: Optional[Candidate] = None
        self.ignored_changes = 0
        self.last_rejection: Optional[str] = None

    @property
    def voice_enabled(self) -> bool:
        return self.state == DrawerState.READY

    def _require(self, *states: DrawerState) -> None:
        if self.state not in states:
            raise TransitionError("%s cannot act in %s" % ("/".join(state.value for state in states), self.state.value))

    def choose_resume(self) -> None:
        self._require(DrawerState.STARTUP_CHOICE)
        self.state = DrawerState.RESUME_VALIDATION

    def resume_ok(self) -> None:
        self._require(DrawerState.RESUME_VALIDATION)
        self.state = DrawerState.READY

    def resume_failed(self) -> None:
        self._require(DrawerState.RESUME_VALIDATION)
        self.state = DrawerState.STARTUP_CHOICE

    def choose_clear(self) -> None:
        self._require(DrawerState.STARTUP_CHOICE)
        self.state = DrawerState.NEEDS_INITIALIZATION

    def begin_initialization(self) -> None:
        self._require(DrawerState.NEEDS_INITIALIZATION)
        self.calibrations.clear()
        self.state = DrawerState.INITIALIZING_WAIT_OPEN

    def init_open(self, layer_no: int) -> None:
        self._require(DrawerState.INITIALIZING_WAIT_OPEN)
        expected = len(self.calibrations) + 1
        if layer_no != expected or not 1 <= layer_no <= self.max_layers:
            raise RecoverableRejection("invalid_initialization_order")
        self.current_layer = layer_no
        self.state = DrawerState.INITIALIZING_OPEN_STABLE

    def init_stable(self, calibration: LayerCalibration) -> None:
        self._require(DrawerState.INITIALIZING_OPEN_STABLE)
        if calibration.layer_no != len(self.calibrations) + 1:
            raise RecoverableRejection("invalid_initialization_order")
        self.calibrations.append(calibration)
        self.state = DrawerState.INITIALIZING_WAIT_CLOSE

    def init_close(self) -> None:
        self._require(DrawerState.INITIALIZING_WAIT_CLOSE)
        self.current_layer = None
        self.state = DrawerState.INITIALIZING_WAIT_OPEN

    def finish_initialization(self) -> None:
        self._require(DrawerState.INITIALIZING_WAIT_OPEN)
        if not self.calibrations:
            raise RecoverableRejection("no_layers_calibrated")
        self.state = DrawerState.READY

    def start_drawer(self, layer_no: int) -> None:
        self._require(DrawerState.READY)
        if layer_no not in {calibration.layer_no for calibration in self.calibrations}:
            self.last_rejection = "drawer_unknown"
            self.state = DrawerState.WAIT_DRAWER_CLOSE
            return
        self.current_layer = layer_no
        self.snapshot_a = None
        self.snapshot_b = None
        self.candidate = None
        self.ignored_changes = 0
        self.last_rejection = None
        self.state = DrawerState.FIRST_CHANGE_MOTION

    def snapshot_a_stable(self, snapshot: Snapshot, layer_no: int) -> None:
        self._require(DrawerState.FIRST_CHANGE_MOTION)
        if layer_no != self.current_layer:
            raise RecoverableRejection("drawer_unknown")
        self.snapshot_a = snapshot
        self.state = DrawerState.WAIT_ITEM_CHANGE

    def item_change_motion(self) -> None:
        self._require(DrawerState.WAIT_ITEM_CHANGE)
        self.state = DrawerState.SECOND_CHANGE_MOTION

    def snapshot_b_stable(self, snapshot: Snapshot) -> None:
        self._require(DrawerState.SECOND_CHANGE_MOTION)
        self.snapshot_b = snapshot
        self.state = DrawerState.DETECTING_CHANGED_CROP

    def reject(self, reason: str) -> None:
        if self.state not in (
            DrawerState.DETECTING_CHANGED_CROP,
            DrawerState.FIRST_CHANGE_MOTION,
            DrawerState.SECOND_CHANGE_MOTION,
            DrawerState.WAIT_ITEM_CHANGE,
        ):
            raise TransitionError("cannot reject transaction in %s" % self.state.value)
        self.last_rejection = reason
        self.state = DrawerState.WAIT_DRAWER_CLOSE

    def accept_candidate(self, candidate: Candidate) -> None:
        self._require(DrawerState.DETECTING_CHANGED_CROP)
        if candidate.layer_no != self.current_layer or candidate.yolo_calls != 1:
            raise ValueError("candidate does not match the active transaction")
        self.candidate = candidate
        self.state = DrawerState.CANDIDATE_CONFIRMED

    def await_drawer_close(self) -> None:
        self._require(DrawerState.CANDIDATE_CONFIRMED)
        self.state = DrawerState.WAIT_DRAWER_CLOSE

    def note_extra_change(self) -> None:
        self._require(DrawerState.CANDIDATE_CONFIRMED, DrawerState.WAIT_DRAWER_CLOSE)
        self.ignored_changes += 1

    def drawer_closed(self) -> Optional[Candidate]:
        self._require(DrawerState.WAIT_DRAWER_CLOSE, DrawerState.CANDIDATE_CONFIRMED)
        candidate = self.candidate
        if candidate is None:
            self._clear_transaction()
            self.state = DrawerState.READY
            return None
        self.state = DrawerState.COMMITTING
        return candidate

    def commit_ok(self) -> None:
        self._require(DrawerState.COMMITTING)
        self._clear_transaction()
        self.state = DrawerState.READY

    def commit_failed(self) -> None:
        self._require(DrawerState.COMMITTING)
        self.state = DrawerState.ERROR

    def fatal_error(self, reason: str) -> None:
        self.last_rejection = reason
        self.state = DrawerState.ERROR

    def _clear_transaction(self) -> None:
        self.current_layer = None
        self.snapshot_a = None
        self.snapshot_b = None
        self.candidate = None
        self.ignored_changes = 0


def make_candidate(change: ChangeResult, detection: Detection, layer_no: int, confidence_threshold: float = 0.5) -> Candidate:
    if detection.confidence < confidence_threshold:
        raise RecoverableRejection("unsupported_or_occluded")
    return Candidate(
        layer_no=layer_no,
        action=change.action,
        class_id=detection.class_id,
        canonical_label=detection.canonical_label,
        confidence=float(detection.confidence),
        signed_depth_change=change.signed_change,
        crop=change.crop,
    )


# Kept here so storage/config/checks share one source of truth.
ENABLED_CLASSES: tuple[tuple[int, str], ...] = (
    (39, "bottle"),
    (40, "wine glass"),
    (41, "cup"),
    (42, "fork"),
    (43, "knife"),
    (44, "spoon"),
    (45, "bowl"),
    (46, "banana"),
    (47, "apple"),
    (49, "orange"),
    (48, "sandwich"),
    (63, "laptop"),
    (64, "mouse"),
    (65, "remote"),
    (66, "keyboard"),
    (67, "cell phone"),
    (73, "book"),
    (74, "clock"),
    (76, "scissors"),
    (79, "toothbrush"),
)
ENABLED_CLASS_IDS = {class_id for class_id, _ in ENABLED_CLASSES}
ENABLED_LABELS = {label for _, label in ENABLED_CLASSES}
