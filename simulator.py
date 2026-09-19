"""Deterministic simulation-first vertical slice for SmartDrawer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .core import (
    Candidate,
    Detection,
    DrawerState,
    DrawerStateMachine,
    LayerCalibration,
    Snapshot,
    build_change_result,
    make_candidate,
    robust_bottom_stat,
    stable_snapshot,
)
from .semantic import demo_catalog
from .storage import Storage


@dataclass(frozen=True)
class SimFrame:
    frame_id: str
    rgb: Any


def filled(height: int, width: int, value: float) -> list[list[float]]:
    return [[float(value) for _ in range(width)] for _ in range(height)]


def mask_filled(height: int, width: int, value: bool = True) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(value for _ in range(width)) for _ in range(height))


def with_patch(
    matrix: Sequence[Sequence[float]],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
) -> list[list[float]]:
    result = [list(row) for row in matrix]
    for row in range(y, y + height):
        for column in range(x, x + width):
            result[row][column] = float(value)
    return result


class FakeDetector:
    """Simulation detector: one recorded crop, no model or PC fallback."""

    def __init__(self, detection: Detection) -> None:
        self.detection = detection
        self.calls: list[tuple[str, object]] = []

    def detect(self, crop: object, source: str) -> list[Detection]:
        self.calls.append((source, crop))
        return [self.detection]


def _stable_frames(base: Sequence[Sequence[float]], tag: str, noise: float = 0.0) -> tuple[list[SimFrame], list[list[list[float]]]]:
    frames: list[SimFrame] = []
    depths: list[list[list[float]]] = []
    for index in range(3):
        frame_depth = [list(row) for row in base]
        if noise:
            frame_depth[0][0] += noise * (index - 1)
        frames.append(SimFrame("%s-%d" % (tag, index), tag))
        depths.append(frame_depth)
    return tuple(frames), tuple(depths)


def run_demo() -> dict[str, object]:
    height, width = 12, 16
    # High relative depth means nearer for this simulation manifest.
    interior = mask_filled(height, width, True)
    drawer_mask = mask_filled(height, width, True)
    baselines = {1: 0.20, 2: 0.50, 3: 0.80}
    engine = DrawerStateMachine()

    storage = Storage(":memory:")
    try:
        engine.choose_clear()
        storage.clear_for_initialization()
        engine.begin_initialization()
        calibrations: list[LayerCalibration] = []
        for layer_no in (1, 2, 3):
            engine.init_open(layer_no)
            base = filled(height, width, baselines[layer_no])
            rgb, depths = _stable_frames(base, "init-%d" % layer_no, noise=0.002)
            snapshot = stable_snapshot(rgb, depths)
            baseline = robust_bottom_stat(snapshot.depth, interior, farthest_is_larger=True)
            calibration = LayerCalibration(layer_no, baseline, drawer_mask, interior, 0.08, 0.04)
            engine.init_stable(calibration)
            calibrations.append(calibration)
            engine.init_close()
        engine.finish_initialization()
        storage.finish_initialization(calibrations)

        # Runtime: layer 2, one laptop put into a 2x2 changed region.
        engine.start_drawer(2)
        base_a = filled(height, width, baselines[2])
        rgb_a, depths_a = _stable_frames(base_a, "A")
        snapshot_a = stable_snapshot(rgb_a, depths_a)
        engine.snapshot_a_stable(snapshot_a, 2)
        engine.item_change_motion()
        base_b = with_patch(base_a, x=6, y=5, width=2, height=2, value=0.78)
        rgb_b, depths_b = _stable_frames(base_b, "B")
        snapshot_b = stable_snapshot(rgb_b, depths_b)
        engine.snapshot_b_stable(snapshot_b)
        change = build_change_result(
            snapshot_a,
            snapshot_b,
            interior,
            near_is_positive=True,
            object_change_threshold=0.10,
            direction_threshold=0.05,
            min_change_area=2,
            component_merge_gap=1,
            morphology_radius=0,
        )
        detector = FakeDetector(Detection(63, "laptop", 0.94))
        detections = detector.detect(change.crop, change.crop.source)
        detection = detections[0]
        candidate = make_candidate(change, detection, 2)
        engine.accept_candidate(candidate)
        engine.await_drawer_close()
        engine.note_extra_change()  # third change is deliberately ignored
        committed = engine.drawer_closed()
        assert committed == candidate
        storage.commit_candidate(committed)
        engine.commit_ok()

        # Take the same laptop: action is signed negative and source is RGB_A.
        engine.start_drawer(2)
        engine.snapshot_a_stable(snapshot_b, 2)
        engine.item_change_motion()
        engine.snapshot_b_stable(snapshot_a)
        take_change = build_change_result(
            snapshot_b,
            snapshot_a,
            interior,
            near_is_positive=True,
            object_change_threshold=0.10,
            direction_threshold=0.05,
            min_change_area=2,
            component_merge_gap=1,
            morphology_radius=0,
        )
        assert take_change.action == "take"
        assert take_change.crop.source == "rgb_a"
        take_detections = detector.detect(take_change.crop, take_change.crop.source)
        take_candidate = make_candidate(take_change, take_detections[0], 2)
        engine.accept_candidate(take_candidate)
        engine.await_drawer_close()
        committed_take = engine.drawer_closed()
        assert committed_take is not None
        storage.commit_candidate(committed_take)
        engine.commit_ok()

        catalog = demo_catalog()
        laptop = catalog.resolve_token("電腦")
        return {
            "state": engine.state.value,
            "drawer_count": storage.calibration_count(),
            "put_action": change.action,
            "put_crop_source": change.crop.source,
            "take_action": take_change.action,
            "take_crop_source": take_change.crop.source,
            "detector_calls": len(detector.calls),
            "ignored_changes": 1,
            "laptop_after_take": storage.inventory(2, 63),
            "event_count": len(storage.event_rows()),
            "semantic_laptop": None if laptop is None else laptop.canonical_label,
            "voice_enabled": engine.voice_enabled,
        }
    finally:
        storage.close()


if __name__ == "__main__":
    print(run_demo())
