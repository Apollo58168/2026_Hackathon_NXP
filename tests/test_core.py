#!/usr/bin/env python3
"""Dependency-free acceptance checks for the simulation-first vertical slice."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_drawer.core import (  # noqa: E402
    Candidate,
    Crop,
    Detection,
    DrawerState,
    DrawerStateMachine,
    LayerCalibration,
    RecoverableRejection,
    TransitionError,
    build_change_result,
    match_layer,
    robust_bottom_stat,
    select_unique_detection,
    stable_snapshot,
)
from smart_drawer.semantic import SemanticCatalog  # noqa: E402
from smart_drawer.simulator import filled, mask_filled, run_demo, with_patch  # noqa: E402
from smart_drawer.storage import Storage  # noqa: E402


def raises(error, function):
    try:
        function()
    except error:
        return
    raise AssertionError("expected %s" % error.__name__)


def calibration(layer: int) -> LayerCalibration:
    mask = mask_filled(4, 4, True)
    return LayerCalibration(layer, float(layer), mask, mask, 0.1, 0.05)


def check_state_machine_and_initialization() -> None:
    machine = DrawerStateMachine()
    assert machine.state == DrawerState.STARTUP_CHOICE
    assert not machine.voice_enabled
    raises(TransitionError, machine.begin_initialization)
    machine.choose_clear()
    machine.begin_initialization()
    raises(RecoverableRejection, lambda: machine.init_open(2))
    assert machine.state == DrawerState.INITIALIZING_WAIT_OPEN
    for layer in (1, 2, 3):
        machine.init_open(layer)
        machine.init_stable(calibration(layer))
        machine.init_close()
    machine.finish_initialization()
    assert machine.state == DrawerState.READY
    assert [item.layer_no for item in machine.calibrations] == [1, 2, 3]
    assert machine.voice_enabled


def check_stability_baseline_and_layer_match() -> None:
    rgb = ["first", "middle", "last"]
    depths = [
        [[1.0, 2.0], [3.0, 100.0]],
        [[1.0, 2.0], [3.0, 102.0]],
        [[1.0, 2.0], [3.0, 101.0]],
    ]
    snapshot = stable_snapshot(rgb, depths)
    assert snapshot.rgb == "middle"
    assert snapshot.depth[1][1] == 101.0

    # Four farthest pixels form the robust band; the one-pixel minimum is not used.
    depth = [[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 9.0]]
    mask = [[False, False, True, True], [False, False, True, True]]
    assert robust_bottom_stat(depth, mask, farthest_percentile=50) == 8.0
    assert match_layer(0.51, {1: 0.2, 2: 0.5, 3: 0.8}, 0.08) == 2
    raises(RecoverableRejection, lambda: match_layer(0.65, {1: 0.2, 2: 0.5, 3: 0.8}, 0.08))
    raises(RecoverableRejection, lambda: match_layer(0.5, {1: 0.4, 2: 0.6}, 0.2))


def check_changed_crop_and_signed_action() -> None:
    base = filled(8, 10, 0.5)
    changed = with_patch(base, x=4, y=3, width=2, height=2, value=0.9)
    interior = mask_filled(8, 10, True)
    a = stable_snapshot(["a0", "a1", "a2"], [base, base, base])
    b = stable_snapshot(["b0", "b1", "b2"], [changed, changed, changed])
    put = build_change_result(
        a, b, interior, near_is_positive=True, object_change_threshold=0.1,
        direction_threshold=0.05, min_change_area=2, morphology_radius=0,
    )
    assert put.action == "put"
    assert put.crop.source == "rgb_b"
    assert put.signed_change > 0.3
    assert put.crop.size >= 2
    take = build_change_result(
        b, a, interior, near_is_positive=True, object_change_threshold=0.1,
        direction_threshold=0.05, min_change_area=2, morphology_radius=0,
    )
    assert take.action == "take"
    assert take.crop.source == "rgb_a"
    two = with_patch(changed, x=0, y=0, width=1, height=1, value=0.9)
    raises(
        RecoverableRejection,
        lambda: build_change_result(
            a,
            stable_snapshot(["c0", "c1", "c2"], [two, two, two]),
            interior,
            near_is_positive=True,
            object_change_threshold=0.1,
            direction_threshold=0.05,
            min_change_area=1,
            morphology_radius=0,
        ),
    )


def check_detection_gates() -> None:
    enabled = {63}
    assert select_unique_detection([Detection(63, "laptop", 0.9)], enabled, confidence_threshold=0.5).class_id == 63
    raises(RecoverableRejection, lambda: select_unique_detection([], enabled, confidence_threshold=0.5))
    raises(RecoverableRejection, lambda: select_unique_detection([Detection(63, "laptop", 0.9), Detection(63, "laptop", 0.8)], enabled, confidence_threshold=0.5))
    raises(RecoverableRejection, lambda: select_unique_detection([Detection(1, "person", 0.99)], enabled, confidence_threshold=0.5))


def check_sqlite_commit_clear_and_atomicity() -> None:
    mask = mask_filled(4, 4, True)
    with Storage(":memory:") as storage:
        storage.finish_initialization([LayerCalibration(2, 0.5, mask, mask, 0.1, 0.05)])
        put = Candidate(2, "put", 63, "laptop", 0.9, 0.4, Crop(0, 0, 2, "rgb_b"))
        storage.commit_candidate(put)
        assert storage.inventory(2, 63) == 1
        take = Candidate(2, "take", 63, "laptop", 0.9, -0.4, Crop(0, 0, 2, "rgb_a"))
        storage.commit_candidate(take)
        assert storage.inventory(2, 63) == 0
        assert len(storage.event_rows()) == 2

        storage.commit_candidate(take)
        assert storage.inventory(2, 63) == 0
        assert storage.event_rows()[-1]["applied"] == 0
        assert storage.event_rows()[-1]["reason"] == "untracked_take"

        storage.connection.execute(
            """CREATE TRIGGER abort_event BEFORE INSERT ON event
               BEGIN SELECT RAISE(ABORT, 'test rollback'); END;"""
        )
        failing_put = Candidate(2, "put", 63, "laptop", 0.9, 0.4, Crop(0, 0, 2, "rgb_b"))
        raises(sqlite3.DatabaseError, lambda: storage.commit_candidate(failing_put))
        assert storage.inventory(2, 63) == 0
        assert len(storage.event_rows()) == 3
        storage.connection.execute("DROP TRIGGER abort_event")
        storage.clear_for_initialization()
        assert storage.calibration_count() == 0
        assert len(storage.event_rows()) == 3


def check_simulation_semantics_and_models() -> None:
    result = run_demo()
    assert result["drawer_count"] == 3
    assert result["put_action"] == "put"
    assert result["put_crop_source"] == "rgb_b"
    assert result["take_action"] == "take"
    assert result["take_crop_source"] == "rgb_a"
    assert result["detector_calls"] == 2
    assert result["laptop_after_take"] == 0
    assert result["event_count"] == 2
    assert result["semantic_laptop"] == "laptop"
    assert result["voice_enabled"] is True

    catalog = SemanticCatalog.load(ROOT / "config/semantic_catalog.json")
    for token in ("電腦", "筆電", "notebook"):
        match = catalog.resolve_token(token)
        assert match is not None and match.canonical_label == "laptop"
    laptop_vector = [0.0] * 20
    laptop_vector[11] = 1.0
    assert catalog.resolve(laptop_vector).canonical_label == "laptop"
    assert catalog.resolve([0.0] * 20) is None

    enabled = json.loads((ROOT / "config/enabled_classes.json").read_text(encoding="utf-8"))
    assert len(enabled["classes"]) == 20
    assert len((ROOT / "models/coco_labels.txt").read_text(encoding="utf-8").splitlines()) == 80
    for filename, expected in {
        "midas_v2_1_small_quant_vela.tflite": "2719ff97aba7b31f61007b85162a6cb34dc1895e7b8e9eb102a90b861c6d7aac",
        "yolov8n_coco_int8.tflite": "357ffdc968542da6c74f3a83eb566a85be4f33663cb5b866521bcc80354e237c",
    }.items():
        digest = hashlib.sha256((ROOT / "models" / filename).read_bytes()).hexdigest()
        assert digest == expected


def check_resume_and_voice_lockout() -> None:
    machine = DrawerStateMachine()
    machine.choose_resume()
    assert not machine.voice_enabled
    machine.resume_ok()
    assert machine.voice_enabled
    machine.start_drawer(1)  # no calibration: safe unknown path
    assert machine.state == DrawerState.WAIT_DRAWER_CLOSE
    assert not machine.voice_enabled
    assert machine.drawer_closed() is None
    assert machine.state == DrawerState.READY


def main() -> None:
    check_state_machine_and_initialization()
    check_stability_baseline_and_layer_match()
    check_changed_crop_and_signed_action()
    check_detection_gates()
    check_sqlite_commit_clear_and_atomicity()
    check_simulation_semantics_and_models()
    check_resume_and_voice_lockout()
    print("check_core: all checks passed")


if __name__ == "__main__":
    main()
