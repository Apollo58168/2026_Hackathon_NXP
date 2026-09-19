"""SQLite persistence for SmartDrawer inventory and audit events."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

from .core import Candidate, ENABLED_CLASSES, LayerCalibration


SCHEMA_VERSION = "1"


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS drawer (
    drawer_id INTEGER PRIMARY KEY,
    layer_no INTEGER NOT NULL UNIQUE CHECK(layer_no BETWEEN 1 AND 6),
    bottom_depth_baseline REAL NOT NULL,
    drawer_mask_blob BLOB NOT NULL,
    interior_mask_blob BLOB NOT NULL,
    open_threshold REAL NOT NULL,
    close_threshold REAL NOT NULL,
    calibrated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS item_catalog (
    class_id INTEGER PRIMARY KEY,
    canonical_label TEXT NOT NULL UNIQUE,
    enabled INTEGER NOT NULL CHECK(enabled IN (0, 1))
);
CREATE TABLE IF NOT EXISTS inventory (
    drawer_id INTEGER NOT NULL REFERENCES drawer(drawer_id),
    class_id INTEGER NOT NULL REFERENCES item_catalog(class_id),
    quantity INTEGER NOT NULL CHECK(quantity >= 0),
    detector_confidence REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(drawer_id, class_id)
);
CREATE TABLE IF NOT EXISTS event (
    event_id TEXT PRIMARY KEY,
    occurred_at TEXT NOT NULL,
    layer_no INTEGER NOT NULL CHECK(layer_no BETWEEN 1 AND 6),
    action TEXT NOT NULL CHECK(action IN ('put','take')),
    class_id INTEGER NOT NULL REFERENCES item_catalog(class_id),
    delta INTEGER NOT NULL,
    applied INTEGER NOT NULL CHECK(applied IN (0, 1)),
    detector_confidence REAL NOT NULL,
    signed_depth_change REAL NOT NULL,
    reason TEXT,
    CHECK((action = 'put' AND delta = 1) OR (action = 'take' AND delta = -1))
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mask_blob(mask: object) -> bytes:
    return json.dumps(mask, separators=(",", ":")).encode("utf-8")


class Storage:
    """One SQLite connection owned by the state/UI thread."""

    def __init__(self, path: Union[str, Path] = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self.set_metadata("schema_version", SCHEMA_VERSION)
        self.seed_catalog()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def set_metadata(self, key: str, value: object) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        self.connection.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        row = self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def seed_catalog(self, classes: Iterable[tuple[int, str]] = ENABLED_CLASSES) -> None:
        self.connection.executemany(
            "INSERT INTO item_catalog(class_id, canonical_label, enabled) VALUES (?, ?, 1) "
            "ON CONFLICT(class_id) DO UPDATE SET canonical_label=excluded.canonical_label, enabled=1",
            list(classes),
        )
        self.connection.commit()

    def save_calibration(self, calibration: LayerCalibration) -> None:
        now = _now()
        self.connection.execute(
            """INSERT INTO drawer(
                   drawer_id, layer_no, bottom_depth_baseline, drawer_mask_blob,
                   interior_mask_blob, open_threshold, close_threshold, calibrated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(drawer_id) DO UPDATE SET
                   layer_no=excluded.layer_no,
                   bottom_depth_baseline=excluded.bottom_depth_baseline,
                   drawer_mask_blob=excluded.drawer_mask_blob,
                   interior_mask_blob=excluded.interior_mask_blob,
                   open_threshold=excluded.open_threshold,
                   close_threshold=excluded.close_threshold,
                   calibrated_at=excluded.calibrated_at""",
            (
                calibration.layer_no,
                calibration.layer_no,
                calibration.bottom_depth_baseline,
                sqlite3.Binary(_mask_blob(calibration.drawer_mask)),
                sqlite3.Binary(_mask_blob(calibration.interior_mask)),
                calibration.open_threshold,
                calibration.close_threshold,
                now,
            ),
        )
        self.connection.commit()

    def clear_for_initialization(self) -> None:
        """Clear calibration/inventory but intentionally retain event audit."""
        with self.connection:
            self.connection.execute("DELETE FROM inventory")
            self.connection.execute("DELETE FROM drawer")
            self.connection.execute("DELETE FROM metadata WHERE key IN ('drawer_count', 'initialized_at')")

    def finish_initialization(self, calibrations: Iterable[LayerCalibration]) -> None:
        calibrations = list(calibrations)
        if not 1 <= len(calibrations) <= 6:
            raise ValueError("SmartDrawer supports one to six layers")
        with self.connection:
            self.connection.execute("DELETE FROM inventory")
            self.connection.execute("DELETE FROM drawer")
            now = _now()
            for calibration in calibrations:
                self.connection.execute(
                    """INSERT INTO drawer(
                           drawer_id, layer_no, bottom_depth_baseline, drawer_mask_blob,
                           interior_mask_blob, open_threshold, close_threshold, calibrated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        calibration.layer_no,
                        calibration.layer_no,
                        calibration.bottom_depth_baseline,
                        sqlite3.Binary(_mask_blob(calibration.drawer_mask)),
                        sqlite3.Binary(_mask_blob(calibration.interior_mask)),
                        calibration.open_threshold,
                        calibration.close_threshold,
                        now,
                    ),
                )
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('drawer_count', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(len(calibrations)),),
            )
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('initialized_at', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (now,),
            )

    def calibration_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM drawer").fetchone()
        return int(row[0])

    def commit_candidate(self, candidate: Candidate) -> str:
        """Atomically apply one candidate and append exactly one event."""
        if candidate.action not in ("put", "take"):
            raise ValueError("invalid candidate action")
        event_id = str(uuid.uuid4())
        occurred_at = _now()
        delta = 1 if candidate.action == "put" else -1
        with self.connection:
            catalog = self.connection.execute(
                "SELECT class_id FROM item_catalog WHERE class_id = ? AND enabled = 1",
                (candidate.class_id,),
            ).fetchone()
            if catalog is None:
                raise ValueError("candidate class is not enabled")
            drawer = self.connection.execute(
                "SELECT drawer_id FROM drawer WHERE layer_no = ?",
                (candidate.layer_no,),
            ).fetchone()
            if drawer is None:
                raise ValueError("candidate layer is not calibrated")
            drawer_id = int(drawer[0])
            existing = self.connection.execute(
                "SELECT quantity FROM inventory WHERE drawer_id = ? AND class_id = ?",
                (drawer_id, candidate.class_id),
            ).fetchone()
            quantity = int(existing[0]) if existing is not None else 0
            applied = 1
            reason = None
            if candidate.action == "take" and quantity <= 0:
                applied = 0
                reason = "untracked_take"
            else:
                new_quantity = quantity + delta
                if new_quantity < 0:
                    raise ValueError("inventory quantity would become negative")
                self.connection.execute(
                    """INSERT INTO inventory(
                           drawer_id, class_id, quantity, detector_confidence, updated_at
                       ) VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(drawer_id, class_id) DO UPDATE SET
                           quantity=excluded.quantity,
                           detector_confidence=excluded.detector_confidence,
                           updated_at=excluded.updated_at""",
                    (drawer_id, candidate.class_id, new_quantity, candidate.confidence, occurred_at),
                )
            self.connection.execute(
                """INSERT INTO event(
                       event_id, occurred_at, layer_no, action, class_id, delta,
                       applied, detector_confidence, signed_depth_change, reason
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    occurred_at,
                    candidate.layer_no,
                    candidate.action,
                    candidate.class_id,
                    delta,
                    applied,
                    candidate.confidence,
                    candidate.signed_depth_change,
                    reason,
                ),
            )
        return event_id

    def inventory_rows(self) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """SELECT d.layer_no, i.class_id, c.canonical_label, i.quantity,
                          i.detector_confidence, i.updated_at
                   FROM inventory AS i
                   JOIN drawer AS d ON d.drawer_id = i.drawer_id
                   JOIN item_catalog AS c ON c.class_id = i.class_id
                   WHERE i.quantity > 0
                   ORDER BY d.layer_no, c.canonical_label"""
            )
        )

    def inventory(self, layer_no: int, class_id: int) -> int:
        row = self.connection.execute(
            """SELECT i.quantity FROM inventory AS i
               JOIN drawer AS d ON d.drawer_id = i.drawer_id
               WHERE d.layer_no = ? AND i.class_id = ?""",
            (layer_no, class_id),
        ).fetchone()
        return 0 if row is None else int(row[0])

    def query_item(self, class_id: int) -> list[tuple[int, int]]:
        rows = self.connection.execute(
            """SELECT d.layer_no, i.quantity
               FROM inventory AS i
               JOIN drawer AS d ON d.drawer_id = i.drawer_id
               WHERE i.class_id = ? AND i.quantity > 0
               ORDER BY d.layer_no""",
            (class_id,),
        )
        return [(int(row[0]), int(row[1])) for row in rows]

    def event_rows(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM event ORDER BY rowid"))

    def clear_inventory_only(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM inventory")

    def check_integrity(self) -> None:
        row = self.connection.execute("PRAGMA integrity_check").fetchone()
        if row[0] != "ok":
            raise RuntimeError("SQLite integrity check failed: %s" % row[0])
