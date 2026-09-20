"""Offline-built token embedding catalog and runtime cosine lookup."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

from .core import ENABLED_CLASSES


ALIASES: dict[str, tuple[str, ...]] = {
    "bottle": ("瓶子", "水瓶", "bottle"),
    "wine glass": ("酒杯", "wine glass"),
    "cup": ("杯子", "cup"),
    "fork": ("叉子", "fork"),
    "knife": ("刀子", "knife"),
    "spoon": ("湯匙", "勺子", "spoon"),
    "bowl": ("碗", "bowl"),
    "banana": ("香蕉", "banana"),
    "apple": ("蘋果", "apple"),
    "orange": ("橘子", "柳橙", "orange"),
    "sandwich": ("三明治", "sandwich"),
    "laptop": ("電腦", "筆電", "notebook", "laptop"),
    "mouse": ("滑鼠", "mouse"),
    "remote": ("遙控器", "remote"),
    "keyboard": ("鍵盤", "keyboard"),
    "cell phone": ("手機", "電話", "phone", "cell phone"),
    "book": ("書", "book"),
    "clock": ("時鐘", "clock"),
    "scissors": ("剪刀", "scissors"),
    "toothbrush": ("牙刷", "toothbrush"),
}


@dataclass(frozen=True)
class SemanticMatch:
    canonical_label: str
    class_id: int
    score: float
    margin: float


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("vectors must have equal non-zero dimensions")
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right)) / (left_norm * right_norm)


class SemanticCatalog:
    """Small immutable catalog loaded by the runtime; no text encoder is run."""

    def __init__(
        self,
        entries: Mapping[str, tuple[str, Sequence[float]]],
        class_ids: Mapping[str, int],
        *,
        min_score: float = 0.80,
        min_margin: float = 0.05,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.entries = {
            token: (label, tuple(float(value) for value in vector))
            for token, (label, vector) in entries.items()
        }
        self.class_ids = dict(class_ids)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.metadata = dict(metadata or {})
        if not self.entries:
            raise ValueError("semantic catalog cannot be empty")

    def resolve(self, vector: Sequence[float]) -> Optional[SemanticMatch]:
        by_label: dict[str, float] = {}
        for label, values in self.entries.values():
            by_label[label] = max(by_label.get(label, -1.0), cosine(vector, values))
        scored = sorted(((score, label) for label, score in by_label.items()), reverse=True)
        best_score, best_label = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        margin = best_score - second_score
        if best_score < self.min_score or margin < self.min_margin:
            return None
        return SemanticMatch(
            canonical_label=best_label,
            class_id=self.class_ids[best_label],
            score=best_score,
            margin=margin,
        )

    def resolve_token(self, token: str) -> Optional[SemanticMatch]:
        entry = self.entries.get(token)
        if entry is None:
            return None
        label, vector = entry
        return SemanticMatch(label, self.class_ids[label], 1.0, 1.0)

    def save(self, path: Union[str, Path]) -> None:
        payload = {
            "metadata": self.metadata,
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "class_ids": self.class_ids,
            "entries": {
                token: {"canonical_label": label, "vector": list(vector)}
                for token, (label, vector) in self.entries.items()
            },
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SemanticCatalog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = {
            token: (value["canonical_label"], value["vector"])
            for token, value in payload["entries"].items()
        }
        return cls(
            entries,
            {str(label): int(class_id) for label, class_id in payload["class_ids"].items()},
            min_score=payload.get("min_score", 0.8),
            min_margin=payload.get("min_margin", 0.05),
            metadata=payload.get("metadata"),
        )


def demo_catalog() -> SemanticCatalog:
    """Create deterministic simulation vectors for the fixed vocabulary.

    The vectors are an offline fixture, not a claim that a multilingual model
    has been run.  Production deployment replaces this file using a pinned
    multilingual encoder on the PC; the i.MX93 runtime still loads vectors only.
    """
    class_ids = {label: class_id for class_id, label in ENABLED_CLASSES}
    labels = [label for _, label in ENABLED_CLASSES]
    entries: dict[str, tuple[str, Sequence[float]]] = {}
    for index, label in enumerate(labels):
        vector = [0.0] * len(labels)
        vector[index] = 1.0
        for token in ALIASES[label]:
            entries[token] = (label, vector)
    return SemanticCatalog(
        entries,
        class_ids,
        min_score=0.80,
        min_margin=0.05,
        metadata={
            "catalog_version": "simulation-1",
            "encoder": "deterministic fixture; replace before hardware acceptance",
            "runtime": "vectors only; no text encoder",
        },
    )


def write_demo_catalog(path: Union[str, Path]) -> SemanticCatalog:
    catalog = demo_catalog()
    catalog.save(path)
    return catalog
