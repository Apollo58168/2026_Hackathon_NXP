"""Small ONNX sentence-embedding runtime for semantic label mapping.

The implementation intentionally depends only on NumPy, ONNX Runtime, and the
Hugging Face ``tokenizers`` package.  It does not require PyTorch,
Transformers, or sentence-transformers on the i.MX93 target.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


@dataclass(frozen=True)
class EmbeddingMatch:
    canonical_label: str
    class_id: int
    display_name: str
    matched_phrase: str
    score: float
    margin: float
    inference_ms: float


class OnnxSentenceEncoder:
    """Encode short English text with a quantized MiniLM ONNX model."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        *,
        threads: int = 2,
        max_length: int = 32,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be at least 1")
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.max_length = max_length

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        required_inputs = {"input_ids", "attention_mask", "token_type_ids"}
        actual_inputs = {item.name for item in self.session.get_inputs()}
        if actual_inputs != required_inputs:
            raise ValueError(
                f"unexpected model inputs: expected {sorted(required_inputs)}, "
                f"got {sorted(actual_inputs)}"
            )

    def encode(self, texts: str | Sequence[str]) -> np.ndarray:
        """Return unit-length 384-D embeddings for one or more strings."""
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = list(texts)
        if not batch or any(not text.strip() for text in batch):
            raise ValueError("texts must contain at least one non-empty string")

        encoded = self.tokenizer.encode_batch(batch)
        sequence_length = max(len(item.ids) for item in encoded)

        input_ids = np.zeros((len(batch), sequence_length), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        token_type_ids = np.zeros_like(input_ids)
        for row, item in enumerate(encoded):
            size = len(item.ids)
            input_ids[row, :size] = item.ids
            attention_mask[row, :size] = item.attention_mask
            token_type_ids[row, :size] = item.type_ids

        (token_embeddings,) = self.session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # all-MiniLM-L6-v2 uses attention-masked mean pooling, then L2 norm.
        mask = attention_mask[:, :, None].astype(np.float32)
        pooled = (token_embeddings * mask).sum(axis=1)
        pooled /= np.maximum(mask.sum(axis=1), 1e-9)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return pooled.astype(np.float32, copy=False)


class EmbeddingMapper:
    """Map free text to a canonical detector label with cosine similarity."""

    def __init__(
        self,
        encoder: OnnxSentenceEncoder,
        items: Sequence[Mapping[str, object]],
        *,
        min_score: float = 0.45,
        min_margin: float = 0.03,
        phrase_vectors: np.ndarray | None = None,
    ) -> None:
        if not items:
            raise ValueError("semantic catalog cannot be empty")
        self.encoder = encoder
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)

        labels: list[str] = []
        class_ids: list[int] = []
        display_names: list[str] = []
        phrases: list[str] = []
        for item in items:
            label = str(item["canonical_label"])
            display_name = str(item.get("display_name", label))
            item_phrases = [str(value) for value in item.get("phrases", [])]
            if not item_phrases:
                raise ValueError(f"catalog item {label!r} has no phrases")
            for phrase in item_phrases:
                labels.append(label)
                class_ids.append(int(item["class_id"]))
                display_names.append(display_name)
                phrases.append(phrase)

        self.labels = labels
        self.class_ids = class_ids
        self.display_names = display_names
        self.phrases = phrases
        if phrase_vectors is None:
            self.phrase_vectors = encoder.encode(phrases)
        else:
            vectors = np.asarray(phrase_vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(phrases):
                raise ValueError(
                    "cached phrase vectors do not match the semantic catalog: "
                    f"expected {len(phrases)} rows, got {vectors.shape}"
                )
            self.phrase_vectors = vectors

    @classmethod
    def from_json(
        cls,
        encoder: OnnxSentenceEncoder,
        path: str | Path,
        *,
        vector_cache: str | Path | None = None,
    ) -> "EmbeddingMapper":
        catalog_path = Path(path)
        catalog_bytes = catalog_path.read_bytes()
        catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
        payload = json.loads(catalog_bytes)
        phrase_vectors = None
        if vector_cache is not None:
            cache_path = Path(vector_cache)
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as cache:
                    cached_sha256 = str(cache["catalog_sha256"].item())
                    if cached_sha256 != catalog_sha256:
                        raise ValueError(
                            f"stale semantic vector cache {cache_path}; rebuild it"
                        )
                    phrase_vectors = cache["phrase_vectors"]

        mapper = cls(
            encoder,
            payload["items"],
            min_score=payload.get("min_score", 0.45),
            min_margin=payload.get("min_margin", 0.03),
            phrase_vectors=phrase_vectors,
        )
        mapper.catalog_sha256 = catalog_sha256
        return mapper

    def save_vector_cache(self, path: str | Path) -> None:
        """Persist normalized catalog embeddings for fast target startup."""
        catalog_sha256 = getattr(self, "catalog_sha256", None)
        if catalog_sha256 is None:
            raise ValueError("mapper was not loaded with from_json")
        np.savez_compressed(
            Path(path),
            catalog_sha256=np.asarray(catalog_sha256),
            phrase_vectors=self.phrase_vectors,
        )

    def rank(self, text: str) -> tuple[list[EmbeddingMatch], float]:
        started = time.perf_counter()
        query = self.encoder.encode(text)[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        phrase_scores = self.phrase_vectors @ query

        best_by_label: dict[str, tuple[float, int]] = {}
        for index, (label, score) in enumerate(zip(self.labels, phrase_scores)):
            previous = best_by_label.get(label)
            if previous is None or float(score) > previous[0]:
                best_by_label[label] = (float(score), index)

        ordered = sorted(best_by_label.items(), key=lambda item: item[1][0], reverse=True)
        matches: list[EmbeddingMatch] = []
        for position, (label, (score, index)) in enumerate(ordered):
            next_score = ordered[position + 1][1][0] if position + 1 < len(ordered) else -1.0
            matches.append(
                EmbeddingMatch(
                    canonical_label=label,
                    class_id=self.class_ids[index],
                    display_name=self.display_names[index],
                    matched_phrase=self.phrases[index],
                    score=score,
                    margin=score - next_score,
                    inference_ms=inference_ms,
                )
            )
        return matches, inference_ms

    def resolve(self, text: str) -> EmbeddingMatch | None:
        matches, _ = self.rank(text)
        return self.accept(matches)

    def accept(self, matches: Sequence[EmbeddingMatch]) -> EmbeddingMatch | None:
        """Apply the configured confidence gates to an existing ranking."""
        if not matches:
            return None
        best = matches[0]
        if best.score < self.min_score or best.margin < self.min_margin:
            return None
        return best
