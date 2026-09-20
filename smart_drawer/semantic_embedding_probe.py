#!/usr/bin/env python3
"""Probe the English ASR-text to detector-label semantic mapper."""
from __future__ import annotations

import argparse
from pathlib import Path

from .semantic_embedding import EmbeddingMapper, OnnxSentenceEncoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2-onnx-q8"
DEFAULT_CATALOG = ROOT / "config" / "semantic_catalog_en.json"
DEFAULT_VECTOR_CACHE = ROOT / "config" / "semantic_catalog_en_minilm_q8.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="phone", help="Moonshine transcript")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--vector-cache", type=Path, default=DEFAULT_VECTOR_CACHE)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Encode all catalog phrases and overwrite --vector-cache",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = OnnxSentenceEncoder(
        args.model_dir / "onnx" / "model_qint8_arm64.onnx",
        args.model_dir / "tokenizer.json",
        threads=args.threads,
    )
    mapper = EmbeddingMapper.from_json(
        encoder,
        args.catalog,
        vector_cache=None if args.rebuild_cache else args.vector_cache,
    )
    if args.rebuild_cache:
        mapper.save_vector_cache(args.vector_cache)
        print(f"vector cache: wrote {args.vector_cache} ({len(mapper.phrases)} phrases)")
    ranked, inference_ms = mapper.rank(args.query)
    accepted = mapper.accept(ranked)

    print(f"query: {args.query!r}")
    print(f"query embedding latency: {inference_ms:.2f} ms ({args.threads} CPU threads)")
    print("rank  score   canonical label  matched catalog phrase")
    for index, match in enumerate(ranked[: max(args.top_k, 1)], start=1):
        print(
            f"{index:>4}  {match.score:>6.3f}  "
            f"{match.canonical_label:<15}  {match.matched_phrase}"
        )
    if accepted is None:
        print("result: rejected by score/margin thresholds")
    else:
        print(
            "result: "
            f"{args.query!r} -> {accepted.display_name!r} "
            f"(COCO {accepted.class_id}, score={accepted.score:.3f}, "
            f"margin={accepted.margin:.3f})"
        )


if __name__ == "__main__":
    main()
