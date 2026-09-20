#!/usr/bin/env python3
"""Run and benchmark Moonshine Tiny Q8 with ONNX Runtime.

The probe intentionally uses the uncached decoder graph. This keeps the runtime
adapter small and portable while still exercising the real quantized encoder
and decoder end to end.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from semantic_embedding import EmbeddingMapper, OnnxSentenceEncoder


SAMPLE_RATE = 16_000
DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "moonshine-tiny-onnx-q8"
DEFAULT_EMBEDDING_MODEL_DIR = (
    Path(__file__).parent / "models" / "all-MiniLM-L6-v2-onnx-q8"
)
DEFAULT_SEMANTIC_CATALOG = Path(__file__).parent / "config" / "semantic_catalog_en.json"
DEFAULT_SEMANTIC_VECTOR_CACHE = (
    Path(__file__).parent / "config" / "semantic_catalog_en_minilm_q8.npz"
)


def load_wav(path: Path, target_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Load PCM16 WAV audio as mono float32 and resample when necessary."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        compression = wav_file.getcomptype()
        raw = wav_file.readframes(frame_count)

    if compression != "NONE" or sample_width != 2:
        raise ValueError(
            f"{path} must be uncompressed PCM16 WAV; got compression={compression}, "
            f"sample_width={sample_width}"
        )
    if channels < 1 or sample_rate < 1 or frame_count < 1:
        raise ValueError(f"{path} has invalid WAV metadata")

    audio = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
    audio = audio.astype(np.float32).mean(axis=1) / 32768.0
    duration_seconds = len(audio) / sample_rate

    if sample_rate != target_rate:
        output_samples = round(len(audio) * target_rate / sample_rate)
        source_positions = np.arange(len(audio), dtype=np.float64)
        target_positions = np.linspace(0, len(audio) - 1, output_samples)
        audio = np.interp(target_positions, source_positions, audio).astype(np.float32)

    return np.ascontiguousarray(audio), duration_seconds


class MoonshineTiny:
    """Minimal greedy decoder for the Hugging Face Moonshine ONNX export."""

    def __init__(self, model_dir: Path, threads: int = 0) -> None:
        encoder_path = model_dir / "onnx" / "encoder_model_quantized.onnx"
        decoder_path = model_dir / "onnx" / "decoder_model_quantized.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        for path in (encoder_path, decoder_path, tokenizer_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing Moonshine artifact: {path}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1

        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(
            str(encoder_path), sess_options=options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(decoder_path), sess_options=options, providers=providers
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.start_token_id = 1
        self.eos_token_id = 2

    def transcribe(
        self, audio: np.ndarray, max_new_tokens: int
    ) -> tuple[str, list[int], float, float]:
        encoder_started = time.perf_counter()
        hidden_states = self.encoder.run(
            ["last_hidden_state"], {"input_values": audio[None, :]}
        )[0]
        encoder_seconds = time.perf_counter() - encoder_started

        decoder_started = time.perf_counter()
        token_ids = [self.start_token_id]
        for _ in range(max_new_tokens):
            logits = self.decoder.run(
                ["logits"],
                {
                    "input_ids": np.asarray([token_ids], dtype=np.int64),
                    "encoder_hidden_states": hidden_states,
                },
            )[0]
            next_token = int(np.argmax(logits[0, -1]))
            token_ids.append(next_token)
            if next_token == self.eos_token_id:
                break
        else:
            raise RuntimeError(
                f"Decoder did not emit EOS within {max_new_tokens} new tokens"
            )
        decoder_seconds = time.perf_counter() - decoder_started

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return text, token_ids, encoder_seconds, decoder_seconds


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    audio, duration_seconds = load_wav(args.audio)

    load_started = time.perf_counter()
    model = MoonshineTiny(args.model_dir, threads=args.threads)
    load_seconds = time.perf_counter() - load_started

    for _ in range(args.warmup):
        model.transcribe(audio, args.max_new_tokens)

    samples: list[dict[str, float]] = []
    transcripts: list[str] = []
    token_count = 0
    for _ in range(args.runs):
        text, token_ids, encoder_seconds, decoder_seconds = model.transcribe(
            audio, args.max_new_tokens
        )
        inference_seconds = encoder_seconds + decoder_seconds
        samples.append(
            {
                "encoder_seconds": encoder_seconds,
                "decoder_seconds": decoder_seconds,
                "inference_seconds": inference_seconds,
                "rtf": inference_seconds / duration_seconds,
            }
        )
        transcripts.append(text)
        token_count = len(token_ids) - 1

    if len(set(transcripts)) != 1:
        raise RuntimeError(f"Non-deterministic transcripts observed: {transcripts}")

    medians = {
        key: statistics.median(sample[key] for sample in samples)
        for key in samples[0]
    }
    result: dict[str, Any] = {
        "model": "Moonshine Tiny",
        "format": "ONNX Q8",
        "model_source": "onnx-community/moonshine-tiny-ONNX",
        "model_revision": "a6da1241cd305dcd64eab1edbd615f2bb9aabb95",
        "provider": model.encoder.get_providers()[0],
        "platform": platform.platform(),
        "audio": str(args.audio),
        "audio_seconds": duration_seconds,
        "audio_samples_16khz": len(audio),
        "load_seconds": load_seconds,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "generated_tokens_including_eos": token_count,
        "transcript": transcripts[0],
        "median": medians,
        "runs": samples,
    }
    if args.semantic:
        embedding_load_started = time.perf_counter()
        semantic_encoder = OnnxSentenceEncoder(
            args.embedding_model_dir / "onnx" / "model_qint8_arm64.onnx",
            args.embedding_model_dir / "tokenizer.json",
            threads=args.embedding_threads,
        )
        mapper = EmbeddingMapper.from_json(
            semantic_encoder,
            args.semantic_catalog,
            vector_cache=args.semantic_vector_cache,
        )
        embedding_load_seconds = time.perf_counter() - embedding_load_started
        semantic_result: dict[str, Any] = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 QInt8 ARM64",
            "embedding_provider": semantic_encoder.session.get_providers()[0],
            "load_seconds_including_catalog": embedding_load_seconds,
        }
        if transcripts[0].strip():
            ranked, semantic_inference_ms = mapper.rank(transcripts[0])
            match = mapper.accept(ranked)
            semantic_result.update(
                {
                    "query_inference_ms": semantic_inference_ms,
                    "accepted": match is not None,
                    "rejection_reason": None if match is not None else "score_or_margin",
                    "match": asdict(match) if match is not None else None,
                    "top3": [asdict(item) for item in ranked[:3]],
                }
            )
        else:
            semantic_result.update(
                {
                    "query_inference_ms": None,
                    "accepted": False,
                    "rejection_reason": "empty_asr_transcript",
                    "match": None,
                    "top3": [],
                }
            )
        result["semantic"] = semantic_result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing tokenizer.json and quantized ONNX graphs",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_MODEL_DIR / "jfk.wav",
        help="Uncompressed PCM16 WAV file (mono/stereo and non-16 kHz are supported)",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0, help="0 uses ORT defaults")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Map the transcript to a canonical COCO label with MiniLM",
    )
    parser.add_argument(
        "--embedding-model-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_MODEL_DIR,
    )
    parser.add_argument(
        "--semantic-catalog",
        type=Path,
        default=DEFAULT_SEMANTIC_CATALOG,
    )
    parser.add_argument(
        "--semantic-vector-cache",
        type=Path,
        default=DEFAULT_SEMANTIC_VECTOR_CACHE,
    )
    parser.add_argument("--embedding-threads", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.max_new_tokens < 1:
        parser.error("--runs and --max-new-tokens must be positive; --warmup cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    median = result["median"]
    print(f"Transcript: {result['transcript']}")
    print(f"Audio: {result['audio_seconds']:.3f} s")
    print(f"Model load: {result['load_seconds']:.3f} s")
    print(f"Encoder median: {median['encoder_seconds']:.3f} s")
    print(f"Decoder median: {median['decoder_seconds']:.3f} s")
    print(f"End-to-end median: {median['inference_seconds']:.3f} s")
    print(f"RTF median: {median['rtf']:.3f}x (lower is faster)")
    print(f"RTFx median: {1.0 / median['rtf']:.2f}x realtime")
    if "semantic" in result:
        semantic = result["semantic"]
        if semantic["query_inference_ms"] is not None:
            print(f"Embedding: {semantic['query_inference_ms']:.2f} ms")
        if semantic["accepted"]:
            match = semantic["match"]
            print(
                "Semantic match: "
                f"{result['transcript']!r} -> {match['display_name']!r} "
                f"(COCO {match['class_id']}, score={match['score']:.3f}, "
                f"margin={match['margin']:.3f})"
            )
        else:
            print(f"Semantic match: rejected ({semantic['rejection_reason']})")


if __name__ == "__main__":
    main()
