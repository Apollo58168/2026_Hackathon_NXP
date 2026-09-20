#!/usr/bin/env python3
"""Transcribe WAV audio with the fixed-window Moonshine Tiny LiteRT model."""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # NXP BSP images may expose the smaller TFLite runtime.
    from tflite_runtime.interpreter import Interpreter


WINDOW_SAMPLES = 80_000
MAX_TOKENS = 64
START_TOKEN = 1
EOS_TOKEN = 2
ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "models" / "moonshine_tiny_5s_i8.tflite"
DEFAULT_TOKENIZER = ROOT / "models" / "moonshine-tiny-onnx-q8" / "tokenizer.json"
DEFAULT_AUDIO = ROOT / "models" / "moonshine-tiny-onnx-q8" / "jfk.wav"


def load_wav(path: Path) -> tuple[np.ndarray, float]:
    """Load uncompressed PCM16 WAV as 16 kHz mono float32."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        compression = wav_file.getcomptype()
        raw = wav_file.readframes(frame_count)
    if compression != "NONE" or sample_width != 2:
        raise ValueError(f"{path} must be uncompressed PCM16 WAV")

    audio = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
    audio = audio.astype(np.float32).mean(axis=1) / 32768.0
    duration_seconds = len(audio) / sample_rate
    if sample_rate != 16_000:
        output_samples = round(len(audio) * 16_000 / sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, output_samples),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return np.ascontiguousarray(audio), duration_seconds


class MoonshineTFLite:
    def __init__(self, model_path: Path, tokenizer_path: Path, threads: int) -> None:
        kwargs = {"model_path": str(model_path)}
        if threads > 0:
            kwargs["num_threads"] = threads
        self.interpreter = Interpreter(**kwargs)
        self.interpreter.allocate_tensors()

        signatures = self.interpreter.get_signature_list()
        if set(signatures) != {"encode", "decode"}:
            raise RuntimeError(f"Expected encode/decode signatures, got {signatures}")
        self.encode = self.interpreter.get_signature_runner("encode")
        self.decode = self.interpreter.get_signature_runner("decode")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        causal = np.tril(np.ones((MAX_TOKENS, MAX_TOKENS), dtype=bool))
        self.mask = np.where(causal, 0.0, -1e9).astype(np.float32)[None, None]

    def transcribe_window(self, audio: np.ndarray) -> tuple[str, dict[str, float]]:
        window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        window[: len(audio)] = audio

        started = time.perf_counter()
        states = self.encode(args_0=window[None, :])["output_0"]
        encoder_seconds = time.perf_counter() - started

        tokens = np.zeros((1, MAX_TOKENS), dtype=np.int32)
        tokens[0, 0] = START_TOKEN
        decoded: list[int] = []
        decoder_started = time.perf_counter()
        for position in range(1, MAX_TOKENS):
            logits = self.decode(
                args_0=states,
                args_1=tokens,
                args_2=self.mask,
            )["output_0"]
            next_token = int(np.argmax(logits[0, position - 1]))
            if next_token == EOS_TOKEN:
                break
            tokens[0, position] = next_token
            decoded.append(next_token)
        decoder_seconds = time.perf_counter() - decoder_started

        text = self.tokenizer.decode(decoded, skip_special_tokens=True).strip()
        return text, {
            "encoder_seconds": encoder_seconds,
            "decoder_seconds": decoder_seconds,
            "total_seconds": encoder_seconds + decoder_seconds,
            "tokens": len(decoded),
        }

    def transcribe(self, audio: np.ndarray) -> tuple[str, list[dict[str, float]]]:
        parts: list[str] = []
        timings: list[dict[str, float]] = []
        for start in range(0, max(len(audio), 1), WINDOW_SAMPLES):
            text, window_timing = self.transcribe_window(
                audio[start : start + WINDOW_SAMPLES]
            )
            if text:
                parts.append(text)
            timings.append(window_timing)
        return " ".join(parts), timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    audio, audio_seconds = load_wav(args.audio)
    load_started = time.perf_counter()
    model = MoonshineTFLite(args.model, args.tokenizer, args.threads)
    load_seconds = time.perf_counter() - load_started
    transcript, windows = model.transcribe(audio)
    inference_seconds = sum(item["total_seconds"] for item in windows)
    result = {
        "model": str(args.model),
        "runtime": "LiteRT CPU",
        "threads": args.threads,
        "audio_seconds": audio_seconds,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "rtf": inference_seconds / audio_seconds,
        "transcript": transcript,
        "windows": windows,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Transcript: {transcript}")
        print(f"Audio: {audio_seconds:.3f} s")
        print(f"Model load: {load_seconds:.3f} s")
        print(f"Inference: {inference_seconds:.3f} s")
        print(f"RTF: {result['rtf']:.3f}x (lower is faster)")


if __name__ == "__main__":
    main()
