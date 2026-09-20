#!/usr/bin/env python3
"""Listen for "hello", then cosine-map the following words to COCO.

This is deliberately not an intent/NLP pipeline.  Moonshine produces text,
the fixed wake phrase gates the command, and only the words after the wake
phrase are embedded and compared with the YOLOv8 COCO vector catalog.

Speech recognition uses the fixed-window Moonshine Tiny i8 TFLite model so
the same runtime can be deployed to the target board.
"""
from __future__ import annotations

import argparse
import math
import platform
import re
import subprocess
import time
from collections import deque
from pathlib import Path
from threading import Event
from typing import Iterator

import numpy as np

from moonshine_tflite_probe import DEFAULT_MODEL, DEFAULT_TOKENIZER, MoonshineTFLite, load_wav
from semantic_embedding import EmbeddingMapper, EmbeddingMatch, OnnxSentenceEncoder


ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2-onnx-q8"
DEFAULT_CATALOG = ROOT / "config" / "semantic_catalog_en.json"
DEFAULT_VECTOR_CACHE = ROOT / "config" / "semantic_catalog_en_minilm_q8.npz"
SAMPLE_RATE = 16_000
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def words(text: str) -> list[str]:
    """Normalize punctuation/case without performing language understanding."""
    return WORD_RE.findall(text.casefold())


def extract_after_wake(text: str, wake_phrase: str) -> str | None:
    """Return normalized words following the first exact wake-token sequence."""
    transcript_words = words(text)
    wake_words = words(wake_phrase)
    if not wake_words:
        raise ValueError("wake phrase must contain at least one word")
    limit = len(transcript_words) - len(wake_words) + 1
    for index in range(max(limit, 0)):
        if transcript_words[index : index + len(wake_words)] == wake_words:
            return " ".join(transcript_words[index + len(wake_words) :])
    return None


class WakeCommandProcessor:
    def __init__(
        self,
        mapper: EmbeddingMapper,
        *,
        wake_phrase: str = "hello",
        wake_timeout: float = 1.0,
        cooldown: float = 3.0,
    ) -> None:
        self.mapper = mapper
        self.wake_phrase = wake_phrase
        self.wake_timeout = wake_timeout
        self.cooldown = cooldown
        self.armed_until = 0.0
        self.cooldown_until = 0.0

    def process(self, transcript: str, *, now: float | None = None) -> tuple[str, EmbeddingMatch | None] | None:
        """Process one ASR utterance.

        Returns ``None`` when the wake phrase was not present, ``("", None)``
        when the wake phrase arms the next utterance, or ``(query, match)``.
        """
        timestamp = time.monotonic() if now is None else now
        armed = timestamp <= self.armed_until
        if not armed and timestamp < self.cooldown_until:
            return None

        query = extract_after_wake(transcript, self.wake_phrase)
        if query is not None:
            if not query:
                self.armed_until = timestamp + self.wake_timeout
                self.cooldown_until = self.armed_until + self.cooldown
                return "", None
            self.armed_until = 0.0
            self.cooldown_until = timestamp + self.cooldown
        elif armed:
            query = " ".join(words(transcript))
            self.armed_until = 0.0
            if not query:
                return "", None
        else:
            return None

        ranked, _ = self.mapper.rank(query)
        return query, self.mapper.accept(ranked)


def microphone_command(input_format: str, input_device: str) -> list[str]:
    if input_format == "auto":
        input_format = "avfoundation" if platform.system() == "Darwin" else "alsa"
    if input_format == "alsa":
        return [
            "arecord",
            "-q",
            "-D",
            input_device,
            "-f",
            "S16_LE",
            "-c",
            "1",
            "-r",
            str(SAMPLE_RATE),
            "-t",
            "raw",
        ]
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        input_format,
        "-i",
        input_device,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]


def live_utterances(
    command: list[str],
    *,
    threshold_db: float,
    silence_ms: int,
    preroll_ms: int,
    min_speech_ms: int,
    max_utterance_seconds: float,
    stop_event: Event | None = None,
) -> Iterator[np.ndarray]:
    """Yield microphone utterances using a small energy-based endpoint detector."""
    frame_ms = 20
    frame_samples = SAMPLE_RATE * frame_ms // 1000
    frame_bytes = frame_samples * 2
    preroll_frames = max(1, preroll_ms // frame_ms)
    silence_frames = max(1, silence_ms // frame_ms)
    min_frames = max(1, min_speech_ms // frame_ms)
    max_frames = max(1, round(max_utterance_seconds * 1000 / frame_ms))

    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg did not provide a PCM output pipe")

    before_speech: deque[bytes] = deque(maxlen=preroll_frames)
    utterance: list[bytes] = []
    speech_frames = 0
    trailing_silence = 0
    active = False
    try:
        while stop_event is None or not stop_event.is_set():
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) != frame_bytes:
                if process.poll() is not None:
                    raise RuntimeError(f"ffmpeg microphone capture exited with {process.returncode}")
                continue

            samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
            rms = math.sqrt(float(np.mean(samples * samples))) / 32768.0
            level_db = 20.0 * math.log10(max(rms, 1e-9))
            voiced = level_db >= threshold_db

            if not active:
                before_speech.append(chunk)
                if not voiced:
                    continue
                active = True
                utterance = list(before_speech)
                speech_frames = 1
                trailing_silence = 0
                continue

            utterance.append(chunk)
            if voiced:
                speech_frames += 1
                trailing_silence = 0
            else:
                trailing_silence += 1

            endpoint = trailing_silence >= silence_frames and speech_frames >= min_frames
            timed_out = len(utterance) >= max_frames
            if endpoint or timed_out:
                pcm = b"".join(utterance)
                yield np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                before_speech.clear()
                utterance = []
                speech_frames = 0
                trailing_silence = 0
                active = False
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def print_result(
    transcript: str,
    outcome: tuple[str, EmbeddingMatch | None] | None,
    *,
    wake_phrase: str,
) -> None:
    print(f"ASR: {transcript!r}")
    if outcome is None:
        print(f"Wake: ignored (say {wake_phrase!r} first)")
        return
    query, match = outcome
    if not query:
        print("Wake: detected; listening for the object name...")
        return
    print(f"Query after wake phrase: {query!r}")
    if match is None:
        print("Result: rejected by cosine score/margin thresholds")
        return
    print(
        f"Result: {match.display_name} "
        f"(YOLOv8={match.canonical_label!r}, class_id={match.class_id}, "
        f"cosine={match.score:.3f}, margin={match.margin:.3f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Skip ASR and test a transcript")
    source.add_argument("--audio", type=Path, help="Transcribe one PCM16 WAV")
    parser.add_argument("--wake-phrase", default="hello")
    parser.add_argument(
        "--wake-timeout",
        type=float,
        default=1.0,
        help="Seconds to accept one follow-up object name after the wake phrase",
    )
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds before accepting another wake phrase")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the Moonshine Tiny i8 TFLite model",
    )
    parser.add_argument("--embedding-model-dir", type=Path, default=DEFAULT_EMBEDDING_MODEL_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--vector-cache", type=Path, default=DEFAULT_VECTOR_CACHE)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--input-format", choices=("auto", "avfoundation", "alsa"), default="auto")
    parser.add_argument("--input-device", help="macOS ':0'; Linux usually 'default'")
    parser.add_argument("--vad-threshold-db", type=float, default=-42.0)
    parser.add_argument("--silence-ms", type=int, default=500)
    parser.add_argument("--preroll-ms", type=int, default=120)
    parser.add_argument("--min-speech-ms", type=int, default=200)
    parser.add_argument("--max-utterance-seconds", type=float, default=1.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.wake_timeout <= 0 or args.cooldown < 0:
        parser.error("threads and wake-timeout must be positive; cooldown cannot be negative")
    return args


def self_test() -> None:
    class FakeMapper:
        def rank(self, text: str) -> tuple[list[str], float]:
            return [text], 0.0

        def accept(self, matches: list[str]) -> str:
            return matches[0]

    processor = WakeCommandProcessor(FakeMapper(), wake_timeout=1.0, cooldown=3.0)  # type: ignore[arg-type]
    assert processor.process("hello", now=10.0) == ("", None)
    assert processor.process("telephone", now=10.5) == ("telephone", "telephone")
    assert processor.process("hello remote", now=12.0) is None
    assert processor.process("hello remote", now=14.1) == ("remote", "remote")
    print("hey_drawer: self-test OK")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    embedding_encoder = OnnxSentenceEncoder(
        args.embedding_model_dir / "onnx" / "model_qint8_arm64.onnx",
        args.embedding_model_dir / "tokenizer.json",
        threads=args.threads,
    )
    mapper = EmbeddingMapper.from_json(
        embedding_encoder,
        args.catalog,
        vector_cache=args.vector_cache,
    )
    processor = WakeCommandProcessor(
        mapper,
        wake_phrase=args.wake_phrase,
        wake_timeout=args.wake_timeout,
        cooldown=args.cooldown,
    )

    if args.text is not None:
        print_result(
            args.text,
            processor.process(args.text),
            wake_phrase=args.wake_phrase,
        )
        return

    moonshine = MoonshineTFLite(
        args.model,
        DEFAULT_TOKENIZER,
        args.threads,
    )
    if args.audio is not None:
        audio, _ = load_wav(args.audio)
        transcript, _ = moonshine.transcribe(audio)
        print_result(
            transcript,
            processor.process(transcript),
            wake_phrase=args.wake_phrase,
        )
        return

    input_format = args.input_format
    if input_format == "auto":
        input_format = "avfoundation" if platform.system() == "Darwin" else "alsa"
    input_device = args.input_device or (
        ":0" if input_format == "avfoundation" else "plughw:CARD=WEBCAM,DEV=0"
    )
    command = microphone_command(input_format, input_device)
    print(
        f"Listening on {input_format}:{input_device}. "
        f"Say '{args.wake_phrase}, remote'. Press Ctrl-C to stop."
    )
    try:
        for audio in live_utterances(
            command,
            threshold_db=args.vad_threshold_db,
            silence_ms=args.silence_ms,
            preroll_ms=args.preroll_ms,
            min_speech_ms=args.min_speech_ms,
            max_utterance_seconds=args.max_utterance_seconds,
        ):
            transcript, _ = moonshine.transcribe(audio)
            if transcript:
                print_result(
                    transcript,
                    processor.process(transcript),
                    wake_phrase=args.wake_phrase,
                )
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
