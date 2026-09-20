# Moonshine Tiny Q8 ONNX test bundle

This directory contains the files used by `moonshine_onnx_probe.py`.

- ONNX source: `onnx-community/moonshine-tiny-ONNX`
- Pinned revision: `a6da1241cd305dcd64eab1edbd615f2bb9aabb95`
- Upstream model: `moonshine-ai/moonshine-tiny` (27M parameters, English)
- Runtime graphs: `onnx/encoder_model_quantized.onnx` and
  `onnx/decoder_model_quantized.onnx`
- Test audio: the 44.1 kHz stereo JFK WAV used in the Hugging Face
  Transformers.js documentation; the probe downmixes and resamples it to
  16 kHz.

The supplied OpenASR repository (`OpenASR/moonshine-tiny`) contains `.oasr`
GGUF-backed packs, not ONNX files. This bundle uses the Hugging Face ONNX
Community export of the same upstream Moonshine Tiny model so that it can be
tested with ONNX Runtime.

Run the bundled test:

```bash
uv run python moonshine_onnx_probe.py
```

Run with two CPU threads to match the target's two Cortex-A55 core count (the
host result is not a substitute for a board benchmark):

```bash
uv run python moonshine_onnx_probe.py --threads 2 --json
```

The repository ignores `.onnx` binaries. To restore the exact Q8 graphs, run:

```bash
uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="onnx-community/moonshine-tiny-ONNX", revision="a6da1241cd305dcd64eab1edbd615f2bb9aabb95", local_dir="models/moonshine-tiny-onnx-q8", allow_patterns=["onnx/encoder_model_quantized.onnx", "onnx/decoder_model_quantized.onnx"])'
```

Artifact hashes:

| File | SHA-256 |
|---|---|
| `encoder_model_quantized.onnx` | `c6fc4b7bc5af75c0591fd157a1f3829b533d18e9769a888fd95a62e470dd4f4a` |
| `decoder_model_quantized.onnx` | `2e7db65f157a0c1b7a2ca2e9bf682766de9304664df8e74605838ea75751affa` |
| `jfk.wav` | `aa81c2552465568567e670f3823117e633900d16bd6202346a72f3c8464c74c8` |

Validated on 2026-09-20 with ONNX Runtime CPUExecutionProvider on Apple
Silicon. The 11.0-second JFK clip produced the expected transcript. Across
three measured runs after one warmup, median inference was 0.239 seconds
(RTF 0.0218, about 46x realtime) with ONNX Runtime's default thread settings.
Restricting the same Mac to two inference threads gave a five-run median of
0.351 seconds (RTF 0.0319). These are host smoke-test measurements, not i.MX93
or Cortex-A55 benchmark results.

This is an English ASR model. It is not a replacement for the project's
Mandarin keyword-spotting model.
