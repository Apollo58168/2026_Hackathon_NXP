# all-MiniLM-L6-v2 ONNX QInt8 (ARM64)

- Source: `sentence-transformers/all-MiniLM-L6-v2`
- Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Model: `onnx/model_qint8_arm64.onnx`
- SHA-256: `4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474`
- License: Apache-2.0
- Runtime: ONNX Runtime CPU, 2 threads on i.MX93 Cortex-A55
- Output: 384-dimensional token embeddings
- Pooling: attention-masked mean pooling followed by L2 normalization

The model binary is intentionally ignored by Git. Download the pinned model
and keep it at the path above before deployment.

Run a text-only check from the repository root:

```bash
uv run python semantic_embedding_probe.py phone
```

Run Moonshine and semantic mapping together:

```bash
uv run python moonshine_onnx_probe.py \
  --audio /tmp/moonshine-mic.wav --threads 2 --semantic
```

Run the wake-gated microphone loop (no intent parser):

```bash
uv run python hey_drawer.py
```

It only accepts words after the exact normalized wake phrase. For example,
`hello, remote` embeds `remote`, while `remote` by itself is ignored.
When `hello` is spoken alone, the following object name must arrive within
one second. The listener then ignores new wake phrases for three seconds.
Use `--input-device ':0'` on macOS when needed, or an ALSA device such as
`--input-device 'hw:3,0'` on i.MX93.

The vector catalog starts from all 80 YOLOv8 COCO labels and contains 149
names or common aliases. For this drawer, `keyboard` is canonicalized to
`remote` (class 65), leaving 79 runtime classes. The precomputed vector database is
`config/semantic_catalog_en_minilm_q8.npz`; its source vocabulary remains the
human-editable `config/semantic_catalog_en.json`.

Expected mappings:

```text
phone -> cellphone (COCO class 67)
computer -> laptop (COCO class 63)
tv remote -> remote (COCO class 65)
television controller -> remote (COCO class 65)
keyboard -> remote (COCO class 65)
```

After editing aliases, rebuild the vector database:

```bash
uv run python semantic_embedding_probe.py --rebuild-cache "tv remote"
```

On i.MX93 this graph is for ONNX Runtime on the two Cortex-A55 cores. It is
not a Vela/Ethos-U graph. Install or use NXP's ONNX Runtime build plus NumPy
and `tokenizers`, then copy this directory, `semantic_embedding.py`,
`semantic_embedding_probe.py`, `config/semantic_catalog_en.json`, and the
precomputed `.npz` vector database to the target.
