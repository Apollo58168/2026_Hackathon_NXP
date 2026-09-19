# Depth Anything Metric Small → eIQ AI Toolkit

## Transfer to the eIQ host / i.MX93 workflow machine

- `models/depth_anything_v2_metric_hypersim_vits_518.onnx`
- `models/depth_anything_v2_metric_hypersim_vits_518.eiq.json`
- `prepare_eiq_calibration.py`
- `eiq_quantize_to_tflite.py`
- `depth_anything_metric.py`
- `pyproject.toml` and `uv.lock`

The ONNX model has one fixed input named `image`: float32 RGB NCHW
`[1, 3, 518, 518]`. Its output is `depth_meters`: float32 `[1, 518, 518]`.

## Prepare calibration inputs

Put 50–100 representative C270 images into one directory. Include closed and
open drawers, expected lighting, empty interiors, and the kinds of objects
that will be used in the demo. Do not use synthetic gradients or unrelated
internet images for quantization.

```bash
uv sync
uv run python prepare_eiq_calibration.py \
  --images /path/to/c270-calibration-images \
  --output artifacts/eiq_depth_metric_small_calibration.zip
```

The ZIP contains `image/sample_XXXX.npy`. Each array is already preprocessed
to the exact ONNX contract, so AI Hub must not resize, reorder channels, or
apply ImageNet normalization a second time.

## Submit the two eIQ passes

Start the eIQ AI Toolkit backend first. The dry run is local and creates no
server resources; `--execute` uploads the model and dataset, then starts jobs.

```bash
uv run python eiq_quantize_to_tflite.py \
  --backend http://localhost:8000 \
  --calibration artifacts/eiq_depth_metric_small_calibration.zip

uv run python eiq_quantize_to_tflite.py \
  --backend http://localhost:8000 \
  --calibration artifacts/eiq_depth_metric_small_calibration.zip \
  --execute
```

Expected artifacts:

- `artifacts/depth_anything_metric_small_int8.onnx`
- `artifacts/depth_anything_v2_metric_hypersim_vits_518_int8.tflite`

Only after eIQ reports a successful TFLite conversion should the resulting
TFLite be processed with the board's matching Vela version and benchmarked
with the Ethos-U delegate log. A successful conversion alone does not prove
full NPU delegation.
