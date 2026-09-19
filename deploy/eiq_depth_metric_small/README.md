# i.MX93 eIQ conversion bundle — Depth Anything V2 Metric Small

This folder is self-contained for eIQ AI Toolkit conversion.  Copy the entire
`eiq_depth_metric_small` directory to the host where the eIQ AI Toolkit backend
is running (for example, the i.MX93 development environment).  It does not
need PyTorch or the original checkpoint.

## Contents

- `model/depth_anything_v2_metric_hypersim_vits_518.onnx` — checked FP32 ONNX.
- `model/depth_anything_v2_metric_hypersim_vits_518.eiq.json` — input/output contract.
- `calibration/depth_metric_small_calibration.zip` — two prepared calibration
  tensors made from `IMG_7413.jpeg` and `IMG_7412.jpeg`.
- `scripts/eiq_quantize_to_tflite.py` — uploads the model and calibration data,
  runs `ONNX2Quant`, then `TFLiteConversion`, and downloads the artifacts.
- `artifacts/` — generated INT8 ONNX and TFLite files will be written here.

Original JPEGs are not included: their EXIF metadata may include GPS.  The
calibration ZIP contains only normalized NumPy tensors.  Its provenance is in
`calibration/manifest.json`.

## Run on the i.MX93 / eIQ host

Start the eIQ AI Toolkit backend first, then run these commands from this
folder:

```sh
uv sync

# Verify paths and the calibration archive only; this does not contact eIQ.
uv run python scripts/eiq_quantize_to_tflite.py \
  --calibration calibration/depth_metric_small_calibration.zip

# Upload and convert.  The default backend is http://localhost:8000.
uv run python scripts/eiq_quantize_to_tflite.py \
  --backend http://localhost:8000 \
  --calibration calibration/depth_metric_small_calibration.zip \
  --execute
```

If the backend is on another machine, replace `--backend` with its reachable
URL.  The backend must expose the `ONNX2Quant` and `TFLiteConversion` passes.
On success, retrieve:

```text
artifacts/depth_anything_metric_small_int8.onnx
artifacts/depth_anything_v2_metric_hypersim_vits_518_int8.tflite
```

## Model contract

Input is one `float32` tensor named `image`, shape `[1, 3, 518, 518]` in RGB
NCHW. Resize to 518 × 518, divide pixels by 255, then normalize with ImageNet
mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

Output is `depth_meters`, shape `[1, 518, 518]`; smaller values mean nearer.
This is a monocular metric estimate, not a calibrated depth sensor.

## Calibration note

The two supplied images prove the conversion flow, but are not enough for a
production INT8 calibration set. Before deployment, make a new archive using
roughly 50–100 representative C270 camera frames (lighting, drawer positions,
and contents) with exactly the same preprocessing.  Only enable an NPU/Vela
delegate after the board-specific compiler output and accuracy/latency checks
have succeeded.
