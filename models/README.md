# SmartDrawer model artifacts

Downloaded artifacts:

| Artifact | Source | SHA-256 | Status |
|---|---|---|---|
| `midas_v2_1_small_quant_vela.tflite` | [NXP MiDaS v2.1 Small](https://huggingface.co/nxp/midas-v2-imx) for i.MX93 LF-6.18.20 | `2719ff97aba7b31f61007b85162a6cb34dc1895e7b8e9eb102a90b861c6d7aac` | Vela artifact; board delegate benchmark still required |
| `yolov8n_coco_int8.tflite` | [EdgeFirst YOLOv8 COCO](https://huggingface.co/EdgeFirst/yolov8-det) | `357ffdc968542da6c74f3a83eb566a85be4f33663cb5b866521bcc80354e237c` | Full INT8 COCO model; compile/validate with i.MX93 Vela before claiming NPU |
| `coco_labels.txt` | same YOLO repository | `bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a` | 80 labels from the same artifact family |
| `reference/lp_kws_detection_en.elf` | [NXP low-power KWS assets](https://github.com/nxp-imx-support/nxp-demo-experience-assets) | `57deace46b2dd9f3089ed2dab023d6ec61b9fa53e75e45a67f64bedb4e5da0c2` | English reference only; not used for Mandarin MVP |

The Mandarin wake-word/20-class KWS artifact is **not publicly available in
`Ref/` or the NXP sample assets**. Generate it with the NXP VIT Model
Generation Tool (Chinese/Mandarin, i.MX93, Linux BSP) before hardware
acceptance. The simulator intentionally does not pretend the English model is
an equivalent replacement, and it never falls back to PC/cloud inference.

For the detector, the downloaded file is not yet marked Vela-compiled. On the
board, compile it with the installed i.MX93 Vela version and retain the
benchmark/delegate log before setting `vela_compiled` to true in
`config/model_manifest.json`.
