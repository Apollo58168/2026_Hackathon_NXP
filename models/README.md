# SmartDrawer model artifacts

Downloaded artifacts:

| Artifact | Source | SHA-256 | Status |
|---|---|---|---|
| `midas_v2_1_small_quant_vela.tflite` | [NXP MiDaS v2.1 Small](https://huggingface.co/nxp/midas-v2-imx) for i.MX93 LF-6.18.20 | `2719ff97aba7b31f61007b85162a6cb34dc1895e7b8e9eb102a90b861c6d7aac` | Vela artifact; live NNStreamer/Ethos-U delegate run validated |
| `ssdlite_mobilenet_v2_coco_quant_uint8_float32_no_postprocess_vela.tflite` | NXP GoPoint object-detection download | `57d7506f0c52ee4d910b24002bcdaebd2cb11552796161762568b33221749234` | Selected board runtime detector; Ethos-U delegate validated |
| `coco_labels_list.txt` | same GoPoint model family | `c7e79c855f73cbba9f33d649d60e1676eb0a974021a41696d1ac0d4b7f7e0211` | 91-entry SSD label map including `???` background placeholders |
| `box_priors.txt` | same GoPoint model family | `e4e4e6e43a9a8dbedffc7e5902c6ee97c631526234a456bfdb1b1d011d367aa9` | Four-row, 1917-anchor SSD priors |
| `yolov8n_coco_int8.tflite` | [EdgeFirst YOLOv8 COCO](https://huggingface.co/EdgeFirst/yolov8-det) | `357ffdc968542da6c74f3a83eb566a85be4f33663cb5b866521bcc80354e237c` | Retained reference; not selected because GoPoint's SSDLite path is the verified board path |
| `coco_labels.txt` | same YOLO repository | `bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a` | 80 labels for the retained YOLO reference |
| `reference/lp_kws_detection_en.elf` | [NXP low-power KWS assets](https://github.com/nxp-imx-support/nxp-demo-experience-assets) | `57deace46b2dd9f3089ed2dab023d6ec61b9fa53e75e45a67f64bedb4e5da0c2` | English reference only; not used for Mandarin MVP |

The Mandarin wake-word/20-class KWS artifact is **not publicly available in
`Ref/` or the NXP sample assets**. Generate it with the NXP VIT Model
Generation Tool (Chinese/Mandarin, i.MX93, Linux BSP) before hardware
acceptance. The simulator intentionally does not pretend the English model is
an equivalent replacement, and it never falls back to PC/cloud inference.

The selected detector follows the GoPoint decoder contract: sigmoid SSD
logits, scales `10,10,5,5`, the supplied 1917 priors, and CPU NMS after the
Ethos-U delegated inference. A 20-run Python adapter benchmark measured
23.58/23.72/24.05 ms min/median/max on the board; this includes preprocessing
and postprocessing, not just the NPU kernel.

Hardware probe entry point:

```bash
python3 /root/midas_nnstreamer.py --transaction-roi X Y W H
```

`X Y W H` are one fixed 256x256 analysis ROI. The probe captures stable depth
A/B, aligns B with the ROI border, extracts only the changed square crop, and
calls the SSDLite adapter once. It intentionally does not claim layer
calibration or SQLite commit until real drawer baselines are measured.
