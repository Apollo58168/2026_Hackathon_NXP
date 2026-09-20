"""YOLOv8 INT8 TFLite detector for the i.MX93 Ethos-U delegate."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional, Sequence


class YoloV8TFLiteDetector:
    """Decode the standard YOLOv8 detection head: 4 xywh + 80 class scores."""

    def __init__(
        self,
        model_path: str,
        *,
        delegate_path: Optional[str] = "/usr/lib/libethosu_delegate.so",
        confidence: float = 0.50,
        nms_iou: float = 0.50,
        enabled_labels: Iterable[str],
        threads: int = 2,
    ) -> None:
        self.np = __import__("numpy")
        self.cv2 = __import__("cv2")
        import tflite_runtime.interpreter as tflite
        from .ssdlite_detector import COCO_LABELS

        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if not 0.0 <= nms_iou <= 1.0:
            raise ValueError("nms_iou must be in [0, 1]")
        self.model_path = model_path
        self.confidence = float(confidence)
        self.nms_iou = float(nms_iou)
        self.enabled_labels = frozenset(enabled_labels)
        self.labels = COCO_LABELS
        self._delegate = tflite.load_delegate(delegate_path) if delegate_path else None
        kwargs = {"model_path": model_path, "num_threads": threads}
        if self._delegate is not None:
            kwargs["experimental_delegates"] = [self._delegate]
        self.interpreter = tflite.Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        input_shape = tuple(int(value) for value in self.input["shape"])
        output_shape = tuple(int(value) for value in self.output["shape"])
        if len(input_shape) != 4 or input_shape[0] != 1:
            raise ValueError(f"expected four-dimensional YOLO input, got {input_shape}")
        if input_shape[-1] == 3:
            self.input_layout = "NHWC"
            self.input_height, self.input_width = input_shape[1:3]
        elif input_shape[1] == 3:
            self.input_layout = "NCHW"
            self.input_height, self.input_width = input_shape[2:4]
        else:
            raise ValueError(f"expected three-channel YOLO input, got {input_shape}")
        if len(output_shape) != 3 or output_shape[0] != 1 or 84 not in output_shape:
            raise ValueError(f"expected standard YOLOv8 [1,84,N] output, got {output_shape}")
        self.backend = "NPU" if self._delegate is not None else "CPU"
        self.calls = 0

    def _letterbox(self, rgb):
        height, width = rgb.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = self.cv2.resize(rgb, (resized_width, resized_height), interpolation=self.cv2.INTER_LINEAR)
        pad_x = (self.input_width - resized_width) / 2.0
        pad_y = (self.input_height - resized_height) / 2.0
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        image = self.cv2.copyMakeBorder(
            resized, top, bottom, left, right, self.cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        if image.shape[:2] != (self.input_height, self.input_width):
            raise RuntimeError(f"letterbox produced {image.shape[:2]}")
        return image, scale, float(left), float(top)

    def _quantize_input(self, image):
        np = self.np
        real = image.astype(np.float32) / 255.0
        dtype = self.input["dtype"]
        if self.input_layout == "NCHW":
            real = real.transpose(2, 0, 1)
        if np.issubdtype(dtype, np.floating):
            return real[None].astype(dtype)
        scale, zero = self.input.get("quantization", (0.0, 0))
        if not scale:
            raise ValueError("integer YOLO input has no quantization scale")
        limits = np.iinfo(dtype)
        # Quantize normalized RGB. Direct image.astype(int8) would wrap pixels
        # above 127 and is not equivalent to this affine conversion.
        quantized = np.rint(real / float(scale) + float(zero))
        return np.clip(quantized, limits.min, limits.max).astype(dtype)[None]

    def _dequantize_output(self, raw):
        values = raw.astype(self.np.float32)
        scale, zero = self.output.get("quantization", (0.0, 0))
        if scale:
            values = (values - float(zero)) * float(scale)
        values = values[0]
        if values.shape[0] != 84:
            values = values.T
        if values.shape[0] != 84:
            raise RuntimeError(f"unexpected YOLO output shape: {values.shape}")
        return values

    @staticmethod
    def _iou(first: Sequence[float], second: Sequence[float]) -> float:
        x0, y0 = max(first[0], second[0]), max(first[1], second[1])
        x1, y1 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        return intersection / max(first_area + second_area - intersection, 1e-9)

    def _decode(self, values, frame_shape, scale: float, pad_x: float, pad_y: float):
        from .ssdlite_detector import DetectedObject

        np = self.np
        boxes = values[:4].T
        class_scores = values[4:].T  # YOLOv8 has no separate objectness channel.
        if class_scores.size and (float(class_scores.min()) < -0.05 or float(class_scores.max()) > 1.05):
            raise RuntimeError(
                f"YOLOv8 class scores outside [0,1]: {class_scores.min():.3f}..{class_scores.max():.3f}"
            )
        class_scores = np.clip(class_scores, 0.0, 1.0)
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        indices = np.flatnonzero(confidences >= self.confidence)
        self.last_candidate_count = int(len(indices))
        self.last_score_range = (float(class_scores.min()), float(class_scores.max()))
        if len(indices) > 300:
            raise RuntimeError(
                f"implausible YOLO output: {len(indices)} boxes exceed confidence "
                f"{self.confidence:.2f}; model export or Vela runtime is incompatible"
            )
        normalized = boxes.size > 0 and float(np.max(np.abs(boxes))) <= 2.0
        frame_height, frame_width = frame_shape[:2]
        candidates = []
        for index in indices:
            class_id = int(class_ids[index])
            if class_id >= len(self.labels):
                continue
            label = self.labels[class_id]
            if label not in self.enabled_labels:
                continue
            center_x, center_y, width, height = (float(value) for value in boxes[index])
            if normalized:
                center_x *= self.input_width
                width *= self.input_width
                center_y *= self.input_height
                height *= self.input_height
            x0 = (center_x - width / 2.0 - pad_x) / scale
            y0 = (center_y - height / 2.0 - pad_y) / scale
            x1 = (center_x + width / 2.0 - pad_x) / scale
            y1 = (center_y + height / 2.0 - pad_y) / scale
            x0, x1 = sorted((max(0.0, min(frame_width, x0)), max(0.0, min(frame_width, x1))))
            y0, y1 = sorted((max(0.0, min(frame_height, y0)), max(0.0, min(frame_height, y1))))
            if x1 - x0 < 1.0 or y1 - y0 < 1.0:
                continue
            candidates.append(
                DetectedObject(
                    label,
                    class_id,
                    float(confidences[index]),
                    (round(x0), round(y0), round(x1 - x0), round(y1 - y0)),
                )
            )

        kept = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            candidate_xyxy = (
                candidate.box[0],
                candidate.box[1],
                candidate.box[0] + candidate.box[2],
                candidate.box[1] + candidate.box[3],
            )
            if any(
                existing.class_id == candidate.class_id
                and self._iou(
                    candidate_xyxy,
                    (
                        existing.box[0],
                        existing.box[1],
                        existing.box[0] + existing.box[2],
                        existing.box[1] + existing.box[3],
                    ),
                )
                > self.nms_iou
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return kept

    def detect(self, rgb):
        if getattr(rgb, "ndim", 0) != 3 or rgb.shape[2] != 3:
            raise ValueError("expected HWC RGB image")
        image, scale, pad_x, pad_y = self._letterbox(rgb)
        self.interpreter.set_tensor(self.input["index"], self._quantize_input(image))
        started = time.perf_counter()
        self.interpreter.invoke()
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        values = self._dequantize_output(self.interpreter.get_tensor(self.output["index"]))
        self.calls += 1
        return self._decode(values, rgb.shape, scale, pad_x, pad_y)
