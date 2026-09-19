"""One-shot cropped SSD-Lite detector using the same model as GoPoint.

GoPoint's i.MX93 Object Detection demo uses the COCO SSD-Lite model plus the
Ethos-U delegate.  This adapter uses that exact model contract, but exposes
raw detections to SmartDrawer instead of scanning/displaying the whole camera
stream.  It is deliberately not a YOLO implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
COCO_CLASS_IDS = {label: index for index, label in enumerate(COCO_LABELS)}

DEFAULT_ENABLED_LABELS = frozenset(
    (
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "orange", "sandwich", "laptop", "mouse", "remote", "keyboard", "cell phone", "book",
        "clock", "scissors", "toothbrush",
    )
)


@dataclass(frozen=True)
class DetectedObject:
    label: str
    class_id: int
    confidence: float
    # x, y, width, height in the original RGB crop coordinates.
    box: tuple[int, int, int, int]


class SSDLiteDetector:
    """Run GoPoint SSD-Lite once on one already-selected RGB crop."""

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        priors_path: str,
        *,
        delegate_path: Optional[str] = "/usr/lib/libethosu_delegate.so",
        confidence: float = 0.50,
        nms_iou: float = 0.50,
        enabled_labels: Iterable[str] = DEFAULT_ENABLED_LABELS,
        threads: int = 2,
    ) -> None:
        self.np = __import__("numpy")
        self.cv2 = __import__("cv2")
        import tflite_runtime.interpreter as tflite

        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        if not Path(labels_path).is_file():
            raise FileNotFoundError(labels_path)
        if not Path(priors_path).is_file():
            raise FileNotFoundError(priors_path)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if not 0.0 <= nms_iou <= 1.0:
            raise ValueError("nms_iou must be in [0, 1]")

        self.model_path = model_path
        self.confidence = float(confidence)
        self.nms_iou = float(nms_iou)
        self.enabled_labels = frozenset(enabled_labels)
        self._delegate = tflite.load_delegate(delegate_path) if delegate_path else None
        kwargs = {"model_path": model_path, "num_threads": threads}
        if self._delegate is not None:
            kwargs["experimental_delegates"] = [self._delegate]
        self.interpreter = tflite.Interpreter(**kwargs)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.outputs = self.interpreter.get_output_details()
        shape = tuple(int(value) for value in self.input["shape"])
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ValueError("expected NHWC input, got %s" % (shape,))
        self.input_height, self.input_width = shape[1], shape[2]
        self.labels = Path(labels_path).read_text(encoding="utf-8").splitlines()
        prior_rows = []
        for line in Path(priors_path).read_text(encoding="utf-8").splitlines():
            values = [float(value) for value in line.replace(",", " ").split()]
            if values:
                prior_rows.append(values)
        self.priors = self.np.asarray(prior_rows, dtype=self.np.float32)
        if self.priors.shape[0] != 4:
            raise ValueError("SSD priors must have four rows")
        self.box_output = next((item for item in self.outputs if int(item["shape"][-1]) == 4), None)
        self.score_output = next((item for item in self.outputs if int(item["shape"][-1]) > 4), None)
        if self.box_output is None or self.score_output is None:
            raise ValueError("could not identify SSD box/score outputs")
        self.logit_threshold = math.log(self.confidence / (1.0 - self.confidence))
        self.backend = "NPU" if self._delegate is not None else "CPU"
        self.calls = 0

    def _dequantize(self, values: Any, details: dict[str, Any]) -> Any:
        scale, zero = details.get("quantization", (0.0, 0))
        if scale:
            return (values.astype(self.np.float32) - zero) * scale
        return values.astype(self.np.float32)

    def _resize(self, rgb: Any) -> tuple[Any, tuple[float, float], int, int]:
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("empty RGB crop")
        # Match GoPoint's PXP -> 300x300 contract.  DESIGN.md normally gives
        # us a square changed crop, but stretching also keeps this adapter
        # compatible with the board demo for rectangular test crops.
        scale_x = self.input_width / float(width)
        scale_y = self.input_height / float(height)
        resized = self.cv2.resize(rgb, (self.input_width, self.input_height), interpolation=self.cv2.INTER_LINEAR)
        return resized, (scale_x, scale_y), 0, 0

    def _input_tensor(self, image: Any) -> Any:
        dtype = self.input["dtype"]
        if self.np.issubdtype(dtype, self.np.floating):
            return (image.astype(self.np.float32) / 255.0)[None].astype(dtype)
        scale, zero = self.input.get("quantization", (0.0, 0))
        if not scale:
            return image[None].astype(dtype)
        # The GoPoint tensor is uint8 with scale 1/255: pass camera bytes
        # directly.  Quantizing those bytes a second time would saturate the
        # tensor at 255 and silently destroy detections.
        if abs(float(scale) - (1.0 / 255.0)) < 1e-6 and zero == 0:
            return image[None].astype(dtype)
        limits = self.np.iinfo(dtype)
        real = image.astype(self.np.float32) / 255.0
        return self.np.clip(self.np.rint(real / scale + zero), limits.min, limits.max).astype(dtype)[None]

    @staticmethod
    def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
        lx, ly, lw, lh = left
        rx, ry, rw, rh = right
        x0, y0 = max(lx, rx), max(ly, ry)
        x1, y1 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        union = lw * lh + rw * rh - intersection
        return intersection / union if union else 0.0

    def _nms(self, detections: Sequence[DetectedObject]) -> list[DetectedObject]:
        kept: list[DetectedObject] = []
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(self._iou(candidate.box, existing.box) > self.nms_iou for existing in kept):
                continue
            kept.append(candidate)
        return kept

    def detect(self, rgb_crop: Any) -> list[DetectedObject]:
        """Run exactly one inference on ``rgb_crop``; never scans another image."""
        if getattr(rgb_crop, "ndim", 0) != 3 or rgb_crop.shape[2] != 3:
            raise ValueError("expected HWC RGB crop")
        image, (scale_x, scale_y), pad_x, pad_y = self._resize(rgb_crop)
        self.interpreter.set_tensor(self.input["index"], self._input_tensor(image))
        self.interpreter.invoke()
        boxes = self._dequantize(self.interpreter.get_tensor(self.box_output["index"]), self.box_output)
        scores = self._dequantize(self.interpreter.get_tensor(self.score_output["index"]), self.score_output)
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1, scores.shape[-1])
        count = min(len(boxes), len(scores), self.priors.shape[1])
        boxes, scores, priors = boxes[:count], scores[:count], self.priors[:, :count]
        class_columns = scores[:, 1:]
        class_indices = self.np.argmax(class_columns, axis=1) + 1
        logits = scores[self.np.arange(count), class_indices]
        candidates: list[DetectedObject] = []
        crop_height, crop_width = int(rgb_crop.shape[0]), int(rgb_crop.shape[1])
        for index in self.np.flatnonzero(logits >= self.logit_threshold):
            output_class = int(class_indices[index])
            if output_class >= len(self.labels):
                continue
            label = self.labels[output_class].strip()
            if label not in self.enabled_labels or label not in COCO_CLASS_IDS:
                continue
            prior_y, prior_x, prior_h, prior_w = priors[:, index]
            raw_y, raw_x, raw_h, raw_w = boxes[index]
            y_center = raw_y / 10.0 * prior_h + prior_y
            x_center = raw_x / 10.0 * prior_w + prior_x
            height = math.exp(float(raw_h) / 5.0) * float(prior_h)
            width = math.exp(float(raw_w) / 5.0) * float(prior_w)
            ymin = max(0.0, min(1.0, y_center - height / 2.0))
            xmin = max(0.0, min(1.0, x_center - width / 2.0))
            ymax = max(0.0, min(1.0, y_center + height / 2.0))
            xmax = max(0.0, min(1.0, x_center + width / 2.0))
            # Undo the letterbox before returning coordinates to changed-crop space.
            x0 = int(round((xmin * self.input_width - pad_x) / scale_x))
            y0 = int(round((ymin * self.input_height - pad_y) / scale_y))
            x1 = int(round((xmax * self.input_width - pad_x) / scale_x))
            y1 = int(round((ymax * self.input_height - pad_y) / scale_y))
            x0, y0 = max(0, min(crop_width - 1, x0)), max(0, min(crop_height - 1, y0))
            x1, y1 = max(x0 + 1, min(crop_width, x1)), max(y0 + 1, min(crop_height, y1))
            confidence = 1.0 / (1.0 + math.exp(-float(logits[index])))
            candidates.append(DetectedObject(label, COCO_CLASS_IDS[label], confidence, (x0, y0, x1 - x0, y1 - y0)))
        self.calls += 1
        return self._nms(candidates)


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="one-shot GoPoint SSD-Lite crop detector")
    parser.add_argument("image")
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--priors", required=True)
    args = parser.parse_args()
    detector = SSDLiteDetector(args.model, args.labels, args.priors)
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    print(detector.detect(image))
