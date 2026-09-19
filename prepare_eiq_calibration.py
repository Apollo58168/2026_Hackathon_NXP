#!/usr/bin/env python3
"""Build an eIQ AI Hub calibration ZIP for Metric Small ONNX quantization."""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from depth_anything_metric import preprocess_bgr


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def source_images(directory: Path) -> list[Path]:
    images = sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no supported images found under {directory}")
    return images


def build_dataset(images_dir: Path, output_zip: Path, *, limit: int, overwrite: bool) -> int:
    if not images_dir.is_dir():
        raise NotADirectoryError(images_dir)
    if output_zip.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_zip}; pass --overwrite")
    selected = source_images(images_dir)[:limit]
    stage = output_zip.with_name(output_zip.stem + "_stage")
    if stage.exists():
        shutil.rmtree(stage)
    input_dir = stage / "image"  # Must exactly equal the ONNX input node name.
    input_dir.mkdir(parents=True)
    try:
        written = 0
        for index, image_path in enumerate(selected):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            tensor = preprocess_bgr(image)
            np.save(input_dir / f"sample_{written:04d}.npy", tensor)
            written += 1
        if not written:
            raise RuntimeError("none of the selected files could be decoded as images")
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for array_path in sorted(input_dir.glob("*.npy")):
                archive.write(array_path, array_path.relative_to(stage).as_posix())
        return written
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create image/*.npy calibration data for eIQ ONNX2Quant")
    parser.add_argument("--images", type=Path, required=True, help="representative drawer-camera image directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/eiq_depth_metric_small_calibration.zip"),
        help="output ZIP for eIQ AI Hub",
    )
    parser.add_argument("--limit", type=int, default=100, help="maximum number of images to include")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    count = build_dataset(args.images, args.output, limit=args.limit, overwrite=args.overwrite)
    print(
        f"calibration ZIP: {args.output} ({count} samples)\n"
        "layout: image/sample_XXXX.npy, float32 [1,3,518,518], preprocessed RGB NCHW"
    )


if __name__ == "__main__":
    main()
