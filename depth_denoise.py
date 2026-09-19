"""Small, board-safe denoising helpers for relative MiDaS depth."""
from __future__ import annotations

import cv2
import numpy as np


def normalize_relative_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize one relative-depth frame without letting outliers set the scale."""
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("MiDaS returned no finite depth values")
    low, high = np.percentile(depth[finite], (5, 95))
    result = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0).astype(np.float32)
    result[~finite] = 0.0
    return result


def spatial_filter(depth: np.ndarray, diameter: int = 5, sigma: float = 0.08) -> np.ndarray:
    """Remove depth speckle while preserving object edges."""
    depth = np.asarray(depth, dtype=np.float32)
    if diameter <= 1:
        return depth.copy()
    if diameter % 2 == 0:
        diameter += 1
    return cv2.bilateralFilter(depth, diameter, float(sigma), float(sigma))


def denoise_relative_depth(depth: np.ndarray, diameter: int = 5, sigma: float = 0.08) -> np.ndarray:
    return spatial_filter(normalize_relative_depth(depth), diameter, sigma)


if __name__ == "__main__":
    sample = np.full((16, 16), 0.5, dtype=np.float32)
    sample[8, 8] = 1.0
    filtered = denoise_relative_depth(sample, 5, 0.08)
    assert filtered.shape == sample.shape and np.isfinite(filtered).all()
    print("depth_denoise: self-test OK")
