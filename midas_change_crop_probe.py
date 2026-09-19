#!/usr/bin/env python3
"""Live MiDaS depth-change crop experiment for a drawer kept open.

This deliberately does *not* initialize drawers, identify a layer, run YOLO,
or update inventory.  The operator captures a stable before image (Snapshot A)
with the mouse.  Once an item is moved and the scene settles, the probe
captures Snapshot B and shows the depth-difference crop that would be passed to
an object detector.

The supplied default model is a CPU-compatible MiDaS TFLite model for local
macOS development.  It is intentionally not the Vela/Ethos-U model under
``models/``: that model has an ``ethos-u`` custom op that LiteRT on macOS
cannot load.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


DEFAULT_MIDAS_MODEL = Path("../midasModel.tflite")
DEFAULT_DEPTH_ANYTHING_MODEL = Path("models/depth_anything_v2_metric_hypersim_vits.pth")
BUTTONS = {
    "capture_a": (12, 10, 154, 38),
    "reset": (178, 10, 110, 38),
    "save": (300, 10, 110, 38),
}

PHASE_FEEDBACK = {
    "ready_for_a": ("STEP 1 — Keep the drawer open and still, then click Capture A", (0, 180, 255)),
    "capture_a": ("CAPTURING A — Keep hands out until stable capture completes", (0, 180, 255)),
    "wait_motion": ("SNAPSHOT A COMPLETE — You can now put in or take out ONE item", (44, 170, 44)),
    "capture_b": ("CHANGE SEEN — Remove your hand; waiting for a stable Snapshot B", (0, 180, 255)),
    "complete": ("SNAPSHOT B COMPLETE — Review the crop, then Save or Reset", (180, 100, 40)),
    "roi_failed": ("DRAWER INTERIOR NOT FOUND — Reset, keep the empty drawer still, then retry", (40, 50, 210)),
}


@dataclass(frozen=True)
class Thresholds:
    stable_median: float = 0.015
    stable_ratio: float = 0.08
    motion_median: float = 0.030
    motion_ratio: float = 0.015
    change_depth: float = 0.040
    min_change_area: int = 80
    morphology_size: int = 5
    padding: float = 0.25
    noise_multiplier: float = 1.5
    noise_warmup_frames: int = 8
    change_noise_multiplier: float = 4.0
    hysteresis_low_ratio: float = 0.50


@dataclass(frozen=True)
class Snapshot:
    frame: np.ndarray
    depth: np.ndarray
    noise_p95: float
    noise_map: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CropResult:
    delta: np.ndarray
    difference: np.ndarray
    confidence: np.ndarray
    candidate_mask: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    offset: float
    threshold: float
    component_score: float
    candidate_count: int


@dataclass(frozen=True)
class DrawerRegion:
    """Detected drawer quadrilateral in camera pixels and its depth-space mask."""

    quad: np.ndarray
    mask: np.ndarray
    score: float = 0.0
    method: str = "lines"


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("MiDaS returned no finite depth values")
    low, high = np.percentile(finite, (2, 98))
    normalized = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def normalize_relative_depth(depth: np.ndarray) -> np.ndarray:
    """Put each relative MiDaS frame on a robust 0..1 comparison scale.

    This model's raw output is in arbitrary units (a black image, for example,
    can span hundreds of units).  Percentile normalization makes the stability
    and change thresholds interpretable across frames; the later robust global
    offset removal handles the remaining frame-to-frame shift.
    """
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("MiDaS returned no finite depth values")
    low, high = np.percentile(finite, (5, 95))
    return np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0).astype(np.float32)


def spatial_filter(depth: np.ndarray, diameter: int, sigma: float) -> np.ndarray:
    """Suppress speckle without blurring a depth edge as much as Gaussian blur."""
    if diameter <= 1:
        return depth.astype(np.float32, copy=True)
    if diameter % 2 == 0:
        diameter += 1
    return cv2.bilateralFilter(depth.astype(np.float32), diameter, sigma, sigma)


def stability_metrics(
    current: np.ndarray,
    previous: np.ndarray,
    threshold: float,
    mask: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    difference = np.abs(current - previous)
    if mask is not None:
        difference = difference[mask]
    if difference.size == 0:
        raise ValueError("depth comparison mask has no pixels")
    return float(np.median(difference)), float(np.mean(difference > threshold))


def is_stable(
    current: np.ndarray,
    previous: np.ndarray,
    thresholds: Thresholds,
    *,
    median_limit: Optional[float] = None,
    pixel_limit: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> bool:
    median_limit = thresholds.stable_median if median_limit is None else median_limit
    pixel_limit = thresholds.stable_median if pixel_limit is None else pixel_limit
    median_delta, ratio = stability_metrics(current, previous, pixel_limit, mask)
    return median_delta <= median_limit and ratio <= thresholds.stable_ratio


def has_motion(
    current: np.ndarray,
    previous: np.ndarray,
    thresholds: Thresholds,
    mask: Optional[np.ndarray] = None,
) -> bool:
    median_delta, ratio = stability_metrics(current, previous, thresholds.motion_median, mask)
    return median_delta >= thresholds.motion_median or ratio >= thresholds.motion_ratio


def is_rectangular(quad: np.ndarray) -> bool:
    """Reject arbitrary four-sided contours; drawer corners should be near 90°."""
    points = quad.reshape(4, 2).astype(np.float32)
    for index in range(4):
        first = points[(index - 1) % 4] - points[index]
        second = points[(index + 1) % 4] - points[index]
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator < 1e-6 or abs(float(np.dot(first, second)) / denominator) > 0.40:
            return False
    return True


def detect_drawer_quad(frame: np.ndarray, minimum_area_ratio: float) -> Optional[np.ndarray]:
    """Find a full or partially visible drawer rectangle using CPU OpenCV only."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    midpoint = float(np.median(gray))
    edges = cv2.Canny(gray, max(0, int(midpoint * 0.66)), min(255, max(20, int(midpoint * 1.33))))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = frame.shape[0] * frame.shape[1]
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        area = float(cv2.contourArea(quad))
        if area < image_area * minimum_area_ratio or not is_rectangular(quad):
            continue
        rectangle_area = float(cv2.minAreaRect(quad)[1][0] * cv2.minAreaRect(quad)[1][1])
        rectangularity = area / max(rectangle_area, 1.0)
        candidates.append((area * rectangularity, quad.reshape(4, 2)))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    # An open drawer often hides one edge.  In that case an exact four-point
    # contour is unavailable, so recover an axis-aligned rectangle from two
    # parallel long edges plus an orthogonal supporting edge (three physical
    # edges / at least two reliable corners).  We intentionally do not accept
    # only two unconnected points: that would have to guess most of the ROI.
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, min(frame.shape[:2]) // 8),
        minLineLength=max(30, min(frame.shape[:2]) // 5),
        maxLineGap=max(12, min(frame.shape[:2]) // 20),
    )
    if lines is None:
        return None
    horizontal: list[tuple[float, float, float, float]] = []
    vertical: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        dx, dy = float(x1 - x0), float(y1 - y0)
        length = float(np.hypot(dx, dy))
        if length < max(30, min(frame.shape[:2]) // 5):
            continue
        if abs(dy) <= abs(dx) * 0.20:
            horizontal.append((min(x0, x1), max(x0, x1), (y0 + y1) / 2.0, length))
        elif abs(dx) <= abs(dy) * 0.20:
            vertical.append(((x0 + x1) / 2.0, min(y0, y1), max(y0, y1), length))

    def valid_box(x0: float, y0: float, x1: float, y1: float) -> Optional[np.ndarray]:
        if x1 <= x0 or y1 <= y0:
            return None
        area = (x1 - x0) * (y1 - y0)
        if area < image_area * minimum_area_ratio:
            return None
        return np.rint([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype(np.int32)

    partial_candidates: list[tuple[float, np.ndarray]] = []
    edge_tolerance = max(12.0, min(frame.shape[:2]) * 0.06)
    # Top/bottom edges plus either left or right edge.
    for first_index, first in enumerate(horizontal):
        for second in horizontal[first_index + 1 :]:
            top, bottom = sorted((first, second), key=lambda item: item[2])
            if bottom[2] - top[2] < frame.shape[0] * 0.12:
                continue
            x0 = float(np.median((top[0], bottom[0])))
            x1 = float(np.median((top[1], bottom[1])))
            support = [line for line in vertical if abs(line[0] - x0) <= edge_tolerance or abs(line[0] - x1) <= edge_tolerance]
            quad = valid_box(x0, top[2], x1, bottom[2])
            if quad is not None and support:
                partial_candidates.append((float(cv2.contourArea(quad)) * (1.0 + max(line[3] for line in support) / frame.shape[0]), quad))
    # Left/right edges plus either top or bottom edge.
    for first_index, first in enumerate(vertical):
        for second in vertical[first_index + 1 :]:
            left, right = sorted((first, second), key=lambda item: item[0])
            if right[0] - left[0] < frame.shape[1] * 0.12:
                continue
            y0 = float(np.median((left[1], right[1])))
            y1 = float(np.median((left[2], right[2])))
            support = [line for line in horizontal if abs(line[2] - y0) <= edge_tolerance or abs(line[2] - y1) <= edge_tolerance]
            quad = valid_box(left[0], y0, right[0], y1)
            if quad is not None and support:
                partial_candidates.append((float(cv2.contourArea(quad)) * (1.0 + max(line[3] for line in support) / frame.shape[1]), quad))
    return max(partial_candidates, key=lambda item: item[0])[1] if partial_candidates else None


def depth_mask_for_quad(
    quad: np.ndarray,
    frame_shape: tuple[int, int],
    depth_shape: tuple[int, int],
    inset: int,
) -> Optional[np.ndarray]:
    frame_height, frame_width = frame_shape
    depth_height, depth_width = depth_shape
    scaled = quad.astype(np.float32).copy()
    scaled[:, 0] *= depth_width / frame_width
    scaled[:, 1] *= depth_height / frame_height
    mask = np.zeros((depth_height, depth_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(scaled).astype(np.int32), 255)
    if inset > 0:
        size = inset * 2 + 1
        mask = cv2.erode(mask, np.ones((size, size), dtype=np.uint8))
    boolean = mask.astype(bool)
    return boolean if int(boolean.sum()) >= 100 else None


def detect_drawer_interior_plane(
    frame: np.ndarray,
    depth: np.ndarray,
    minimum_area_ratio: float,
    inset: int,
    plane_tolerance: float = 0.035,
) -> Optional[DrawerRegion]:
    """Find the drawer floor as an enclosed inverse-depth planar region.

    For a pinhole camera, inverse depth over a 3-D plane is approximately
    linear in image coordinates.  RANSAC proposes several such planes; their
    connected regions are then ranked using enclosure, rectangularity,
    RGB/depth boundary support, and Shi-Tomasi corner support.  This does not
    require all four drawer corners to be visible.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    if int(valid.sum()) < 100:
        return None
    max_side = 180
    scale = min(1.0, max_side / max(depth.shape))
    small_width = max(32, round(depth.shape[1] * scale))
    small_height = max(32, round(depth.shape[0] * scale))
    small_depth = cv2.resize(depth, (small_width, small_height), interpolation=cv2.INTER_AREA)
    small_valid = cv2.resize(valid.astype(np.uint8), (small_width, small_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    inverse = np.zeros_like(small_depth, dtype=np.float32)
    inverse[small_valid] = 1.0 / np.maximum(small_depth[small_valid], 1e-6)
    low, high = np.percentile(inverse[small_valid], (5, 95))
    inverse = np.clip((inverse - low) / max(float(high - low), 1e-6), 0.0, 1.0).astype(np.float32)

    yy, xx = np.mgrid[0:small_height, 0:small_width].astype(np.float32)
    xx = xx / max(small_width - 1, 1) * 2.0 - 1.0
    yy = yy / max(small_height - 1, 1) * 2.0 - 1.0
    design = np.stack((xx, yy, np.ones_like(xx)), axis=-1)

    rgb_small = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2GRAY)
    rgb_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    grad_x = cv2.Sobel(inverse, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(inverse, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    depth_edge_threshold = max(float(np.percentile(gradient[small_valid], 82)), 1e-6)
    depth_edges = (gradient >= depth_edge_threshold).astype(np.uint8) * 255
    combined_edges = cv2.dilate(cv2.bitwise_or(rgb_edges, depth_edges), np.ones((3, 3), np.uint8))

    corner_map = np.zeros((small_height, small_width), dtype=np.uint8)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=0.02, minDistance=8)
    if corners is not None:
        for corner in corners.reshape(-1, 2):
            cv2.circle(corner_map, tuple(np.rint(corner).astype(int)), 4, 255, cv2.FILLED)

    rng = np.random.default_rng(20260919)
    remaining = small_valid.copy()
    candidates: list[tuple[float, np.ndarray]] = []
    minimum_pixels = max(80, round(small_width * small_height * minimum_area_ratio))
    for _plane_index in range(5):
        coordinates = np.argwhere(remaining)
        if len(coordinates) < minimum_pixels:
            break
        best_inliers: Optional[np.ndarray] = None
        best_count = 0
        for _ in range(120):
            sample_indices = rng.choice(len(coordinates), size=3, replace=False)
            sample = coordinates[sample_indices]
            matrix = design[sample[:, 0], sample[:, 1]]
            if abs(float(np.linalg.det(matrix))) < 1e-5:
                continue
            coefficients = np.linalg.solve(matrix, inverse[sample[:, 0], sample[:, 1]])
            residual = np.abs(np.sum(design * coefficients, axis=2) - inverse)
            inliers = remaining & (residual <= plane_tolerance)
            count = int(inliers.sum())
            if count > best_count:
                best_count, best_inliers = count, inliers
        if best_inliers is None or best_count < minimum_pixels:
            break

        points = np.argwhere(best_inliers)
        coefficients, *_ = np.linalg.lstsq(
            design[points[:, 0], points[:, 1]],
            inverse[points[:, 0], points[:, 1]],
            rcond=None,
        )
        residual = np.abs(np.sum(design * coefficients, axis=2) - inverse)
        plane_mask = (remaining & (residual <= plane_tolerance)).astype(np.uint8)
        plane_mask = cv2.morphologyEx(plane_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        plane_mask = cv2.morphologyEx(plane_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(plane_mask, connectivity=8)
        for component in range(1, component_count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < minimum_pixels or area > small_width * small_height * 0.80:
                continue
            component_mask = (component_labels == component).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            rectangle = cv2.minAreaRect(contour)
            rectangle_area = max(float(rectangle[1][0] * rectangle[1][1]), 1.0)
            rectangularity = min(1.0, area / rectangle_area)
            boundary = cv2.morphologyEx(component_mask, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(bool)
            boundary_support = float(np.mean(combined_edges[boundary] > 0)) if boundary.any() else 0.0
            corner_support = float(np.mean(corner_map[boundary] > 0)) if boundary.any() else 0.0
            x, y, width, height, _ = (int(value) for value in stats[component])
            touches = sum((x <= 2, y <= 2, x + width >= small_width - 2, y + height >= small_height - 2))
            border_factor = (1.0, 0.45, 0.12, 0.04, 0.01)[touches]
            plane_residual = float(np.median(residual[component_mask.astype(bool)]))
            planarity = max(0.05, 1.0 - plane_residual / plane_tolerance)
            fill = area / max(float(width * height), 1.0)
            score = (
                np.sqrt(float(area))
                * (0.25 + 0.75 * rectangularity)
                * (0.30 + 0.70 * fill)
                * (0.20 + 0.80 * boundary_support)
                * (0.70 + 8.0 * corner_support)
                * planarity
                * border_factor
            )
            candidates.append((score, component_mask.astype(bool)))
        remaining &= ~best_inliers

    if not candidates:
        return None
    score, small_mask = max(candidates, key=lambda item: item[0])
    mask = cv2.resize(small_mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    if inset > 0:
        size = inset * 2 + 1
        mask = cv2.erode(mask, np.ones((size, size), dtype=np.uint8))
    if int(mask.sum()) < 100:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    depth_quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    frame_quad = depth_quad.copy()
    frame_quad[:, 0] *= frame.shape[1] / depth.shape[1]
    frame_quad[:, 1] *= frame.shape[0] / depth.shape[0]
    return DrawerRegion(frame_quad, mask.astype(bool), float(score), "inverse-depth-plane")


def make_snapshot(frames: list[np.ndarray], depths: list[np.ndarray]) -> Snapshot:
    if len(frames) != len(depths) or not depths:
        raise ValueError("snapshot needs matching RGB and depth frames")
    stack = np.stack(depths).astype(np.float32)
    median_depth = np.median(stack, axis=0).astype(np.float32)
    # Per-pixel robust sigma.  A single global noise number is dominated by
    # unstable cabinet edges and can hide a shallow object on a quiet floor.
    noise_map = (1.4826 * np.median(np.abs(stack - median_depth), axis=0)).astype(np.float32)
    return Snapshot(
        frame=frames[len(frames) // 2].copy(),
        depth=median_depth,
        noise_p95=float(np.percentile(noise_map, 95)),
        noise_map=noise_map,
    )


def padded_bbox(x: int, y: int, width: int, height: int, shape: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    image_height, image_width = shape
    pad_x = int(round(width * padding))
    pad_y = int(round(height * padding))
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(image_width, x + width + pad_x), min(image_height, y + height + pad_y)
    return x0, y0, x1, y1


def build_crop(
    before: Snapshot,
    after: Snapshot,
    thresholds: Thresholds,
    drawer_mask: Optional[np.ndarray] = None,
) -> Optional[CropResult]:
    """Create a single-item crop from depth B - depth A.

    MiDaS is relative depth and can shift globally between snapshots.  With no
    initialized background ROI available, remove the robust global offset.  A
    single drawer item occupies far less than half the image, so its pixels do
    not influence this median materially.
    """
    raw_delta = after.depth - before.depth
    values = raw_delta[drawer_mask] if drawer_mask is not None else raw_delta.ravel()
    if values.size == 0:
        raise ValueError("drawer mask has no depth pixels")
    offset = float(np.median(values))
    delta = raw_delta - offset
    difference = np.abs(delta)
    noise_a = before.noise_map
    if noise_a is None:
        noise_a = np.full_like(before.depth, before.noise_p95)
    noise_b = after.noise_map
    if noise_b is None:
        noise_b = np.full_like(after.depth, after.noise_p95)
    combined_noise = np.sqrt(np.square(noise_a) + np.square(noise_b))
    high_threshold = np.maximum(
        thresholds.change_depth,
        combined_noise * thresholds.change_noise_multiplier,
    )
    low_threshold = np.maximum(
        thresholds.change_depth * thresholds.hysteresis_low_ratio,
        combined_noise * thresholds.change_noise_multiplier * thresholds.hysteresis_low_ratio,
    )
    confidence = difference / np.maximum(high_threshold, 1e-6)
    strong = difference >= high_threshold
    weak = difference >= low_threshold
    if drawer_mask is not None:
        strong &= drawer_mask
        weak &= drawer_mask

    # Hysteresis: grow complete low-threshold objects only from a reliable
    # high-threshold seed.  This keeps shallow object edges without admitting
    # unrelated weak flicker across the frame.
    weak_count, weak_labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    seeded_labels = np.unique(weak_labels[strong])
    seeded_labels = seeded_labels[seeded_labels != 0]
    if weak_count <= 1 or seeded_labels.size == 0:
        return None
    mask = np.isin(weak_labels, seeded_labels).astype(np.uint8)
    if drawer_mask is not None:
        mask &= drawer_mask.astype(np.uint8)
    kernel = np.ones((thresholds.morphology_size, thresholds.morphology_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if drawer_mask is not None:
        mask &= drawer_mask.astype(np.uint8)
    selected_thresholds = high_threshold[drawer_mask] if drawer_mask is not None else high_threshold.ravel()
    threshold = float(np.median(selected_thresholds))
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= thresholds.min_change_area]
    if not candidates:
        return None

    # A depth model often flickers along large cabinet/image boundaries.  The
    # inserted object is normally a thick, compact, signed-consistent region
    # with a corresponding RGB change.  Rank components by those properties
    # instead of blindly choosing the largest area.
    depth_height, depth_width = before.depth.shape
    border_margin = max(2, thresholds.morphology_size)
    interior_candidates = []
    for index in candidates:
        x, y, width, height, _area = (int(value) for value in stats[index])
        touches_border = (
            x <= border_margin
            or y <= border_margin
            or x + width >= depth_width - border_margin
            or y + height >= depth_height - border_margin
        )
        if not touches_border:
            interior_candidates.append(index)
    # If every region touches a border, retain them as a fallback; otherwise
    # boundary artifacts cannot compete with an interior object.
    ranked_candidates = interior_candidates or candidates

    gray_a = cv2.cvtColor(before.frame, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(after.frame, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.resize(gray_a, (depth_width, depth_height), interpolation=cv2.INTER_AREA)
    gray_b = cv2.resize(gray_b, (depth_width, depth_height), interpolation=cv2.INTER_AREA)
    rgb_difference = cv2.GaussianBlur(cv2.absdiff(gray_a, gray_b).astype(np.float32), (5, 5), 0)
    rgb_scale = max(float(np.percentile(rgb_difference, 90)), 1.0)
    rgb_change = np.clip(rgb_difference / rgb_scale, 0.0, 1.0)

    gradient_x = cv2.Sobel(before.depth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(before.depth, cv2.CV_32F, 0, 1, ksize=3)
    baseline_edges = cv2.magnitude(gradient_x, gradient_y)
    edge_threshold = max(float(np.percentile(baseline_edges, 85)), 1e-6)

    def component_score(index: int) -> float:
        pixels = labels == index
        x, y, width, height, area = (int(value) for value in stats[index])
        binary = pixels.astype(np.uint8)
        thickness = float(cv2.distanceTransform(binary, cv2.DIST_L2, 5).max()) * 2.0
        compactness = area / max(float(width * height), 1.0)
        strength = float(np.median(confidence[pixels]))
        signed = delta[pixels]
        direction_consistency = max(float(np.mean(signed > 0)), float(np.mean(signed < 0)))
        rgb_support = float(np.mean(rgb_change[pixels]))
        edge_overlap = float(np.mean(baseline_edges[pixels] >= edge_threshold))
        thickness_factor = min(1.0, thickness / max(6.0, min(depth_height, depth_width) * 0.03))
        return (
            np.sqrt(float(area))
            * strength
            * (0.25 + 0.75 * compactness)
            * (0.15 + 0.85 * thickness_factor)
            * (0.40 + 0.60 * rgb_support)
            * direction_consistency
            * max(0.15, 1.0 - edge_overlap)
        )

    scores = {index: component_score(index) for index in ranked_candidates}
    component = max(ranked_candidates, key=scores.__getitem__)
    x, y, width, height, _area = (int(value) for value in stats[component])
    x0, y0, x1, y1 = padded_bbox(x, y, width, height, before.depth.shape, thresholds.padding)
    selected = (labels == component).astype(np.uint8) * 255
    return CropResult(
        delta,
        difference,
        confidence,
        mask * 255,
        selected,
        (x0, y0, x1, y1),
        offset,
        threshold,
        scores[component],
        len(candidates),
    )


class MidasTFLite:
    """CPU LiteRT adapter for the user-provided 256x256 float MiDaS model."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"MiDaS model not found: {model_path}")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        shape = tuple(int(value) for value in self.input["shape"])
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3 or self.input["dtype"] is not np.float32:
            raise ValueError(f"expected float32 1xHxWx3 MiDaS input, got {shape} {self.input['dtype']}")
        self.height, self.width = shape[1], shape[2]
        self.relative_depth = True
        self.name = "MiDaS TFLite"

    def infer(self, frame: np.ndarray) -> np.ndarray:
        image = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.interpreter.set_tensor(self.input["index"], image[None])
        self.interpreter.invoke()
        depth = np.asarray(self.interpreter.get_tensor(self.output["index"]), dtype=np.float32).squeeze()
        if depth.shape != (self.height, self.width) or not np.isfinite(depth).all():
            raise RuntimeError(f"invalid MiDaS output: {depth.shape}")
        return depth


class DepthAnythingMetric:
    """Lazy adapter so the MiDaS-only path does not need PyTorch at import time."""

    def __init__(self, model_path: Path) -> None:
        from depth_anything_metric import DepthAnythingMetricSmall

        self.estimator = DepthAnythingMetricSmall(model_path)
        self.width = self.height = self.estimator.input_size
        self.relative_depth = False
        self.name = f"Depth Anything V2 Metric Small ({self.estimator.backend})"

    def infer(self, frame: np.ndarray) -> np.ndarray:
        return self.estimator.infer(frame)


def make_depth_estimator(args: argparse.Namespace) -> MidasTFLite | DepthAnythingMetric:
    if args.depth_model == "midas":
        return MidasTFLite(args.model or DEFAULT_MIDAS_MODEL)
    return DepthAnythingMetric(args.model or DEFAULT_DEPTH_ANYTHING_MODEL)


class ChangeProbe:
    """Manual A capture followed by automatic stable B capture."""

    def __init__(
        self,
        stable_frames: int,
        thresholds: Thresholds,
        *,
        drawer_roi_mode: str = "full",
        drawer_roi_min_area: float = 0.08,
        drawer_roi_inset: int = 4,
        drawer_plane_tolerance: float = 0.035,
    ) -> None:
        if stable_frames < 3:
            raise ValueError("stable_frames must be at least 3")
        self.required = stable_frames
        self.thresholds = thresholds
        self.drawer_roi_mode = drawer_roi_mode
        self.drawer_roi_min_area = drawer_roi_min_area
        self.drawer_roi_inset = drawer_roi_inset
        self.drawer_plane_tolerance = drawer_plane_tolerance
        self.phase = "ready_for_a"
        self.message = "Click Capture A while the opened drawer is still."
        self.before: Optional[Snapshot] = None
        self.after: Optional[Snapshot] = None
        self.result: Optional[CropResult] = None
        self.frames: Deque[np.ndarray] = deque(maxlen=stable_frames)
        self.depths: Deque[np.ndarray] = deque(maxlen=stable_frames)
        self.previous: Optional[np.ndarray] = None
        self.drawer_region: Optional[DrawerRegion] = None
        self.noise_medians: Deque[float] = deque(maxlen=thresholds.noise_warmup_frames)
        self.noise_pixel_p95: Deque[float] = deque(maxlen=thresholds.noise_warmup_frames)
        self.last_median_delta = 0.0
        self.last_ratio = 0.0
        self.effective_median_limit = thresholds.stable_median
        self.effective_pixel_limit = thresholds.stable_median

    def reset(self) -> None:
        self.phase = "ready_for_a"
        self.message = "Click Capture A while the opened drawer is still."
        self.before = self.after = self.result = None
        self.drawer_region = None
        self.frames.clear()
        self.depths.clear()
        self.noise_medians.clear()
        self.noise_pixel_p95.clear()
        self.last_median_delta = self.last_ratio = 0.0
        self.effective_median_limit = self.thresholds.stable_median
        self.effective_pixel_limit = self.thresholds.stable_median

    def request_before(self) -> None:
        if self.phase in {"ready_for_a", "complete", "roi_failed"}:
            self.phase = "capture_a"
            self.message = f"Capturing stable A: 0/{self.required}. Keep hands out."
            self.frames.clear()
            self.depths.clear()
            self.noise_medians.clear()
            self.noise_pixel_p95.clear()
            self.before = self.after = self.result = None
            self.drawer_region = None
            
    def _observe_noise(self, depth: np.ndarray) -> None:
        if self.previous is None:
            return
        difference = np.abs(depth - self.previous)
        if self.drawer_region is not None:
            difference = difference[self.drawer_region.mask]
        self.noise_medians.append(float(np.median(difference)))
        self.noise_pixel_p95.append(float(np.percentile(difference, 95)))
        # Use the observed 95th-percentile frame noise as the per-pixel gate.
        # The median gate follows the less-sensitive median of those samples.
        if len(self.noise_medians) >= self.thresholds.noise_warmup_frames:
            self.effective_median_limit = max(
                self.thresholds.stable_median,
                float(np.percentile(self.noise_medians, 90)) * self.thresholds.noise_multiplier,
            )
            self.effective_pixel_limit = max(
                self.thresholds.stable_median,
                float(np.percentile(self.noise_pixel_p95, 90)) * self.thresholds.noise_multiplier,
            )

    def _capture_if_stable(self, frame: np.ndarray, depth: np.ndarray) -> Optional[Snapshot]:
        # Learn noise only while the operator is explicitly holding the drawer
        # still for A.  During B, a hand moving out of frame must never teach
        # the gate that large motion is ordinary noise.
        if self.phase == "capture_a":
            self._observe_noise(depth)
        if self.previous is None:
            self.frames.clear()
            self.depths.clear()
            return None
        drawer_mask = None if self.drawer_region is None else self.drawer_region.mask
        self.last_median_delta, self.last_ratio = stability_metrics(
            depth,
            self.previous,
            self.effective_pixel_limit,
            drawer_mask,
        )
        if not is_stable(
            depth,
            self.previous,
            self.thresholds,
            median_limit=self.effective_median_limit,
            pixel_limit=self.effective_pixel_limit,
            mask=drawer_mask,
        ):
            self.frames.clear()
            self.depths.clear()
            return None
        self.frames.append(frame.copy())
        self.depths.append(depth.copy())
        if len(self.depths) != self.required:
            self.message = f"Capturing stable {'A' if self.phase == 'capture_a' else 'B'}: {len(self.depths)}/{self.required}."
            return None
        snapshot = make_snapshot(list(self.frames), list(self.depths))
        self.frames.clear()
        self.depths.clear()
        return snapshot

    def process(self, frame: np.ndarray, depth: np.ndarray) -> None:
        if self.phase == "capture_a":
            snapshot = self._capture_if_stable(frame, depth)
            if snapshot is not None:
                self.before = snapshot
                if self.drawer_roi_mode == "plane":
                    self.drawer_region = detect_drawer_interior_plane(
                        snapshot.frame,
                        snapshot.depth,
                        self.drawer_roi_min_area,
                        self.drawer_roi_inset,
                        self.drawer_plane_tolerance,
                    )
                    if self.drawer_region is None:
                        self.phase = "roi_failed"
                        self.message = "No enclosed planar drawer interior was found. Reset and retry with a clearer empty drawer view."
                        self.previous = depth.copy()
                        return
                elif self.drawer_roi_mode == "auto":
                    quad = detect_drawer_quad(snapshot.frame, self.drawer_roi_min_area)
                    mask = None if quad is None else depth_mask_for_quad(
                        quad,
                        snapshot.frame.shape[:2],
                        snapshot.depth.shape,
                        self.drawer_roi_inset,
                    )
                    if mask is None:
                        self.phase = "roi_failed"
                        self.message = "Drawer rectangle was not found. Ensure all four edges are visible, then click Reset."
                        self.previous = depth.copy()
                        return
                    self.drawer_region = DrawerRegion(quad, mask)
                self.phase = "wait_motion"
                if self.drawer_region is None:
                    self.message = "Snapshot A saved with full-frame change search. Put in or take out one item."
                else:
                    self.message = (
                        f"Drawer interior found ({self.drawer_region.method}, score={self.drawer_region.score:.1f}). "
                        "Put in or take out one item."
                    )
        elif self.phase == "wait_motion" and self.previous is not None:
            # Once A is captured, require motion materially above the just
            # measured stable noise so flicker cannot trigger Snapshot B.
            motion_threshold = max(self.thresholds.motion_median, self.effective_pixel_limit * 2.5)
            motion_thresholds = Thresholds(
                stable_median=self.thresholds.stable_median,
                stable_ratio=self.thresholds.stable_ratio,
                motion_median=motion_threshold,
                motion_ratio=self.thresholds.motion_ratio,
                change_depth=self.thresholds.change_depth,
                min_change_area=self.thresholds.min_change_area,
                morphology_size=self.thresholds.morphology_size,
                padding=self.thresholds.padding,
                noise_multiplier=self.thresholds.noise_multiplier,
                noise_warmup_frames=self.thresholds.noise_warmup_frames,
            )
            drawer_mask = None if self.drawer_region is None else self.drawer_region.mask
            if has_motion(depth, self.previous, motion_thresholds, drawer_mask):
                self.phase = "capture_b"
                self.frames.clear()
                self.depths.clear()
                self.message = f"Motion seen. Remove your hand; waiting for stable B (0/{self.required})."
        elif self.phase == "capture_b":
            snapshot = self._capture_if_stable(frame, depth)
            if snapshot is not None:
                self.after = snapshot
                assert self.before is not None
                self.result = build_crop(
                    self.before,
                    snapshot,
                    self.thresholds,
                    None if self.drawer_region is None else self.drawer_region.mask,
                )
                self.phase = "complete"
                self.message = "Crop detected; click Save or Reset." if self.result else "No changed region passed the threshold; click Reset."
        self.previous = depth.copy()


def inside(point: tuple[int, int], rectangle: tuple[int, int, int, int]) -> bool:
    x, y = point
    left, top, width, height = rectangle
    return left <= x < left + width and top <= y < top + height


def draw_button(image: np.ndarray, label: str, rectangle: tuple[int, int, int, int], enabled: bool = True) -> None:
    x, y, width, height = rectangle
    cv2.rectangle(image, (x, y), (x + width, y + height), (54, 128, 54) if enabled else (70, 70, 70), cv2.FILLED)
    cv2.putText(image, label, (x + 9, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def fit_panel(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def render(frame: np.ndarray, depth: np.ndarray, probe: ChangeProbe, latency_ms: float) -> np.ndarray:
    height, width = frame.shape[:2]
    annotated = frame.copy()
    if probe.drawer_region is not None:
        overlay = annotated.copy()
        rgb_mask = cv2.resize(
            probe.drawer_region.mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        overlay[rgb_mask] = (20, 180, 20)
        annotated = cv2.addWeighted(annotated, 0.72, overlay, 0.28, 0)
        cv2.polylines(annotated, [probe.drawer_region.quad.astype(np.int32)], True, (0, 220, 0), 2, cv2.LINE_AA)
    if probe.result is not None:
        x0, y0, x1, y1 = probe.result.bbox
        # result coordinates are model-space; scale them to camera RGB space.
        sx, sy = width / depth.shape[1], height / depth.shape[0]
        cv2.rectangle(annotated, (round(x0 * sx), round(y0 * sy)), (round(x1 * sx), round(y1 * sy)), (0, 0, 255), 2)
    depth_view = fit_panel(colorize_depth(depth), (width, height))
    if probe.drawer_region is not None:
        scaled_quad = probe.drawer_region.quad.astype(np.float32).copy()
        scaled_quad[:, 0] *= width / depth.shape[1]
        scaled_quad[:, 1] *= height / depth.shape[0]
        cv2.polylines(depth_view, [np.rint(scaled_quad).astype(np.int32)], True, (0, 220, 0), 2, cv2.LINE_AA)
    if probe.result is None:
        diff_view = np.zeros_like(frame)
        if probe.drawer_region is None:
            mask_view = np.zeros_like(frame)
        else:
            drawer_view = probe.drawer_region.mask.astype(np.uint8) * 255
            mask_view = fit_panel(cv2.applyColorMap(drawer_view, cv2.COLORMAP_SUMMER), (width, height))
    else:
        scaled = np.clip(probe.result.confidence / 2.0, 0.0, 1.0)
        diff_view = fit_panel(cv2.applyColorMap(np.rint(scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO), (width, height))
        # Dim gray shows every thresholded component; white is the component
        # selected by object-aware ranking.
        candidate_view = np.where(probe.result.candidate_mask > 0, 80, 0).astype(np.uint8)
        candidate_view[probe.result.mask > 0] = 255
        mask_view = fit_panel(cv2.cvtColor(candidate_view, cv2.COLOR_GRAY2BGR), (width, height))
        x0, y0, x1, y1 = probe.result.bbox
        sx, sy = width / depth.shape[1], height / depth.shape[0]
        cv2.rectangle(mask_view, (round(x0 * sx), round(y0 * sy)), (round(x1 * sx), round(y1 * sy)), (0, 0, 255), 2)
    if probe.before is None:
        snapshot_a_view = np.zeros_like(frame)
    else:
        snapshot_a_view = fit_panel(colorize_depth(probe.before.depth), (width, height))
    snapshot_b_depth = probe.after.depth if probe.after is not None else depth
    snapshot_b_view = fit_panel(colorize_depth(snapshot_b_depth), (width, height))
    instruction, colour = PHASE_FEEDBACK[probe.phase]
    feedback = np.zeros((46, width * 2, 3), dtype=np.uint8)
    cv2.rectangle(feedback, (0, 0), (width * 2, 46), colour, cv2.FILLED)
    cv2.putText(feedback, instruction, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (15, 15, 15), 2, cv2.LINE_AA)
    header = np.zeros((55, width * 2, 3), dtype=np.uint8)
    cv2.putText(header, probe.message[:150], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        header,
        f"state={probe.phase}  MiDaS={latency_ms:.0f}ms  depth Δ={probe.last_median_delta:.4f}/{probe.effective_median_limit:.4f}  noisy pixels={probe.last_ratio:.1%}/{probe.thresholds.stable_ratio:.1%}",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    for name, rectangle in BUTTONS.items():
        draw_button(header, {"capture_a": "Capture A", "reset": "Reset", "save": "Save"}[name], rectangle, name != "save" or probe.result is not None)
    labels_top = np.zeros((24, width * 2, 3), dtype=np.uint8)
    for text, x in (("RGB + crop", 12), ("MiDaS relative depth (live)", width + 12)):
        cv2.putText(labels_top, text, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    labels_change = np.zeros((24, width * 2, 3), dtype=np.uint8)
    right_label = (
        "drawer interior candidate"
        if probe.result is None and probe.drawer_region is not None
        else "change mask: gray=candidate, white=selected"
    )
    for text, x in (("noise-normalized abs(B - A)", 12), (right_label, width + 12)):
        cv2.putText(labels_change, text, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    labels_snapshots = np.zeros((24, width * 2, 3), dtype=np.uint8)
    cv2.putText(labels_snapshots, "Snapshot A depth", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        labels_snapshots,
        "Snapshot B depth" if probe.after is not None else "Snapshot B depth (live until captured)",
        (width + 12, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((
        feedback,
        header,
        labels_top,
        np.hstack((annotated, depth_view)),
        labels_change,
        np.hstack((diff_view, mask_view)),
        labels_snapshots,
        np.hstack((snapshot_a_view, snapshot_b_view)),
    ))


def save_result(output_dir: Path, probe: ChangeProbe) -> Optional[Path]:
    if probe.before is None or probe.after is None or probe.result is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir()
    result = probe.result
    x0, y0, x1, y1 = result.bbox
    sx = probe.after.frame.shape[1] / probe.after.depth.shape[1]
    sy = probe.after.frame.shape[0] / probe.after.depth.shape[0]
    rgb_bbox = (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))
    bx0, by0, bx1, by1 = rgb_bbox
    cv2.imwrite(str(run_dir / "rgb_a.png"), probe.before.frame)
    cv2.imwrite(str(run_dir / "rgb_b.png"), probe.after.frame)
    cv2.imwrite(str(run_dir / "depth_a.png"), colorize_depth(probe.before.depth))
    cv2.imwrite(str(run_dir / "depth_b.png"), colorize_depth(probe.after.depth))
    cv2.imwrite(str(run_dir / "difference.png"), cv2.applyColorMap(np.rint(np.clip(result.difference / max(result.threshold * 2, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    cv2.imwrite(str(run_dir / "confidence.png"), cv2.applyColorMap(np.rint(np.clip(result.confidence / 2.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    cv2.imwrite(str(run_dir / "candidate_mask.png"), result.candidate_mask)
    cv2.imwrite(str(run_dir / "mask.png"), result.mask)
    if probe.drawer_region is not None:
        cv2.imwrite(str(run_dir / "drawer_roi_mask.png"), probe.drawer_region.mask.astype(np.uint8) * 255)
    cv2.imwrite(str(run_dir / "crop_rgb_b.png"), probe.after.frame[by0:by1, bx0:bx1])
    np.save(run_dir / "depth_a.npy", probe.before.depth)
    np.save(run_dir / "depth_b.npy", probe.after.depth)
    (run_dir / "metadata.txt").write_text(
        f"model-space bbox={result.bbox}\nrgb-space bbox={rgb_bbox}\n"
        f"global_offset={result.offset:.7f}\nthreshold={result.threshold:.7f}\n"
        f"component_score={result.component_score:.7f}\ncandidate_count={result.candidate_count}\n"
        f"A_noise_p95={probe.before.noise_p95:.7f}\nB_noise_p95={probe.after.noise_p95:.7f}\n",
        encoding="utf-8",
    )
    return run_dir


def self_test() -> None:
    thresholds = Thresholds(change_depth=0.05, min_change_area=12, morphology_size=3, padding=0.0)
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    before_depth = np.full((20, 32), 0.5, dtype=np.float32)
    after_depth = before_depth + 0.02  # global MiDaS drift must be removed
    after_depth[6:12, 10:18] += 0.20
    before = Snapshot(frame, before_depth, 0.002)
    after = Snapshot(frame, after_depth, 0.003)
    result = build_crop(before, after, thresholds)
    assert result is not None and result.bbox == (10, 6, 18, 12)
    assert np.isclose(result.threshold, 0.05) and abs(result.offset - 0.02) < 1e-6
    # A large frame-edge artifact must not beat a smaller, thick interior item.
    wide_before_depth = np.full((40, 60), 0.5, dtype=np.float32)
    wide_after_depth = wide_before_depth.copy()
    wide_after_depth[0:9, 5:55] += 0.25
    wide_after_depth[20:31, 25:37] += 0.20
    wide_frame = np.zeros((80, 120, 3), dtype=np.uint8)
    wide_frame_b = wide_frame.copy()
    wide_frame_b[40:62, 50:74] = 255
    ranked = build_crop(
        Snapshot(wide_frame, wide_before_depth, 0.002),
        Snapshot(wide_frame_b, wide_after_depth, 0.003),
        thresholds,
    )
    assert ranked is not None and ranked.bbox == (25, 20, 37, 31)
    assert ranked.candidate_count == 2
    # Per-pixel MAD must suppress a noisy frame edge without hiding a shallow
    # object on a locally quiet drawer floor.
    local_noise = np.full_like(wide_before_depth, 0.001)
    local_noise[0:9, :] = 0.060
    shallow_after = wide_before_depth.copy()
    shallow_after[0:9, 5:55] += 0.10
    shallow_after[20:31, 25:37] += 0.020
    shallow = build_crop(
        Snapshot(wide_frame, wide_before_depth, 0.060, local_noise),
        Snapshot(wide_frame_b, shallow_after, 0.060, local_noise),
        Thresholds(change_depth=0.012, min_change_area=12, morphology_size=3, padding=0.0),
    )
    assert shallow is not None and shallow.bbox == (25, 20, 37, 31)
    rgb_rectangle = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.rectangle(rgb_rectangle, (20, 20), (140, 80), (255, 255, 255), 2)
    quad = detect_drawer_quad(rgb_rectangle, 0.08)
    assert quad is not None
    partial_rectangle = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.line(partial_rectangle, (20, 20), (140, 20), (255, 255, 255), 2)
    cv2.line(partial_rectangle, (20, 20), (20, 80), (255, 255, 255), 2)
    cv2.line(partial_rectangle, (140, 20), (140, 80), (255, 255, 255), 2)
    partial_quad = detect_drawer_quad(partial_rectangle, 0.08)
    assert partial_quad is not None
    plane_frame = np.full((120, 160, 3), 150, dtype=np.uint8)
    cv2.rectangle(plane_frame, (35, 30), (125, 100), (30, 30, 30), 3)
    plane_depth = np.full((120, 160), 1.0, dtype=np.float32)
    plane_depth[32:99, 37:124] = 1.5
    plane_region = detect_drawer_interior_plane(plane_frame, plane_depth, 0.05, 2)
    assert plane_region is not None and plane_region.mask[60, 80] and not plane_region.mask[5, 5]
    assert plane_region.method == "inverse-depth-plane"
    drawer_mask = depth_mask_for_quad(quad, rgb_rectangle.shape[:2], before_depth.shape, 1)
    assert drawer_mask is not None and drawer_mask[10, 16] and not drawer_mask[1, 1]
    external_only = before_depth + 0.02
    external_only[1:4, 1:6] += 0.30
    assert build_crop(before, Snapshot(frame, external_only, 0.003), thresholds, drawer_mask) is None
    assert is_stable(before_depth, before_depth + 0.001, thresholds)
    assert has_motion(before_depth, after_depth, thresholds)
    probe = ChangeProbe(3, thresholds)
    probe.request_before()
    for _ in range(4):  # first frame establishes the comparison predecessor
        probe.process(frame, before_depth)
    assert probe.phase == "wait_motion" and probe.before is not None
    probe.process(frame, after_depth)
    assert probe.phase == "capture_b"
    for _ in range(3):
        probe.process(frame, after_depth)
    assert probe.phase == "complete" and probe.result is not None
    print("midas_change_crop_probe: self-test OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual-A / automatic-B MiDaS changed-crop probe")
    parser.add_argument(
        "--depth-model",
        choices=("midas", "depth-anything-metric"),
        default="depth-anything-metric",
        help="select the depth backend (default: depth-anything-metric)",
    )
    parser.add_argument("--model", type=Path, help="override the checkpoint for the selected --depth-model")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--avfoundation", action="store_true", help="force macOS AVFoundation capture")
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until q/Esc; useful for headless camera checks")
    parser.add_argument("--no-display", action="store_true", help="capture/infer without opening the interactive window")
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--stable-median", type=float, default=Thresholds.stable_median)
    parser.add_argument("--stable-ratio", type=float, default=Thresholds.stable_ratio)
    parser.add_argument("--motion-median", type=float, default=Thresholds.motion_median)
    parser.add_argument("--motion-ratio", type=float, default=Thresholds.motion_ratio)
    parser.add_argument("--change-depth", type=float, help="minimum change; defaults to 0.012 m for Metric Depth or 0.04 for normalized MiDaS")
    parser.add_argument("--min-change-area", type=int, default=Thresholds.min_change_area)
    parser.add_argument("--morphology-size", type=int, default=Thresholds.morphology_size)
    parser.add_argument("--padding", type=float, default=Thresholds.padding)
    parser.add_argument(
        "--drawer-roi",
        choices=("plane", "auto", "full"),
        default="plane",
        help="plane finds an enclosed inverse-depth floor; auto uses line geometry; full disables drawer masking",
    )
    parser.add_argument("--drawer-roi-min-area", type=float, default=0.08, help="minimum detected rectangle area as a fraction of RGB frame")
    parser.add_argument("--drawer-roi-inset", type=int, default=4, help="depth pixels eroded from detected drawer borders")
    parser.add_argument("--drawer-plane-tolerance", type=float, default=0.035, help="normalized inverse-depth residual allowed in one floor plane")
    parser.add_argument("--noise-multiplier", type=float, default=Thresholds.noise_multiplier, help="scale measured stable noise into its capture gate")
    parser.add_argument("--noise-warmup-frames", type=int, default=Thresholds.noise_warmup_frames)
    parser.add_argument("--change-noise-multiplier", type=float, default=Thresholds.change_noise_multiplier, help="per-pixel MAD multiplier for a strong change seed")
    parser.add_argument("--hysteresis-low-ratio", type=float, default=Thresholds.hysteresis_low_ratio, help="low/high ratio used to grow a complete object from strong seeds")
    parser.add_argument("--bilateral-diameter", type=int, default=5, help="1 disables spatial bilateral filtering")
    parser.add_argument("--bilateral-sigma", type=float, default=0.08)
    parser.add_argument("--output-dir", type=Path, default=Path("captures/midas-change-probe"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.morphology_size < 1 or args.morphology_size % 2 == 0:
        raise ValueError("--morphology-size must be a positive odd number")
    if args.noise_warmup_frames < 1 or args.noise_multiplier < 1.0:
        raise ValueError("--noise-warmup-frames must be positive and --noise-multiplier must be at least 1")
    if args.change_noise_multiplier < 1.0 or not 0 < args.hysteresis_low_ratio < 1:
        raise ValueError("--change-noise-multiplier must be at least 1 and --hysteresis-low-ratio must be in (0, 1)")
    model = make_depth_estimator(args)
    change_depth = args.change_depth
    if change_depth is None:
        change_depth = 0.040 if model.relative_depth else 0.012
    thresholds = Thresholds(
        stable_median=args.stable_median,
        stable_ratio=args.stable_ratio,
        motion_median=args.motion_median,
        motion_ratio=args.motion_ratio,
        change_depth=change_depth,
        min_change_area=args.min_change_area,
        morphology_size=args.morphology_size,
        padding=args.padding,
        noise_multiplier=args.noise_multiplier,
        noise_warmup_frames=args.noise_warmup_frames,
        change_noise_multiplier=args.change_noise_multiplier,
        hysteresis_low_ratio=args.hysteresis_low_ratio,
    )
    backend = cv2.CAP_AVFOUNDATION if args.avfoundation else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera}")
    if not 0 < args.drawer_roi_min_area <= 1 or args.drawer_roi_inset < 0 or args.drawer_plane_tolerance <= 0:
        raise ValueError("drawer ROI area/tolerance must be positive and inset must be non-negative")
    probe = ChangeProbe(
        args.stable_frames,
        thresholds,
        drawer_roi_mode=args.drawer_roi,
        drawer_roi_min_area=args.drawer_roi_min_area,
        drawer_roi_inset=args.drawer_roi_inset,
        drawer_plane_tolerance=args.drawer_plane_tolerance,
    )
    latest_frame: Optional[np.ndarray] = None
    latest_depth: Optional[np.ndarray] = None
    window = "Depth model: drawer-interior and change-crop probe"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Buttons are rendered below the 46-pixel phase-feedback banner.
        button_point = (x, y - 46)
        if inside(button_point, BUTTONS["capture_a"]):
            probe.request_before()
        elif inside(button_point, BUTTONS["reset"]):
            probe.reset()
        elif inside(button_point, BUTTONS["save"]):
            saved = save_result(args.output_dir, probe)
            if saved is not None:
                probe.message = f"Saved: {saved}"

    if not args.no_display:
        cv2.setMouseCallback(window, mouse)
    print(f"model={model.name} input={model.width}x{model.height}; click Capture A, then move one item; q/Esc quits")
    frames = 0
    try:
        while args.max_frames == 0 or frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame capture failed")
            started = time.perf_counter()
            depth = model.infer(frame)
            if model.relative_depth:
                depth = normalize_relative_depth(depth)
            depth = spatial_filter(depth, args.bilateral_diameter, args.bilateral_sigma)
            latency_ms = (time.perf_counter() - started) * 1000
            latest_frame, latest_depth = frame, depth
            probe.process(frame, depth)
            frames += 1
            if args.no_display:
                continue
            cv2.imshow(window, render(latest_frame, latest_depth, probe, latency_ms))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("a"):
                probe.request_before()
            if key == ord("r"):
                probe.reset()
            if key == ord("s"):
                saved = save_result(args.output_dir, probe)
                if saved is not None:
                    probe.message = f"Saved: {saved}"
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
