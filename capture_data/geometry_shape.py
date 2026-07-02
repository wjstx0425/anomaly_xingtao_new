# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Slot-aware contour geometry scoring for C789 part crops."""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class GeometryTemplate:
    """Normal slot geometry template."""

    slot: str
    expected_body: np.ndarray
    allowed_body: np.ndarray
    edge_band: np.ndarray
    source_count: int
    expected_occupancy: float = 0.85
    allowed_occupancy: float = 0.05

    @property
    def shape(self) -> tuple[int, int]:
        """Return template height and width."""
        return self.expected_body.shape


@dataclass
class GeometryScore:
    """Less/more geometry score for one crop image."""

    slot: str
    geometry_score: float
    geometry_type: str
    less_score: float = 0.0
    more_score: float = 0.0
    missing_area: int = 0
    extra_area: int = 0
    dx: int = 0
    dy: int = 0
    iou: float = 0.0
    missing_component_count: int = 0
    extra_component_count: int = 0
    geometry_region: str = ""
    missing_region: str = ""
    extra_region: str = ""
    region_scores: dict[str, float] = field(default_factory=dict)
    source_path: str = ""
    image_path: str = ""
    dataset_label: str = ""
    gt_label: int | None = None
    geometry_threshold: float | None = None
    geometry_pred_label: int | None = None

    def as_row(self) -> dict[str, str | int | float]:
        """Return a CSV-friendly row."""
        return {
            "source_path": self.source_path,
            "image_path": self.image_path,
            "slot": self.slot,
            "dataset_label": self.dataset_label,
            "gt_label": "" if self.gt_label is None else self.gt_label,
            "geometry_score": self.geometry_score,
            "geometry_pred_label": "" if self.geometry_pred_label is None else self.geometry_pred_label,
            "geometry_type": self.geometry_type,
            "geometry_threshold": "" if self.geometry_threshold is None else self.geometry_threshold,
            "geometry_missing_area": self.missing_area,
            "geometry_extra_area": self.extra_area,
            "geometry_region": self.geometry_region,
            "missing_region": self.missing_region,
            "extra_region": self.extra_region,
            "less_score": self.less_score,
            "more_score": self.more_score,
            "align_dx": self.dx,
            "align_dy": self.dy,
            "align_iou": self.iou,
            "missing_component_count": self.missing_component_count,
            "extra_component_count": self.extra_component_count,
        }


@lru_cache(maxsize=1)
def _load_part_crop_module() -> ModuleType:
    """Load the fixed-crop preset module."""
    spec = importlib.util.spec_from_file_location(
        "prepare_part_crops_for_geometry_shape",
        REPO_ROOT / "capture_data" / "prepare_part_crops.py",
    )
    if spec is None or spec.loader is None:
        msg = "Could not load capture_data/prepare_part_crops.py"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preset_choices() -> tuple[str, ...]:
    """Return available crop/mask preset names."""
    return tuple(sorted(_load_part_crop_module().PRESETS))


def get_preset(preset_name: str) -> Any:
    """Return a crop preset by name."""
    presets = _load_part_crop_module().PRESETS
    if preset_name not in presets:
        msg = f"Unknown preset {preset_name!r}. Available presets: {', '.join(sorted(presets))}"
        raise ValueError(msg)
    return presets[preset_name]


def slot_name_from_path(path: str | Path) -> str | None:
    """Infer a slot name from a crop image path."""
    text = str(path)
    match = re.search(r"(?:^|[_/\-])slot(?P<slot>[0-9]+)(?:$|[_/\-.])", text)
    if match is None:
        return None
    return f"slot{int(match.group('slot')):02d}"


def iter_image_paths(root: Path) -> list[Path]:
    """Return image paths under ``root`` in stable order."""
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_view_label_root(data_root: Path, view: str, label: str) -> Path:
    """Resolve a Folder-style view/label directory when present."""
    if "_" not in view:
        return data_root
    hand, position = view.split("_", maxsplit=1)
    candidates = [data_root / hand / position / label]
    if position == "bottom":
        candidates.append(data_root / hand / "bottom_ZS32" / label)
    candidates.append(data_root / label)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return data_root


def _read_image(path: Path) -> np.ndarray:
    """Read an image as OpenCV BGR."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read image: {path}"
        raise ValueError(msg)
    return image


def _foreground_region_mask(
    image: np.ndarray,
    *,
    border_margin: int,
    foreground_threshold_scale: float,
) -> np.ndarray:
    """Build a conservative foreground mask that rejects dark fixture/background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(max(18.0, otsu_threshold * foreground_threshold_scale), 18.0, 55.0))
    raw = (gray > threshold).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if component_count <= 1:
        foreground = raw.astype(bool)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(areas.max())
        area_floor = max(64, int(gray.size * 0.001), int(largest_area * 0.03))
        foreground = np.zeros_like(raw, dtype=bool)
        for label_index in range(1, component_count):
            if int(stats[label_index, cv2.CC_STAT_AREA]) >= area_floor:
                foreground |= labels == label_index

    height, width = gray.shape
    border = np.zeros((height, width), dtype=bool)
    margin = max(0, border_margin)
    if margin == 0:
        border[:, :] = True
    elif height > margin * 2 and width > margin * 2:
        border[margin : height - margin, margin : width - margin] = True
    material = foreground & border & (gray > max(20.0, threshold * 0.9))
    return cv2.morphologyEx(material.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1) > 0


def solid_body_mask(mask: np.ndarray, *, close_px: int = 41) -> np.ndarray:
    """Return a filled exterior silhouette for a thresholded part mask."""
    body = mask.astype(np.uint8)
    if close_px > 0:
        kernel_size = max(1, close_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        pad = kernel_size
        padded = cv2.copyMakeBorder(body, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        padded = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel, iterations=1)
        body = padded[pad : pad + body.shape[0], pad : pad + body.shape[1]]

    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(body)
    min_area = max(64.0, float(body.size) * 0.001)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(solid, [contour], -1, 255, -1)
    return solid > 0


def preset_hole_mask(shape: tuple[int, int], slot_name: str, preset_name: str, dilation: int) -> np.ndarray:
    """Return a boolean mask for preset hole/inpaint regions in one slot crop."""
    preset = get_preset(preset_name)
    slot = next((candidate for candidate in preset.slots if candidate.name == slot_name), None)
    hole_masks = preset.slot_hole_masks.get(slot_name, ())
    if slot is None or not hole_masks:
        return np.zeros(shape, dtype=bool)

    height, width = shape
    scale_x = width / max(slot.box.width, 1)
    scale_y = height / max(slot.box.height, 1)
    mask = np.zeros(shape, dtype=np.uint8)
    for ellipse in hole_masks:
        center = (round(ellipse.cx * scale_x), round(ellipse.cy * scale_y))
        axes = (max(1, round(ellipse.rx * scale_x)), max(1, round(ellipse.ry * scale_y)))
        cv2.ellipse(mask, center, axes, ellipse.angle, 0, 360, 255, -1)
    if dilation > 0:
        kernel = np.ones((max(1, dilation), max(1, dilation)), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 0


def build_foreground_mask(
    image: np.ndarray,
    *,
    slot_name: str | None,
    preset: str,
    hole_dilation: int = 32,
    border_margin: int = 8,
    foreground_threshold_scale: float = 0.45,
) -> np.ndarray:
    """Build the valid foreground body mask for one crop image."""
    if preset == "none":
        return np.ones(image.shape[:2], dtype=bool)

    mask = solid_body_mask(
        _foreground_region_mask(
            image,
            border_margin=border_margin,
            foreground_threshold_scale=foreground_threshold_scale,
        ),
    )
    if slot_name:
        mask &= ~preset_hole_mask(mask.shape, slot_name, preset, hole_dilation)
    return mask


def _external_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Return exterior contours for a binary silhouette."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) >= max(64.0, mask.size * 0.001)]


def _draw_external_contours(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw exterior mask contours on an image."""
    for contour in _external_contours(mask):
        cv2.drawContours(image, [contour], -1, color, thickness, cv2.LINE_AA)


def shift_mask(mask: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    """Return ``mask`` translated by ``dx`` and ``dy`` pixels."""
    height, width = mask.shape
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0


def _shift_slices(shape: tuple[int, int], dx: int, dy: int) -> tuple[slice, slice, slice, slice] | None:
    """Return source and destination slices for shifting without allocation."""
    height, width = shape
    src_x1 = max(0, -dx)
    src_x2 = min(width, width - dx)
    dst_x1 = max(0, dx)
    dst_x2 = min(width, width + dx)
    src_y1 = max(0, -dy)
    src_y2 = min(height, height - dy)
    dst_y1 = max(0, dy)
    dst_y2 = min(height, height + dy)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return None
    return slice(src_y1, src_y2), slice(src_x1, src_x2), slice(dst_y1, dst_y2), slice(dst_x1, dst_x2)


def _iou_for_shift(moving: np.ndarray, reference: np.ndarray, dx: int, dy: int, reference_sum: int) -> float:
    """Return IoU after shifting ``moving`` by ``dx`` and ``dy``."""
    slices = _shift_slices(moving.shape, dx, dy)
    if slices is None:
        return 0.0
    src_y, src_x, dst_y, dst_x = slices
    moving_view = moving[src_y, src_x]
    reference_view = reference[dst_y, dst_x]
    intersection = int(np.count_nonzero(moving_view & reference_view))
    shifted_sum = int(np.count_nonzero(moving_view))
    union = shifted_sum + reference_sum - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _candidate_offsets(radius: int, step: int) -> list[int]:
    """Return stable translation candidates."""
    if radius <= 0:
        return [0]
    values = set(range(-radius, radius + 1, max(1, step)))
    values.update({-radius, 0, radius})
    return sorted(values)


def best_translation(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    search_radius: int = 25,
    coarse_step: int = 5,
) -> tuple[int, int, float]:
    """Find the translation that maximizes mask IoU."""
    if moving.shape != reference.shape:
        msg = f"Mask shape mismatch: moving={moving.shape}, reference={reference.shape}"
        raise ValueError(msg)
    reference_sum = int(np.count_nonzero(reference))
    best_dx = 0
    best_dy = 0
    best_iou = -1.0
    for dy in _candidate_offsets(search_radius, coarse_step):
        for dx in _candidate_offsets(search_radius, coarse_step):
            iou = _iou_for_shift(moving, reference, dx, dy, reference_sum)
            if iou > best_iou:
                best_dx = dx
                best_dy = dy
                best_iou = iou

    fine_radius = max(0, coarse_step - 1)
    min_dx = max(-search_radius, best_dx - fine_radius)
    max_dx = min(search_radius, best_dx + fine_radius)
    min_dy = max(-search_radius, best_dy - fine_radius)
    max_dy = min(search_radius, best_dy + fine_radius)
    for dy in range(min_dy, max_dy + 1):
        for dx in range(min_dx, max_dx + 1):
            iou = _iou_for_shift(moving, reference, dx, dy, reference_sum)
            if iou > best_iou:
                best_dx = dx
                best_dy = dy
                best_iou = iou
    return best_dx, best_dy, max(0.0, best_iou)


def edge_band_for_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Return a boundary band around a binary mask."""
    radius = max(1, pixels)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    body = solid_body_mask(mask).astype(np.uint8)
    dilated = cv2.dilate(body, kernel, iterations=1) > 0
    eroded = cv2.erode(body, kernel, iterations=1) > 0
    return dilated & ~eroded


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a mask to ``shape`` when needed."""
    if mask.shape == shape:
        return mask
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _region_name(cx: float, cy: float, shape: tuple[int, int], grid: tuple[int, int]) -> str:
    """Return a stable region name for a component centroid."""
    rows, cols = grid
    height, width = shape
    row = min(rows - 1, max(0, int(cy / max(1.0, height / rows))))
    col = min(cols - 1, max(0, int(cx / max(1.0, width / cols))))
    return f"r{row:02d}_c{col:02d}"


def _component_stats_by_region(
    mask: np.ndarray,
    min_component_area: int,
    grid: tuple[int, int],
) -> tuple[int, int, str, dict[str, float]]:
    """Return largest component stats and max area per coarse image region."""
    if not np.any(mask):
        return 0, 0, "", {}
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    largest_area = 0
    largest_region = ""
    accepted_count = 0
    region_scores: dict[str, float] = {}
    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        accepted_count += 1
        cx, cy = centroids[index]
        region = _region_name(float(cx), float(cy), mask.shape, grid)
        region_scores[region] = max(region_scores.get(region, 0.0), float(area))
        if area > largest_area:
            largest_area = area
            largest_region = region
    return largest_area, accepted_count, largest_region, region_scores


def score_mask_against_template(
    observed_mask: np.ndarray,
    template: GeometryTemplate,
    *,
    search_radius: int = 25,
    coarse_step: int = 5,
    min_component_area: int = 64,
    tolerance_px: int = 3,
    region_grid: tuple[int, int] = (4, 8),
) -> GeometryScore:
    """Score one observed body mask against a normal slot template."""
    observed = _resize_mask(observed_mask.astype(bool), template.shape)
    dx, dy, iou = best_translation(observed, template.expected_body, search_radius=search_radius, coarse_step=coarse_step)
    aligned = shift_mask(observed, dx=dx, dy=dy)

    tolerance = max(0, tolerance_px)
    if tolerance > 0:
        kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
        observed_for_less = cv2.dilate(aligned.astype(np.uint8), kernel, iterations=1) > 0
        allowed_for_more = cv2.dilate(template.allowed_body.astype(np.uint8), kernel, iterations=1) > 0
    else:
        observed_for_less = aligned
        allowed_for_more = template.allowed_body

    missing_mask = template.expected_body & template.edge_band & ~observed_for_less
    extra_mask = aligned & template.edge_band & ~allowed_for_more
    missing_area, missing_count, missing_region, missing_region_scores = _component_stats_by_region(
        missing_mask,
        min_component_area,
        region_grid,
    )
    extra_area, extra_count, extra_region, extra_region_scores = _component_stats_by_region(
        extra_mask,
        min_component_area,
        region_grid,
    )

    if missing_area <= 0 and extra_area <= 0:
        geometry_type = "none"
        geometry_region = ""
    elif missing_area >= extra_area:
        geometry_type = "less"
        geometry_region = missing_region
    else:
        geometry_type = "more"
        geometry_region = extra_region

    region_scores = {f"less:{region}": area for region, area in missing_region_scores.items()}
    region_scores.update({f"more:{region}": area for region, area in extra_region_scores.items()})

    return GeometryScore(
        slot=template.slot,
        geometry_score=float(max(missing_area, extra_area)),
        geometry_type=geometry_type,
        less_score=float(missing_area),
        more_score=float(extra_area),
        missing_area=missing_area,
        extra_area=extra_area,
        dx=dx,
        dy=dy,
        iou=iou,
        missing_component_count=missing_count,
        extra_component_count=extra_count,
        geometry_region=geometry_region,
        missing_region=missing_region,
        extra_region=extra_region,
        region_scores=region_scores,
    )


def group_paths_by_slot(paths: Iterable[Path]) -> dict[str, list[Path]]:
    """Group image paths by inferred slot name."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        slot = slot_name_from_path(path)
        if slot:
            grouped[slot].append(path)
    return {slot: sorted(slot_paths) for slot, slot_paths in sorted(grouped.items())}


def build_template_for_slot(
    image_paths: Sequence[Path],
    *,
    slot: str,
    preset: str,
    search_radius: int = 25,
    coarse_step: int = 5,
    expected_occupancy: float = 0.85,
    allowed_occupancy: float = 0.85,
    allowed_dilation: int = 8,
    edge_band_px: int = 35,
    hole_dilation: int = 32,
    border_margin: int = 8,
    foreground_threshold_scale: float = 0.45,
) -> GeometryTemplate:
    """Build one slot geometry template from normal crops."""
    if not image_paths:
        msg = f"No images provided for {slot}"
        raise ValueError(msg)

    masks: list[np.ndarray] = []
    for path in image_paths:
        image = _read_image(path)
        mask = build_foreground_mask(
            image,
            slot_name=slot,
            preset=preset,
            hole_dilation=hole_dilation,
            border_margin=border_margin,
            foreground_threshold_scale=foreground_threshold_scale,
        )
        masks.append(mask)

    areas = [int(np.count_nonzero(mask)) for mask in masks]
    reference_index = sorted(range(len(masks)), key=lambda index: areas[index])[len(masks) // 2]
    reference = masks[reference_index]
    accumulator = np.zeros(reference.shape, dtype=np.float32)
    for mask in masks:
        resized = _resize_mask(mask, reference.shape)
        dx, dy, _ = best_translation(resized, reference, search_radius=search_radius, coarse_step=coarse_step)
        accumulator += shift_mask(resized, dx=dx, dy=dy).astype(np.float32)

    occupancy = accumulator / max(1, len(masks))
    expected = occupancy >= expected_occupancy
    allowed = occupancy >= allowed_occupancy
    if allowed_dilation > 0:
        kernel = np.ones((allowed_dilation * 2 + 1, allowed_dilation * 2 + 1), np.uint8)
        allowed = cv2.dilate(allowed.astype(np.uint8), kernel, iterations=1) > 0
    invalid = preset_hole_mask(reference.shape, slot, preset, hole_dilation)
    expected &= ~invalid
    allowed &= ~invalid
    edge_band = edge_band_for_mask(expected, edge_band_px) & ~invalid

    return GeometryTemplate(
        slot=slot,
        expected_body=expected.astype(bool),
        allowed_body=allowed.astype(bool),
        edge_band=edge_band.astype(bool),
        source_count=len(masks),
        expected_occupancy=expected_occupancy,
        allowed_occupancy=allowed_occupancy,
    )


def save_template(template: GeometryTemplate, output_dir: Path) -> Path:
    """Save one slot template as a compressed npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{template.slot}_geometry_template.npz"
    np.savez_compressed(
        path,
        slot=np.array(template.slot),
        expected_body=template.expected_body.astype(np.uint8),
        allowed_body=template.allowed_body.astype(np.uint8),
        edge_band=template.edge_band.astype(np.uint8),
        source_count=np.array(template.source_count),
        expected_occupancy=np.array(template.expected_occupancy),
        allowed_occupancy=np.array(template.allowed_occupancy),
    )
    return path


def load_template(path: Path) -> GeometryTemplate:
    """Load one slot geometry template."""
    with np.load(path, allow_pickle=False) as data:
        return GeometryTemplate(
            slot=str(data["slot"].item()),
            expected_body=data["expected_body"].astype(bool),
            allowed_body=data["allowed_body"].astype(bool),
            edge_band=data["edge_band"].astype(bool),
            source_count=int(data["source_count"].item()),
            expected_occupancy=float(data["expected_occupancy"].item()),
            allowed_occupancy=float(data["allowed_occupancy"].item()),
        )


def load_templates(template_dir: Path) -> dict[str, GeometryTemplate]:
    """Load all slot templates in ``template_dir``."""
    templates = {}
    for path in sorted(template_dir.glob("*_geometry_template.npz")):
        template = load_template(path)
        templates[template.slot] = template
    if not templates:
        msg = f"No geometry templates found in {template_dir}"
        raise FileNotFoundError(msg)
    return templates


def write_template_overlay(template: GeometryTemplate, output_path: Path) -> None:
    """Write a compact template preview image."""
    height, width = template.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    expected = solid_body_mask(template.expected_body)
    allowed = solid_body_mask(template.allowed_body)
    overlay[expected] = (70, 70, 70)
    band = template.edge_band & ~expected
    overlay[band] = (35, 65, 70)
    _draw_external_contours(overlay, allowed, color=(0, 180, 0), thickness=1)
    _draw_external_contours(overlay, expected, color=(0, 255, 255), thickness=3)
    cv2.putText(
        overlay,
        f"{template.slot} n={template.source_count}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_dataset_label(path: Path) -> str:
    """Infer dataset label from Folder-style path parts."""
    parts = set(path.parts)
    if "defect" in parts:
        return "defect"
    if "normal_test" in parts:
        return "normal_test"
    if "normal" in parts:
        return "normal"
    return ""


def infer_gt_label(path: Path) -> int | None:
    """Infer binary ground-truth label from a Folder-style path."""
    label = infer_dataset_label(path)
    if label == "defect":
        return 1
    if label in {"normal", "normal_test"}:
        return 0
    return None


def evaluate_image(
    image_path: Path,
    template: GeometryTemplate,
    *,
    preset: str,
    search_radius: int = 25,
    coarse_step: int = 5,
    min_component_area: int = 64,
    tolerance_px: int = 3,
    region_grid: tuple[int, int] = (4, 8),
    hole_dilation: int = 32,
    border_margin: int = 8,
    foreground_threshold_scale: float = 0.45,
) -> GeometryScore:
    """Evaluate one image against its slot template."""
    image = _read_image(image_path)
    mask = build_foreground_mask(
        image,
        slot_name=template.slot,
        preset=preset,
        hole_dilation=hole_dilation,
        border_margin=border_margin,
        foreground_threshold_scale=foreground_threshold_scale,
    )
    score = score_mask_against_template(
        mask,
        template,
        search_radius=search_radius,
        coarse_step=coarse_step,
        min_component_area=min_component_area,
        tolerance_px=tolerance_px,
        region_grid=region_grid,
    )
    score.source_path = str(image_path.resolve())
    score.image_path = str(image_path.resolve())
    score.dataset_label = infer_dataset_label(image_path)
    score.gt_label = infer_gt_label(image_path)
    return score


def calibrate_thresholds(scores: Iterable[GeometryScore], *, margin_ratio: float = 0.05) -> dict[str, float]:
    """Calibrate slotwise thresholds from locked normal scores."""
    max_by_slot: dict[str, float] = {}
    for score in scores:
        max_by_slot[score.slot] = max(max_by_slot.get(score.slot, 0.0), float(score.geometry_score))
    return {slot: round(max_score * (1.0 + margin_ratio), 6) for slot, max_score in sorted(max_by_slot.items())}


def calibrate_region_thresholds(
    scores: Iterable[GeometryScore],
    *,
    margin_ratio: float = 0.05,
) -> dict[tuple[str, str, str], float]:
    """Calibrate thresholds by slot, geometry type, and coarse boundary region."""
    max_by_key: dict[tuple[str, str, str], float] = {}
    for score in scores:
        for key_text, area in score.region_scores.items():
            geometry_type, region = key_text.split(":", maxsplit=1)
            key = (score.slot, geometry_type, region)
            max_by_key[key] = max(max_by_key.get(key, 0.0), float(area))
    return {key: round(max_score * (1.0 + margin_ratio), 6) for key, max_score in sorted(max_by_key.items())}


def load_thresholds(path: Path) -> dict[tuple[str, str, str], float]:
    """Load slot thresholds from CSV."""
    thresholds: dict[tuple[str, str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            slot = row.get("slot", "")
            value = row.get("geometry_threshold") or row.get("threshold")
            if slot and value not in {None, ""}:
                geometry_type = row.get("geometry_type") or "*"
                region = row.get("geometry_region") or "*"
                thresholds[(slot, geometry_type, region)] = float(value)
    if not thresholds:
        msg = f"No thresholds found in {path}"
        raise ValueError(msg)
    return thresholds


def apply_thresholds(scores: Iterable[GeometryScore], thresholds: dict[tuple[str, str, str], float]) -> None:
    """Attach thresholds and prediction labels to scores in-place."""
    for score in scores:
        candidates: list[tuple[float, float, str, str, float]] = []
        for key_text, area in score.region_scores.items():
            geometry_type, region = key_text.split(":", maxsplit=1)
            threshold = (
                thresholds.get((score.slot, geometry_type, region))
                or thresholds.get((score.slot, geometry_type, "*"))
                or thresholds.get((score.slot, "*", "*"))
            )
            if threshold is None:
                continue
            ratio = area / threshold if threshold > 0 else (1_000_000_000.0 if area > 0 else 0.0)
            candidates.append((ratio, area, geometry_type, region, threshold))
        if not candidates:
            continue
        ratio, area, geometry_type, region, threshold = max(candidates, key=lambda item: item[0])
        score.geometry_score = float(area)
        score.geometry_type = geometry_type
        score.geometry_region = region
        score.geometry_threshold = threshold
        score.geometry_pred_label = int(area > threshold)
