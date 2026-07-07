# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explainable traditional operator branches for C789 inspection crops."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from capture_data import geometry_shape as geometry
from capture_data.prepare_part_crops import PRESETS, crop_slot, mask_holes


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
BRANCH_FIELDNAMES = [
    "part_id",
    "side",
    "view",
    "slot_id",
    "branch",
    "pred_label",
    "score",
    "threshold",
    "defect_type",
    "evidence_type",
    "gt_defect_type",
    "reason",
    "source_path",
    "evidence_path",
    "status",
]
CASE_FIELDNAMES = [
    "case_id",
    "label",
    "part_id",
    "side",
    "view",
    "slot_id",
    "pred_label",
    "positive_branches",
    "gt_defect_type",
    "source_path",
]
SUMMARY_FIELDNAMES = ["metric", "value"]
CONFUSION_FIELDNAMES = ["gt_defect_type", "primary_evidence", "count"]
KNOWN_DEFECT_TYPES = {"corner", "crack", "deform", "less", "more", "surface"}
DEFAULT_CONFIG: dict[str, Any] = {
    "calibration": {
        "max_images_per_slot": 48,
    },
    "operators": {
        "registration": {
            "enabled": True,
            "min_area_ratio": 0.12,
            "max_area_ratio": 0.78,
            "expected_center_x_ratio": 0.5,
            "expected_center_y_ratio": 0.5,
            "max_center_shift_ratio": 0.28,
            "mode": "warn",
        },
        "geometry": {
            "enabled": True,
            "min_area_ratio": 0.12,
            "max_area_ratio": 0.78,
            "min_solidity": 0.45,
            "threshold_margin": 0.05,
            "threshold_mode": "slot",
            "search_radius": 25,
            "coarse_step": 5,
            "min_component_area": 64,
            "tolerance_px": 3,
            "hole_dilation": 32,
            "border_margin": 8,
            "foreground_threshold_scale": 0.45,
        },
        "crack": {
            "enabled": True,
            "dark_delta": 45,
            "min_length_ratio": 0.32,
            "min_aspect_ratio": 4.0,
            "min_component_area": 20,
            "threshold_margin": 0.05,
        },
        "surface_texture": {
            "enabled": True,
            "residual_std_max": 42.0,
            "threshold_margin": 0.05,
        },
        "feature_presence": {
            "enabled": True,
            "min_dark_feature_count": 1,
            "max_dark_feature_count": 14,
        },
    },
}


@dataclass(frozen=True)
class TraditionalResult:
    """One fusion-compatible traditional operator row."""

    part_id: str
    side: str
    view: str
    slot_id: str
    branch: str
    pred_label: int
    score: float
    threshold: float | None
    defect_type: str
    evidence_type: str
    gt_defect_type: str
    reason: str
    source_path: str
    evidence_path: str
    status: str = ""


@dataclass(frozen=True)
class SlotFeatures:
    """Reusable image measurements for one slot crop."""

    image: np.ndarray
    gray: np.ndarray
    metal_mask: np.ndarray
    area_ratio: float
    bbox: tuple[int, int, int, int] | None
    bbox_center_ratio: tuple[float, float] | None
    solidity: float
    foreground_mean: float


@dataclass(frozen=True)
class TraditionalRuntime:
    """Runtime calibration artifacts for slot-aware traditional operators."""

    preset_name: str
    templates: dict[str, geometry.GeometryTemplate]
    geometry_thresholds: dict[geometry.ThresholdKey, float]
    crack_thresholds: dict[str, float]
    surface_thresholds: dict[str, float]
    crack_ignore_masks: dict[str, np.ndarray]
    calibration_dir: Path | None = None


@dataclass(frozen=True)
class CasePrediction:
    """One case-level prediction aggregated from branch rows."""

    case_id: str
    label: str
    part_id: str
    side: str
    view: str
    slot_id: str
    pred_label: int
    positive_branches: tuple[str, ...]
    gt_defect_type: str
    source_path: str


def iter_image_paths(root: Path) -> list[Path]:
    """Return candidate images under a file or directory."""
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_config(path: Path | None) -> dict[str, Any]:
    """Load a traditional-operator config, falling back to defaults."""
    if path is None or not path.exists():
        return dict(DEFAULT_CONFIG)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        msg = f"Traditional operator config must be a mapping: {path}"
        raise ValueError(msg)
    return _deep_merge(DEFAULT_CONFIG, dict(data))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating inputs."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _operator_config(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return config for one operator branch."""
    operators = config.get("operators", {})
    if not isinstance(operators, Mapping):
        return {}
    item = operators.get(name, {})
    return item if isinstance(item, Mapping) else {}


def _enabled(config: Mapping[str, Any], name: str) -> bool:
    """Return whether an operator branch is enabled."""
    return bool(_operator_config(config, name).get("enabled", True))


def _part_id_from_path(path: Path) -> str:
    """Infer part id from common C789 file names."""
    match = re.search(r"(part[0-9]+)", path.stem)
    if match is not None:
        return match.group(1)
    stem = path.stem
    for token in ("_top_", "_bottom_", "_slot"):
        if token in stem:
            stem = stem.split(token, maxsplit=1)[0]
    return stem


def _slot_from_path(path: Path) -> str | None:
    """Infer slot id from a path if present."""
    match = re.search(r"(?:^|[_/\-])slot(?P<slot>[0-9]+)(?:$|[_/\-.])", str(path))
    if match is None:
        return None
    return f"slot{int(match.group('slot')):02d}"


def infer_label_from_path(path: Path, label_override: str = "auto") -> str:
    """Infer benchmark label from path parts unless explicitly overridden."""
    normalized = label_override.strip().lower()
    if normalized != "auto":
        return normalized
    parts = {part.lower() for part in path.parts}
    if "defect" in parts:
        return "defect"
    if parts & {"normal", "normal_test", "stress_normal", "locked", "train"}:
        return "normal"
    if "invalid" in parts:
        return "invalid"
    return "unknown"


def infer_gt_defect_type_from_path(path: Path, label_override: str = "auto") -> str:
    """Infer the human-labeled defect type from the file name for offline evaluation."""
    stem = path.stem.lower()
    prefix = stem.split("_", maxsplit=1)[0]
    if prefix in KNOWN_DEFECT_TYPES:
        return prefix
    label = infer_label_from_path(path, label_override)
    if label in {"normal", "invalid"}:
        return label
    return ""


def _geometry_int(config: Mapping[str, Any], name: str, default: int) -> int:
    """Return an integer geometry operator setting."""
    return int(_operator_config(config, "geometry").get(name, default))


def _geometry_float(config: Mapping[str, Any], name: str, default: float) -> float:
    """Return a float geometry operator setting."""
    return float(_operator_config(config, "geometry").get(name, default))


def _threshold_from_values(values: Sequence[float], margin_ratio: float) -> float:
    """Return a max-normal threshold with a configurable margin."""
    if not values:
        return 0.0
    return round(float(max(values)) * (1.0 + margin_ratio), 6)


def _sample_calibration_paths(paths: Sequence[Path], max_images: int) -> list[Path]:
    """Return stable evenly spaced calibration paths for one slot."""
    ordered = sorted(paths)
    if max_images <= 0 or len(ordered) <= max_images:
        return list(ordered)
    if max_images == 1:
        return [ordered[len(ordered) // 2]]
    indexes = np.linspace(0, len(ordered) - 1, num=max_images, dtype=int)
    return [ordered[int(index)] for index in indexes]


def _write_operator_thresholds(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write crack/surface slot thresholds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["slot_id", "branch", "threshold", "max_normal", "threshold_margin", "n_normal"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_binary_mask(path: Path, mask: np.ndarray) -> None:
    """Write a boolean mask as a binary PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def _load_binary_masks(mask_dir: Path) -> dict[str, np.ndarray]:
    """Load slot keyed binary masks from a directory."""
    if not mask_dir.is_dir():
        return {}
    masks: dict[str, np.ndarray] = {}
    for path in sorted(mask_dir.glob("slot*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks[path.stem] = mask > 0
    return masks


def _write_calibration_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a compact calibration summary."""
    lines = [
        "# Traditional Operator Calibration",
        "",
        "| slot | branch | n_normal | max_normal | threshold |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['slot_id']} | {row['branch']} | {row['n_normal']} | "
            f"{float(row['max_normal']):.6g} | {float(row['threshold']):.6g} |",
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_operator_thresholds(path: Path | None) -> tuple[dict[str, float], dict[str, float]]:
    """Load optional crack/surface thresholds from calibration CSV."""
    if path is None or not path.is_file():
        return {}, {}
    crack: dict[str, float] = {}
    surface: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            slot_id = (row.get("slot_id") or "").strip()
            branch = (row.get("branch") or "").strip()
            value = row.get("threshold")
            if not slot_id or value in {None, ""}:
                continue
            if branch == "crack":
                crack[slot_id] = float(value)
            elif branch == "surface_texture":
                surface[slot_id] = float(value)
    return crack, surface


def _build_runtime_from_normal(
    normal_root: Path,
    *,
    preset_name: str,
    config: Mapping[str, Any],
    calibration_dir: Path,
    max_images_per_slot: int,
    show_progress: bool = False,
) -> TraditionalRuntime:
    """Build templates and slot thresholds from trusted normal slot crops."""
    image_paths = iter_image_paths(normal_root)
    grouped = geometry.group_paths_by_slot(image_paths)
    if not grouped:
        msg = f"No slot images found for calibration under {normal_root}"
        raise FileNotFoundError(msg)

    template_dir = calibration_dir / "templates"
    templates: dict[str, geometry.GeometryTemplate] = {}
    sampled_grouped = {
        slot_id: _sample_calibration_paths(slot_paths, max_images_per_slot) for slot_id, slot_paths in grouped.items()
    }
    for slot_id, slot_paths in sampled_grouped.items():
        if show_progress:
            print(
                f"[traditional][calibrate] build template {slot_id}: "
                f"{len(slot_paths)}/{len(grouped[slot_id])} normal",
            )
        template = geometry.build_template_for_slot(
            slot_paths,
            slot=slot_id,
            preset=preset_name,
            search_radius=_geometry_int(config, "search_radius", 25),
            coarse_step=_geometry_int(config, "coarse_step", 5),
            expected_occupancy=_geometry_float(config, "expected_occupancy", 0.85),
            allowed_occupancy=_geometry_float(config, "allowed_occupancy", 0.85),
            allowed_dilation=_geometry_int(config, "allowed_dilation", 8),
            edge_band_px=_geometry_int(config, "edge_band_px", 35),
            hole_dilation=_geometry_int(config, "hole_dilation", 32),
            border_margin=_geometry_int(config, "border_margin", 8),
            foreground_threshold_scale=_geometry_float(config, "foreground_threshold_scale", 0.45),
        )
        templates[slot_id] = template
        geometry.save_template(template, template_dir)
        geometry.write_template_overlay(template, template_dir / "overlays" / f"{slot_id}_template_overlay.png")

    geometry_scores: list[geometry.GeometryScore] = []
    normal_cache: dict[str, list[tuple[SlotFeatures, np.ndarray]]] = {slot_id: [] for slot_id in templates}
    crack_ignore_masks: dict[str, np.ndarray] = {}
    for slot_id, slot_paths in sampled_grouped.items():
        if show_progress:
            print(f"[traditional][calibrate] score normal {slot_id}: {len(slot_paths)}")
        template = templates[slot_id]
        for image_path in slot_paths:
            image = _read_image(image_path)
            features = compute_slot_features(image)
            observed_mask = geometry.build_foreground_mask(
                image,
                slot_name=slot_id,
                preset=preset_name,
                hole_dilation=_geometry_int(config, "hole_dilation", 32),
                border_margin=_geometry_int(config, "border_margin", 8),
                foreground_threshold_scale=_geometry_float(config, "foreground_threshold_scale", 0.45),
            )
            geometry_scores.append(
                geometry.score_mask_against_template(
                    observed_mask,
                    template,
                    search_radius=_geometry_int(config, "search_radius", 25),
                    coarse_step=_geometry_int(config, "coarse_step", 5),
                    min_component_area=_geometry_int(config, "min_component_area", 64),
                    tolerance_px=_geometry_int(config, "tolerance_px", 3),
                ),
            )
            roi = _template_roi(template, features.gray.shape, features.metal_mask, erosion_px=15)
            normal_cache[slot_id].append((features, roi))

    for slot_id, items in normal_cache.items():
        stable_mask = np.zeros_like(items[0][0].gray, dtype=bool)
        for features, roi in items:
            raw_mask, _raw_score = _crack_mask(features, config, roi)
            stable_mask |= raw_mask
        crack_ignore_masks[slot_id] = cv2.dilate(
            stable_mask.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        ) > 0
        _write_binary_mask(calibration_dir / "crack_ignore_masks" / f"{slot_id}.png", crack_ignore_masks[slot_id])

    crack_scores: dict[str, list[float]] = {slot_id: [] for slot_id in templates}
    surface_scores: dict[str, list[float]] = {slot_id: [] for slot_id in templates}
    for slot_id, items in normal_cache.items():
        template = templates[slot_id]
        surface_roi = _template_roi(template, items[0][0].gray.shape, items[0][0].metal_mask, erosion_px=5)
        for features, crack_roi in items:
            _crack_regions, crack_score = _crack_mask(features, config, crack_roi, crack_ignore_masks[slot_id])
            surface_score, _surface_regions = _surface_texture_score(features, 1_000_000_000.0, surface_roi)
            crack_scores[slot_id].append(crack_score)
            surface_scores[slot_id].append(surface_score)

    threshold_mode = str(_operator_config(config, "geometry").get("threshold_mode", "slot")).lower()
    if threshold_mode == "region":
        geometry_thresholds = geometry.calibrate_region_thresholds(
            geometry_scores,
            margin_ratio=_geometry_float(config, "threshold_margin", 0.05),
        )
    else:
        slot_thresholds = geometry.calibrate_thresholds(
            geometry_scores,
            margin_ratio=_geometry_float(config, "threshold_margin", 0.05),
        )
        geometry_thresholds = geometry.add_threshold_fallbacks(
            {
                (slot_id, geometry.WILDCARD, geometry.WILDCARD): threshold
                for slot_id, threshold in slot_thresholds.items()
            },
        )
    threshold_rows = []
    crack_margin = float(_operator_config(config, "crack").get("threshold_margin", 0.05))
    surface_margin = float(_operator_config(config, "surface_texture").get("threshold_margin", 0.05))
    crack_thresholds: dict[str, float] = {}
    surface_thresholds: dict[str, float] = {}
    for slot_id in sorted(templates):
        crack_thresholds[slot_id] = _threshold_from_values(crack_scores.get(slot_id, []), crack_margin)
        surface_thresholds[slot_id] = _threshold_from_values(surface_scores.get(slot_id, []), surface_margin)
        threshold_rows.extend(
            [
                {
                    "slot_id": slot_id,
                    "branch": "crack",
                    "threshold": crack_thresholds[slot_id],
                    "max_normal": max(crack_scores.get(slot_id, [0.0])),
                    "threshold_margin": crack_margin,
                    "n_normal": len(crack_scores.get(slot_id, [])),
                },
                {
                    "slot_id": slot_id,
                    "branch": "surface_texture",
                    "threshold": surface_thresholds[slot_id],
                    "max_normal": max(surface_scores.get(slot_id, [0.0])),
                    "threshold_margin": surface_margin,
                    "n_normal": len(surface_scores.get(slot_id, [])),
                },
            ],
        )

    geometry.write_csv(
        calibration_dir / "geometry_thresholds.csv",
        _geometry_threshold_rows(geometry_scores, geometry_thresholds),
        [
            "slot_id",
            "defect_type",
            "region_id",
            "threshold",
            "max_normal",
            "n_normal",
            "threshold_margin",
        ],
    )
    _write_operator_thresholds(calibration_dir / "operator_thresholds.csv", threshold_rows)
    _write_calibration_summary(calibration_dir / "calibration_summary.md", threshold_rows)
    return TraditionalRuntime(
        preset_name=preset_name,
        templates=templates,
        geometry_thresholds=geometry_thresholds,
        crack_thresholds=crack_thresholds,
        surface_thresholds=surface_thresholds,
        crack_ignore_masks=crack_ignore_masks,
        calibration_dir=calibration_dir,
    )


def _geometry_threshold_rows(
    scores: Sequence[geometry.GeometryScore],
    thresholds: Mapping[geometry.ThresholdKey, float],
) -> list[dict[str, str | int | float]]:
    """Return compact geometry threshold rows."""
    grouped: dict[geometry.ThresholdKey, list[float]] = {}
    for score in scores:
        for key_text, area in score.region_scores.items():
            defect_type, region = key_text.split(":", maxsplit=1)
            grouped.setdefault((score.slot, defect_type, region), []).append(float(area))
    rows = []
    for (slot_id, defect_type, region), threshold in sorted(thresholds.items()):
        values = grouped.get((slot_id, defect_type, region), [])
        rows.append(
            {
                "slot_id": slot_id,
                "defect_type": defect_type,
                "region_id": region,
                "threshold": threshold,
                "max_normal": max(values) if values else 0.0,
                "n_normal": len(values),
                "threshold_margin": "",
            },
        )
    return rows


def build_runtime_context(args: argparse.Namespace, config: Mapping[str, Any]) -> TraditionalRuntime | None:
    """Build or load slot calibration artifacts requested by the CLI."""
    calibrate_root = getattr(args, "calibrate_normal_root", None)
    if calibrate_root is not None:
        calibration_config = config.get("calibration", {})
        if not isinstance(calibration_config, Mapping):
            calibration_config = {}
        max_images = int(
            getattr(args, "calibration_max_images_per_slot", None)
            or calibration_config.get("max_images_per_slot", 48),
        )
        return _build_runtime_from_normal(
            Path(calibrate_root),
            preset_name=args.preset,
            config=config,
            calibration_dir=args.output_dir / "calibration",
            max_images_per_slot=max_images,
            show_progress=bool(getattr(args, "show_progress", False)),
        )

    template_dir = getattr(args, "template_dir", None)
    if template_dir is None:
        return None
    templates = geometry.load_templates(Path(template_dir))
    geometry_thresholds_path = getattr(args, "geometry_thresholds", None)
    geometry_thresholds = geometry.load_thresholds(Path(geometry_thresholds_path)) if geometry_thresholds_path else {}
    operator_thresholds = Path(template_dir).parent / "operator_thresholds.csv"
    crack_thresholds, surface_thresholds = _load_operator_thresholds(operator_thresholds)
    crack_ignore_masks = _load_binary_masks(Path(template_dir).parent / "crack_ignore_masks")
    return TraditionalRuntime(
        preset_name=args.preset,
        templates=templates,
        geometry_thresholds=geometry_thresholds,
        crack_thresholds=crack_thresholds,
        surface_thresholds=surface_thresholds,
        crack_ignore_masks=crack_ignore_masks,
        calibration_dir=Path(template_dir).parent,
    )


def _read_image(image_path: Path) -> np.ndarray:
    """Read an OpenCV BGR image or raise an explicit error."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        msg = f"Could not read image: {image_path}"
        raise ValueError(msg)
    return image


def _metal_mask(gray: np.ndarray) -> np.ndarray:
    """Segment the bright metal foreground from the dark fixture."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _threshold, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = max(20, int(np.percentile(gray, 55)))
    mask = (otsu > 0) & (gray >= floor)
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_u8 > 0


def compute_slot_features(image: np.ndarray) -> SlotFeatures:
    """Compute reusable foreground and contour measurements."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _metal_mask(gray)
    height, width = gray.shape
    area_ratio = float(mask.mean())
    bbox = None
    center = None
    solidity = 0.0
    foreground_values = gray[mask]
    foreground_mean = float(foreground_values.mean()) if foreground_values.size else 0.0
    if mask.any():
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bbox = (x1, y1, x2, y2)
        center = ((x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height))
        contours, _hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = sum(cv2.contourArea(contour) for contour in contours)
        hull_area = 0.0
        for contour in contours:
            if len(contour) >= 3:
                hull_area += cv2.contourArea(cv2.convexHull(contour))
        solidity = float(contour_area / hull_area) if hull_area > 0 else 0.0
    return SlotFeatures(
        image=image,
        gray=gray,
        metal_mask=mask,
        area_ratio=area_ratio,
        bbox=bbox,
        bbox_center_ratio=center,
        solidity=solidity,
        foreground_mean=foreground_mean,
    )


def _runtime(config: Mapping[str, Any]) -> TraditionalRuntime | None:
    """Return runtime calibration artifacts from a loaded config."""
    value = config.get("_runtime")
    return value if isinstance(value, TraditionalRuntime) else None


def _resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to an image shape."""
    if mask.shape == shape:
        return mask.astype(bool)
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _template_roi(
    template: geometry.GeometryTemplate | None,
    shape: tuple[int, int],
    fallback: np.ndarray,
    *,
    erosion_px: int = 5,
) -> np.ndarray:
    """Return the material ROI used by local operators."""
    if template is None:
        return fallback.astype(bool)
    roi = _resize_bool(template.allowed_body | template.expected_body, shape)
    if erosion_px > 0:
        kernel = np.ones((erosion_px, erosion_px), np.uint8)
        roi = cv2.erode(roi.astype(np.uint8), kernel, iterations=1) > 0
    return roi


def _template_for_slot(config: Mapping[str, Any], slot_id: str) -> geometry.GeometryTemplate | None:
    """Return the calibrated template for a slot, when present."""
    runtime = _runtime(config)
    if runtime is None:
        return None
    return runtime.templates.get(slot_id)


def _slot_threshold(thresholds: Mapping[str, float], slot_id: str, fallback: float) -> float:
    """Return a slot-specific threshold with a stable fallback."""
    return float(thresholds.get(slot_id, fallback))


def _make_result(
    *,
    image_path: Path,
    part_id: str,
    side: str,
    view: str,
    slot_id: str,
    branch: str,
    pred_label: int,
    score: float,
    threshold: float | None,
    defect_type: str,
    evidence_type: str,
    reason: str,
    evidence_path: Path | None,
    status: str = "",
) -> TraditionalResult:
    """Create a normalized result row."""
    source_path = str(image_path.resolve(strict=False))
    return TraditionalResult(
        part_id=part_id,
        side=side,
        view=view,
        slot_id=slot_id,
        branch=branch,
        pred_label=int(pred_label),
        score=float(score),
        threshold=threshold,
        defect_type=defect_type,
        evidence_type=evidence_type,
        gt_defect_type=infer_gt_defect_type_from_path(image_path),
        reason=reason,
        source_path=source_path,
        evidence_path="" if evidence_path is None else str(evidence_path.resolve(strict=False)),
        status=status,
    )


def _geometry_evidence_type(raw_delta: str) -> str:
    """Map internal geometry deltas to evidence names that do not imply GT classes."""
    if raw_delta == "less":
        return "missing_mask"
    if raw_delta == "more":
        return "extra_mask"
    if raw_delta == "corner":
        return "corner_delta"
    return "shape_delta"


def _evidence_path(evidence_dir: Path | None, image_path: Path, slot_id: str, branch: str) -> Path | None:
    """Return a stable evidence image path for one branch."""
    if evidence_dir is None:
        return None
    return evidence_dir / f"{image_path.stem}_{slot_id}_{branch}.png"


def _write_evidence(path: Path | None, image: np.ndarray, mask: np.ndarray | None, label: str) -> None:
    """Write an overlay evidence image."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    if mask is not None and mask.any():
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), overlay)


def _write_geometry_evidence(
    path: Path | None,
    image: np.ndarray,
    missing_mask: np.ndarray,
    extra_mask: np.ndarray,
    label: str,
) -> None:
    """Write geometry evidence with missing and extra regions in distinct colors."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    missing = _resize_bool(missing_mask, image.shape[:2])
    extra = _resize_bool(extra_mask, image.shape[:2])
    if missing.any():
        overlay[missing] = (0.25 * overlay[missing] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
    if extra.any():
        overlay[extra] = (0.25 * overlay[extra] + 0.75 * np.array([255, 0, 0])).astype(np.uint8)
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), overlay)


def _registration_result(
    image_path: Path,
    features: SlotFeatures,
    config: Mapping[str, Any],
    *,
    part_id: str,
    side: str,
    view: str,
    slot_id: str,
    evidence_dir: Path | None,
) -> TraditionalResult:
    """Evaluate coarse slot pose and foreground coverage."""
    op_config = _operator_config(config, "registration")
    min_area = float(op_config.get("min_area_ratio", 0.12))
    max_area = float(op_config.get("max_area_ratio", 0.78))
    expected_x = float(op_config.get("expected_center_x_ratio", 0.5))
    expected_y = float(op_config.get("expected_center_y_ratio", 0.5))
    max_shift = float(op_config.get("max_center_shift_ratio", 0.28))
    mode = str(op_config.get("mode", "warn")).upper()
    area_bad = features.area_ratio < min_area or features.area_ratio > max_area
    shift = 1.0
    if features.bbox_center_ratio is not None:
        shift_x = abs(features.bbox_center_ratio[0] - expected_x)
        shift_y = abs(features.bbox_center_ratio[1] - expected_y)
        shift = max(shift_x, shift_y)
    is_bad = area_bad or shift > max_shift
    status = "PASS" if not is_bad else ("FAIL" if mode == "FAIL" else "WARN")
    pred_label = int(is_bad and status == "FAIL")
    threshold = max_shift
    reason = (
        f"registration area_ratio={features.area_ratio:.4f} expected=[{min_area:.4f},{max_area:.4f}], "
        f"center_shift={shift:.4f} max={max_shift:.4f}"
    )
    evidence = _evidence_path(evidence_dir, image_path, slot_id, "registration")
    bbox_mask = np.zeros(features.gray.shape, dtype=bool)
    if features.bbox is not None:
        x1, y1, x2, y2 = features.bbox
        bbox_mask[y1:y2, x1:x2] = True
    _write_evidence(evidence, features.image, bbox_mask & ~features.metal_mask, status)
    return _make_result(
        image_path=image_path,
        part_id=part_id,
        side=side,
        view=view,
        slot_id=slot_id,
        branch="registration",
        pred_label=pred_label,
        score=shift,
        threshold=threshold,
        defect_type="pose",
        evidence_type="pose_shift",
        reason=reason,
        evidence_path=evidence,
        status=status,
    )


def _geometry_result(
    image_path: Path,
    features: SlotFeatures,
    config: Mapping[str, Any],
    *,
    part_id: str,
    side: str,
    view: str,
    slot_id: str,
    evidence_dir: Path | None,
) -> TraditionalResult:
    """Evaluate simple foreground shape constraints."""
    runtime = _runtime(config)
    template = _template_for_slot(config, slot_id)
    if runtime is not None and template is not None:
        return _template_geometry_result(
            image_path,
            features,
            config,
            template=template,
            runtime=runtime,
            part_id=part_id,
            side=side,
            view=view,
            slot_id=slot_id,
            evidence_dir=evidence_dir,
        )
    op_config = _operator_config(config, "geometry")
    min_area = float(op_config.get("min_area_ratio", 0.12))
    max_area = float(op_config.get("max_area_ratio", 0.78))
    min_solidity = float(op_config.get("min_solidity", 0.45))
    area_low = features.area_ratio < min_area
    area_high = features.area_ratio > max_area
    solidity_low = features.solidity < min_solidity
    pred_label = int(area_low or area_high or solidity_low)
    score = max(max(0.0, min_area - features.area_ratio), max(0.0, features.area_ratio - max_area))
    score = max(score, max(0.0, min_solidity - features.solidity))
    reason = (
        f"geometry area_ratio={features.area_ratio:.4f} expected=[{min_area:.4f},{max_area:.4f}], "
        f"solidity={features.solidity:.4f} min={min_solidity:.4f}"
    )
    evidence = _evidence_path(evidence_dir, image_path, slot_id, "geometry")
    _write_evidence(evidence, features.image, features.metal_mask, "NG" if pred_label else "OK")
    return _make_result(
        image_path=image_path,
        part_id=part_id,
        side=side,
        view=view,
        slot_id=slot_id,
        branch="geometry",
        pred_label=pred_label,
        score=score,
        threshold=0.0,
        defect_type="geometry_delta",
        evidence_type=_geometry_evidence_type("less" if area_low or solidity_low else "more"),
        reason=reason,
        evidence_path=evidence,
    )


def _template_geometry_result(
    image_path: Path,
    features: SlotFeatures,
    config: Mapping[str, Any],
    *,
    template: geometry.GeometryTemplate,
    runtime: TraditionalRuntime,
    part_id: str,
    side: str,
    view: str,
    slot_id: str,
    evidence_dir: Path | None,
) -> TraditionalResult:
    """Evaluate slot geometry using a calibrated template."""
    op_config = _operator_config(config, "geometry")
    observed_mask = geometry.build_foreground_mask(
        features.image,
        slot_name=slot_id,
        preset=runtime.preset_name,
        hole_dilation=int(op_config.get("hole_dilation", 32)),
        border_margin=int(op_config.get("border_margin", 8)),
        foreground_threshold_scale=float(op_config.get("foreground_threshold_scale", 0.45)),
    )
    diff = geometry.diff_mask_against_template(
        observed_mask,
        template,
        search_radius=int(op_config.get("search_radius", 25)),
        coarse_step=int(op_config.get("coarse_step", 5)),
        min_component_area=int(op_config.get("min_component_area", 64)),
        tolerance_px=int(op_config.get("tolerance_px", 3)),
    )
    score = geometry.GeometryScore(
        slot=slot_id,
        geometry_score=diff.geometry_score,
        geometry_type=diff.geometry_type,
        less_score=diff.less_score,
        more_score=diff.more_score,
        missing_area=diff.missing_area,
        extra_area=diff.extra_area,
        dx=diff.dx,
        dy=diff.dy,
        iou=diff.iou,
        missing_component_count=diff.missing_component_count,
        extra_component_count=diff.extra_component_count,
        geometry_region=diff.geometry_region,
        missing_region=diff.missing_region,
        extra_region=diff.extra_region,
        region_scores=diff.region_scores,
    )
    if runtime.geometry_thresholds:
        geometry.apply_thresholds([score], runtime.geometry_thresholds)
    pred_label = int(score.geometry_pred_label or 0)
    threshold = score.geometry_threshold
    raw_delta = score.geometry_type if score.geometry_type != "none" else "none"
    evidence_type = _geometry_evidence_type(raw_delta)
    reason = (
        "geometry_template "
        f"raw_delta={raw_delta} evidence={evidence_type} region={score.geometry_region or '-'} "
        f"score={score.geometry_score:.4f} threshold={'' if threshold is None else f'{threshold:.4f}'} "
        f"dx={score.dx} dy={score.dy} iou={score.iou:.4f}"
    )
    evidence = _evidence_path(evidence_dir, image_path, slot_id, "geometry")
    _write_geometry_evidence(
        evidence,
        features.image,
        diff.missing_mask,
        diff.extra_mask,
        "NG_GEOMETRY" if pred_label else "OK",
    )
    return _make_result(
        image_path=image_path,
        part_id=part_id,
        side=side,
        view=view,
        slot_id=slot_id,
        branch="geometry",
        pred_label=pred_label,
        score=score.geometry_score,
        threshold=threshold,
        defect_type="geometry_delta" if raw_delta != "none" else "none",
        evidence_type=evidence_type,
        reason=reason,
        evidence_path=evidence,
    )


def _crack_mask(
    features: SlotFeatures,
    config: Mapping[str, Any],
    roi_mask: np.ndarray | None = None,
    ignore_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return a dark-line crack mask and normalized score."""
    op_config = _operator_config(config, "crack")
    dark_delta = float(op_config.get("dark_delta", 45))
    foreground_mean = features.foreground_mean or float(features.gray.mean())
    support = roi_mask.astype(bool).copy() if roi_mask is not None else features.metal_mask.copy()
    if features.bbox is not None:
        x1, y1, x2, y2 = features.bbox
        bbox_mask = np.zeros_like(support)
        bbox_mask[y1:y2, x1:x2] = True
        if roi_mask is not None:
            support &= bbox_mask
        else:
            support |= bbox_mask
    support = cv2.dilate(support.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    dark = (features.gray.astype(np.float32) < foreground_mean - dark_delta) & support
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    blackhat = cv2.morphologyEx(features.gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    enhanced = (blackhat > max(18, dark_delta * 0.35)) & support
    mask = dark | enhanced
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1) > 0
    if ignore_mask is not None:
        mask &= ~_resize_bool(ignore_mask, mask.shape)
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    height, width = features.gray.shape
    min_area = float(op_config.get("min_component_area", 20))
    min_aspect = float(op_config.get("min_aspect_ratio", 4.0))
    best_score = 0.0
    keep = np.zeros_like(mask)
    for index in range(1, component_count):
        x, y, w, h, area = stats[index]
        if area < min_area:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        fill_ratio = float(area / max(1, w * h))
        if aspect < min_aspect and fill_ratio > 0.25:
            continue
        length_ratio = float(np.hypot(w, h) / max(width, height))
        best_score = max(best_score, length_ratio)
        keep[y : y + h, x : x + w] |= mask[y : y + h, x : x + w]
    return keep, best_score


def _crack_result(
    image_path: Path,
    features: SlotFeatures,
    config: Mapping[str, Any],
    *,
    part_id: str,
    side: str,
    view: str,
    slot_id: str,
    evidence_dir: Path | None,
) -> TraditionalResult:
    """Evaluate elongated dark crack evidence."""
    fallback_threshold = float(_operator_config(config, "crack").get("min_length_ratio", 0.32))
    runtime = _runtime(config)
    threshold = fallback_threshold
    ignore_mask = None
    if runtime is not None:
        threshold = _slot_threshold(runtime.crack_thresholds, slot_id, fallback_threshold)
        ignore_mask = runtime.crack_ignore_masks.get(slot_id)
    roi = _template_roi(_template_for_slot(config, slot_id), features.gray.shape, features.metal_mask, erosion_px=15)
    mask, score = _crack_mask(features, config, roi, ignore_mask)
    pred_label = int(score > threshold)
    reason = f"crack length_ratio={score:.4f} threshold={threshold:.4f}"
    evidence = _evidence_path(evidence_dir, image_path, slot_id, "crack")
    _write_evidence(evidence, features.image, mask, "NG_CRACK" if pred_label else "OK")
    return _make_result(
        image_path=image_path,
        part_id=part_id,
        side=side,
        view=view,
        slot_id=slot_id,
        branch="crack",
        pred_label=pred_label,
        score=score,
        threshold=threshold,
        defect_type="crack",
        evidence_type="dark_line",
        reason=reason,
        evidence_path=evidence,
    )


def _surface_texture_score(
    features: SlotFeatures,
    threshold: float,
    roi_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return local texture score and evidence mask within the material ROI."""
    roi = roi_mask.astype(bool) if roi_mask is not None else features.metal_mask
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(features.gray)
    residual = cv2.Laplacian(clahe, cv2.CV_64F)
    values = residual[roi]
    score = float(values.std()) if values.size else 0.0
    texture_mask = (np.abs(residual) > threshold) & roi
    return score, texture_mask


def _surface_texture_result(
    image_path: Path,
    features: SlotFeatures,
    config: Mapping[str, Any],
    *,
    part_id: str,
    side: str,
    view: str,
    slot_id: str,
    evidence_dir: Path | None,
) -> TraditionalResult:
    """Evaluate local texture residual energy."""
    fallback_threshold = float(_operator_config(config, "surface_texture").get("residual_std_max", 42.0))
    runtime = _runtime(config)
    threshold = fallback_threshold
    if runtime is not None:
        threshold = _slot_threshold(runtime.surface_thresholds, slot_id, fallback_threshold)
    roi = _template_roi(_template_for_slot(config, slot_id), features.gray.shape, features.metal_mask, erosion_px=5)
    score, texture_mask = _surface_texture_score(features, threshold, roi)
    pred_label = int(score > threshold)
    reason = f"surface_texture residual_std={score:.4f} threshold={threshold:.4f}"
    evidence = _evidence_path(evidence_dir, image_path, slot_id, "surface_texture")
    _write_evidence(evidence, features.image, texture_mask, "SUSPECT" if pred_label else "OK")
    return _make_result(
        image_path=image_path,
        part_id=part_id,
        side=side,
        view=view,
        slot_id=slot_id,
        branch="surface_texture",
        pred_label=pred_label,
        score=score,
        threshold=threshold,
        defect_type="surface",
        evidence_type="texture_residual",
        reason=reason,
        evidence_path=evidence,
        status="SUSPECT" if pred_label else "",
    )


def _feature_presence_result(
    image_path: Path,
    features: SlotFeatures,
    config: Mapping[str, Any],
    *,
    part_id: str,
    side: str,
    view: str,
    slot_id: str,
    evidence_dir: Path | None,
) -> TraditionalResult:
    """Check count of dark hole-like features within metal foreground."""
    op_config = _operator_config(config, "feature_presence")
    min_count = int(op_config.get("min_dark_feature_count", 1))
    max_count = int(op_config.get("max_dark_feature_count", 8))
    dark_threshold = max(8.0, features.foreground_mean - 55.0)
    mask = (features.gray < dark_threshold) & features.metal_mask
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    count = 0
    keep = np.zeros_like(mask)
    for index in range(1, component_count):
        area = stats[index, cv2.CC_STAT_AREA]
        if 30 <= area <= 5000:
            count += 1
            x, y, w, h = (
                stats[index, cv2.CC_STAT_LEFT],
                stats[index, cv2.CC_STAT_TOP],
                stats[index, cv2.CC_STAT_WIDTH],
                stats[index, cv2.CC_STAT_HEIGHT],
            )
            keep[y : y + h, x : x + w] |= mask[y : y + h, x : x + w]
    pred_label = int(count < min_count or count > max_count)
    reason = f"feature_presence dark_feature_count={count} expected=[{min_count},{max_count}]"
    evidence = _evidence_path(evidence_dir, image_path, slot_id, "feature_presence")
    _write_evidence(evidence, features.image, keep, "NG_FEATURE" if pred_label else "OK")
    return _make_result(
        image_path=image_path,
        part_id=part_id,
        side=side,
        view=view,
        slot_id=slot_id,
        branch="feature_presence",
        pred_label=pred_label,
        score=float(count),
        threshold=float(min_count),
        defect_type="feature_presence",
        evidence_type="dark_feature_count",
        reason=reason,
        evidence_path=evidence,
    )


def evaluate_slot_image(
    image_path: Path,
    *,
    side: str,
    view: str,
    slot_id: str,
    config: Mapping[str, Any],
    evidence_dir: Path | None = None,
    part_id: str | None = None,
    image: np.ndarray | None = None,
    preset_name: str = "c789_left_top_3x2",
) -> list[TraditionalResult]:
    """Evaluate enabled traditional operators for one slot crop."""
    slot_image = _read_image(image_path) if image is None else image
    features = compute_slot_features(slot_image)
    resolved_part_id = part_id or _part_id_from_path(image_path)
    results: list[TraditionalResult] = []
    common = {
        "part_id": resolved_part_id,
        "side": side,
        "view": view,
        "slot_id": slot_id,
        "evidence_dir": evidence_dir,
    }
    if _enabled(config, "registration"):
        results.append(_registration_result(image_path, features, config, **common))
    if _enabled(config, "geometry"):
        results.append(_geometry_result(image_path, features, config, **common))
    if _enabled(config, "crack"):
        results.append(_crack_result(image_path, features, config, **common))
    if _enabled(config, "surface_texture"):
        results.append(_surface_texture_result(image_path, features, config, **common))
    if _enabled(config, "feature_presence"):
        results.append(_feature_presence_result(image_path, features, config, **common))
    return results


def _looks_like_full_capture(image: np.ndarray, preset_name: str) -> bool:
    """Return whether an image can be processed as a full C789 fixture capture."""
    if preset_name not in PRESETS:
        return False
    preset = PRESETS[preset_name]
    height, width = image.shape[:2]
    return width >= preset.roi.x2 and height >= preset.roi.y2 and bool(preset.slots)


def evaluate_image(
    image_path: Path,
    *,
    preset_name: str,
    side: str,
    view: str,
    config: Mapping[str, Any],
    evidence_dir: Path | None = None,
    input_mode: str = "auto",
) -> list[TraditionalResult]:
    """Evaluate one full capture or one pre-cropped slot image."""
    image = _read_image(image_path)
    slot_from_name = _slot_from_path(image_path)
    as_full = input_mode == "full" or (input_mode == "auto" and slot_from_name is None)
    if as_full and _looks_like_full_capture(image, preset_name):
        preset = PRESETS[preset_name]
        part_id = _part_id_from_path(image_path)
        results: list[TraditionalResult] = []
        for slot in preset.slots:
            crop = crop_slot(image, preset.roi, slot)
            crop = mask_holes(crop, preset.slot_hole_masks.get(slot.name, ()), "median", 9)
            results.extend(
                evaluate_slot_image(
                    image_path,
                    side=side,
                    view=view,
                    slot_id=slot.name,
                    config=config,
                    evidence_dir=evidence_dir,
                    part_id=part_id,
                    image=crop,
                    preset_name=preset_name,
                ),
            )
        return results
    slot_id = slot_from_name or "slot00"
    return evaluate_slot_image(
        image_path,
        side=side,
        view=view,
        slot_id=slot_id,
        config=config,
        evidence_dir=evidence_dir,
        preset_name=preset_name,
    )


def write_branch_csv(results: Sequence[TraditionalResult], output_csv: Path) -> None:
    """Write traditional operator results to a fusion-compatible CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BRANCH_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "part_id": result.part_id,
                    "side": result.side,
                    "view": result.view,
                    "slot_id": result.slot_id,
                    "branch": result.branch,
                    "pred_label": result.pred_label,
                    "score": result.score,
                    "threshold": "" if result.threshold is None else result.threshold,
                    "defect_type": result.defect_type,
                    "evidence_type": result.evidence_type,
                    "gt_defect_type": result.gt_defect_type,
                    "reason": result.reason,
                    "source_path": result.source_path,
                    "evidence_path": result.evidence_path,
                    "status": result.status,
                },
            )


def aggregate_case_predictions(
    results: Sequence[TraditionalResult],
    *,
    label_override: str = "auto",
) -> list[CasePrediction]:
    """Aggregate branch rows into case-level predictions for FP/FN reporting."""
    grouped: dict[tuple[str, str], list[TraditionalResult]] = {}
    for result in results:
        grouped.setdefault((result.source_path, result.slot_id), []).append(result)
    cases: list[CasePrediction] = []
    for (source_path, slot_id), rows in sorted(grouped.items()):
        first = rows[0]
        positives = tuple(sorted({row.branch for row in rows if row.pred_label == 1}))
        label = infer_label_from_path(Path(source_path), label_override)
        case_id = f"{Path(source_path).stem}_{slot_id}"
        cases.append(
            CasePrediction(
                case_id=case_id,
                label=label,
                part_id=first.part_id,
                side=first.side,
                view=first.view,
                slot_id=slot_id,
                pred_label=int(bool(positives)),
                positive_branches=positives,
                gt_defect_type=infer_gt_defect_type_from_path(Path(source_path), label_override),
                source_path=source_path,
            ),
        )
    return cases


def summarize_cases(cases: Sequence[CasePrediction], *, row_count: int) -> dict[str, float | int]:
    """Compute compact false-positive and false-negative metrics."""
    normal_cases = [case for case in cases if case.label == "normal"]
    defect_cases = [case for case in cases if case.label == "defect"]
    invalid_cases = [case for case in cases if case.label == "invalid"]
    unknown_cases = [case for case in cases if case.label not in {"normal", "defect", "invalid"}]
    normal_false_positive = sum(case.pred_label for case in normal_cases)
    defect_detected = sum(case.pred_label for case in defect_cases)
    defect_false_negative = len(defect_cases) - defect_detected
    invalid_rejected = sum(case.pred_label for case in invalid_cases)
    positive_cases = sum(case.pred_label for case in cases)
    return {
        "row_count": row_count,
        "case_total": len(cases),
        "positive_cases": positive_cases,
        "normal_total": len(normal_cases),
        "normal_false_positive": normal_false_positive,
        "false_positive_rate": _safe_ratio(normal_false_positive, len(normal_cases)),
        "defect_total": len(defect_cases),
        "defect_detected": defect_detected,
        "defect_false_negative": defect_false_negative,
        "false_negative_rate": _safe_ratio(defect_false_negative, len(defect_cases)),
        "defect_recall": _safe_ratio(defect_detected, len(defect_cases)),
        "invalid_total": len(invalid_cases),
        "invalid_rejected": invalid_rejected,
        "unknown_total": len(unknown_cases),
    }


def primary_evidence_by_case(results: Sequence[TraditionalResult]) -> dict[tuple[str, str], str]:
    """Return the primary positive evidence per case without pretending it is the GT class."""
    grouped: dict[tuple[str, str], list[TraditionalResult]] = {}
    for result in results:
        grouped.setdefault((result.source_path, result.slot_id), []).append(result)
    primary: dict[tuple[str, str], str] = {}
    for key, rows in grouped.items():
        positives = [row for row in rows if row.pred_label == 1]
        if not positives:
            primary[key] = "MISS"
            continue
        positives.sort(key=lambda row: _evidence_priority(row.branch))
        first = positives[0]
        evidence_type = first.evidence_type or first.defect_type or "positive"
        primary[key] = f"{first.branch}:{evidence_type}"
    return primary


def _evidence_priority(branch: str) -> int:
    """Rank evidence branches for human-facing case summaries."""
    priority = {
        "registration": 0,
        "geometry": 1,
        "crack": 2,
        "surface_texture": 3,
        "feature_presence": 4,
    }
    return priority.get(branch, 50)


def defect_evidence_confusion(
    cases: Sequence[CasePrediction],
    results: Sequence[TraditionalResult],
) -> list[dict[str, str]]:
    """Count GT defect types against primary evidence labels for offline analysis."""
    primary = primary_evidence_by_case(results)
    counts: dict[tuple[str, str], int] = {}
    for case in cases:
        if case.label != "defect":
            continue
        gt_defect_type = case.gt_defect_type or "unknown"
        evidence = primary.get((case.source_path, case.slot_id), "MISS")
        counts[(gt_defect_type, evidence)] = counts.get((gt_defect_type, evidence), 0) + 1
    return [
        {"gt_defect_type": gt, "primary_evidence": evidence, "count": str(count)}
        for (gt, evidence), count in sorted(counts.items())
    ]


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a ratio while keeping empty denominators explicit as zero."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def write_defect_evidence_confusion_csv(rows: Sequence[Mapping[str, str]], output_csv: Path) -> None:
    """Write GT defect type vs primary evidence counts."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CONFUSION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_case_predictions_csv(cases: Sequence[CasePrediction], output_csv: Path) -> None:
    """Write per-case predictions used for FP/FN reporting."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CASE_FIELDNAMES)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "label": case.label,
                    "part_id": case.part_id,
                    "side": case.side,
                    "view": case.view,
                    "slot_id": case.slot_id,
                    "pred_label": case.pred_label,
                    "positive_branches": ";".join(case.positive_branches),
                    "gt_defect_type": case.gt_defect_type,
                    "source_path": case.source_path,
                },
            )


def write_summary_csv(summary: Mapping[str, float | int], output_csv: Path) -> None:
    """Write summary metrics as metric/value rows."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": _format_metric(value)})


def write_summary_markdown(
    summary: Mapping[str, float | int],
    output_md: Path,
    confusion_rows: Sequence[Mapping[str, str]] | None = None,
) -> None:
    """Write a human-readable traditional-operator summary."""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Traditional Operators Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | {_format_metric(value)} |")
    if confusion_rows:
        lines.extend(
            [
                "",
                "## GT Defect Type vs Primary Evidence",
                "",
                "This table is for offline analysis. `primary_evidence` describes the traditional branch evidence, "
                "not a claimed defect-class prediction.",
                "",
                "| gt_defect_type | primary_evidence | count |",
                "| --- | --- | ---: |",
            ],
        )
        for row in confusion_rows:
            lines.append(
                f"| `{row['gt_defect_type']}` | `{row['primary_evidence']}` | {row['count']} |",
            )
    lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def _format_metric(value: float | int) -> str:
    """Return stable metric text."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def run_traditional_operators(args: argparse.Namespace) -> list[TraditionalResult]:
    """Run traditional operators for all images under an input root."""
    config = load_config(args.config)
    runtime = build_runtime_context(args, config)
    if runtime is not None:
        config = dict(config)
        config["_runtime"] = runtime
    evidence_dir = args.output_dir / "evidence" if args.save_evidence else None
    results: list[TraditionalResult] = []
    image_paths = iter_image_paths(args.input_root)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    show_progress = bool(getattr(args, "show_progress", False))
    label_override = str(getattr(args, "label", "auto"))
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        if show_progress:
            print(f"[traditional] {index}/{total} {image_path}")
        results.extend(
            evaluate_image(
                image_path,
                preset_name=args.preset,
                side=args.side,
                view=args.view,
                config=config,
                evidence_dir=evidence_dir,
                input_mode=args.input_mode,
            ),
        )
    write_branch_csv(results, args.output_dir / "traditional_predictions.csv")
    cases = aggregate_case_predictions(results, label_override=label_override)
    summary = summarize_cases(cases, row_count=len(results))
    confusion_rows = defect_evidence_confusion(cases, results)
    write_case_predictions_csv(cases, args.output_dir / "traditional_cases.csv")
    write_summary_csv(summary, args.output_dir / "traditional_summary.csv")
    write_defect_evidence_confusion_csv(
        confusion_rows,
        args.output_dir / "traditional_defect_evidence_confusion.csv",
    )
    write_summary_markdown(summary, args.output_dir / "traditional_summary.md", confusion_rows)
    if show_progress:
        print(
            "[traditional] summary "
            f"normal_fp={summary['normal_false_positive']}/{summary['normal_total']} "
            f"defect_fn={summary['defect_false_negative']}/{summary['defect_total']}",
        )
    return results
