# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rule-based image quality gates for inspection captures."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PASS_STATUS = "PASS"
WARN_STATUS = "WARN"
FAIL_STATUS = "FAIL"
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
QUALITY_FIELDNAMES = [
    "part_id",
    "side",
    "view",
    "branch",
    "status",
    "fail_label",
    "reason",
    "source_path",
    "image_path",
    "brightness_mean",
    "brightness_std",
    "saturation_ratio",
    "dark_ratio",
    "blur_laplacian_var",
    "highlight_ratio",
    "foreground_coverage",
]
CALIBRATED_MIN_MAX_METRICS = ("brightness_mean", "brightness_std")
CALIBRATED_MAX_METRICS = ("saturation_ratio", "dark_ratio", "highlight_ratio")
CALIBRATED_MIN_METRICS = ("blur_laplacian_var",)


@dataclass(frozen=True)
class ImageQualityMetrics:
    """Whole-image quality metrics for one capture."""

    image_path: str
    side: str | None
    view: str | None
    brightness_mean: float
    brightness_std: float
    saturation_ratio: float
    dark_ratio: float
    blur_laplacian_var: float
    highlight_ratio: float
    foreground_coverage: float | None = None


@dataclass(frozen=True)
class QualityGateResult:
    """PASS/WARN/FAIL quality decision for one capture."""

    status: str
    reasons: list[str]
    metrics: ImageQualityMetrics


def iter_image_paths(root: Path) -> list[Path]:
    """Return readable image candidate paths in stable order."""
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def infer_side_view(path: Path) -> tuple[str | None, str | None]:
    """Infer side and view labels from a path when possible."""
    text = str(path).lower()
    side = None
    if re.search(r"(?:^|[_/\-])top(?:$|[_/\-.])", text):
        side = "top"
    elif re.search(r"(?:^|[_/\-])bottom(?:$|[_/\-.])", text):
        side = "bottom"

    view = None
    for candidate in (
        "uniform",
        "darkfield",
        "left_bar",
        "right_bar",
        "left_top",
        "left_bottom",
        "no_hand_top",
        "no_hand_bottom",
    ):
        if candidate in text:
            view = candidate
            break
    return side, view


def _foreground_coverage(foreground_mask: np.ndarray | None, shape: tuple[int, int]) -> float | None:
    """Return foreground coverage when a mask is supplied."""
    if foreground_mask is None:
        return None
    mask = foreground_mask.astype(bool)
    if mask.shape != shape:
        msg = f"foreground_mask shape {mask.shape} does not match image shape {shape}"
        raise ValueError(msg)
    return float(mask.mean())


def compute_quality_metrics(image_path: Path, *, foreground_mask: np.ndarray | None = None) -> ImageQualityMetrics:
    """Compute whole-image quality metrics without running a model."""
    if not image_path.is_file():
        msg = f"Could not read image for quality gate: {image_path}"
        raise ValueError(msg)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        msg = f"Could not read image for quality gate: {image_path}"
        raise ValueError(msg)
    if image.ndim != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        msg = f"Invalid image shape for quality gate: {image.shape}"
        raise ValueError(msg)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    saturation_ratio = float((image >= 250).any(axis=2).mean())
    dark_ratio = float((gray <= 5).mean())
    blur_laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    highlight_cutoff = max(brightness_mean + 2.5 * brightness_std, 245.0)
    highlight_ratio = float((gray >= highlight_cutoff).mean())
    side, view = infer_side_view(image_path)
    return ImageQualityMetrics(
        image_path=str(image_path.resolve(strict=False)),
        side=side,
        view=view,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        saturation_ratio=saturation_ratio,
        dark_ratio=dark_ratio,
        blur_laplacian_var=blur_laplacian_var,
        highlight_ratio=highlight_ratio,
        foreground_coverage=_foreground_coverage(foreground_mask, gray.shape),
    )


def _metric_value(metrics: ImageQualityMetrics, name: str) -> float | None:
    """Return a named metric value."""
    value = getattr(metrics, name)
    return None if value is None else float(value)


def _format_limit(value: Any) -> str:
    """Return a compact threshold string."""
    return f"{float(value):.6g}"


def evaluate_quality_gate(metrics: ImageQualityMetrics, config: Mapping[str, Any]) -> QualityGateResult:
    """Evaluate quality metrics against a threshold config."""
    mode = str(config.get("mode", "fail")).lower()
    if mode == "off":
        return QualityGateResult(PASS_STATUS, [], metrics)

    reasons: list[str] = []
    metric_config = config.get("metrics", {})
    if not isinstance(metric_config, Mapping):
        msg = "quality config 'metrics' must be a mapping"
        raise ValueError(msg)

    for name, rule in metric_config.items():
        if not isinstance(rule, Mapping):
            continue
        value = _metric_value(metrics, str(name))
        if value is None:
            continue
        minimum = rule.get("min")
        maximum = rule.get("max")
        if minimum not in {None, ""} and value < float(minimum):
            reasons.append(f"{name} below min: {_format_limit(value)} < {_format_limit(minimum)}")
        if maximum not in {None, ""} and value > float(maximum):
            reasons.append(f"{name} above max: {_format_limit(value)} > {_format_limit(maximum)}")

    if not reasons:
        status = PASS_STATUS
    elif mode == "warn":
        status = WARN_STATUS
    else:
        status = FAIL_STATUS
    return QualityGateResult(status, reasons, metrics)


def evaluate_image_quality(image_path: Path, config: Mapping[str, Any]) -> QualityGateResult:
    """Evaluate one image, converting read errors into FAIL results."""
    try:
        metrics = compute_quality_metrics(image_path)
    except ValueError as error:
        metrics = ImageQualityMetrics(
            image_path=str(image_path.resolve(strict=False)),
            side=None,
            view=None,
            brightness_mean=0.0,
            brightness_std=0.0,
            saturation_ratio=0.0,
            dark_ratio=1.0,
            blur_laplacian_var=0.0,
            highlight_ratio=0.0,
            foreground_coverage=None,
        )
        return QualityGateResult(FAIL_STATUS, [str(error)], metrics)
    return evaluate_quality_gate(metrics, config)


def _part_id_from_path(path: Path) -> str:
    """Return a stable part id from an image path."""
    stem = path.stem
    for token in ("_top_", "_bottom_", "_uniform", "_darkfield", "_left_bar", "_right_bar"):
        if token in stem:
            stem = stem.split(token, maxsplit=1)[0]
    return stem


def quality_result_row(result: QualityGateResult) -> dict[str, str | float | int]:
    """Return a CSV row for one quality result."""
    metrics = result.metrics
    image_path = Path(metrics.image_path)
    reason = "; ".join(result.reasons)
    return {
        "part_id": _part_id_from_path(image_path),
        "side": "" if metrics.side is None else metrics.side,
        "view": "" if metrics.view is None else metrics.view,
        "branch": "quality_gate",
        "status": result.status,
        "fail_label": int(result.status == FAIL_STATUS),
        "reason": reason,
        "source_path": metrics.image_path,
        "image_path": metrics.image_path,
        "brightness_mean": metrics.brightness_mean,
        "brightness_std": metrics.brightness_std,
        "saturation_ratio": metrics.saturation_ratio,
        "dark_ratio": metrics.dark_ratio,
        "blur_laplacian_var": metrics.blur_laplacian_var,
        "highlight_ratio": metrics.highlight_ratio,
        "foreground_coverage": "" if metrics.foreground_coverage is None else metrics.foreground_coverage,
    }


def write_quality_gate_csv(results: Sequence[QualityGateResult], output_csv: Path) -> None:
    """Write quality gate results to a fusion-compatible CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=QUALITY_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow(quality_result_row(result))


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def _metric_values(metrics: Sequence[ImageQualityMetrics], name: str) -> list[float]:
    """Return all present values for one metric."""
    values = []
    for item in metrics:
        value = _metric_value(item, name)
        if value is not None:
            values.append(value)
    return values


def calibrate_quality_config(
    normal_metrics: Sequence[ImageQualityMetrics],
    *,
    mode: str = "warn",
    margin_ratio: float = 0.05,
) -> dict[str, Any]:
    """Build a conservative quality config from normal/stress-normal metrics."""
    if not normal_metrics:
        msg = "At least one normal/stress-normal image is required to calibrate quality thresholds."
        raise ValueError(msg)

    metrics_config: dict[str, dict[str, float]] = {}
    for name in CALIBRATED_MIN_MAX_METRICS:
        values = _metric_values(normal_metrics, name)
        if values:
            minimum = max(0.0, _percentile(values, 0.005) * (1.0 - margin_ratio))
            maximum = _percentile(values, 0.995) * (1.0 + margin_ratio)
            metrics_config[name] = {"min": round(minimum, 6), "max": round(maximum, 6)}
    for name in CALIBRATED_MAX_METRICS:
        values = _metric_values(normal_metrics, name)
        if values:
            metrics_config[name] = {"max": round(_percentile(values, 0.995) * (1.0 + margin_ratio), 6)}
    for name in CALIBRATED_MIN_METRICS:
        values = _metric_values(normal_metrics, name)
        if values:
            metrics_config[name] = {"min": round(max(0.0, _percentile(values, 0.005) * (1.0 - margin_ratio)), 6)}
    return {
        "version": 1,
        "mode": mode,
        "metrics": metrics_config,
    }


def write_quality_config(config: Mapping[str, Any], output_path: Path) -> None:
    """Write a quality config as YAML when available, otherwise JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ModuleNotFoundError:
        output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    output_path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
