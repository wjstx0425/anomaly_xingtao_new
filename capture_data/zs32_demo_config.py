# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the single editable ZS32 Demo configuration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEW_ORDER = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)
SECONDARY_VIEWS = frozenset(("front_secondary", "back_secondary"))


@dataclass(frozen=True, slots=True)
class YoloConfig:
    """YOLO model settings used by the Demo runtime."""

    weights: Path
    imgsz: int
    candidate_conf: float


@dataclass(frozen=True, slots=True)
class DemoThresholds:
    """Per-view decision thresholds for the three Demo branches."""

    template: Mapping[str, float]
    patchcore: Mapping[str, float]
    yolo: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class DemoConfig:
    """Resolved model, geometry, and threshold settings for the ZS32 Demo."""

    path: Path
    patchcore_process_count: int
    topology: Path
    roi_config: Path
    template_dir: Path
    patchcore: Mapping[str, Path]
    yolo: YoloConfig
    thresholds: DemoThresholds


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return cast("dict[str, Any]", payload)


def _object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return cast("dict[str, Any]", value)


def _resolve_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _existing_file(root: Path, value: object, *, field: str) -> Path:
    path = _resolve_path(root, value, field=field)
    if not path.is_file():
        raise FileNotFoundError(f"{field} file does not exist: {path}")
    return path


def _existing_directory(root: Path, value: object, *, field: str) -> Path:
    path = _resolve_path(root, value, field=field)
    if not path.is_dir():
        raise FileNotFoundError(f"{field} directory does not exist: {path}")
    return path


def _canonical_view_object(value: object, *, field: str) -> dict[str, Any]:
    records = _object(value, field=field)
    if tuple(records) != VIEW_ORDER:
        raise ValueError(f"{field} must contain exactly the canonical eight views in order: {VIEW_ORDER}")
    return records


def _validate_topology(path: Path) -> None:
    payload = _read_json_object(path, label="topology")
    required_views = payload.get("required_views")
    if not isinstance(required_views, list) or tuple(required_views) != VIEW_ORDER:
        raise ValueError(f"topology must require exactly the canonical eight views in order: {VIEW_ORDER}")

    camera_slots = payload.get("camera_slots")
    if not isinstance(camera_slots, list) or len(camera_slots) != 4:
        raise ValueError("topology must declare exactly four camera slots")
    serials: set[str] = set()
    mapped_views: list[str] = []
    for index, slot in enumerate(camera_slots):
        record = _object(slot, field=f"topology camera_slots[{index}]")
        serial = record.get("serial")
        if not isinstance(serial, str) or not serial.strip() or serial in serials:
            raise ValueError("topology camera serials must be non-empty and unique")
        serials.add(serial)
        views = _object(record.get("views"), field=f"topology camera_slots[{index}].views")
        if tuple(views) != ("front", "back"):
            raise ValueError(f"topology camera_slots[{index}] must map front and back")
        if not all(isinstance(view, str) for view in views.values()):
            raise TypeError(f"topology camera_slots[{index}] view names must be strings")
        mapped_views.extend(cast("str", view) for view in views.values())

    missing_secondary = SECONDARY_VIEWS.difference(mapped_views)
    if missing_secondary:
        raise ValueError(f"topology is missing secondary view mappings: {sorted(missing_secondary)}")
    if len(mapped_views) != len(VIEW_ORDER) or set(mapped_views) != set(VIEW_ORDER):
        raise ValueError(f"topology camera mappings must produce exactly the canonical eight views: {VIEW_ORDER}")


def _validate_roi_config(path: Path) -> None:
    payload = _read_json_object(path, label="ROI config")
    image_size = _object(payload.get("image_size"), field="ROI image_size")
    width = image_size.get("width")
    height = image_size.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
    ):
        raise ValueError("ROI image_size width and height must be positive integers")

    views = _canonical_view_object(payload.get("views"), field="ROI views")
    for view, record_value in views.items():
        record = _object(record_value, field=f"ROI {view}")
        roi = record.get("roi")
        if (
            not isinstance(roi, list)
            or len(roi) != 4
            or any(not isinstance(coordinate, int) or isinstance(coordinate, bool) for coordinate in roi)
        ):
            raise ValueError(f"ROI for {view} must contain four integer coordinates")
        x1, y1, x2, y2 = cast("list[int]", roi)
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
            raise ValueError(f"ROI for {view} is outside image bounds or has non-positive area: {roi}")


def _threshold_mapping(value: object, *, branch: str, maximum: float) -> Mapping[str, float]:
    records = _canonical_view_object(value, field=f"{branch} thresholds")
    validated: dict[str, float] = {}
    for view, raw_threshold in records.items():
        if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
            raise TypeError(f"{branch} threshold for {view} must be a number")
        threshold = float(raw_threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= maximum:
            raise ValueError(f"{branch} threshold for {view} must be finite and in [0, {maximum:g}]")
        validated[view] = threshold
    return MappingProxyType(validated)


def load_demo_config(path: str | Path, repo_root: str | Path | None = None) -> DemoConfig:
    """Load a Demo JSON without reading or checking any asset hashes.

    Args:
        path: Demo JSON path. Relative paths are interpreted from ``repo_root``.
        repo_root: Repository root used to resolve every relative path. Defaults
            to the repository containing this module.

    Returns:
        A validated configuration with absolute, resolved asset paths.

    Raises:
        FileNotFoundError: If the configuration or a declared asset is absent.
        TypeError: If a field has the wrong JSON type.
        ValueError: If JSON syntax, views, geometry, or thresholds are invalid.
    """

    root = Path(repo_root).expanduser().resolve() if repo_root is not None else REPO_ROOT
    config_path = _resolve_path(root, str(path), field="Demo config")
    if not config_path.is_file():
        raise FileNotFoundError(f"Demo config file does not exist: {config_path}")
    payload = _read_json_object(config_path, label="Demo config")
    if payload.get("schema_version") != 1:
        raise ValueError("Demo config schema_version must be exactly 1")
    if payload.get("mode") != "demo":
        raise ValueError("Demo config mode must be exactly 'demo'")
    patchcore_process_count = payload.get("patchcore_process_count")
    if not isinstance(patchcore_process_count, int) or isinstance(patchcore_process_count, bool):
        raise TypeError("patchcore_process_count must be an integer")
    if patchcore_process_count not in {1, 2, 4, 8}:
        raise ValueError("patchcore_process_count must be 1, 2, 4, or 8")

    topology = _existing_file(root, payload.get("topology"), field="topology")
    roi_config = _existing_file(root, payload.get("roi_config"), field="roi")
    _validate_topology(topology)
    _validate_roi_config(roi_config)

    models = _object(payload.get("models"), field="models")
    template_dir = _existing_directory(root, models.get("template_dir"), field="template")
    patchcore_payload = _canonical_view_object(models.get("patchcore"), field="patchcore models")
    patchcore = MappingProxyType(
        {
            view: _existing_file(root, checkpoint, field=f"patchcore {view}")
            for view, checkpoint in patchcore_payload.items()
        },
    )
    yolo_payload = _object(models.get("yolo"), field="yolo")
    yolo_weights = _existing_file(root, yolo_payload.get("weights"), field="yolo weights")
    imgsz = yolo_payload.get("imgsz")
    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz <= 0:
        raise ValueError("yolo imgsz must be a positive integer")
    candidate_conf_raw = yolo_payload.get("candidate_conf")
    if isinstance(candidate_conf_raw, bool) or not isinstance(candidate_conf_raw, (int, float)):
        raise TypeError("yolo candidate_conf must be a number")
    candidate_conf = float(candidate_conf_raw)
    if not math.isfinite(candidate_conf) or not 0 <= candidate_conf <= 1:
        raise ValueError("yolo candidate_conf must be finite and in [0, 1]")

    thresholds_payload = _object(payload.get("thresholds"), field="thresholds")
    if tuple(thresholds_payload) != ("template", "patchcore", "yolo"):
        raise ValueError("thresholds must contain exactly template, patchcore, and yolo in order")
    thresholds = DemoThresholds(
        template=_threshold_mapping(thresholds_payload["template"], branch="template", maximum=2),
        patchcore=_threshold_mapping(thresholds_payload["patchcore"], branch="patchcore", maximum=2),
        yolo=_threshold_mapping(thresholds_payload["yolo"], branch="yolo", maximum=1),
    )
    return DemoConfig(
        path=config_path,
        patchcore_process_count=patchcore_process_count,
        topology=topology,
        roi_config=roi_config,
        template_dir=template_dir,
        patchcore=patchcore,
        yolo=YoloConfig(weights=yolo_weights, imgsz=imgsz, candidate_conf=candidate_conf),
        thresholds=thresholds,
    )
