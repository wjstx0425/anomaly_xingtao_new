"""Manual ignore-mask assets for BMW eight-view EfficientAD diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np


Point = tuple[int, int]
Polygon = Sequence[Point]


@dataclass(frozen=True, slots=True)
class IgnoreMaskAsset:
    """One validated, immutable set of original-ROI-sized ignore masks."""

    index_path: Path
    index_sha256: str
    public_roi_config_sha256: str
    source_capture_id: str
    masks: Mapping[str, np.ndarray]
    mask_sha256_by_view: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MaskedAnomalyMap:
    """An anomaly map with operator-selected pixels excluded from scoring."""

    score: float
    raw_max: float
    hotspot_x: int
    hotspot_y: int
    ignored_map_pixel_count: int
    masked_map: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_binary_ignore_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != np.uint8 or array.ndim != 2 or tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"ignore mask must be uint8 with shape {expected_shape}, got {array.dtype}/{array.shape}")
    if not set(np.unique(array).tolist()).issubset({0, 255}):
        raise ValueError("ignore mask must contain only 0 and 255")
    return array


def polygons_to_ignore_mask(shape: tuple[int, int], polygons: Sequence[Polygon]) -> np.ndarray:
    """Rasterize zero or more source-image polygons using ``255=ignore``."""
    if (
        len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or shape[0] <= 0
        or shape[1] <= 0
    ):
        raise ValueError("shape must contain positive integer height and width")
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    contours: list[np.ndarray] = []
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("each ignore polygon must contain at least three points")
        points: list[Point] = []
        for point in polygon:
            if (
                len(point) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in point)
            ):
                raise ValueError("polygon points must be integer (x, y) pairs")
            x, y = point
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError(f"polygon point {(x, y)} lies outside {width}x{height}")
            points.append((x, y))
        contours.append(np.asarray(points, dtype=np.int32).reshape((-1, 1, 2)))
    if contours:
        cv2.fillPoly(mask, contours, 255)
    return mask


def mask_anomaly_map(anomaly_map: np.ndarray, ignore_mask: np.ndarray) -> MaskedAnomalyMap:
    """Exclude an original-ROI-sized manual mask from one returned anomaly map."""
    map_value = np.asarray(anomaly_map)
    if map_value.ndim != 2 or map_value.size == 0 or not np.isfinite(map_value).all():
        raise ValueError("anomaly_map must be a finite non-empty 2D array")
    mask_value = np.asarray(ignore_mask)
    if mask_value.dtype != np.uint8 or mask_value.ndim != 2 or mask_value.size == 0:
        raise ValueError("ignore_mask must be a non-empty uint8 2D array")
    if not set(np.unique(mask_value).tolist()).issubset({0, 255}):
        raise ValueError("ignore_mask must contain only 0 and 255")
    resized = cv2.resize(
        mask_value,
        (map_value.shape[1], map_value.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    valid = resized == 0
    if not np.any(valid):
        raise ValueError("ignore mask excludes every anomaly-map pixel")
    scoring = np.where(valid, map_value, -np.inf)
    hotspot_y, hotspot_x = np.unravel_index(int(np.argmax(scoring)), scoring.shape)
    masked = np.where(valid, map_value, 0).copy()
    masked.flags.writeable = False
    return MaskedAnomalyMap(
        score=float(scoring[hotspot_y, hotspot_x]),
        raw_max=float(np.max(map_value)),
        hotspot_x=int(hotspot_x),
        hotspot_y=int(hotspot_y),
        ignored_map_pixel_count=int(np.count_nonzero(~valid)),
        masked_map=masked,
    )


def _preview(image: np.ndarray, mask: np.ndarray, polygons: Sequence[Polygon]) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3].copy()
    tinted = base.copy()
    tinted[mask == 255] = (0, 0, 255)
    overlay = cv2.addWeighted(base, 0.65, tinted, 0.35, 0.0)
    for polygon in polygons:
        contour = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [contour], True, (0, 255, 255), 3, cv2.LINE_AA)
    return overlay


def save_ignore_mask_asset(
    output_dir: Path,
    *,
    images: Mapping[str, np.ndarray],
    polygons_by_view: Mapping[str, Sequence[Polygon]],
    source_paths: Mapping[str, Path],
    source_capture_id: str,
    public_roi_config: Path,
) -> Path:
    """Atomically publish a no-overwrite manual ignore-mask bundle."""
    views = tuple(images)
    if not views or tuple(polygons_by_view) != views or tuple(source_paths) != views:
        raise ValueError("images, polygons, and source paths must use one exact view order")
    if not source_capture_id.strip():
        raise ValueError("source_capture_id must not be empty")
    roi_path = Path(public_roi_config).expanduser().resolve()
    if not roi_path.is_file():
        raise ValueError(f"public ROI config does not exist: {roi_path}")
    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"ignore-mask asset already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (staging / "masks").mkdir()
        (staging / "previews").mkdir()
        records: dict[str, object] = {}
        for view in views:
            image = np.asarray(images[view])
            if image.dtype != np.uint8 or image.ndim not in {2, 3} or image.size == 0:
                raise ValueError(f"{view} source ROI must be a non-empty uint8 image")
            source_path = Path(source_paths[view]).expanduser().resolve()
            if not source_path.is_file():
                raise ValueError(f"{view} source ROI does not exist: {source_path}")
            polygons = [list(polygon) for polygon in polygons_by_view[view]]
            mask = polygons_to_ignore_mask(image.shape[:2], polygons)
            mask_path = staging / "masks" / f"{view}.png"
            preview_path = staging / "previews" / f"{view}.jpg"
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"failed to write ignore mask: {view}")
            if not cv2.imwrite(str(preview_path), _preview(image, mask, polygons)):
                raise OSError(f"failed to write ignore-mask preview: {view}")
            records[view] = {
                "mask_path": f"masks/{view}.png",
                "mask_sha256": _sha256(mask_path),
                "preview_path": f"previews/{view}.jpg",
                "source_roi_path": str(source_path),
                "source_roi_sha256": _sha256(source_path),
                "roi_width": int(image.shape[1]),
                "roi_height": int(image.shape[0]),
                "polygon_count": len(polygons),
                "polygons": [[[int(x), int(y)] for x, y in polygon] for polygon in polygons],
                "ignored_pixel_count": int(np.count_nonzero(mask)),
            }
        payload = {
            "schema_version": "bmw.efficientad_manual_ignore_masks/1.0",
            "purpose": "exclude_operator_selected_regions_from_efficientad_anomaly_map_scoring",
            "mask_semantics": {"0": "inspect", "255": "ignore"},
            "source_capture_id": source_capture_id,
            "public_roi_config": str(roi_path),
            "public_roi_config_sha256": _sha256(roi_path),
            "views": records,
        }
        index = staging / "index.json"
        index.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "index.json"


def load_ignore_mask_asset(
    index_path: Path,
    *,
    expected_views: Sequence[str],
    expected_roi_config_sha256: str,
    expected_shapes: Mapping[str, tuple[int, int]],
) -> IgnoreMaskAsset:
    """Load manual ignore masks and fail closed on order, size, or SHA drift."""
    path = Path(index_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ignore-mask index: {path}: {error}") from error
    if payload.get("schema_version") != "bmw.efficientad_manual_ignore_masks/1.0":
        raise ValueError("unsupported ignore-mask schema")
    if payload.get("mask_semantics") != {"0": "inspect", "255": "ignore"}:
        raise ValueError("ignore-mask semantics are invalid")
    if payload.get("public_roi_config_sha256") != expected_roi_config_sha256:
        raise ValueError("ignore-mask public ROI SHA mismatch")
    source_capture_id = payload.get("source_capture_id")
    if not isinstance(source_capture_id, str) or not source_capture_id.strip():
        raise ValueError("ignore-mask source_capture_id is invalid")
    records = payload.get("views")
    if not isinstance(records, dict) or tuple(records) != tuple(expected_views):
        raise ValueError("ignore-mask views do not match the expected order")
    if tuple(expected_shapes) != tuple(expected_views):
        raise ValueError("expected_shapes do not match the expected view order")
    masks: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for view in expected_views:
        record = records[view]
        if not isinstance(record, dict):
            raise ValueError(f"ignore-mask record is invalid: {view}")
        relative = record.get("mask_path")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"ignore-mask path is invalid: {view}")
        mask_path = (path.parent / relative).resolve()
        digest = _sha256(mask_path)
        if digest != record.get("mask_sha256"):
            raise ValueError(f"ignore-mask SHA mismatch: {view}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"ignore mask is unreadable: {view}")
        _validate_binary_ignore_mask(mask, expected_shapes[view])
        if (record.get("roi_height"), record.get("roi_width")) != expected_shapes[view]:
            raise ValueError(f"ignore-mask declared shape mismatch: {view}")
        mask.setflags(write=False)
        masks[view] = mask
        digests[view] = digest
    return IgnoreMaskAsset(
        index_path=path,
        index_sha256=_sha256(path),
        public_roi_config_sha256=expected_roi_config_sha256,
        source_capture_id=source_capture_id,
        masks=MappingProxyType(masks),
        mask_sha256_by_view=MappingProxyType(digests),
    )
