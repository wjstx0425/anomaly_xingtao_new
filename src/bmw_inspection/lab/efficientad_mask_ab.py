"""Pure helpers for the BMW EfficientAD foreground-mask offline A/B experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np


@dataclass(frozen=True)
class MapAggregates:
    """Comparable foreground/background summaries from one returned anomaly map."""

    global_max: float
    foreground_max: float
    background_max: float | None
    foreground_quantile: float
    background_quantile: float | None
    foreground_top_k_mean: float
    background_top_k_mean: float | None
    foreground_component_score: float
    background_component_score: float | None
    hotspot_x: int
    hotspot_y: int
    hotspot_region: str
    foreground_pixel_count: int
    background_pixel_count: int
    foreground_activation_sum: float
    background_activation_sum: float
    background_activation_fraction: float
    background_max_advantage: float

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Return JSON/CSV-friendly scalar values."""
        return asdict(self)


@dataclass(frozen=True)
class ForegroundMaskAsset:
    """Validated immutable set of original-ROI-sized foreground masks."""

    index_path: Path
    index_sha256: str
    public_roi_config_sha256: str
    fixed_fill_value: int
    masks: Mapping[str, np.ndarray]
    mask_sha256_by_view: Mapping[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_binary_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    """Validate an original-ROI-sized `{0, 255}` foreground mask."""
    array = np.asarray(mask)
    if array.ndim != 2 or tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"mask shape must be {expected_shape}, got {array.shape}")
    values = np.unique(array)
    if not set(values.tolist()).issubset({0, 255}):
        raise ValueError("mask must be binary with values 0 and 255")
    if not np.any(array == 255):
        raise ValueError("mask foreground must not be empty")
    return array == 255


def apply_fixed_fill(image: np.ndarray, mask: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Fill pixels outside the foreground while preserving image shape and dtype."""
    array = np.asarray(image)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ValueError("image must be grayscale, BGR, or BGRA")
    if array.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    if isinstance(fill_value, bool) or not isinstance(fill_value, int) or not 0 <= fill_value <= 255:
        raise ValueError("fill_value must be an integer in [0, 255]")
    foreground = validate_binary_mask(mask, array.shape[:2])
    result = array.copy()
    result[~foreground] = fill_value
    return result


def build_candidate_mask(
    reference_images: Sequence[np.ndarray],
    *,
    working_size: int = 512,
    erosion_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a review-required GrabCut candidate from same-view trusted-OK ROI images."""
    if not reference_images:
        raise ValueError("reference_images must not be empty")
    first = np.asarray(reference_images[0])
    if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("reference images must be uint8 BGR images")
    original_shape = first.shape
    if any(
        np.asarray(image).shape != original_shape or np.asarray(image).dtype != np.uint8
        for image in reference_images
    ):
        raise ValueError("reference images must have one common uint8 BGR shape")
    if working_size < 32:
        raise ValueError("working_size must be at least 32")
    if erosion_px < 0:
        raise ValueError("erosion_px must be non-negative")

    height, width = original_shape[:2]
    scale = working_size / max(height, width)
    work_width = max(32, int(round(width * scale)))
    work_height = max(32, int(round(height * scale)))
    resized = [cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_AREA) for image in reference_images]
    median_work = np.median(np.stack(resized), axis=0).astype(np.uint8)
    gray = cv2.cvtColor(median_work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    initialization = np.full(gray.shape, cv2.GC_BGD, dtype=np.uint8)
    initialization[blurred > 10] = cv2.GC_PR_FGD
    initialization[blurred > 60] = cv2.GC_FGD
    for border in (initialization[0, :], initialization[-1, :], initialization[:, 0], initialization[:, -1]):
        border[border != cv2.GC_FGD] = cv2.GC_BGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        median_work,
        initialization,
        None,
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_MASK,
    )
    candidate = np.where(
        np.logical_or(initialization == cv2.GC_FGD, initialization == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    close_size = max(3, int(round(working_size * 11 / 512)))
    if close_size % 2 == 0:
        close_size += 1
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count <= 1:
        raise ValueError("candidate foreground segmentation is empty")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    kept = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(kept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(kept, contours, -1, 255, cv2.FILLED)
    if erosion_px:
        diameter = erosion_px * 2 + 1
        kept = cv2.erode(kept, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    mask = cv2.resize(kept, (width, height), interpolation=cv2.INTER_NEAREST)
    validate_binary_mask(mask, (height, width))
    median = cv2.resize(median_work, (width, height), interpolation=cv2.INTER_LINEAR)
    return mask, median


def load_foreground_mask_asset(
    index_path: Path,
    *,
    expected_views: Sequence[str],
    expected_roi_sha256: str,
) -> ForegroundMaskAsset:
    """Load an exact-view mask bundle and fail closed on any path/SHA/shape drift."""
    path = Path(index_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"foreground mask index does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "bmw.efficientad_foreground_masks/1.0":
        raise ValueError("unsupported foreground mask schema")
    if payload.get("candidate_only") is not True:
        raise ValueError("foreground mask asset must be explicitly candidate_only")
    if payload.get("public_roi_config_sha256") != expected_roi_sha256:
        raise ValueError("foreground mask public ROI SHA does not match")
    fill_value = payload.get("fixed_fill_value")
    if isinstance(fill_value, bool) or not isinstance(fill_value, int) or not 0 <= fill_value <= 255:
        raise ValueError("foreground mask fixed_fill_value must be an integer in [0, 255]")
    view_payload = payload.get("views")
    if not isinstance(view_payload, dict) or tuple(view_payload) != tuple(expected_views):
        raise ValueError("foreground mask views must match the exact expected order")
    masks: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for view in expected_views:
        record = view_payload[view]
        mask_path = (path.parent / record["mask_path"]).resolve()
        digest = _sha256(mask_path)
        if digest != record.get("mask_sha256"):
            raise ValueError(f"foreground mask SHA mismatch: {view}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"foreground mask is unreadable: {view}")
        shape = (record.get("roi_height"), record.get("roi_width"))
        validate_binary_mask(mask, shape)
        mask.setflags(write=False)
        masks[view] = mask
        digests[view] = digest
    return ForegroundMaskAsset(
        index_path=path,
        index_sha256=_sha256(path),
        public_roi_config_sha256=expected_roi_sha256,
        fixed_fill_value=fill_value,
        masks=MappingProxyType(masks),
        mask_sha256_by_view=MappingProxyType(digests),
    )


def _top_k_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(values.size * fraction)))
    partitioned = np.partition(values, values.size - count)
    return float(np.mean(partitioned[-count:], dtype=np.float64))


def _component_score(
    anomaly_map: np.ndarray,
    region: np.ndarray,
    *,
    threshold: float,
    minimum_area: int,
) -> float:
    active = np.logical_and(region, anomaly_map >= threshold).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(active, connectivity=8)
    scores = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            scores.append(float(np.mean(anomaly_map[labels == label], dtype=np.float64)))
    return max(scores, default=0.0)


def _optional_region_metrics(
    anomaly_map: np.ndarray,
    region: np.ndarray,
    *,
    quantile: float,
    top_k_fraction: float,
    component_threshold: float,
    minimum_component_area: int,
) -> tuple[float, float, float, float] | None:
    values = anomaly_map[region]
    if values.size == 0:
        return None
    return (
        float(np.max(values)),
        float(np.quantile(values, quantile)),
        _top_k_mean(values, top_k_fraction),
        _component_score(
            anomaly_map,
            region,
            threshold=component_threshold,
            minimum_area=minimum_component_area,
        ),
    )


def aggregate_anomaly_map(
    anomaly_map: np.ndarray,
    roi_mask: np.ndarray,
    *,
    quantile: float = 0.995,
    top_k_fraction: float = 0.001,
    component_threshold: float = 0.5,
    minimum_component_area: int = 8,
) -> MapAggregates:
    """Aggregate one post-processed map without conflating it with `pred_score`."""
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("anomaly_map must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("anomaly_map must contain only finite values")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    if not 0 < top_k_fraction <= 1:
        raise ValueError("top_k_fraction must be in (0, 1]")
    if not math.isfinite(component_threshold):
        raise ValueError("component_threshold must be finite")
    if isinstance(minimum_component_area, bool) or not isinstance(minimum_component_area, int):
        raise ValueError("minimum_component_area must be a positive integer")
    if minimum_component_area <= 0:
        raise ValueError("minimum_component_area must be a positive integer")

    original_mask = np.asarray(roi_mask)
    validate_binary_mask(original_mask, original_mask.shape)
    resized = cv2.resize(original_mask, (values.shape[1], values.shape[0]), interpolation=cv2.INTER_NEAREST)
    foreground = resized == 255
    background = ~foreground
    foreground_metrics = _optional_region_metrics(
        values,
        foreground,
        quantile=quantile,
        top_k_fraction=top_k_fraction,
        component_threshold=component_threshold,
        minimum_component_area=minimum_component_area,
    )
    if foreground_metrics is None:
        raise ValueError("resized mask foreground must not be empty")
    background_metrics = _optional_region_metrics(
        values,
        background,
        quantile=quantile,
        top_k_fraction=top_k_fraction,
        component_threshold=component_threshold,
        minimum_component_area=minimum_component_area,
    )
    foreground_max, foreground_quantile, foreground_top_k, foreground_component = foreground_metrics
    if background_metrics is None:
        background_max = background_quantile = background_top_k = background_component = None
    else:
        background_max, background_quantile, background_top_k, background_component = background_metrics

    hotspot_y, hotspot_x = np.unravel_index(int(np.argmax(values)), values.shape)
    foreground_activation = float(np.sum(np.maximum(values[foreground], 0), dtype=np.float64))
    background_activation = float(np.sum(np.maximum(values[background], 0), dtype=np.float64))
    activation_total = foreground_activation + background_activation
    return MapAggregates(
        global_max=float(values[hotspot_y, hotspot_x]),
        foreground_max=foreground_max,
        background_max=background_max,
        foreground_quantile=foreground_quantile,
        background_quantile=background_quantile,
        foreground_top_k_mean=foreground_top_k,
        background_top_k_mean=background_top_k,
        foreground_component_score=foreground_component,
        background_component_score=background_component,
        hotspot_x=int(hotspot_x),
        hotspot_y=int(hotspot_y),
        hotspot_region="foreground" if foreground[hotspot_y, hotspot_x] else "background",
        foreground_pixel_count=int(np.count_nonzero(foreground)),
        background_pixel_count=int(np.count_nonzero(background)),
        foreground_activation_sum=foreground_activation,
        background_activation_sum=background_activation,
        background_activation_fraction=(background_activation / activation_total if activation_total > 0 else 0.0),
        background_max_advantage=max(0.0, background_max - foreground_max) if background_max is not None else 0.0,
    )


def classify_diagnostic_hotspot(hotspot_region: str) -> tuple[str, str]:
    """Classify only the mask-excluded spatial case; all foreground semantics need a human."""
    if hotspot_region == "background":
        return "fixture_or_background", "candidate_mask_spatial_rule"
    if hotspot_region == "foreground":
        return "uncertain", "requires_human_business_label"
    raise ValueError("hotspot_region must be foreground or background")


__all__ = [
    "ForegroundMaskAsset",
    "MapAggregates",
    "aggregate_anomaly_map",
    "apply_fixed_fill",
    "build_candidate_mask",
    "classify_diagnostic_hotspot",
    "load_foreground_mask_asset",
    "validate_binary_mask",
]
