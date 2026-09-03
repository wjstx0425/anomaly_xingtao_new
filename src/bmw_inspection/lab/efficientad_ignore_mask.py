"""Apply configured BMW ignore masks to EfficientAD anomaly maps."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class MaskedAnomalyMap:
    score: float
    raw_max: float
    hotspot_x: int
    hotspot_y: int
    ignored_map_pixel_count: int
    masked_map: np.ndarray


def mask_anomaly_map(anomaly_map: np.ndarray, ignore_mask: np.ndarray) -> MaskedAnomalyMap:
    """Exclude masked pixels without any publication or digest contract."""
    map_value = np.asarray(anomaly_map)
    mask_value = np.asarray(ignore_mask)
    if map_value.ndim != 2 or map_value.size == 0 or not np.isfinite(map_value).all():
        raise ValueError("anomaly_map must be a finite non-empty 2D array")
    if mask_value.dtype != np.uint8 or mask_value.ndim != 2 or mask_value.size == 0:
        raise ValueError("ignore_mask must be a non-empty uint8 2D array")
    if not set(np.unique(mask_value).tolist()).issubset({0, 255}):
        raise ValueError("ignore_mask must contain only 0 and 255")
    resized = cv2.resize(mask_value, (map_value.shape[1], map_value.shape[0]), interpolation=cv2.INTER_NEAREST)
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


__all__ = ["MaskedAnomalyMap", "mask_anomaly_map"]
