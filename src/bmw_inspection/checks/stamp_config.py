# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit stamp geometry, confidence settings, and CPU execution options."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _rectangle(value: object, size: tuple[int, int], name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(type(v) is not int for v in value):
        raise ValueError(f"{name} must contain four integer coordinates")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
        raise ValueError(f"{name} is outside {size} or empty: {value}")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class StampReaderConfig:
    """Explicit camera orientation and manually configured stamp/code regions."""

    hand: str
    view_id: str
    reference_image_size: tuple[int, int]
    roi_xyxy: tuple[int, int, int, int]
    rotation_clockwise_degrees: int
    code_region_xyxy: tuple[int, int, int, int]
    min_score: float = 0.95
    border_margin_px: int = 3
    min_code_width_fraction: float = 0.70
    normalization_policy: str = "none"
    line_quad_xy: tuple[tuple[float, float], ...] | None = None
    line_image_size: tuple[int, int] = (450, 104)
    consensus_min_score: float = 0.90
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 2

    def __post_init__(self) -> None:
        """Reject malformed coordinates instead of rescaling or inferring a hand."""
        if self.hand not in {"left", "right"} or self.view_id != "back":
            raise ValueError("Explicit left/right hand and back view_id are required")
        for name in ("intra_op_num_threads", "inter_op_num_threads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        size = self.reference_image_size
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError("reference_image_size must be two positive integers [width, height]")
        if type(self.rotation_clockwise_degrees) is not int or self.rotation_clockwise_degrees != 90:
            raise ValueError("rotation_clockwise_degrees must be 90")
        roi = _rectangle(self.roi_xyxy, tuple(size), "roi_xyxy")
        upright_size = (roi[3] - roi[1], roi[2] - roi[0])
        region = _rectangle(self.code_region_xyxy, upright_size, "code_region_xyxy")
        if (
            type(self.min_score) not in (float, int)
            or not math.isfinite(self.min_score)
            or not 0 <= self.min_score <= 1
        ):
            raise ValueError("min_score must be finite and between zero and one")
        if (
            type(self.min_code_width_fraction) not in (float, int)
            or not math.isfinite(self.min_code_width_fraction)
            or not 0 <= self.min_code_width_fraction <= 1
        ):
            raise ValueError("min_code_width_fraction must be finite and between zero and one")
        if type(self.border_margin_px) is not int or self.border_margin_px < 0:
            raise ValueError("border_margin_px must be a nonnegative integer")
        if self.normalization_policy not in ("none", "ascii_alphanumeric"):
            raise ValueError("normalization_policy must be none or ascii_alphanumeric")
        if (
            type(self.consensus_min_score) not in (int, float)
            or not math.isfinite(self.consensus_min_score) or not 0 <= self.consensus_min_score <= 1
        ):
            raise ValueError("consensus_min_score must be finite and between zero and one")
        if (
            not isinstance(self.line_image_size, (list, tuple)) or len(self.line_image_size) != 2
            or any(type(value) is not int or value < 2 for value in self.line_image_size)
        ):
            raise ValueError("line_image_size must contain two integer dimensions >= 2")
        object.__setattr__(self, "line_image_size", tuple(self.line_image_size))
        if self.line_quad_xy is not None:
            quad = np.asarray(self.line_quad_xy, dtype=np.float32)
            if quad.shape != (4, 2) or not np.isfinite(quad).all():
                raise ValueError("line_quad_xy must contain four finite XY points")
            if (
                np.any(quad < 0) or np.any(quad[:, 0] >= upright_size[0])
                or np.any(quad[:, 1] >= upright_size[1]) or not cv2.isContourConvex(quad)
                or cv2.contourArea(quad, oriented=True) <= 1
                or not (quad[0, 0] < quad[1, 0] and quad[3, 0] < quad[2, 0]
                        and quad[0, 1] < quad[3, 1] and quad[1, 1] < quad[2, 1])
            ):
                raise ValueError("line_quad_xy must be convex TL, TR, BR, BL points inside upright ROI")
            object.__setattr__(self, "line_quad_xy", tuple(tuple(float(value) for value in point) for point in quad))
        object.__setattr__(self, "reference_image_size", tuple(size))
        object.__setattr__(self, "roi_xyxy", roi)
        object.__setattr__(self, "code_region_xyxy", region)

    @classmethod
    def from_json(cls, path: str | Path) -> StampReaderConfig:
        """Load a config with no implicit side selection or expected text."""
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
