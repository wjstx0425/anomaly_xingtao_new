"""Editable perspective ROI used by the BMW laboratory light-streak detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class RotatedBrightStreakRoi:
    """A clockwise source quadrilateral and its fixed 81 by 613 destination."""

    points_xy: tuple[tuple[int, int], ...]
    source_width: int
    source_height: int
    output_width: int
    output_height: int

    def __post_init__(self) -> None:
        for name in ("source_width", "source_height", "output_width", "output_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (self.output_width, self.output_height) != (81, 613):
            raise ValueError("output dimensions must be 81x613")
        if len(self.points_xy) != 4:
            raise ValueError("points_xy must contain four points")
        points: list[tuple[int, int]] = []
        for point in self.points_xy:
            if not isinstance(point, tuple) or len(point) != 2:
                raise ValueError("points_xy must contain integer pairs")
            x, y = point
            if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, int) or not isinstance(y, int):
                raise ValueError("points_xy must contain integer pairs")
            if not 0 <= x < self.source_width or not 0 <= y < self.source_height:
                raise ValueError("points_xy must be in source image bounds")
            points.append((x, y))
        crosses = []
        for index in range(4):
            first, second, third = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
            crosses.append(
                (second[0] - first[0]) * (third[1] - second[1])
                - (second[1] - first[1]) * (third[0] - second[0])
            )
        if any(value == 0 for value in crosses) or not (
            all(value > 0 for value in crosses) or all(value < 0 for value in crosses)
        ):
            raise ValueError("points_xy must form a convex quadrilateral")
        signed_area_twice = sum(
            points[index][0] * points[(index + 1) % 4][1]
            - points[(index + 1) % 4][0] * points[index][1]
            for index in range(4)
        )
        if signed_area_twice <= 0:
            raise ValueError("points_xy must be clockwise")
        if (
            (points[0][1] + points[1][1]) / 2 >= (points[2][1] + points[3][1]) / 2
            or points[0][0] >= points[1][0]
            or points[3][0] >= points[2][0]
            or max(points[0][1], points[1][1]) >= min(points[2][1], points[3][1])
        ):
            raise ValueError(
                "points_xy point order must be left-top, right-top, right-bottom, left-bottom"
            )


def rectify_bright_streak_roi(image: np.ndarray, asset: RotatedBrightStreakRoi) -> np.ndarray:
    """Rectify the configured quadrilateral into an 81 by 613 pixel image."""
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise TypeError("image must be a numpy array with at least two dimensions")
    if image.shape[:2] != (asset.source_height, asset.source_width):
        raise ValueError("image dimensions do not match the ROI asset")
    destination = np.array([[0, 0], [80, 0], [80, 612], [0, 612]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.asarray(asset.points_xy, dtype=np.float32), destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (asset.output_width, asset.output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def load_rotated_bright_streak_roi(path: Path) -> RotatedBrightStreakRoi:
    """Load only the geometry used during inference."""
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取光痕旋转ROI：{resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("光痕旋转ROI必须是JSON对象")
    try:
        points = payload["points_xy"]
        if not isinstance(points, list):
            raise ValueError("points_xy必须是数组")
        return RotatedBrightStreakRoi(
            points_xy=tuple(tuple(point) for point in points),
            source_width=payload["source_width"],
            source_height=payload["source_height"],
            output_width=payload["output_width"],
            output_height=payload["output_height"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("光痕旋转ROI字段无效") from error


def write_rotated_bright_streak_roi(path: Path, asset: RotatedBrightStreakRoi) -> Path:
    """Write one directly editable ROI JSON file."""
    if not isinstance(asset, RotatedBrightStreakRoi):
        raise TypeError("asset must be RotatedBrightStreakRoi")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(asset), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
