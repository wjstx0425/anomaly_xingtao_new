"""Versioned asset and perspective rectification for a rotated bright-streak ROI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np


_SCHEMA = "bmw.bright_streak_rotated_roi/1.0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class RotatedBrightStreakRoi:
    """A clockwise source quadrilateral and its fixed 81 by 613 destination."""

    points_xy: tuple[tuple[int, int], ...]
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_image: str
    source_image_sha256: str

    def __post_init__(self) -> None:
        for name in ("source_width", "source_height", "output_width", "output_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (self.output_width, self.output_height) != (81, 613):
            raise ValueError("output dimensions must be 81x613")
        if not isinstance(self.source_image, str) or not self.source_image:
            raise ValueError("source_image must be a non-empty string")
        if not isinstance(self.source_image_sha256, str) or not _SHA256_RE.fullmatch(self.source_image_sha256):
            raise ValueError("source_image_sha256 must be a lowercase 64-character SHA-256")
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

        cross_products = []
        for index in range(4):
            first = points[index]
            second = points[(index + 1) % 4]
            third = points[(index + 2) % 4]
            cross_products.append(
                (second[0] - first[0]) * (third[1] - second[1])
                - (second[1] - first[1]) * (third[0] - second[0])
            )
        if any(cross == 0 for cross in cross_products) or not (all(cross > 0 for cross in cross_products) or all(cross < 0 for cross in cross_products)):
            raise ValueError("points_xy must form a convex quadrilateral (凸四边形)")

        signed_area_twice = sum(
            points[index][0] * points[(index + 1) % 4][1]
            - points[(index + 1) % 4][0] * points[index][1]
            for index in range(4)
        )
        if signed_area_twice <= 0:
            raise ValueError("points_xy must be clockwise")


_ASSET_FIELDS = frozenset({"schema", *RotatedBrightStreakRoi.__dataclass_fields__})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_payload(asset: RotatedBrightStreakRoi) -> dict[str, object]:
    return {"schema": _SCHEMA, **asdict(asset)}


def rectify_bright_streak_roi(image: np.ndarray, asset: RotatedBrightStreakRoi) -> np.ndarray:
    """Rectify ``asset``'s source quadrilateral into its 81 by 613 pixel image."""
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


def load_rotated_bright_streak_roi(
    path: Path,
    expected_sha256: str | None = None,
) -> RotatedBrightStreakRoi:
    """Load a verified ROI asset and verify its referenced source image."""
    asset_path = Path(path)
    if not asset_path.is_file() or asset_path.is_symlink():
        raise ValueError("ROI asset must be a regular file")
    actual_sha256 = _sha256(asset_path)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256")
        if actual_sha256 != expected_sha256:
            raise ValueError("ROI asset SHA-256 does not match expected_sha256")
    try:
        payload = json.loads(asset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ROI asset must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload) != _ASSET_FIELDS:
        raise ValueError("ROI asset fields must exactly match the schema")
    if payload["schema"] != _SCHEMA:
        raise ValueError(f"ROI asset schema must be {_SCHEMA}")
    try:
        raw_points = payload["points_xy"]
        if not isinstance(raw_points, list):
            raise ValueError("points_xy must be an array")
        asset = RotatedBrightStreakRoi(
            points_xy=tuple(tuple(point) for point in raw_points),
            source_width=payload["source_width"],
            source_height=payload["source_height"],
            output_width=payload["output_width"],
            output_height=payload["output_height"],
            source_image=payload["source_image"],
            source_image_sha256=payload["source_image_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("ROI asset fields are invalid") from error

    source_path = Path(asset.source_image)
    if not source_path.is_absolute():
        source_path = asset_path.parent / source_path
    if not source_path.is_file() or source_path.is_symlink() or _sha256(source_path) != asset.source_image_sha256:
        raise ValueError("ROI source image SHA-256 does not match")
    source_image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if source_image is None or source_image.shape[:2] != (asset.source_height, asset.source_width):
        raise ValueError("ROI source image dimensions do not match")
    return asset


def write_rotated_bright_streak_roi(path: Path, asset: RotatedBrightStreakRoi) -> Path:
    """Atomically publish one ROI JSON object without replacing an existing asset."""
    if not isinstance(asset, RotatedBrightStreakRoi):
        raise TypeError("asset must be RotatedBrightStreakRoi")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_asset_payload(asset), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(f"ROI asset already exists: {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination
