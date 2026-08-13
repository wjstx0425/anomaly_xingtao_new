"""Tests for the versioned, rotated bright-streak ROI asset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.bright_streak_rotated_roi import (
    RotatedBrightStreakRoi,
    load_rotated_bright_streak_roi,
    rectify_bright_streak_roi,
    write_rotated_bright_streak_roi,
)


def test_rectify_manual_quadrilateral_to_v3_shape() -> None:
    image = np.zeros((100, 120), dtype=np.uint8)
    image[10:90, 40:70] = 180
    asset = RotatedBrightStreakRoi(
        points_xy=((40, 10), (70, 15), (65, 90), (35, 85)),
        source_width=120,
        source_height=100,
        output_width=81,
        output_height=613,
        source_image="sample.png",
        source_image_sha256="a" * 64,
    )

    rectified = rectify_bright_streak_roi(image, asset)

    assert rectified.shape == (613, 81)
    assert rectified.dtype == np.uint8


def test_rejects_non_convex_or_out_of_bounds_points() -> None:
    with pytest.raises(ValueError, match="凸四边形"):
        RotatedBrightStreakRoi(
            points_xy=((10, 10), (50, 50), (50, 10), (10, 50)),
            source_width=100,
            source_height=100,
            output_width=81,
            output_height=613,
            source_image="sample.png",
            source_image_sha256="a" * 64,
        )


def test_writer_publishes_exact_asset_and_loader_verifies_source(tmp_path: Path) -> None:
    source = tmp_path / "sample.png"
    image = np.zeros((100, 120), dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    asset = RotatedBrightStreakRoi(
        points_xy=((40, 10), (70, 15), (65, 90), (35, 85)),
        source_width=120,
        source_height=100,
        output_width=81,
        output_height=613,
        source_image=source.name,
        source_image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    destination = tmp_path / "roi.json"

    assert write_rotated_bright_streak_roi(destination, asset) == destination
    expected_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert load_rotated_bright_streak_roi(destination, expected_sha256=expected_sha256) == asset
    with pytest.raises(FileExistsError):
        write_rotated_bright_streak_roi(destination, asset)


def test_loader_rejects_non_exact_schema_fields(tmp_path: Path) -> None:
    source = tmp_path / "sample.png"
    assert cv2.imwrite(str(source), np.zeros((100, 120), dtype=np.uint8))
    payload = {
        "schema": "bmw.bright_streak_rotated_roi/1.0",
        "points_xy": [[40, 10], [70, 15], [65, 90], [35, 85]],
        "source_width": 120,
        "source_height": 100,
        "output_width": 81,
        "output_height": 613,
        "source_image": source.name,
        "source_image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "unexpected": True,
    }
    path = tmp_path / "roi.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        load_rotated_bright_streak_roi(path)
