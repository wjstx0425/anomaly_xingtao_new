"""Focused checks for the directly editable rotated light-streak ROI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bmw_inspection.lab.bright_streak_rotated_roi import (
    RotatedBrightStreakRoi,
    load_rotated_bright_streak_roi,
    rectify_bright_streak_roi,
    write_rotated_bright_streak_roi,
)


def _asset() -> RotatedBrightStreakRoi:
    return RotatedBrightStreakRoi(
        points_xy=((40, 10), (70, 15), (65, 90), (35, 85)),
        source_width=120,
        source_height=100,
        output_width=81,
        output_height=613,
    )


def test_roi_json_is_directly_editable_without_sha_or_source_image(tmp_path: Path) -> None:
    path = write_rotated_bright_streak_roi(tmp_path / "roi.json", _asset())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "sha256" not in str(payload)
    assert load_rotated_bright_streak_roi(path) == _asset()

    payload["points_xy"][0] = [39, 10]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_rotated_bright_streak_roi(path).points_xy[0] == (39, 10)


def test_rectify_keeps_geometry_and_input_size_checks() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    assert rectify_bright_streak_roi(image, _asset()).shape == (613, 81, 3)

    with pytest.raises(ValueError, match="dimensions"):
        rectify_bright_streak_roi(np.zeros((99, 120, 3), dtype=np.uint8), _asset())
