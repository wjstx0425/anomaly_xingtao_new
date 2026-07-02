# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for C789 slot-aware geometry shape scoring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_geometry_module() -> ModuleType:
    """Load the geometry shape helper from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "geometry_shape.py"
    spec = importlib.util.spec_from_file_location("capture_data_geometry_shape", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load geometry shape script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rectangle_mask() -> np.ndarray:
    """Return a simple rectangular part mask."""
    mask = np.zeros((100, 130), dtype=bool)
    mask[20:80, 30:100] = True
    return mask


def _edge_band(mask: np.ndarray, pixels: int = 10) -> np.ndarray:
    """Return a boundary band around a binary mask."""
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    return dilated & ~eroded


def test_foreground_mask_excludes_background_and_preset_holes() -> None:
    """Foreground extraction should ignore black background and masked C789 holes."""
    geometry = load_geometry_module()
    image = np.zeros((700, 1510, 3), dtype=np.uint8)
    image[80:620, 120:1390] = 120
    cv2.ellipse(image, (482, 520), (55, 55), 0, 0, 360, (125, 125, 125), -1)

    mask = geometry.build_foreground_mask(
        image,
        slot_name="slot01",
        preset="c789_left_top_3x2",
        hole_dilation=8,
    )

    assert bool(mask[250, 500])
    assert not bool(mask[10, 10])
    assert not bool(mask[520, 482])


def test_foreground_mask_keeps_multiple_large_material_components() -> None:
    """Geometry foreground should keep separated top and bottom metal regions."""
    geometry = load_geometry_module()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[15:50, 20:140] = 95
    image[72:108, 24:136] = 90
    image[5:9, 5:9] = 110

    mask = geometry.build_foreground_mask(
        image,
        slot_name=None,
        preset="c789_left_top_3x2",
        border_margin=0,
    )

    assert bool(mask[30, 80])
    assert bool(mask[90, 80])
    assert not bool(mask[6, 6])


def test_foreground_mask_solidifies_internal_shadow_seams() -> None:
    """Geometry masks should represent the part silhouette, not internal dark seams."""
    geometry = load_geometry_module()
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    image[20:100, 25:155] = 100
    image[54:66, 25:155] = 0

    mask = geometry.build_foreground_mask(
        image,
        slot_name=None,
        preset="c789_left_top_3x2",
        border_margin=0,
    )

    assert bool(mask[30, 90])
    assert bool(mask[60, 90])
    assert not bool(mask[10, 10])


def test_edge_band_uses_outer_silhouette_not_internal_gaps() -> None:
    """The geometry edge band should not be dominated by internal dark seams."""
    geometry = load_geometry_module()
    body = np.zeros((120, 180), dtype=bool)
    body[20:100, 25:155] = True
    body[54:66, 25:155] = False

    edge_band = geometry.edge_band_for_mask(body, pixels=8)

    assert bool(edge_band[20, 90])
    assert not bool(edge_band[60, 90])


def test_less_score_uses_missing_edge_component_area() -> None:
    """Removing body pixels on the edge should produce a less geometry score."""
    geometry = load_geometry_module()
    expected = _rectangle_mask()
    template = geometry.GeometryTemplate(
        slot="slot01",
        expected_body=expected,
        allowed_body=expected,
        edge_band=_edge_band(expected),
        source_count=1,
    )
    observed = expected.copy()
    observed[20:38, 30:58] = False

    score = geometry.score_mask_against_template(
        observed,
        template,
        search_radius=0,
        min_component_area=4,
        tolerance_px=0,
    )

    assert score.geometry_type == "less"
    assert score.geometry_score == score.missing_area
    assert score.missing_area >= 300


def test_more_score_uses_extra_edge_component_area() -> None:
    """Adding body pixels outside the allowed edge should produce a more score."""
    geometry = load_geometry_module()
    expected = _rectangle_mask()
    template = geometry.GeometryTemplate(
        slot="slot01",
        expected_body=expected,
        allowed_body=expected,
        edge_band=_edge_band(expected),
        source_count=1,
    )
    observed = expected.copy()
    observed[25:48, 100:120] = True

    score = geometry.score_mask_against_template(
        observed,
        template,
        search_radius=0,
        min_component_area=4,
        tolerance_px=0,
    )

    assert score.geometry_type == "more"
    assert score.geometry_score == score.extra_area
    assert score.extra_area >= 200


def test_translation_alignment_keeps_shifted_normal_from_scoring() -> None:
    """A pure shifted normal mask should align before less/more scoring."""
    geometry = load_geometry_module()
    expected = _rectangle_mask()
    template = geometry.GeometryTemplate(
        slot="slot01",
        expected_body=expected,
        allowed_body=expected,
        edge_band=_edge_band(expected),
        source_count=1,
    )
    observed = geometry.shift_mask(expected, dx=7, dy=-5)

    score = geometry.score_mask_against_template(observed, template, search_radius=10, min_component_area=4)

    assert score.geometry_score == 0
    assert score.geometry_type == "none"
    assert score.dx == -7
    assert score.dy == 5


def test_calibrate_thresholds_uses_slotwise_locked_max_plus_margin() -> None:
    """Locked normal geometry thresholds should be calibrated per slot."""
    geometry = load_geometry_module()
    rows = [
        geometry.GeometryScore(slot="slot01", geometry_score=10.0, geometry_type="none"),
        geometry.GeometryScore(slot="slot01", geometry_score=12.0, geometry_type="none"),
        geometry.GeometryScore(slot="slot02", geometry_score=5.0, geometry_type="none"),
    ]

    thresholds = geometry.calibrate_thresholds(rows, margin_ratio=0.05)

    assert thresholds["slot01"] == 12.6
    assert thresholds["slot02"] == 5.25


def test_region_thresholds_do_not_let_one_edge_region_hide_another() -> None:
    """Region calibration should keep thresholds local to each boundary area."""
    geometry = load_geometry_module()
    locked_scores = [
        geometry.GeometryScore(
            slot="slot01",
            geometry_score=1000.0,
            geometry_type="more",
            region_scores={"more:r00_c00": 1000.0, "more:r00_c01": 20.0},
        ),
        geometry.GeometryScore(
            slot="slot01",
            geometry_score=30.0,
            geometry_type="more",
            region_scores={"more:r00_c01": 30.0},
        ),
    ]
    defect_score = geometry.GeometryScore(
        slot="slot01",
        geometry_score=80.0,
        geometry_type="more",
        region_scores={"more:r00_c01": 80.0},
    )

    thresholds = geometry.calibrate_region_thresholds(locked_scores, margin_ratio=0.05)
    geometry.apply_thresholds([defect_score], thresholds)

    assert thresholds[("slot01", "more", "r00_c00")] == 1050.0
    assert thresholds[("slot01", "more", "r00_c01")] == 31.5
    assert defect_score.geometry_pred_label == 1
    assert defect_score.geometry_region == "r00_c01"
