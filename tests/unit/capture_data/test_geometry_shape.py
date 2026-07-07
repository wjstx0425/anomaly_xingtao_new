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


def load_evaluate_geometry_module() -> ModuleType:
    """Load the geometry evaluator script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "evaluate_geometry_shape.py"
    sys.path.insert(0, str(script_path.parent))
    spec = importlib.util.spec_from_file_location("capture_data_evaluate_geometry_shape", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load geometry evaluator script from {script_path}"
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


def test_region_thresholds_keep_slot_fallback_when_normals_have_no_region_scores() -> None:
    """Perfect locked-normal masks should still produce slot and global fallback thresholds."""
    geometry = load_geometry_module()
    locked_scores = [
        geometry.GeometryScore(slot="slot01", geometry_score=0.0, geometry_type="less"),
        geometry.GeometryScore(slot="slot02", geometry_score=0.0, geometry_type="more"),
    ]

    thresholds = geometry.calibrate_region_thresholds(locked_scores, margin_ratio=0.05)

    assert thresholds[("slot01", "*", "*")] == 0.0
    assert thresholds[("slot02", "*", "*")] == 0.0
    assert thresholds[("*", "*", "*")] == 0.0


def test_apply_thresholds_marks_empty_region_scores_as_ok_with_slot_fallback() -> None:
    """A sample without region deviations should still be evaluated as geometry OK."""
    geometry = load_geometry_module()
    score = geometry.GeometryScore(slot="slot01", geometry_score=0.0, geometry_type="less")

    geometry.apply_thresholds(
        [score],
        {
            ("slot01", "*", "*"): 0.0,
            ("*", "*", "*"): 0.0,
        },
    )

    assert score.geometry_pred_label == 0
    assert score.geometry_threshold == 0.0
    assert score.threshold_lookup_level == "slot"
    assert score.threshold_source == "slot01:*:*"


def test_apply_thresholds_uses_ordered_fallback_levels() -> None:
    """Threshold lookup should fall back through exact, slot/type, slot/region, slot, type, and global."""
    geometry = load_geometry_module()
    thresholds = {
        ("slot01", "less", "r00_c00"): 100.0,
        ("slot01", "less", "*"): 80.0,
        ("slot01", "*", "r00_c02"): 70.0,
        ("slot01", "*", "*"): 60.0,
        ("*", "less", "*"): 50.0,
        ("*", "*", "*"): 40.0,
    }
    scores = [
        geometry.GeometryScore(
            slot="slot01",
            geometry_score=90.0,
            geometry_type="less",
            region_scores={"less:r00_c00": 90.0},
        ),
        geometry.GeometryScore(
            slot="slot01",
            geometry_score=90.0,
            geometry_type="less",
            region_scores={"less:r00_c09": 90.0},
        ),
        geometry.GeometryScore(
            slot="slot01",
            geometry_score=90.0,
            geometry_type="more",
            region_scores={"more:r00_c02": 90.0},
        ),
        geometry.GeometryScore(
            slot="slot01",
            geometry_score=90.0,
            geometry_type="more",
            region_scores={"more:r00_c09": 90.0},
        ),
        geometry.GeometryScore(
            slot="slot02",
            geometry_score=90.0,
            geometry_type="less",
            region_scores={"less:r00_c09": 90.0},
        ),
        geometry.GeometryScore(
            slot="slot02",
            geometry_score=90.0,
            geometry_type="more",
            region_scores={"more:r00_c09": 90.0},
        ),
    ]

    geometry.apply_thresholds(scores, thresholds)

    assert [score.threshold_lookup_level for score in scores] == [
        "exact",
        "slot_type",
        "slot_region",
        "slot",
        "defect_type",
        "global",
    ]
    assert [score.geometry_threshold for score in scores] == [100.0, 80.0, 70.0, 60.0, 50.0, 40.0]
    assert scores[0].geometry_pred_label == 0
    assert all(score.geometry_pred_label == 1 for score in scores[1:])


def test_apply_thresholds_keeps_zero_exact_threshold() -> None:
    """A configured 0.0 exact threshold should not be skipped in favor of a fallback."""
    geometry = load_geometry_module()
    score = geometry.GeometryScore(
        slot="slot01",
        geometry_score=1.0,
        geometry_type="less",
        region_scores={"less:r00_c00": 1.0},
    )

    geometry.apply_thresholds(
        [score],
        {
            ("slot01", "less", "r00_c00"): 0.0,
            ("slot01", "less", "*"): 10.0,
        },
    )

    assert score.geometry_threshold == 0.0
    assert score.threshold_lookup_level == "exact"
    assert score.geometry_pred_label == 1


def test_load_thresholds_adds_fallback_rows_for_legacy_csv(tmp_path: Path) -> None:
    """Legacy exact threshold CSV rows should gain slot/type/global fallback entries at load time."""
    geometry = load_geometry_module()
    thresholds_csv = tmp_path / "geometry_thresholds.csv"
    thresholds_csv.write_text(
        "\n".join(
            [
                "slot,geometry_type,geometry_region,geometry_threshold",
                "slot01,less,r00_c00,10.0",
                "slot01,less,r00_c01,12.0",
                "slot02,more,r00_c00,20.0",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    thresholds = geometry.load_thresholds(thresholds_csv)

    assert thresholds[("slot01", "less", "r00_c00")] == 10.0
    assert thresholds[("slot01", "less", "*")] == 12.0
    assert thresholds[("slot01", "*", "r00_c00")] == 10.0
    assert thresholds[("slot01", "*", "*")] == 12.0
    assert thresholds[("*", "less", "*")] == 12.0
    assert thresholds[("*", "*", "*")] == 20.0


def test_threshold_rows_include_new_schema_and_fallback_diagnostics() -> None:
    """Calibrated threshold CSV rows should expose both legacy and MVP-2 diagnostic columns."""
    geometry = load_geometry_module()
    evaluator = load_evaluate_geometry_module()
    locked_scores = [
        geometry.GeometryScore(
            slot="slot01",
            geometry_score=10.0,
            geometry_type="less",
            region_scores={"less:r00_c00": 10.0, "less:r00_c01": 20.0},
        ),
        geometry.GeometryScore(
            slot="slot01",
            geometry_score=30.0,
            geometry_type="more",
            region_scores={"more:r00_c00": 30.0},
        ),
    ]
    thresholds = geometry.calibrate_region_thresholds(locked_scores, margin_ratio=0.0)

    rows = evaluator._threshold_rows(locked_scores, thresholds, 0.0)
    row_by_key = {(row["slot_id"], row["defect_type"], row["region_id"]): row for row in rows}

    exact_row = row_by_key[("slot01", "less", "r00_c00")]
    fallback_row = row_by_key[("slot01", "less", "*")]
    assert exact_row["slot"] == "slot01"
    assert exact_row["geometry_type"] == "less"
    assert exact_row["geometry_threshold"] == exact_row["threshold"]
    assert exact_row["threshold_source"] == "calibrated_exact"
    assert fallback_row["threshold_source"] == "auto_fallback"
    assert fallback_row["n_normal"] == 2
    assert fallback_row["max_normal"] == 20.0
