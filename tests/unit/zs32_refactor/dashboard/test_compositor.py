# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pixel and geometry tests for the ZS32 evidence compositor."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from zs32_inspection.dashboard.compositor import (
    compose_view,
    draw_yolo_detections,
    fit_letterbox,
    overlay_red_mask,
    place_crop_mask,
)
from zs32_inspection.dashboard.contracts import BranchEvidence, BranchState, EvidenceLayer, ViewResult
from zs32_inspection.dashboard.parser import load_inspection_result


@pytest.fixture
def source_view(eight_view_result_dir: Path) -> ViewResult:
    return load_inspection_result(eight_view_result_dir).views[0]


@pytest.fixture
def secondary_view(eight_view_result_dir: Path) -> ViewResult:
    return load_inspection_result(eight_view_result_dir).views[3]


def test_place_crop_mask_uses_half_open_roi_and_nearest() -> None:
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    full = place_crop_mask(mask, (2, 1, 6, 5), source_shape=(6, 8))

    assert full.dtype == np.bool_
    assert full.shape == (6, 8)
    assert full[1, 2]
    assert not full[1, 5]
    assert not full[0].any()
    assert not full[:, :2].any()


@pytest.mark.parametrize("roi", [(-1, 0, 2, 2), (0, 0, 9, 2), (2, 2, 2, 3), (3, 2, 2, 3)])
def test_place_crop_mask_rejects_invalid_roi(roi: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="ROI"):
        place_crop_mask(np.ones((2, 2), dtype=np.uint8), roi, source_shape=(6, 8))


def test_overlay_red_mask_uses_045_alpha_without_changing_input() -> None:
    image = np.zeros((1, 2, 3), dtype=np.uint8)
    original = image.copy()

    output = overlay_red_mask(image, np.array([[True, False]]), alpha=0.45)

    assert output[0, 0].tolist() == [0, 0, 115]
    assert output[0, 1].tolist() == [0, 0, 0]
    assert np.array_equal(image, original)


def test_fit_letterbox_returns_exact_scale_and_offsets_without_stretching() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    fitted = fit_letterbox(image, width=100, height=80)

    assert fitted.image.shape == (50, 100, 3)
    assert fitted.scale == pytest.approx(5.0)
    assert fitted.offset == (0, 15)


@pytest.mark.parametrize(
    ("image", "expected_pixel"),
    [
        (np.full((2, 3), 7, dtype=np.uint8), [7, 7, 7]),
        (np.full((2, 3, 4), [1, 2, 3, 255], dtype=np.uint8), [1, 2, 3]),
    ],
)
def test_fit_letterbox_normalizes_gray_and_alpha(image: np.ndarray, expected_pixel: list[int]) -> None:
    fitted = fit_letterbox(image, width=6, height=4)

    assert fitted.image.shape == (4, 6, 3)
    assert fitted.image[0, 0].tolist() == expected_pixel


def test_empty_yolo_detections_do_not_change_pixels(source_view: ViewResult) -> None:
    original = cv2.imread(str(source_view.source_path))

    rendered = draw_yolo_detections(original, (), (5, 5, 25, 15))

    assert np.array_equal(rendered, original)


def test_yolo_bbox_maps_crop_coordinates_and_only_changes_box_region() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    detections = ({"xyxy": [1, 2, 5, 6], "class_name": "scratch", "confidence": 0.8},)

    rendered = draw_yolo_detections(image, detections, (3, 1, 11, 9))

    changed = np.any(rendered != image, axis=2)
    assert changed[3:7, 4:8].any()
    assert not changed[:3].any()
    assert not changed[:, :4].any()
    assert not changed[7:].any()
    assert not changed[:, 8:].any()
    assert not np.any(image)


def test_yolo_bbox_displays_class_and_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    labels: list[str] = []
    original_put_text = cv2.putText

    def record_label(*args: object, **kwargs: object) -> np.ndarray:
        labels.append(str(args[1]))
        return original_put_text(*args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_label)

    draw_yolo_detections(
        image,
        ({"xyxy": [1, 2, 15, 12], "class_name": "scratch", "confidence": 0.8},),
        (3, 1, 23, 16),
    )

    assert labels == ["scratch 0.80"]


@pytest.mark.parametrize(
    "detection",
    [
        {},
        {"xyxy": [1, 2, 1, 5]},
        {"xyxy": [float("nan"), 2, 4, 5]},
        {"xyxy": [True, 2, 4, 5]},
        {"xyxy": "1,2,4,5"},
    ],
)
def test_invalid_yolo_detection_fails_closed(detection: dict[str, object]) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)

    rendered = draw_yolo_detections(image, (detection,), (3, 1, 11, 9))

    assert np.array_equal(rendered, image)


def test_secondary_non_original_layer_is_unsupported(secondary_view: ViewResult) -> None:
    composed = compose_view(secondary_view, EvidenceLayer.FUSION)

    assert composed.notice == "暂未接入模型"
    assert np.array_equal(composed.image, cv2.imread(str(secondary_view.source_path)))


def test_secondary_original_layer_shows_real_source(secondary_view: ViewResult) -> None:
    composed = compose_view(secondary_view, EvidenceLayer.ORIGINAL)

    assert composed.notice == ""
    assert np.array_equal(composed.image, cv2.imread(str(secondary_view.source_path)))


def test_patchcore_and_fusion_use_available_mask(source_view: ViewResult) -> None:
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[0, 0] = 255
    patchcore = source_view.branches["patchcore"]
    assert patchcore.mask_path is not None
    assert cv2.imwrite(str(patchcore.mask_path), mask)
    original = cv2.imread(str(source_view.source_path))

    patchcore_view = compose_view(source_view, EvidenceLayer.PATCHCORE)
    fusion_view = compose_view(source_view, EvidenceLayer.FUSION)

    assert patchcore_view.image[5, 5, 2] > original[5, 5, 2]
    assert fusion_view.image[5, 5, 2] > original[5, 5, 2]


def test_template_without_mask_keeps_original_pixels_and_reports_status(source_view: ViewResult) -> None:
    composed = compose_view(source_view, EvidenceLayer.TEMPLATE)

    assert composed.notice == "REVIEW"
    assert np.array_equal(composed.image, cv2.imread(str(source_view.source_path)))


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (BranchState.SKIPPED, "未执行"),
        (BranchState.UNSUPPORTED, "不支持"),
        (BranchState.ERROR, "证据错误"),
    ],
)
def test_unavailable_branch_keeps_original_and_has_explicit_notice(
    source_view: ViewResult,
    state: BranchState,
    expected: str,
) -> None:
    branch = BranchEvidence("patchcore", state, state.value.upper(), None, "backend reason")
    view = replace(source_view, branches={**source_view.branches, "patchcore": branch})

    composed = compose_view(view, EvidenceLayer.PATCHCORE)

    assert expected in composed.notice
    assert "backend reason" in composed.notice
    assert np.array_equal(composed.image, cv2.imread(str(source_view.source_path)))


def test_compose_view_rejects_undecodable_source_without_mutating_file(
    source_view: ViewResult,
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    view = replace(source_view, source_path=invalid)
    before = invalid.read_bytes()

    with pytest.raises(ValueError, match="decode"):
        compose_view(view, EvidenceLayer.ORIGINAL)

    assert invalid.read_bytes() == before
