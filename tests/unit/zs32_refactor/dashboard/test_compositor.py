# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pixel and geometry tests for the ZS32 evidence compositor."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from zs32_inspection.dashboard import compositor
from zs32_inspection.dashboard.compositor import (
    compose_view,
    draw_yolo_detections,
    fit_letterbox,
    load_source_image,
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


def test_huge_yolo_coordinates_fail_closed_without_changing_input() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    original = image.copy()
    detection = {"xyxy": [10**10000, 1, 10**10001, 5]}

    rendered = draw_yolo_detections(image, (detection,), (3, 1, 11, 9))

    assert np.array_equal(rendered, original)
    assert np.array_equal(image, original)


def _with_yolo_detections(
    source_view: ViewResult,
    detections: tuple[dict[str, object], ...],
) -> ViewResult:
    yolo = replace(source_view.branches["yolo"], roi_xyxy=(5, 5, 25, 15), detections=detections)
    return replace(source_view, branches={**source_view.branches, "yolo": yolo})


@pytest.mark.parametrize("layer", [EvidenceLayer.YOLO, EvidenceLayer.FUSION])
@pytest.mark.parametrize(
    "detection",
    [
        {"xyxy": [10**10000, 1, 10**10001, 5]},
        {"xyxy": [1, 2, 3]},
    ],
    ids=["huge-integer", "malformed"],
)
def test_compose_available_yolo_reports_all_invalid_detections(
    source_view: ViewResult,
    layer: EvidenceLayer,
    detection: dict[str, object],
) -> None:
    view = _with_yolo_detections(source_view, (detection,))
    original = cv2.imread(str(source_view.source_path))

    composed = compose_view(view, layer)

    assert "检测框数据无效" in composed.notice
    assert np.array_equal(composed.image, original)


@pytest.mark.parametrize("layer", [EvidenceLayer.YOLO, EvidenceLayer.FUSION])
def test_compose_available_yolo_draws_valid_and_reports_partial_invalid(
    source_view: ViewResult,
    layer: EvidenceLayer,
) -> None:
    valid = {"xyxy": [1, 2, 8, 9], "class_name": "scratch", "confidence": 0.25}
    invalid = {"xyxy": [10**10000, 1, 10**10001, 5]}
    view = _with_yolo_detections(source_view, (valid, invalid))
    original = cv2.imread(str(source_view.source_path))

    composed = compose_view(view, layer)

    assert "部分检测框无效" in composed.notice
    if layer is EvidenceLayer.YOLO:
        assert not np.array_equal(composed.image, original)
    else:
        assert np.array_equal(composed.image, original)
    assert np.array_equal(cv2.imread(str(source_view.source_path)), original)


@pytest.mark.parametrize("layer", [EvidenceLayer.YOLO, EvidenceLayer.FUSION])
def test_compose_empty_available_yolo_is_not_an_evidence_error(
    source_view: ViewResult,
    layer: EvidenceLayer,
) -> None:
    view = _with_yolo_detections(source_view, ())
    original = cv2.imread(str(source_view.source_path))

    composed = compose_view(view, layer)

    assert "检测框" not in composed.notice
    assert np.array_equal(composed.image, original)


@pytest.mark.parametrize("layer", [EvidenceLayer.YOLO, EvidenceLayer.FUSION])
@pytest.mark.parametrize(
    "confidence",
    [10**10000, float("nan"), float("inf"), "not-a-number"],
    ids=["huge-integer", "nan", "infinity", "non-numeric"],
)
def test_compose_available_yolo_draws_box_and_reports_invalid_label_metadata(
    source_view: ViewResult,
    layer: EvidenceLayer,
    confidence: object,
) -> None:
    detection = {"xyxy": [1, 2, 8, 9], "class_name": "scratch", "confidence": confidence}
    view = _with_yolo_detections(source_view, (detection,))
    original = cv2.imread(str(source_view.source_path))

    composed = compose_view(view, layer)

    assert "检测标签数据无效" in composed.notice
    if layer is EvidenceLayer.YOLO:
        assert not np.array_equal(composed.image, original)
    else:
        assert np.array_equal(composed.image, original)
    assert np.array_equal(cv2.imread(str(source_view.source_path)), original)


@pytest.mark.parametrize("layer", [EvidenceLayer.YOLO, EvidenceLayer.FUSION])
def test_compose_available_yolo_reports_partial_invalid_label_metadata(
    source_view: ViewResult,
    layer: EvidenceLayer,
) -> None:
    valid = {"xyxy": [1, 2, 8, 9], "class_name": "scratch", "confidence": 0.25}
    invalid_label = {"xyxy": [10, 1, 15, 6], "class_name": "dent", "confidence": 10**10000}
    view = _with_yolo_detections(source_view, (valid, invalid_label))
    original = cv2.imread(str(source_view.source_path))

    composed = compose_view(view, layer)

    assert "部分检测标签无效" in composed.notice
    if layer is EvidenceLayer.YOLO:
        assert not np.array_equal(composed.image, original)
    else:
        assert np.array_equal(composed.image, original)


def test_secondary_notice_is_determined_by_branch_state(secondary_view: ViewResult) -> None:
    unsupported = replace(
        secondary_view.branches["fusion"],
        state=BranchState.UNSUPPORTED,
        status="UNSUPPORTED",
        reason="model assets are not commissioned",
    )
    view = replace(secondary_view, branches={**secondary_view.branches, "fusion": unsupported})

    composed = compose_view(view, EvidenceLayer.FUSION)

    assert "Fusion: 不支持" in composed.notice
    assert np.array_equal(composed.image, cv2.imread(str(secondary_view.source_path)))


def test_secondary_original_layer_shows_real_source(secondary_view: ViewResult) -> None:
    composed = compose_view(secondary_view, EvidenceLayer.ORIGINAL)

    assert composed.notice == ""
    assert np.array_equal(composed.image, cv2.imread(str(secondary_view.source_path)))


def test_parser_and_compositor_share_detached_source_decodes(
    eight_view_result_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_reads: list[Path] = []
    original_imread = cv2.imread

    def record_source_reads(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
        candidate = Path(path).resolve()
        if candidate.parent.name == "sources":
            source_reads.append(candidate)
        return original_imread(path, flags)

    monkeypatch.setattr(cv2, "imread", record_source_reads)
    compositor._decode_source_image.cache_clear()

    first = load_inspection_result(eight_view_result_dir)
    source = first.views[1].source_path
    detached = load_source_image(source)
    detached[:, :] = 0
    composed = compose_view(first.views[1], EvidenceLayer.ORIGINAL)
    second = load_inspection_result(eight_view_result_dir)
    repeated = compose_view(second.views[1], EvidenceLayer.ORIGINAL)

    assert len(source_reads) == 8
    assert compositor._decode_source_image.cache_info().maxsize == 8
    assert composed.image[0, 0].tolist() == [20, 20, 20]
    assert repeated.image[0, 0].tolist() == [20, 20, 20]


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


def test_fusion_hides_yolo_when_branch_score_is_below_threshold(source_view: ViewResult) -> None:
    patchcore = replace(
        source_view.branches["patchcore"],
        state=BranchState.SKIPPED,
        status="SKIPPED",
        reason="not required for this compositor case",
    )
    view = replace(source_view, branches={**source_view.branches, "patchcore": patchcore})
    original = cv2.imread(str(source_view.source_path))

    yolo_view = compose_view(view, EvidenceLayer.YOLO)
    fusion_view = compose_view(view, EvidenceLayer.FUSION)

    assert not np.array_equal(yolo_view.image, original)
    assert np.array_equal(fusion_view.image, original)


def test_fusion_hides_individual_yolo_candidates_below_threshold(source_view: ViewResult) -> None:
    patchcore = replace(
        source_view.branches["patchcore"],
        state=BranchState.SKIPPED,
        status="SKIPPED",
        reason="not required for this compositor case",
    )
    yolo = replace(
        source_view.branches["yolo"],
        score=0.8,
        threshold=0.5,
        detections=(
            {"xyxy": [1, 1, 5, 5], "class_name": "scratch", "confidence": 0.2},
            {"xyxy": [10, 1, 14, 5], "class_name": "scratch", "confidence": 0.5},
        ),
    )
    view = replace(
        source_view,
        branches={**source_view.branches, "patchcore": patchcore, "yolo": yolo},
    )
    original = cv2.imread(str(source_view.source_path))

    fusion_view = compose_view(view, EvidenceLayer.FUSION)

    assert np.array_equal(fusion_view.image[6:10, 6:10], original[6:10, 6:10])
    assert not np.array_equal(fusion_view.image[6:10, 15:19], original[6:10, 15:19])


def test_template_without_mask_keeps_original_pixels_and_reports_status(source_view: ViewResult) -> None:
    composed = compose_view(source_view, EvidenceLayer.TEMPLATE)

    assert composed.notice == "REVIEW"
    assert np.array_equal(composed.image, cv2.imread(str(source_view.source_path)))


def test_template_short_circuit_fusion_leads_with_per_view_result(source_view: ViewResult) -> None:
    patchcore = replace(
        source_view.branches["patchcore"],
        state=BranchState.SKIPPED,
        status="SKIPPED",
        reason="global Template gate stopped PatchCore",
    )
    yolo = replace(
        source_view.branches["yolo"],
        state=BranchState.SKIPPED,
        status="SKIPPED",
        reason="global Template gate stopped YOLO",
    )
    fusion = replace(
        source_view.branches["fusion"],
        state=BranchState.AVAILABLE,
        status="NG_TEMPLATE",
        reason="this view's Template result determined Fusion",
    )
    view = replace(
        source_view,
        branches={
            **source_view.branches,
            "patchcore": patchcore,
            "yolo": yolo,
            "fusion": fusion,
        },
    )

    composed = compose_view(view, EvidenceLayer.FUSION)

    assert composed.notice.startswith("Fusion: NG_TEMPLATE")
    assert "PatchCore: 未执行" in composed.notice
    assert "YOLO: 未执行" in composed.notice


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
