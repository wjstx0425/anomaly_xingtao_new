"""Focused tests for the simplified eight-view model runtime."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.views import VIEW_ORDER
from bmw_inspection.lab.efficientad_component_filter import ComponentFilterPolicy
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoFinalStatus,
)
from bmw_inspection.lab.eight_view_demo_models import (
    EightViewEfficientAdPredictor,
    EightViewModelSuite,
    EightViewTemplatePredictor,
    ModelOutput,
    RoundInspectionResult,
    _load_demo_ignore_masks,
    load_part_rois,
)


def _pass(reason: str) -> ModelOutput:
    return ModelOutput(BranchStatus.PASS, 0.0, 0.5, reason, None)


def test_suite_runs_all_25_checks_and_turns_predictor_exception_into_error() -> None:
    rois = {view: (1, 1, 9, 9) for view in VIEW_ORDER}
    images = {view: np.zeros((10, 10, 3), dtype=np.uint8) for view in VIEW_ORDER}
    calls: list[tuple[str, str]] = []
    crop_ids: dict[tuple[str, str], int] = {}
    bright_input: list[np.ndarray] = []

    def template(view: str, _image: np.ndarray) -> ModelOutput:
        calls.append(("template", view))
        crop_ids[("template", view)] = id(_image)
        return _pass("template")

    def yolo(view: str, _image: np.ndarray) -> ModelOutput:
        calls.append(("yolo", view))
        crop_ids[("yolo", view)] = id(_image)
        if view == "back_right":
            raise RuntimeError("synthetic inference failure")
        return _pass("yolo")

    def efficientad(view: str, _image: np.ndarray) -> ModelOutput:
        calls.append(("efficientad", view))
        crop_ids[("efficientad", view)] = id(_image)
        return _pass("efficientad")

    def bright(_image: np.ndarray) -> ModelOutput:
        bright_input.append(_image)
        return _pass("bright")

    suite = EightViewModelSuite(
        rois=rois,
        template_predictor=template,
        bright_streak_predictor=bright,
        yolo_predictor=yolo,
        efficientad_predictor=efficientad,
    )

    inspection = suite.inspect(images, capture_id="capture-001")

    assert len(inspection.results) == 25
    assert len(calls) == 24
    assert inspection.final_status is DemoFinalStatus.ERROR
    failed = [
        row
        for row in inspection.results
        if row.branch is DemoBranch.YOLO and row.view_id == "back_right"
    ]
    assert len(failed) == 1
    assert failed[0].status is BranchStatus.ERROR
    assert "synthetic inference failure" in failed[0].reason
    for view in VIEW_ORDER:
        assert crop_ids[("template", view)] == crop_ids[("yolo", view)]
        assert crop_ids[("template", view)] == crop_ids[("efficientad", view)]
    assert len(bright_input) == 1
    assert bright_input[0] is images["front_left"]


def test_inspect_round_runs_front_13_and_back_12_with_shared_crops() -> None:
    rois = {view: (1, 1, 9, 9) for view in VIEW_ORDER}
    images = {view: np.zeros((10, 10, 3), dtype=np.uint8) for view in VIEW_ORDER}
    crop_ids: dict[tuple[str, str], int] = {}
    bright_input: list[np.ndarray] = []

    def predictor(branch: str):
        def predict(view: str, image: np.ndarray) -> ModelOutput:
            crop_ids[(branch, view)] = id(image)
            return _pass(branch)

        return predict

    def bright(image: np.ndarray) -> ModelOutput:
        bright_input.append(image)
        return _pass("bright")

    suite = EightViewModelSuite(
        rois=rois,
        template_predictor=predictor("template"),
        bright_streak_predictor=bright,
        yolo_predictor=predictor("yolo"),
        efficientad_predictor=predictor("efficientad"),
    )

    front = suite.inspect_round(
        {view: images[view] for view in VIEW_ORDER[:4]},
        "front",
    )
    back = suite.inspect_round(
        {view: images[view] for view in VIEW_ORDER[4:]},
        "back",
    )

    assert isinstance(front, RoundInspectionResult)
    assert front.round_name == "front"
    assert back.round_name == "back"
    assert len(front.results) == 13
    assert len(back.results) == 12
    assert [(row.branch, row.view_id) for row in front.results] == [
        *((DemoBranch.TEMPLATE, view) for view in VIEW_ORDER[:4]),
        (DemoBranch.BRIGHT_STREAK, "front_left"),
        *((DemoBranch.YOLO, view) for view in VIEW_ORDER[:4]),
        *((DemoBranch.EFFICIENTAD, view) for view in VIEW_ORDER[:4]),
    ]
    assert [(row.branch, row.view_id) for row in back.results] == [
        *((DemoBranch.TEMPLATE, view) for view in VIEW_ORDER[4:]),
        *((DemoBranch.YOLO, view) for view in VIEW_ORDER[4:]),
        *((DemoBranch.EFFICIENTAD, view) for view in VIEW_ORDER[4:]),
    ]
    for view in VIEW_ORDER:
        assert crop_ids[("template", view)] == crop_ids[("yolo", view)]
        assert crop_ids[("template", view)] == crop_ids[("efficientad", view)]
    assert bright_input == [images["front_left"]]


def test_finalize_rounds_preserves_global_order_and_defers_trusted_ok(monkeypatch) -> None:
    rois = {view: (1, 1, 9, 9) for view in VIEW_ORDER}
    images = {view: np.zeros((10, 10, 3), dtype=np.uint8) for view in VIEW_ORDER}
    predictor_calls: list[tuple[str, str]] = []
    match_calls: list[tuple[str, str]] = []

    def template(view: str, _image: np.ndarray) -> ModelOutput:
        predictor_calls.append(("template", view))
        if view == "front_left":
            return ModelOutput(BranchStatus.NG, 0.8, 0.5, "template NG", None)
        return _pass("template")

    def yolo(view: str, _image: np.ndarray) -> ModelOutput:
        predictor_calls.append(("yolo", view))
        return _pass("yolo")

    def efficientad(view: str, _image: np.ndarray) -> ModelOutput:
        predictor_calls.append(("efficientad", view))
        return _pass("efficientad")

    def bright(_image: np.ndarray) -> ModelOutput:
        predictor_calls.append(("bright", "front_left"))
        return ModelOutput(BranchStatus.NG, 0.8, 0.5, "bright NG", None)

    class RaisingMatcher:
        def match(
            self,
            view: str,
            _full_image: np.ndarray,
            _roi_image: np.ndarray,
            *,
            comparison_mode: str,
        ) -> None:
            match_calls.append((view, comparison_mode))
            raise RuntimeError("synthetic matcher failure")

    suite = EightViewModelSuite(
        rois=rois,
        template_predictor=template,
        bright_streak_predictor=bright,
        yolo_predictor=yolo,
        efficientad_predictor=efficientad,
        trusted_ok_matcher=RaisingMatcher(),  # type: ignore[arg-type]
    )
    front = suite.inspect_round({view: images[view] for view in VIEW_ORDER[:4]}, "front")
    back = suite.inspect_round({view: images[view] for view in VIEW_ORDER[4:]}, "back")

    assert match_calls == []
    predictor_call_count = len(predictor_calls)
    ticks = iter((100.0, 100.005))
    monkeypatch.setattr(
        "bmw_inspection.lab.eight_view_demo_models.perf_counter",
        lambda: next(ticks),
    )
    inspection = suite.finalize_rounds(images, front, back, "capture-rounds")

    assert len(predictor_calls) == predictor_call_count
    assert [(row.branch, row.view_id) for row in inspection.results] == [
        *((DemoBranch.TEMPLATE, view) for view in VIEW_ORDER),
        (DemoBranch.BRIGHT_STREAK, "front_left"),
        *((DemoBranch.YOLO, view) for view in VIEW_ORDER),
        *((DemoBranch.EFFICIENTAD, view) for view in VIEW_ORDER),
    ]
    assert match_calls == [("front_left", "roi"), ("front_left", "full")]
    assert inspection.final_status is DemoFinalStatus.NG
    assert inspection.elapsed_ms == pytest.approx(front.elapsed_ms + back.elapsed_ms + 5.0)
    assert set(inspection.diagnostic_metadata["trusted_ok_match_errors"]) == {
        "front_left/roi",
        "front_left/full",
    }


def test_roi_loader_uses_direct_coordinates_and_keeps_bounds_check(tmp_path: Path) -> None:
    path = tmp_path / "roi.json"
    payload = {
        "image_width": 20,
        "image_height": 16,
        "part_rois": {view: [1, 2, 19, 15] for view in VIEW_ORDER},
        "reference_manifest_sha256": "stale metadata is ignored",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_part_rois(path)["front"] == (1, 2, 19, 15)

    payload["part_rois"]["back"] = [1, 2, 21, 15]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="超出"):
        load_part_rois(path)


def test_ignore_mask_loader_uses_paths_without_sha_binding(tmp_path: Path) -> None:
    views: dict[str, dict[str, str]] = {}
    for view in VIEW_ORDER:
        mask = np.zeros((8, 9), dtype=np.uint8)
        mask[1:3, 2:4] = 255
        path = tmp_path / f"{view}.png"
        assert cv2.imwrite(str(path), mask)
        views[view] = {"mask_path": path.name, "sha256": "deliberately stale"}
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"views": views}), encoding="utf-8")

    masks = _load_demo_ignore_masks(index, {view: (8, 9) for view in VIEW_ORDER})

    assert tuple(masks) == VIEW_ORDER
    assert int(np.count_nonzero(masks["front"])) == 4


def test_ignore_mask_loader_resizes_to_new_roi_with_nearest_neighbor(tmp_path: Path) -> None:
    views: dict[str, dict[str, str]] = {}
    source = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    for view in VIEW_ORDER:
        path = tmp_path / f"{view}.png"
        assert cv2.imwrite(str(path), source)
        views[view] = {"mask_path": path.name}
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"views": views}), encoding="utf-8")

    masks = _load_demo_ignore_masks(index, {view: (4, 6) for view in VIEW_ORDER})

    assert masks["front"].shape == (4, 6)
    assert set(np.unique(masks["front"]).tolist()) == {0, 255}
    assert np.array_equal(masks["front"], cv2.resize(source, (6, 4), interpolation=cv2.INTER_NEAREST))


def test_template_records_model_and_deployment_thresholds(tmp_path: Path) -> None:
    model_paths: dict[str, Path] = {}
    deployment = {view: 0.3 for view in VIEW_ORDER}
    pattern = np.arange(64, dtype=np.uint8).reshape(8, 8)
    for view in VIEW_ORDER:
        root = tmp_path / view
        root.mkdir()
        assert cv2.imwrite(str(root / "template.png"), pattern)
        model = {
            "view_id": view,
            "input_width": 8,
            "input_height": 8,
            "threshold": 0.12,
            "preprocess": {"target_width": 8, "target_height": 8, "max_shift": 0},
            "templates": [{"path": "template.png"}],
        }
        path = root / "model.json"
        path.write_text(json.dumps(model), encoding="utf-8")
        model_paths[view] = path
    predictor = EightViewTemplatePredictor(model_paths, thresholds=deployment)

    result = predictor.predict("front", pattern)

    assert result.details["model_threshold"] == pytest.approx(0.12)
    assert result.details["deployment_threshold"] == pytest.approx(0.3)
    assert result.details["risk"] == pytest.approx(result.score)
    assert result.details["threshold_exceedance"] == pytest.approx(result.score - 0.3)


def test_template_weighted_regions_keep_one_result_and_legacy_diagnostics(tmp_path: Path) -> None:
    model_paths: dict[str, Path] = {}
    deployment = {view: 0.3 for view in VIEW_ORDER}
    weighted_thresholds = {view: 0.2 for view in VIEW_ORDER}
    pattern = np.arange(1024, dtype=np.uint16).reshape(32, 32).astype(np.uint8)
    changed = pattern.copy()
    changed[1:6, 1:6] = 255 - changed[1:6, 1:6]
    for view in VIEW_ORDER:
        root = tmp_path / view
        root.mkdir()
        assert cv2.imwrite(str(root / "template.png"), pattern)
        model = {
            "view_id": view,
            "input_width": 32,
            "input_height": 32,
            "threshold": 0.12,
            "preprocess": {"target_width": 32, "target_height": 32, "max_shift": 0},
            "templates": [{"path": "template.png"}],
        }
        path = root / "model.json"
        path.write_text(json.dumps(model), encoding="utf-8")
        model_paths[view] = path
    from bmw_inspection.lab.template_region_weighting import TemplateWeightedRegion

    regions = {
        view: ((TemplateWeightedRegion("foot", (0, 0, 14, 14)),) if view == "front" else ())
        for view in VIEW_ORDER
    }
    masks = {view: np.zeros((32, 32), dtype=np.uint8) for view in VIEW_ORDER}
    masks["front"][31, 31] = 255
    predictor = EightViewTemplatePredictor(
        model_paths,
        thresholds=deployment,
        ignore_masks=masks,
        weighted_regions=regions,
        weighted_region_weight=3.0,
        weighted_outside_weight=0.5,
        weighted_thresholds=weighted_thresholds,
    )

    weighted = predictor.predict("front", changed)
    empty_view = predictor.predict("back", changed)

    assert weighted.details["score_source"] == "weighted_region_ccoeff_normed"
    assert weighted.details["region_weight"] == pytest.approx(3.0)
    assert weighted.details["outside_weight"] == pytest.approx(0.5)
    assert weighted.details["effective_region_ratio"] == pytest.approx(6.0)
    assert weighted.details["weighted_region_count"] == 1
    assert weighted.details["weighted_risk"] == pytest.approx(weighted.score)
    assert weighted.details["legacy_risk"] >= 0
    assert weighted.details["masked_similarity"] is not None
    assert weighted.details["ignored_input_pixel_count"] == 1
    assert weighted.threshold == pytest.approx(0.2)
    assert empty_view.details["score_source"] == "unmasked_ccoeff_normed"
    assert empty_view.threshold == pytest.approx(0.3)


def test_efficientad_keeps_masked_score_and_direct_threshold(tmp_path: Path) -> None:
    checkpoints: dict[str, Path] = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / f"{view}.ckpt"
        checkpoint.write_bytes(b"fixture")
        checkpoints[view] = checkpoint
    thresholds = {view: 0.6 for view in VIEW_ORDER}
    masks = {view: np.zeros((8, 8), dtype=np.uint8) for view in VIEW_ORDER}
    masks["front"][0:4, 0:4] = 255

    def factory(_path: Path):
        def predict(_image: np.ndarray):
            anomaly_map = np.zeros((8, 8), dtype=np.float32)
            anomaly_map[0:4, 0:4] = 0.95
            anomaly_map[6, 6] = 0.25
            return 0.95, True, anomaly_map

        return predict

    predictor = EightViewEfficientAdPredictor(
        checkpoints,
        thresholds=thresholds,
        predictor_factory=factory,
        ignore_masks=masks,
        threshold_source="legacy_reuse_for_0820_candidate",
        validation_status="pending_independent_validation",
    )

    result = predictor.predict("front", np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.status is BranchStatus.PASS
    assert result.score == pytest.approx(0.25)
    assert result.threshold == pytest.approx(0.6)
    assert result.details["score_source"] == "manual_ignore_masked_anomaly_map_max"
    assert result.details["threshold_source"] == "legacy_reuse_for_0820_candidate"
    assert result.details["validation_status"] == "pending_independent_validation"


def test_efficientad_ng_overlay_marks_accepted_components_red_and_labels_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints: dict[str, Path] = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / f"{view}.ckpt"
        checkpoint.write_bytes(b"fixture")
        checkpoints[view] = checkpoint
    thresholds = {view: 0.5 for view in VIEW_ORDER}
    policy = ComponentFilterPolicy(
        low_threshold=0.2,
        seed_threshold=0.35,
        p95_threshold=0.5,
        minimum_area=8,
        hard_peak_threshold=0.95,
        line_minimum_length=20,
        line_minimum_area=8,
    )

    def factory(_path: Path):
        def predict(_image: np.ndarray):
            anomaly_map = np.zeros((16, 16), dtype=np.float32)
            anomaly_map[2:6, 2:6] = 0.8  # accepted by area + P95
            anomaly_map[10:12, 10:12] = 0.4  # seeded but rejected by component gates
            return 0.8, True, anomaly_map

        return predict

    rendered_text: list[str] = []
    original_put_text = cv2.putText

    def record_put_text(image: np.ndarray, text: str, *args, **kwargs):
        rendered_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_put_text)
    predictor = EightViewEfficientAdPredictor(
        checkpoints,
        thresholds=thresholds,
        predictor_factory=factory,
        component_policies={view: policy for view in VIEW_ORDER},
    )

    result = predictor.predict("front", np.zeros((160, 160, 3), dtype=np.uint8))

    assert result.status is BranchStatus.NG
    assert result.score == pytest.approx(0.8)
    assert result.overlay is not None
    assert tuple(result.overlay[20, 40]) == (0, 0, 255)  # accepted component, red in BGR
    assert tuple(result.overlay[100, 100]) == (0, 165, 255)  # rejected component, orange in BGR
    assert any(
        all(token in text for token in ("P95=0.800", "threshold=0.500", "exceedance=+0.300"))
        for text in rendered_text
    )
    assert any(
        all(token in text for token in ("P95=0.800", "area=16", "reason=area_p95"))
        for text in rendered_text
    )
