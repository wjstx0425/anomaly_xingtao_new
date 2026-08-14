"""Runtime behavior for the BMW eight-view model suite."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bmw_inspection.lab import eight_view_demo_models as demo_models
from bmw_inspection.lab.bright_streak_rotated_roi import (
    RotatedBrightStreakRoi,
    write_rotated_bright_streak_roi,
)
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import BranchStatus, DemoFinalStatus
from bmw_inspection.lab.efficientad_analysis import fixed_scale_heatmap
from bmw_inspection.lab.efficientad_component_filter import ComponentFilterPolicy
from bmw_inspection.lab.eight_view_demo_models import (
    EightViewBrightStreakPredictor,
    EightViewEfficientAdPredictor,
    EightViewModelSuite,
    EightViewTemplatePredictor,
    EightViewYoloPredictor,
    EightViewRawProfileBrightStreakPredictor,
    EightViewRotatedTrackedProfileCandidatePredictor,
    EightViewTrackedProfileBrightStreakPredictor,
    ModelOutput,
    build_model_suite,
    load_part_rois,
)
from bmw_inspection.lab.trusted_ok_reference import TrustedOkMatch


def _trusted_match(
    view: str,
    current_full: np.ndarray,
    current_roi: np.ndarray,
    comparison_mode: str = "roi",
) -> TrustedOkMatch:
    reference_full = np.full_like(current_full, 10)
    reference_roi = np.full_like(current_roi, 10)
    return TrustedOkMatch(
        view_id=view,
        comparison_mode=comparison_mode,
        physical_part_id=f"trusted-{view}",
        sample_id=f"sample-{view}",
        similarity=0.9,
        shift_x=0,
        shift_y=0,
        current_full_image=current_full,
        reference_full_image=reference_full,
        current_roi=current_roi,
        reference_roi=reference_roi,
        aligned_reference_roi=reference_roi,
        difference_overlay=np.zeros((*current_roi.shape[:2], 3), dtype=np.uint8),
        source_sha256="a" * 64,
        reference_full_sha256="b" * 64,
        reference_roi_sha256="c" * 64,
        index_sha256="d" * 64,
        whitelist_sha256="e" * 64,
    )


def test_model_suite_runs_all_25_checks_without_short_circuiting() -> None:
    calls: Counter[str] = Counter()

    def output(branch: str, view: str, _image: np.ndarray) -> ModelOutput:
        calls[branch] += 1
        status = BranchStatus.NG if branch == "template" and view == "front" else BranchStatus.PASS
        return ModelOutput(
            status,
            0.1,
            0.2,
            f"{branch}/{view}",
            np.zeros((10, 10, 3), dtype=np.uint8),
            raw_pred_label=True if branch == "efficientad" else None,
        )

    suite = EightViewModelSuite(
        rois={view: (0, 0, 20, 20) for view in VIEW_ORDER},
        template_predictor=lambda view, image: output("template", view, image),
        bright_streak_predictor=lambda image: output("bright_streak", "front_left", image),
        yolo_predictor=lambda view, image: output("yolo", view, image),
        efficientad_predictor=lambda view, image: output("efficientad", view, image),
    )
    images = {view: np.zeros((20, 20, 3), dtype=np.uint8) for view in VIEW_ORDER}

    inspection = suite.inspect(images, capture_id="sample")

    assert calls == {"template": 8, "bright_streak": 1, "yolo": 8, "efficientad": 8}
    assert len(inspection.results) == 25
    assert inspection.final_status is DemoFinalStatus.NG
    efficientad_rows = [row for row in inspection.results if row.branch.value == "efficientad"]
    assert all(row.raw_pred_label is True for row in efficientad_rows)


def test_model_suite_converts_one_predictor_exception_to_error_and_continues() -> None:
    calls = Counter()

    def efficientad(view: str, image: np.ndarray) -> ModelOutput:
        del image
        calls[view] += 1
        if view == "front_right":
            raise RuntimeError("broken model")
        return ModelOutput(BranchStatus.PASS, 0.1, 0.5, "pass", None)

    passing = lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.5, "pass", None)
    suite = EightViewModelSuite(
        rois={view: (0, 0, 20, 20) for view in VIEW_ORDER},
        template_predictor=passing,
        bright_streak_predictor=lambda _image: passing(),
        yolo_predictor=passing,
        efficientad_predictor=efficientad,
    )

    inspection = suite.inspect(
        {view: np.zeros((20, 20, 3), dtype=np.uint8) for view in VIEW_ORDER},
        capture_id="sample",
    )

    assert calls.total() == 8
    assert inspection.final_status is DemoFinalStatus.ERROR
    failed = [row for row in inspection.results if row.status is BranchStatus.ERROR]
    assert len(failed) == 1
    assert failed[0].view_id == "front_right"


def test_model_suite_matches_only_unique_actionable_views_after_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(demo_models, "perf_counter", lambda: 1.0)
    calls: list[tuple[str, tuple[int, int], tuple[int, int], str]] = []

    class Matcher:
        def match(
            self,
            view: str,
            current_full: np.ndarray,
            current_roi: np.ndarray,
            *,
            comparison_mode: str,
        ) -> TrustedOkMatch:
            calls.append((view, current_full.shape[:2], current_roi.shape[:2], comparison_mode))
            return _trusted_match(view, current_full, current_roi, comparison_mode)

    def output(branch: str, view: str, _image: np.ndarray) -> ModelOutput:
        status = BranchStatus.PASS
        if view == "front" and branch in {"template", "yolo"}:
            status = BranchStatus.NG
        if branch == "bright_streak":
            status = BranchStatus.NG
        return ModelOutput(status, 0.1, 0.2, f"{branch}/{view}", None)

    kwargs = {
        "rois": {view: (0, 0, 10, 10) for view in VIEW_ORDER},
        "template_predictor": lambda view, image: output("template", view, image),
        "bright_streak_predictor": lambda image: output("bright_streak", "front_left", image),
        "yolo_predictor": lambda view, image: output("yolo", view, image),
        "efficientad_predictor": lambda view, image: output("efficientad", view, image),
    }
    images = {view: np.zeros((20, 20, 3), dtype=np.uint8) for view in VIEW_ORDER}
    baseline = EightViewModelSuite(**kwargs).inspect(images, capture_id="baseline")
    inspection = EightViewModelSuite(**kwargs, trusted_ok_matcher=Matcher()).inspect(images, capture_id="matched")

    assert [row[0] for row in calls] == ["front", "front_left"]
    assert calls[0][1:] == ((20, 20), (10, 10), "roi")
    assert calls[1][1:] == ((20, 20), (10, 10), "full")
    assert tuple(inspection.trusted_ok_by_comparison) == (("front", "roi"), ("front_left", "full"))
    assert inspection.results == baseline.results
    assert inspection.final_status is baseline.final_status is DemoFinalStatus.NG


def test_model_suite_pass_only_inspection_does_not_call_matcher() -> None:
    class Matcher:
        def match(self, *_args: object) -> TrustedOkMatch:
            raise AssertionError("PASS-only inspection must not match trusted references")

    passing = lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.5, "pass", None)
    suite = EightViewModelSuite(
        rois={view: (0, 0, 20, 20) for view in VIEW_ORDER},
        template_predictor=passing,
        bright_streak_predictor=lambda _image: passing(),
        yolo_predictor=passing,
        efficientad_predictor=passing,
        trusted_ok_matcher=Matcher(),
    )

    inspection = suite.inspect(
        {view: np.zeros((20, 20, 3), dtype=np.uint8) for view in VIEW_ORDER},
        capture_id="pass",
    )

    assert inspection.final_status is DemoFinalStatus.OK
    assert dict(inspection.trusted_ok_by_comparison) == {}


def test_matcher_failure_is_diagnostic_only_and_keeps_model_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(demo_models, "perf_counter", lambda: 1.0)

    class BrokenMatcher:
        def match(self, *_args: object, **_kwargs: object) -> TrustedOkMatch:
            raise RuntimeError("reference bank unavailable")

    def template(view: str, _image: np.ndarray) -> ModelOutput:
        return ModelOutput(
            BranchStatus.NG if view == "front" else BranchStatus.PASS,
            0.1,
            0.2,
            view,
            None,
        )

    kwargs = {
        "rois": {view: (0, 0, 20, 20) for view in VIEW_ORDER},
        "template_predictor": template,
        "bright_streak_predictor": lambda _image: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None),
        "yolo_predictor": lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None),
        "efficientad_predictor": lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None),
    }
    images = {view: np.zeros((20, 20, 3), dtype=np.uint8) for view in VIEW_ORDER}
    baseline = EightViewModelSuite(**kwargs).inspect(images, capture_id="baseline")
    inspection = EightViewModelSuite(**kwargs, trusted_ok_matcher=BrokenMatcher()).inspect(
        images, capture_id="broken"
    )

    assert dict(inspection.trusted_ok_by_comparison) == {}
    assert inspection.diagnostic_metadata["trusted_ok_match_errors"] == {
        "front/roi": "reference bank unavailable"
    }
    assert inspection.results == baseline.results
    assert inspection.final_status is baseline.final_status is DemoFinalStatus.NG


def test_build_model_suite_keeps_trusted_ok_disabled_without_explicit_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = SimpleNamespace(
        predict=lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None)
    )
    monkeypatch.setattr(demo_models, "EightViewTemplatePredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "EightViewBrightStreakPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "EightViewYoloPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "EightViewEfficientAdPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(
        demo_models,
        "load_part_rois",
        lambda _path: {view: (0, 0, 10, 10) for view in VIEW_ORDER},
    )
    config = SimpleNamespace(
        template_models={},
        bright_streak_engine="calibrated_rule_v1",
        bright_streak_config=Path("bright.json"),
        yolo_checkpoint=Path("best.pt"),
        yolo_candidate_conf=0.1,
        yolo_final_threshold=0.2,
        yolo_imgsz=640,
        efficientad_checkpoints={},
        efficientad_thresholds={},
        efficientad_base_thresholds={},
        efficientad_threshold_margin=0.0,
        roi_config=Path("roi.json"),
        trusted_ok_reference_index=None,
        trusted_ok_reference_index_sha256=None,
    )

    suite = build_model_suite(config)
    inspection = suite.inspect(
        {view: np.zeros((10, 10, 3), dtype=np.uint8) for view in VIEW_ORDER},
        capture_id="legacy-config",
    )

    assert suite._trusted_ok_matcher is None
    assert suite._trusted_ok_matcher_error is None
    assert len(inspection.results) == 25
    assert inspection.final_status is DemoFinalStatus.OK
    assert dict(inspection.trusted_ok_by_comparison) == {}
    assert dict(inspection.diagnostic_metadata) == {}


def test_build_model_suite_wires_manual_rotated_roi_into_tracked_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = SimpleNamespace(
        predict=lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None)
    )
    received: dict[str, object] = {}

    def tracked_factory(report: Path, **kwargs: object) -> SimpleNamespace:
        received["report"] = report
        received.update(kwargs)
        return predictor

    monkeypatch.setattr(demo_models, "EightViewTrackedProfileBrightStreakPredictor", tracked_factory)
    monkeypatch.setattr(demo_models, "EightViewTemplatePredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "EightViewYoloPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "EightViewEfficientAdPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(
        demo_models,
        "load_part_rois",
        lambda _path: {view: (0, 0, 10, 10) for view in VIEW_ORDER},
    )
    config = SimpleNamespace(
        template_models={},
        bright_streak_engine="tracked_profile_v3_manual_rotated_roi",
        bright_streak_config=Path("tracked-v3.json"),
        bright_streak_rotated_roi=Path("rotated-roi.json"),
        bright_streak_rotated_roi_sha256="a" * 64,
        bright_streak_weak_row_score_override=95.0,
        yolo_checkpoint=Path("best.pt"),
        yolo_candidate_conf=0.1,
        yolo_final_threshold=0.2,
        yolo_imgsz=640,
        efficientad_checkpoints={},
        efficientad_thresholds={},
        efficientad_base_thresholds={},
        efficientad_threshold_margin=0.0,
        roi_config=Path("roi.json"),
        trusted_ok_reference_index=None,
        trusted_ok_reference_index_sha256=None,
    )

    build_model_suite(config)

    assert received == {
        "report": Path("tracked-v3.json"),
        "rotated_roi_path": Path("rotated-roi.json"),
        "rotated_roi_sha256": "a" * 64,
        "weak_row_score_override": 95.0,
    }


def test_build_model_suite_injects_same_manual_masks_into_template_and_efficientad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roi = tmp_path / "roi.json"
    roi.write_text("{}", encoding="utf-8")
    masks = {view: np.zeros((10, 10), dtype=np.uint8) for view in VIEW_ORDER}
    masks["front"][1:3, 1:3] = 255
    asset = SimpleNamespace(masks=masks, index_sha256="a" * 64)
    predictor = SimpleNamespace(
        predict=lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None)
    )
    calls: dict[str, dict[str, object]] = {}

    def template_factory(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls["template"] = kwargs
        return predictor

    def efficientad_factory(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls["efficientad"] = kwargs
        return predictor

    monkeypatch.setattr(demo_models, "EightViewTemplatePredictor", template_factory)
    monkeypatch.setattr(demo_models, "EightViewEfficientAdPredictor", efficientad_factory)
    monkeypatch.setattr(demo_models, "EightViewBrightStreakPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "EightViewYoloPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(
        demo_models,
        "load_part_rois",
        lambda _path: {view: (0, 0, 10, 10) for view in VIEW_ORDER},
    )
    from bmw_inspection.lab import efficientad_ignore_mask

    monkeypatch.setattr(efficientad_ignore_mask, "load_ignore_mask_asset", lambda *_args, **_kwargs: asset)
    config = SimpleNamespace(
        template_models={},
        template_ignore_mask_index=Path("mask.json"),
        template_ignore_mask_index_sha256="a" * 64,
        template_masked_thresholds={view: 0.01 for view in VIEW_ORDER},
        template_masked_threshold_artifact_sha256="b" * 64,
        bright_streak_engine="calibrated_rule_v1",
        bright_streak_config=Path("bright.json"),
        yolo_checkpoint=Path("best.pt"),
        yolo_candidate_conf=0.1,
        yolo_final_threshold=0.2,
        yolo_imgsz=640,
        efficientad_checkpoints={},
        efficientad_thresholds={},
        efficientad_base_thresholds={},
        efficientad_threshold_margin=0.0,
        efficientad_ignore_mask_index=Path("mask.json"),
        efficientad_ignore_mask_index_sha256="a" * 64,
        roi_config=roi,
        trusted_ok_reference_index=None,
        trusted_ok_reference_index_sha256=None,
    )

    build_model_suite(config)

    assert calls["template"]["ignore_masks"] is masks
    assert calls["template"]["ignore_mask_index_sha256"] == "a" * 64
    assert calls["template"]["masked_thresholds"] == config.template_masked_thresholds
    assert calls["template"]["masked_threshold_artifact_sha256"] == "b" * 64
    assert calls["efficientad"]["ignore_masks"] is masks


def test_model_suite_matches_front_left_full_and_roi_independently_when_both_are_ng() -> None:
    calls: list[tuple[str, str]] = []

    class Matcher:
        def match(
            self,
            view: str,
            current_full: np.ndarray,
            current_roi: np.ndarray,
            *,
            comparison_mode: str,
        ) -> TrustedOkMatch:
            calls.append((view, comparison_mode))
            return _trusted_match(view, current_full, current_roi, comparison_mode)

    def template(view: str, _image: np.ndarray) -> ModelOutput:
        return ModelOutput(
            BranchStatus.NG if view == "front_left" else BranchStatus.PASS,
            0.1, 0.2, view, None,
        )

    suite = EightViewModelSuite(
        rois={view: (0, 0, 10, 10) for view in VIEW_ORDER},
        template_predictor=template,
        bright_streak_predictor=lambda _image: ModelOutput(BranchStatus.NG, 0.1, 0.2, "bright", None),
        yolo_predictor=lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None),
        efficientad_predictor=lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None),
        trusted_ok_matcher=Matcher(),
    )

    inspection = suite.inspect(
        {view: np.zeros((20, 20, 3), dtype=np.uint8) for view in VIEW_ORDER},
        capture_id="front-left-dual",
    )

    assert calls == [("front_left", "roi"), ("front_left", "full")]
    assert tuple(inspection.trusted_ok_by_comparison) == (
        ("front_left", "roi"),
        ("front_left", "full"),
    )


def test_build_model_suite_preloads_configured_matcher_before_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Matcher:
        def __init__(self, release_dir: Path, *, expected_index_sha256: str) -> None:
            events.append(f"init:{release_dir.name}:{expected_index_sha256}")
            self.roi_config_sha256 = hashlib.sha256(b"roi-current").hexdigest()

        def preload(self) -> None:
            events.append("preload")

    predictor = SimpleNamespace(
        predict=lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None)
    )
    monkeypatch.setattr(demo_models, "EightViewTemplatePredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "EightViewBrightStreakPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "EightViewYoloPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "EightViewEfficientAdPredictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "TrustedOkMatcher", Matcher)
    monkeypatch.setattr(
        demo_models,
        "load_part_rois",
        lambda _path: {view: (0, 0, 10, 10) for view in VIEW_ORDER},
    )
    index = Path("/trusted/bmw_right_20260810_21_train_normal_approved_v2/reference_index.json")
    roi_path = Path("/tmp/trusted-ok-task4-current-roi.json")
    roi_path.write_bytes(b"roi-current")
    config = SimpleNamespace(
        template_models={}, bright_streak_engine="calibrated_rule_v1", bright_streak_config=Path("bright.json"),
        yolo_checkpoint=Path("best.pt"), yolo_candidate_conf=0.1, yolo_final_threshold=0.2,
        yolo_imgsz=640, efficientad_checkpoints={}, efficientad_thresholds={},
        efficientad_base_thresholds={}, efficientad_threshold_margin=0.0, roi_config=roi_path,
        trusted_ok_reference_index=index, trusted_ok_reference_index_sha256="a" * 64,
        trusted_ok_reference_error=None,
    )

    suite = build_model_suite(config, status_callback=events.append)

    assert events == [
        "正在校验并预热可信OK参考库，首次启动约需27秒……",
        f"init:{index.parent.name}:{'a' * 64}",
        "preload",
        "可信OK参考库预热完成。",
    ]
    assert isinstance(suite._trusted_ok_matcher, Matcher)
    assert suite._trusted_ok_matcher_error is None


def test_build_model_suite_disables_trusted_matcher_when_current_roi_sha_drifted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_roi = tmp_path / "roi.json"
    current_roi.write_bytes(b"current-roi-drifted")
    validated_roi_sha = hashlib.sha256(b"published-roi").hexdigest()

    class Matcher:
        roi_config_sha256 = validated_roi_sha

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def preload(self) -> None:
            pass

    predictor = SimpleNamespace(
        predict=lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None)
    )
    for name in (
        "EightViewTemplatePredictor", "EightViewBrightStreakPredictor",
        "EightViewYoloPredictor", "EightViewEfficientAdPredictor",
    ):
        monkeypatch.setattr(demo_models, name, lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "TrustedOkMatcher", Matcher)
    monkeypatch.setattr(
        demo_models,
        "load_part_rois",
        lambda _path: {view: (0, 0, 10, 10) for view in VIEW_ORDER},
    )
    config = SimpleNamespace(
        template_models={}, bright_streak_engine="calibrated_rule_v1", bright_streak_config=Path("bright.json"),
        yolo_checkpoint=Path("best.pt"), yolo_candidate_conf=0.1, yolo_final_threshold=0.2,
        yolo_imgsz=640, efficientad_checkpoints={}, efficientad_thresholds={},
        efficientad_base_thresholds={}, efficientad_threshold_margin=0.0, roi_config=current_roi,
        trusted_ok_reference_index=Path(
            "/trusted/bmw_right_20260810_21_train_normal_approved_v2/reference_index.json"
        ),
        trusted_ok_reference_index_sha256="a" * 64,
        trusted_ok_reference_error=None,
    )
    messages: list[str] = []

    suite = build_model_suite(config, status_callback=messages.append)
    inspection = suite.inspect(
        {view: np.zeros((10, 10, 3), dtype=np.uint8) for view in VIEW_ORDER},
        capture_id="roi-drift",
    )

    assert suite._trusted_ok_matcher is None
    assert suite._trusted_ok_matcher_error == "可信OK参考不可用：可信参考ROI配置与当前Demo ROI配置SHA256不匹配"
    assert messages[-1] == "可信OK参考库不可用，已仅禁用参考诊断：可信参考ROI配置与当前Demo ROI配置SHA256不匹配"
    assert len(inspection.results) == 25
    assert inspection.final_status is DemoFinalStatus.OK
    assert all(row.status is BranchStatus.PASS for row in inspection.results)


def test_build_model_suite_disables_only_reference_diagnostics_when_preload_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenMatcher:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def preload(self) -> None:
            raise ValueError("参考图损坏")

    predictor = SimpleNamespace(
        predict=lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None)
    )
    for name in (
        "EightViewTemplatePredictor", "EightViewBrightStreakPredictor",
        "EightViewYoloPredictor", "EightViewEfficientAdPredictor",
    ):
        monkeypatch.setattr(demo_models, name, lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(demo_models, "TrustedOkMatcher", BrokenMatcher)
    monkeypatch.setattr(
        demo_models, "load_part_rois", lambda _path: {view: (0, 0, 10, 10) for view in VIEW_ORDER}
    )
    config = SimpleNamespace(
        template_models={}, bright_streak_engine="calibrated_rule_v1", bright_streak_config=Path("bright.json"),
        yolo_checkpoint=Path("best.pt"), yolo_candidate_conf=0.1, yolo_final_threshold=0.2,
        yolo_imgsz=640, efficientad_checkpoints={}, efficientad_thresholds={},
        efficientad_base_thresholds={}, efficientad_threshold_margin=0.0, roi_config=Path("roi.json"),
        trusted_ok_reference_index=Path(
            "/trusted/bmw_right_20260810_21_train_normal_approved_v2/reference_index.json"
        ),
        trusted_ok_reference_index_sha256="a" * 64,
        trusted_ok_reference_error=None,
    )
    messages: list[str] = []

    suite = build_model_suite(config, status_callback=messages.append)

    assert suite._trusted_ok_matcher is None
    assert suite._trusted_ok_matcher_error == "可信OK参考不可用：参考图损坏"
    assert messages[-1] == "可信OK参考库不可用，已仅禁用参考诊断：参考图损坏"


def test_build_model_suite_uses_prevalidated_reference_error_without_constructing_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = SimpleNamespace(
        predict=lambda *_args: ModelOutput(BranchStatus.PASS, 0.1, 0.2, "pass", None)
    )
    for name in (
        "EightViewTemplatePredictor", "EightViewBrightStreakPredictor",
        "EightViewYoloPredictor", "EightViewEfficientAdPredictor",
    ):
        monkeypatch.setattr(demo_models, name, lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(
        demo_models,
        "TrustedOkMatcher",
        lambda *_args, **_kwargs: pytest.fail("prevalidated bad reference must not construct matcher"),
    )
    monkeypatch.setattr(
        demo_models, "load_part_rois", lambda _path: {view: (0, 0, 10, 10) for view in VIEW_ORDER}
    )
    config = SimpleNamespace(
        template_models={}, bright_streak_engine="calibrated_rule_v1", bright_streak_config=Path("bright.json"),
        yolo_checkpoint=Path("best.pt"), yolo_candidate_conf=0.1, yolo_final_threshold=0.2,
        yolo_imgsz=640, efficientad_checkpoints={}, efficientad_thresholds={},
        efficientad_base_thresholds={}, efficientad_threshold_margin=0.0, roi_config=Path("roi.json"),
        trusted_ok_reference_index=Path(
            "/trusted/bmw_right_20260810_21_train_normal_approved_v2/reference_index.json"
        ),
        trusted_ok_reference_index_sha256="a" * 64,
        trusted_ok_reference_error="可信OK参考索引SHA256不匹配",
    )

    suite = build_model_suite(config)

    assert suite._trusted_ok_matcher is None
    assert suite._trusted_ok_matcher_error == "可信OK参考不可用：可信OK参考索引SHA256不匹配"


def test_generic_template_predictor_loads_secondary_view_model() -> None:
    root = Path(__file__).resolve().parents[4]
    model_root = root / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/template"
    crop_root = root / "dataset/bmw_lab_training/bmw_hdr_roi_training_reviewed_v1/crops/front_secondary"
    image_path = next(crop_root.glob("*.png"))
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    predictor = EightViewTemplatePredictor(
        {view: model_root / view / "model.json" for view in VIEW_ORDER}
    )

    output = predictor.predict("front_secondary", image)

    assert output.status in {BranchStatus.PASS, BranchStatus.NG}
    assert output.score is not None
    assert output.threshold is not None
    assert output.overlay is not None


def test_load_part_rois_preserves_canonical_order() -> None:
    root = Path(__file__).resolve().parents[4]

    rois = load_part_rois(root / "configs/bmw/rois/bmw_hdr_eight_view_v1.json")

    assert tuple(rois) == VIEW_ORDER


def test_yolo_predictor_uses_separate_candidate_and_ng_thresholds(tmp_path: Path) -> None:
    class Boxes:
        xyxy = np.array([[1, 1, 8, 8], [2, 2, 6, 6]], dtype=np.float32)
        conf = np.array([0.2, 0.8], dtype=np.float32)
        cls = np.array([0, 0], dtype=np.float32)

    class Result:
        boxes = Boxes()

    class Model:
        names = {0: "defect"}

        def predict(self, **_kwargs):
            return [Result()]

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    predictor = EightViewYoloPredictor(
        checkpoint,
        candidate_conf=0.1,
        final_threshold=0.25,
        imgsz=640,
        model_factory=lambda _path: Model(),
    )

    output = predictor.predict("front", np.zeros((10, 10, 3), dtype=np.uint8))

    assert output.status is BranchStatus.NG
    assert output.score == pytest.approx(0.8)
    assert output.threshold == pytest.approx(0.25)
    assert output.overlay is not None
    assert output.details["evidence_type"] == "真实检测框"
    assert output.details["final_box_count"] == 1
    assert output.details["candidate_box_count"] == 2
    assert output.details["boxes"][1]["confidence"] == pytest.approx(0.8)
    assert "最高置信度 0.8000" in output.reason
    assert "部署阈值 0.2500" in output.reason


def test_bright_streak_predictor_uses_trained_config() -> None:
    root = Path(__file__).resolve().parents[4]
    predictor = EightViewBrightStreakPredictor(
        root / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak/calibrated_config.json"
    )
    image_path = next(
        (root / "dataset/bmw_lab_raw/left/front_left/normal").rglob("*_fused.png")
    )
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    assert image is not None

    output = predictor.predict(image)

    assert output.status in {BranchStatus.PASS, BranchStatus.NG}
    assert output.score is not None
    assert output.overlay is not None


def test_raw_profile_bright_streak_predictor_uses_full_image_fixed_roi(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        '{"roi_xyxy":[10,20,91,633],"raw_profile_v2":{"thresholds":'
        '{"min_row_score":30.0,"min_presence_coverage_ratio":0.9,'
        '"min_longest_run_ratio":0.9,"max_gap_ratio":0.01,"max_gap_count":0}}}',
        encoding="utf-8",
    )
    image = np.full((700, 120), 20, dtype=np.uint8)
    image[20:633, 46:55] = 180
    predictor = EightViewRawProfileBrightStreakPredictor(report)

    output = predictor.predict(image)

    assert output.status is BranchStatus.PASS
    assert output.score == pytest.approx(1.0)
    assert output.threshold == pytest.approx(0.9)
    assert "原灰度" in output.reason
    assert output.overlay is not None
    assert output.overlay.shape == (613, 81, 3)
    assert output.details["evidence_type"] == "规则 ROI 证据"
    assert output.details["roi_xyxy"] == (10, 20, 91, 633)
    assert output.details["coverage_ratio"] == pytest.approx(1.0)
    assert output.details["gap_count"] == 0
    assert "覆盖率阈值 0.900" in output.reason
    assert "最长连续段阈值 0.900" in output.reason


def test_raw_profile_bright_streak_predictor_reports_missing_streak(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        '{"roi_xyxy":[0,0,81,613],"raw_profile_v2":{"thresholds":'
        '{"min_row_score":30.0,"min_presence_coverage_ratio":0.1,'
        '"min_longest_run_ratio":0.1,"max_gap_ratio":0.01,"max_gap_count":0}}}',
        encoding="utf-8",
    )
    predictor = EightViewRawProfileBrightStreakPredictor(report)

    output = predictor.predict(np.full((613, 81), 20, dtype=np.uint8))

    assert output.status is BranchStatus.NG
    assert "未检测到" in output.reason


def _tracked_report(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "bmw_right_batch_20260810_21_bright_v3_tracked_v8"
    profiles = artifact_root / "profiles"
    profiles.mkdir(parents=True)
    metrics = artifact_root / "metrics.csv"
    replay = artifact_root / "replay_summary.json"
    manifest = tmp_path / "bright_streak.csv"
    live_record = tmp_path / "bmw_demo_20260812_211302"
    (live_record / "images").mkdir(parents=True)
    live_files = {
        "front_left_hdr.png": live_record / "images/front_left_hdr.png",
        "front_left_long.png": live_record / "images/front_left_long.png",
        "front_left_short.png": live_record / "images/front_left_short.png",
        "inspection.json": live_record / "inspection.json",
    }
    for name, live_file in live_files.items():
        live_file.write_bytes(f"confirmed-live-normal:{name}".encode())
    metric_rows = []
    final_outcomes = []
    for index in range(61):
        if index < 4:
            split, truth, provenance, predicted = "calibration", "no_streak", "manifest", "NG_NO_STREAK"
        elif index < 21:
            split, truth, provenance, predicted = "calibration", "normal", "manifest", "OK"
        elif index == 21:
            split, truth, provenance, predicted = "calibration", "normal", "user_confirmed_live_normal", "OK"
        elif index < 26:
            split, truth, provenance, predicted = "final_test", "no_streak", "manifest", "NG_NO_STREAK"
        elif index < 41:
            split, truth, provenance, predicted = "final_test", "normal", "manifest", "OK"
        elif index == 41:
            split, truth, provenance, predicted = "final_test", "normal", "manifest", "NG_NO_STREAK"
        else:
            split, truth, provenance, predicted = "live_replay", "unknown", "unconfirmed_live_replay", "OK"
        sample_id = "bmw_demo_20260812_211302" if index == 21 else f"synthetic_{index:04d}"
        profile = profiles / f"profile_{index + 1:04d}.npz"
        np.savez_compressed(profile, path_x=np.zeros(613, dtype=np.int64))
        metric_rows.append(
            {
                "record_id": f"record-{index:04d}",
                "sample_id": sample_id,
                "split": split,
                "truth": truth,
                "provenance_kind": provenance,
                "profile_npz": str(profile),
                "tracked_profile_v3_predicted_status": predicted,
            }
        )
        if split == "final_test":
            expected = "NG_NO_STREAK" if truth == "no_streak" else "OK"
            final_outcomes.append(
                {
                    "correct": predicted == expected,
                    "expected_status": expected,
                    "predicted_status": predicted,
                    "sample_id": sample_id,
                }
            )
    with metrics.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    manifest_rows = [
        {
            "sample_id": row["sample_id"],
            "source_class": row["truth"],
            "split": row["split"],
            "expected_status": "NG_NO_STREAK" if row["truth"] == "no_streak" else "OK",
        }
        for row in metric_rows
        if row["provenance_kind"] == "manifest"
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    replay_payload = {
        "accuracy_denominator": 1,
        "count": 20,
        "known_truth_count": 1,
        "limit": 20,
        "outcomes": [
            {
                "capture_id": (
                    "bmw_demo_20260812_211302" if index == 0 else f"synthetic_{index + 41:04d}"
                ),
                "included_in_accuracy": index == 0,
                "truth": "normal" if index == 0 else "unknown",
                "v3_predicted_status": "OK",
            }
            for index in range(20)
        ],
        "unknown_truth_count": 19,
    }
    replay.write_text(json.dumps(replay_payload), encoding="utf-8")
    root = Path(__file__).resolve().parents[4]
    algorithm_source = root / "src/bmw_inspection/lab/bright_streak_tracked_profile.py"
    evaluator_source = root / "pipeline/bmw_lab_evaluate_bright_streak_tracked_profile.py"
    report = artifact_root / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "algorithm": "tracked_profile_v3",
                "fit_split": "calibration",
                "final_test_used_for_fit": False,
                "real_broken_samples": 0,
                "manifest": str(manifest),
                "roi_xyxy": [1792, 1180, 1873, 1793],
                "geometry": {
                    "candidate_width": 5,
                    "background_width": 10,
                    "background_gap": 3,
                    "smooth_window": 5,
                    "max_step": 2,
                    "step_penalty": 1.0,
                },
                "thresholds": {
                    "strong_row_score": 60.0,
                    "weak_row_score": 30.0,
                    "min_presence_coverage_ratio": 0.08,
                    "min_longest_run_ratio": 0.04,
                    "max_gap_ratio": 0.10,
                    "max_gap_count": 2,
                },
                "artifact_identities": {
                    "metrics_csv_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
                    "replay_summary_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
                    "profile_npz_sha256": {
                        profile.name: hashlib.sha256(profile.read_bytes()).hexdigest()
                        for profile in sorted(profiles.glob("*.npz"))
                    },
                },
                "identities": {
                    "algorithm_source_sha256": hashlib.sha256(algorithm_source.read_bytes()).hexdigest(),
                    "evaluator_source_sha256": hashlib.sha256(evaluator_source.read_bytes()).hexdigest(),
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "roi_config_sha256": hashlib.sha256(
                        json.dumps(
                            {"roi_xyxy": [1792, 1180, 1873, 1793]},
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                },
                "geometry_selection": {},
                "calibration_counts": {
                    "accepted_live_normal": 1,
                    "fit_total": 22,
                    "manifest_no_streak": 4,
                    "manifest_normal": 17,
                },
                "accepted_live_normals": [
                    {
                        "capture_id": "bmw_demo_20260812_211302",
                        "file_sha256": {
                            name: hashlib.sha256(live_file.read_bytes()).hexdigest()
                            for name, live_file in live_files.items()
                        },
                        "predicted_status": "OK",
                        "provenance_kind": "user_confirmed_live_normal",
                        "record_path": str(live_record),
                    }
                ],
                "acceptance_gate": {
                    "confirmed_live_normal_count": 1,
                    "no_streak_count": 8,
                    "normal_not_worse_than_v2_splits": ["calibration", "final_test"],
                    "passed": True,
                },
                "comparison_to_v2": {
                    "calibration": {"v2_normal_false_rejects": 0, "v3_normal_false_rejects": 0},
                    "final_test": {"v2_normal_false_rejects": 1, "v3_normal_false_rejects": 1},
                },
                "final_test": {
                    "count": 20,
                    "normal_count": 16,
                    "no_streak_count": 4,
                    "no_streak_false_accepts": 0,
                    "normal_false_rejects": 1,
                    "outcomes": final_outcomes,
                },
                "replay": replay_payload,
                "cpu_per_image_ms": {},
                "metrics_csv": str(metrics),
                "profiles_dir": str(profiles),
                "replay_summary_json": str(replay),
                "report_json": str(report),
            }
        ),
        encoding="utf-8",
    )
    return report


def _tracked_roi(kind: str) -> np.ndarray:
    image = np.full((613, 81), 20, dtype=np.uint8)
    if kind == "absent":
        return image
    for row in range(613):
        centre = 35 + row // 120
        value = 180
        if kind == "leading_background" and row < 540:
            continue
        if kind == "bridged" and 290 <= row < 295:
            value = 70
        if kind == "broken" and 280 <= row < 360:
            continue
        image[row, centre - 2 : centre + 3] = value
    return image


def _tracked_full_image(kind: str) -> np.ndarray:
    image = np.full((1793, 1873), 20, dtype=np.uint8)
    image[1180:1793, 1792:1873] = _tracked_roi(kind)
    return image


def _rotated_roi_asset(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "front_left_hdr.png"
    assert cv2.imwrite(str(source), _tracked_full_image("diagonal"))
    asset = RotatedBrightStreakRoi(
        points_xy=((1792, 1180), (1872, 1175), (1872, 1788), (1792, 1792)),
        source_width=1873,
        source_height=1793,
        output_width=81,
        output_height=613,
        source_image=source.name,
        source_image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    path = tmp_path / "bright_streak_rotated_roi.json"
    write_rotated_bright_streak_roi(path, asset)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _rotated_candidate_report(tmp_path: Path, roi_asset: Path, roi_sha256: str) -> Path:
    normal_manifest = tmp_path / "normal_manifest.csv"
    normal_manifest.write_text("sample_id\nnormal-1\n", encoding="utf-8")
    no_streak = tmp_path / "no_streak.png"
    no_streak.write_bytes(b"no-streak-source")
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("sample_key\nnormal-1\n", encoding="utf-8")
    report = tmp_path / "candidate_report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "bmw.bright_streak_rotated_v3_candidate/1.0",
                "status": "complete",
                "algorithm": "tracked_profile_v3_manual_rotated_roi",
                "candidate_only": True,
                "normal_count": 1,
                "no_streak_count": 1,
                "no_streak_independent_test_count": 0,
                "fit_data": "all_current_normals_plus_one_current_no_streak",
                "presence_fit_weak_row_score": 95.0,
                "weak_row_score_policy": "reuse_rotated_hdr_field_recalibration_v1",
                "geometry": {
                    "candidate_width": 5,
                    "background_width": 10,
                    "background_gap": 3,
                    "smooth_window": 5,
                    "max_step": 2,
                    "step_penalty": 1.0,
                },
                "thresholds": {
                    "strong_row_score": 120.0,
                    "weak_row_score": 95.0,
                    "min_presence_coverage_ratio": 0.1,
                    "min_longest_run_ratio": 0.1,
                    "max_gap_ratio": 0.1,
                    "max_gap_count": 2,
                },
                "normal_false_rejects": 0,
                "no_streak_false_accepts": 0,
                "normal_manifest": str(normal_manifest),
                "normal_manifest_sha256": hashlib.sha256(normal_manifest.read_bytes()).hexdigest(),
                "no_streak_image": str(no_streak),
                "no_streak_image_sha256": hashlib.sha256(no_streak.read_bytes()).hexdigest(),
                "rotated_roi": str(roi_asset),
                "rotated_roi_sha256": roi_sha256,
                "metrics_csv": str(metrics),
                "report_json": str(report),
            }
        ),
        encoding="utf-8",
    )
    return report


def test_rotated_candidate_predictor_uses_report_thresholds(tmp_path: Path) -> None:
    roi_asset, roi_sha256 = _rotated_roi_asset(tmp_path)
    predictor = EightViewRotatedTrackedProfileCandidatePredictor(
        _rotated_candidate_report(tmp_path, roi_asset, roi_sha256),
        roi_asset,
        roi_sha256,
        expected_report_sha256=hashlib.sha256(
            (tmp_path / "candidate_report.json").read_bytes()
        ).hexdigest(),
    )

    output = predictor.predict(_tracked_full_image("diagonal"))

    assert output.details["candidate_only"] is True
    assert output.details["weak_row_score"] == pytest.approx(95.0)
    assert output.details["rotated_roi_sha256"] == roi_sha256


def test_rotated_v3_reuses_report_thresholds(tmp_path: Path) -> None:
    roi_asset, roi_sha256 = _rotated_roi_asset(tmp_path)
    predictor = EightViewTrackedProfileBrightStreakPredictor(
        _tracked_report(tmp_path),
        rotated_roi_path=roi_asset,
        rotated_roi_sha256=roi_sha256,
    )

    result = predictor.predict(_tracked_full_image("diagonal"))

    assert result.details["roi_mode"] == "manual_rotated_perspective"
    assert result.details["threshold_calibration"] == "existing_v3_not_recalibrated"
    assert result.details["rotated_roi_sha256"] == roi_sha256
    assert result.overlay is not None
    assert result.overlay.shape[:2] == (613, 81)
    assert result.reason.startswith("手动倾斜ROI，沿用V3阈值（未重标定）")


def test_rotated_v3_weak_score_override_preserves_report_provenance(tmp_path: Path) -> None:
    roi_asset, roi_sha256 = _rotated_roi_asset(tmp_path)
    report = _tracked_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["thresholds"]["strong_row_score"] = 136.85
    payload["thresholds"]["weak_row_score"] = 128.2
    report.write_text(json.dumps(payload), encoding="utf-8")
    predictor = EightViewTrackedProfileBrightStreakPredictor(
        report,
        rotated_roi_path=roi_asset,
        rotated_roi_sha256=roi_sha256,
        weak_row_score_override=95.0,
    )

    result = predictor.predict(_tracked_full_image("diagonal"))

    assert result.details["report_weak_row_score"] == pytest.approx(128.2)
    assert result.details["weak_row_score"] == pytest.approx(95.0)
    assert result.details["threshold_calibration"] == "rotated_hdr_field_recalibration_v1"
    assert "倾斜HDR现场重标定" in result.reason


@pytest.mark.parametrize("override", [94.0, 96.0])
def test_rotated_v3_predictor_rejects_unapproved_weak_score_override(
    tmp_path: Path,
    override: float,
) -> None:
    roi_asset, roi_sha256 = _rotated_roi_asset(tmp_path)

    with pytest.raises(ValueError, match="只允许精确值95.0"):
        EightViewTrackedProfileBrightStreakPredictor(
            _tracked_report(tmp_path),
            rotated_roi_path=roi_asset,
            rotated_roi_sha256=roi_sha256,
            weak_row_score_override=override,
        )


def test_rotated_v3_rejects_roi_sha_mismatch(tmp_path: Path) -> None:
    roi_asset, _roi_sha256 = _rotated_roi_asset(tmp_path)

    with pytest.raises(ValueError, match="倾斜光痕ROI.*SHA256"):
        EightViewTrackedProfileBrightStreakPredictor(
            _tracked_report(tmp_path),
            rotated_roi_path=roi_asset,
            rotated_roi_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("kind", "expected_status", "decision"),
    [
        ("diagonal", BranchStatus.PASS, "OK"),
        ("absent", BranchStatus.NG, "NG_NO_STREAK"),
        ("bridged", BranchStatus.PASS, "OK"),
        ("broken", BranchStatus.NG, "NG_BROKEN"),
    ],
)
def test_tracked_profile_predictor_exposes_chinese_metrics_and_colored_path(
    tmp_path: Path,
    kind: str,
    expected_status: BranchStatus,
    decision: str,
) -> None:
    predictor = EightViewTrackedProfileBrightStreakPredictor(_tracked_report(tmp_path))

    output = predictor.predict(_tracked_full_image(kind))

    assert output.status is expected_status
    assert output.details["decision"] == decision
    assert output.overlay is not None
    assert output.overlay.shape == (613, 81, 3)
    for phrase in ("是否存在", "覆盖率", "最长连续段", "最大断点", "断点数", "强阈值", "弱阈值", "桥接行"):
        assert phrase in output.reason
    if kind == "bridged":
        assert output.details["bridged_rows"] > 0
    path_colors = {
        tuple(pixel) for pixel in output.overlay.reshape(-1, 3)
        if tuple(pixel) in {(0, 255, 0), (0, 200, 255), (0, 0, 255)}
    }
    if kind == "diagonal":
        assert (0, 255, 0) in path_colors
    elif kind == "bridged":
        assert {(0, 255, 0), (0, 200, 255)}.issubset(path_colors)
    elif kind == "broken":
        assert {(0, 255, 0), (0, 0, 255)}.issubset(path_colors)


def test_tracked_profile_predictor_rejects_incomplete_artifact_inventory(tmp_path: Path) -> None:
    report = _tracked_report(tmp_path)
    (report.parent / "profiles/profile_0001.npz").unlink()

    with pytest.raises(ValueError, match="artifact inventory"):
        EightViewTrackedProfileBrightStreakPredictor(report)


def test_tracked_profile_overlay_does_not_paint_leading_background_as_gap(tmp_path: Path) -> None:
    predictor = EightViewTrackedProfileBrightStreakPredictor(_tracked_report(tmp_path))

    output = predictor.predict(_tracked_full_image("leading_background"))

    assert output.details["active_start_row"] == 540
    first_path_x = output.details["tracked_centerline_x"][0]
    assert tuple(output.overlay[0, first_path_x]) != (0, 0, 255)


def test_tracked_profile_predictor_rejects_unknown_report_fields(tmp_path: Path) -> None:
    report = _tracked_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="report schema"):
        EightViewTrackedProfileBrightStreakPredictor(report)


def test_tracked_profile_predictor_rejects_changed_manifest_source(tmp_path: Path) -> None:
    report = _tracked_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    Path(payload["manifest"]).write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="source identity SHA256"):
        EightViewTrackedProfileBrightStreakPredictor(report)


def test_tracked_profile_predictor_rejects_same_shape_wrong_fixed_roi(tmp_path: Path) -> None:
    report = _tracked_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["roi_xyxy"] = [0, 0, 81, 613]
    payload["identities"]["roi_config_sha256"] = hashlib.sha256(
        json.dumps(
            {"roi_xyxy": payload["roi_xyxy"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="固定ROI"):
        EightViewTrackedProfileBrightStreakPredictor(report)


def test_tracked_profile_predictor_requires_confirmed_live_normal_identity(tmp_path: Path) -> None:
    report = _tracked_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["accepted_live_normals"] = []
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="现场正常样本"):
        EightViewTrackedProfileBrightStreakPredictor(report)


def test_tracked_profile_predictor_rejects_incomplete_acceptance_evidence(tmp_path: Path) -> None:
    report = _tracked_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["acceptance_gate"] = {"passed": True}
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="验收证据"):
        EightViewTrackedProfileBrightStreakPredictor(report)


def test_tracked_profile_predictor_rejects_metrics_inventory_semantic_mismatch(tmp_path: Path) -> None:
    report = _tracked_report(tmp_path)
    metrics = report.parent / "metrics.csv"
    rows = metrics.read_text(encoding="utf-8").splitlines()
    metrics.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["artifact_identities"]["metrics_csv_sha256"] = hashlib.sha256(
        metrics.read_bytes()
    ).hexdigest()
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics与profile inventory"):
        EightViewTrackedProfileBrightStreakPredictor(report)


def test_tracked_profile_predictor_rejects_duplicated_final_and_replay_identities(tmp_path: Path) -> None:
    report = _tracked_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["final_test"]["outcomes"] = [payload["final_test"]["outcomes"][0]] * 20
    payload["replay"]["outcomes"] = [payload["replay"]["outcomes"][0]] * 20
    replay = report.parent / "replay_summary.json"
    replay.write_text(json.dumps(payload["replay"]), encoding="utf-8")
    payload["artifact_identities"]["replay_summary_sha256"] = hashlib.sha256(
        replay.read_bytes()
    ).hexdigest()
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="逐样本身份不一致"):
        EightViewTrackedProfileBrightStreakPredictor(report)


def test_tracked_profile_predictor_rejects_geometry_that_cannot_fit_fixed_roi(tmp_path: Path) -> None:
    report = _tracked_report(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["geometry"]["candidate_width"] = 79
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="geometry超出固定ROI"):
        EightViewTrackedProfileBrightStreakPredictor(report)


def test_bright_engine_change_preserves_every_field_of_all_24_non_bright_rows() -> None:
    images = {view: np.full((20, 20, 3), index, dtype=np.uint8) for index, view in enumerate(VIEW_ORDER)}

    def output(branch: str, view: str, _image: np.ndarray) -> ModelOutput:
        index = list(VIEW_ORDER).index(view)
        return ModelOutput(
            BranchStatus.NG if index % 3 == 0 else BranchStatus.PASS,
            index / 10,
            0.55,
            f"{branch}/{view}/fixed",
            None,
        )

    def suite(bright_reason: str) -> EightViewModelSuite:
        return EightViewModelSuite(
            rois={view: (0, 0, 20, 20) for view in VIEW_ORDER},
            template_predictor=lambda view, image: output("template", view, image),
            bright_streak_predictor=lambda _image: ModelOutput(
                BranchStatus.PASS, 0.8, 0.1, bright_reason, None
            ),
            yolo_predictor=lambda view, image: output("yolo", view, image),
            efficientad_predictor=lambda view, image: output("efficientad", view, image),
        )

    old_rows = suite("raw_profile_v2").inspect(images, capture_id="same").results
    new_rows = suite("tracked_profile_v3").inspect(images, capture_id="same").results
    fields = lambda row: (
        row.branch.value, row.view_id, row.status.value, row.score, row.threshold, row.reason
    )
    old_non_bright = [fields(row) for row in old_rows if row.branch.value != "bright_streak"]
    new_non_bright = [fields(row) for row in new_rows if row.branch.value != "bright_streak"]

    assert len(old_rows) == len(new_rows) == 25
    assert len(old_non_bright) == len(new_non_bright) == 24
    assert old_non_bright == new_non_bright
    assert [fields(row) for row in old_rows if row.branch.value == "bright_streak"] != [
        fields(row) for row in new_rows if row.branch.value == "bright_streak"
    ]


def test_efficientad_predictor_keeps_one_resident_predictor_per_view(tmp_path: Path) -> None:
    loaded: list[str] = []
    checkpoints = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / view / "model.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        checkpoints[view] = checkpoint

    def factory(path: Path):
        loaded.append(path.parent.name)
        return lambda _image: (0.8, True, np.ones((4, 4), dtype=np.float32))

    predictor = EightViewEfficientAdPredictor(
        checkpoints,
        thresholds={view: 0.5 for view in VIEW_ORDER},
        base_thresholds={view: 0.45 for view in VIEW_ORDER},
        threshold_margin=0.05,
        predictor_factory=factory,
    )
    output = predictor.predict("back_secondary", np.zeros((10, 10, 3), dtype=np.uint8))

    assert loaded == list(VIEW_ORDER)
    assert output.status is BranchStatus.NG
    assert output.score == pytest.approx(0.8)
    assert output.threshold == pytest.approx(0.5)
    assert output.overlay is not None
    assert output.details["evidence_type"] == "诊断热区"
    assert output.details["base_threshold"] == pytest.approx(0.45)
    assert output.details["deployment_threshold"] == pytest.approx(0.5)
    assert output.details["threshold_margin"] == pytest.approx(0.05)
    assert output.details["threshold_exceedance"] == pytest.approx(0.3)
    assert output.details["hotspot_x"] == 0
    assert output.details["hotspot_y"] == 0
    assert output.details["score_source"] == "pred_score"


def test_efficientad_predictor_uses_component_p95_score_and_exposes_component_evidence(
    tmp_path: Path,
) -> None:
    checkpoints = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / view / "model.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        checkpoints[view] = checkpoint
    anomaly_map = np.zeros((9, 9), dtype=np.float32)
    anomaly_map[2:5, 3:6] = 0.70
    anomaly_map[8, 8] = 0.99
    masks = {view: np.zeros((9, 9), dtype=np.uint8) for view in VIEW_ORDER}
    masks["front"][8, 8] = 255
    policy = ComponentFilterPolicy(
        low_threshold=0.30,
        seed_threshold=0.50,
        p95_threshold=0.65,
        minimum_area=8,
        hard_peak_threshold=0.90,
        line_minimum_length=5,
        line_minimum_area=4,
    )
    predictor = EightViewEfficientAdPredictor(
        checkpoints,
        thresholds={view: 0.65 for view in VIEW_ORDER},
        predictor_factory=lambda _path: lambda _image: (0.10, False, anomaly_map),
        component_policies={view: policy for view in VIEW_ORDER},
        component_filter_artifact_sha256="b" * 64,
        ignore_masks=masks,
        ignore_mask_index_sha256="a" * 64,
    )

    output = predictor.predict("front", np.zeros((90, 90, 3), dtype=np.uint8))

    assert output.status is BranchStatus.NG
    assert output.score == pytest.approx(0.70)
    assert output.raw_pred_label is False
    assert output.details["score_source"] == "accepted_component_max_p95"
    assert output.details["component_filter_artifact_sha256"] == "b" * 64
    assert output.details["raw_anomaly_map_max"] == pytest.approx(0.99)
    assert output.details["ignored_map_pixel_count"] == 1
    assert output.details["accepted_component_count"] == 1
    assert output.details["rejected_component_count"] == 0
    component = output.details["accepted_components"][0]
    assert component == {
        "area": 9,
        "peak": pytest.approx(0.70),
        "mean": pytest.approx(0.70),
        "p95": pytest.approx(0.70),
        "bounding_box_xyxy": (3, 2, 6, 5),
        "acceptance_reason": "area_p95",
        "contains_seed": True,
    }
    assert output.details["hotspot_x"] == 3
    assert output.details["hotspot_y"] == 2
    assert output.overlay is not None
    assert np.any(output.overlay[:, :, 1] > output.overlay[:, :, 2])
    masked_map = anomaly_map.copy()
    masked_map[8, 8] = 0.0
    expected = cv2.addWeighted(
        np.zeros((90, 90, 3), dtype=np.uint8),
        0.6,
        cv2.resize(fixed_scale_heatmap(masked_map), (90, 90), interpolation=cv2.INTER_LINEAR),
        0.4,
        0.0,
    )
    assert np.array_equal(output.overlay[89, 89], expected[89, 89])


def test_efficientad_predictor_uses_deployment_threshold_not_pred_label(tmp_path: Path) -> None:
    checkpoints = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / view / "model.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        checkpoints[view] = checkpoint
    anomaly_map = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    predictor = EightViewEfficientAdPredictor(
        checkpoints,
        thresholds={view: 0.8 for view in VIEW_ORDER},
        predictor_factory=lambda _path: lambda _image: (0.6, True, anomaly_map),
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    output = predictor.predict("front", image)

    assert output.status is BranchStatus.PASS
    assert output.score == pytest.approx(0.6)
    assert output.threshold == pytest.approx(0.8)
    assert output.raw_pred_label is True
    assert output.details["hotspot_x"] == 1
    assert output.details["hotspot_y"] == 1
    assert not np.array_equal(output.overlay, cv2.addWeighted(image, 0.6, fixed_scale_heatmap(anomaly_map), 0.4, 0.0))


def test_efficientad_predictor_excludes_manual_ignore_mask_but_keeps_empty_views_on_pred_score(
    tmp_path: Path,
) -> None:
    checkpoints = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / view / "model.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        checkpoints[view] = checkpoint
    anomaly_map = np.array([[0.2, 1.0], [0.7, 0.4]], dtype=np.float32)
    masks = {view: np.zeros((4, 4), dtype=np.uint8) for view in VIEW_ORDER}
    masks["back"][:2, 2:] = 255
    predictor = EightViewEfficientAdPredictor(
        checkpoints,
        thresholds={view: 0.8 for view in VIEW_ORDER},
        predictor_factory=lambda _path: lambda _image: (0.9, True, anomaly_map),
        ignore_masks=masks,
        ignore_mask_index_sha256="a" * 64,
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    masked = predictor.predict("back", image)
    unchanged = predictor.predict("front", image)

    assert masked.status is BranchStatus.PASS
    assert masked.score == pytest.approx(0.7)
    assert masked.details["raw_pred_score"] == pytest.approx(0.9)
    assert masked.details["score_source"] == "manual_ignore_masked_anomaly_map_max"
    assert masked.details["ignored_roi_pixel_count"] == 4
    assert masked.details["hotspot_x"] == 0
    assert masked.details["hotspot_y"] == 1
    assert masked.details["ignore_mask_index_sha256"] == "a" * 64
    assert "手动忽略区外异常图最大值" in masked.reason
    assert unchanged.status is BranchStatus.NG
    assert unchanged.score == pytest.approx(0.9)
    assert unchanged.details["score_source"] == "pred_score"


def test_template_difference_uses_best_match_alignment(tmp_path: Path) -> None:
    model_paths = {}
    base = np.zeros((16, 16), dtype=np.uint8)
    base[4:9, 5:10] = 255
    query = np.zeros_like(base)
    query[4:9, 7:12] = 255
    for view in VIEW_ORDER:
        directory = tmp_path / view
        directory.mkdir()
        assert cv2.imwrite(str(directory / "template.png"), base)
        (directory / "model.json").write_text(
            '{"view_id":"%s","input_width":16,"input_height":16,'
            '"threshold":1.0,"preprocess":{"target_width":16,"target_height":16,"max_shift":3},'
            '"templates":[{"path":"template.png"}]}' % view,
            encoding="utf-8",
        )
        model_paths[view] = directory / "model.json"
    predictor = EightViewTemplatePredictor(model_paths)

    output = predictor.predict("front", query)

    assert output.details["evidence_type"] == "诊断热区"
    assert output.details["best_shift_x"] == 2
    assert output.details["best_shift_y"] == 0
    assert output.details["aligned_mean_absolute_difference"] < 20
    assert "最佳平移 (2, 0)" in output.reason
    assert "诊断热区" in output.reason


def _manual_mask_template_predictor(
    tmp_path: Path,
) -> tuple[EightViewTemplatePredictor, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260813)
    base = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    model_paths: dict[str, Path] = {}
    masks = {view: np.zeros(base.shape, dtype=np.uint8) for view in VIEW_ORDER}
    masks["front"][3:13, 3:13] = 255
    for view in VIEW_ORDER:
        directory = tmp_path / view
        directory.mkdir()
        assert cv2.imwrite(str(directory / "template.png"), cv2.GaussianBlur(base, (3, 3), 0))
        (directory / "model.json").write_text(
            '{"view_id":"%s","input_width":32,"input_height":32,'
            '"threshold":0.02,"preprocess":{"target_width":32,"target_height":32,"max_shift":2},'
            '"templates":[{"path":"template.png"}]}' % view,
            encoding="utf-8",
        )
        model_paths[view] = directory / "model.json"
    predictor = EightViewTemplatePredictor(
        model_paths,
        ignore_masks=masks,
        ignore_mask_index_sha256="a" * 64,
        masked_thresholds={view: 0.02 for view in VIEW_ORDER},
        masked_threshold_artifact_sha256="b" * 64,
    )
    return predictor, base, masks["front"]


def test_template_predictor_excludes_manual_mask_and_preserves_raw_evidence(tmp_path: Path) -> None:
    predictor, base, mask = _manual_mask_template_predictor(tmp_path)
    query = base.copy()
    query[5:11, 5:11] = 255

    output = predictor.predict("front", query)

    assert output.status is BranchStatus.PASS
    assert output.score is not None and output.score <= 0.02
    assert output.threshold == pytest.approx(0.02)
    assert output.details["score_source"] == "manual_ignore_masked_ccoeff_normed"
    assert output.details["raw_unmasked_risk"] > 0.02
    assert output.details["masked_risk"] == pytest.approx(output.score)
    assert output.details["ignored_input_pixel_count"] == 100
    assert output.details["valid_target_pixel_fraction"] == pytest.approx(924 / 1024)
    assert output.details["ignore_mask_index_sha256"] == "a" * 64
    assert output.details["masked_threshold_artifact_sha256"] == "b" * 64
    assert "手动忽略区外" in output.reason


def test_template_predictor_keeps_corruption_outside_manual_mask_as_ng(tmp_path: Path) -> None:
    predictor, base, _mask = _manual_mask_template_predictor(tmp_path)
    query = base.copy()
    query[18:26, 18:26] = 255

    output = predictor.predict("front", query)

    assert output.status is BranchStatus.NG
    assert output.score is not None and output.score > 0.02
    assert output.details["masked_risk"] == pytest.approx(output.score)


def test_efficientad_predictor_threshold_equality_is_ng_when_pred_label_is_false(tmp_path: Path) -> None:
    checkpoints = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / view / "model.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        checkpoints[view] = checkpoint
    predictor = EightViewEfficientAdPredictor(
        checkpoints,
        thresholds={view: 0.8 for view in VIEW_ORDER},
        predictor_factory=lambda _path: lambda _image: (0.8, False, np.zeros((2, 2), dtype=np.float32)),
    )

    output = predictor.predict("front", np.zeros((2, 2, 3), dtype=np.uint8))

    assert output.status is BranchStatus.NG
    assert output.raw_pred_label is False


@pytest.mark.parametrize("invalid", [True, "0.8"])
def test_efficientad_predictor_rejects_non_numeric_threshold_before_loading_models(
    tmp_path: Path,
    invalid: object,
) -> None:
    checkpoints = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / view / "model.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        checkpoints[view] = checkpoint
    thresholds = {view: 0.8 for view in VIEW_ORDER}
    thresholds["front"] = invalid
    loaded: list[Path] = []

    with pytest.raises(ValueError, match="有限数值"):
        EightViewEfficientAdPredictor(
            checkpoints,
            thresholds=thresholds,
            predictor_factory=lambda path: loaded.append(path),
        )

    assert loaded == []
