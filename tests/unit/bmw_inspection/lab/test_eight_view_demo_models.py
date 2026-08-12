"""Runtime behavior for the BMW eight-view model suite."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import BranchStatus, DemoFinalStatus
from bmw_inspection.lab.efficientad_analysis import fixed_scale_heatmap
from bmw_inspection.lab.eight_view_demo_models import (
    EightViewBrightStreakPredictor,
    EightViewEfficientAdPredictor,
    EightViewModelSuite,
    EightViewTemplatePredictor,
    EightViewYoloPredictor,
    EightViewRawProfileBrightStreakPredictor,
    ModelOutput,
    load_part_rois,
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
