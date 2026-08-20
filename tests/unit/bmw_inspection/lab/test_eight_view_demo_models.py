"""Runtime behavior for the BMW eight-view model suite."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import BranchStatus, DemoFinalStatus
from bmw_inspection.lab.eight_view_demo_models import (
    EightViewBrightStreakPredictor,
    EightViewEfficientAdPredictor,
    EightViewModelSuite,
    EightViewTemplatePredictor,
    EightViewYoloPredictor,
    ModelOutput,
    load_part_rois,
)


def test_model_suite_runs_all_25_checks_without_short_circuiting() -> None:
    calls: Counter[str] = Counter()

    def output(branch: str, view: str, _image: np.ndarray) -> ModelOutput:
        calls[branch] += 1
        status = BranchStatus.NG if branch == "template" and view == "front" else BranchStatus.PASS
        return ModelOutput(status, 0.1, 0.2, f"{branch}/{view}", np.zeros((10, 10, 3), dtype=np.uint8))

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

    predictor = EightViewEfficientAdPredictor(checkpoints, predictor_factory=factory)
    output = predictor.predict("back_secondary", np.zeros((10, 10, 3), dtype=np.uint8))

    assert loaded == list(VIEW_ORDER)
    assert output.status is BranchStatus.NG
    assert output.score == pytest.approx(0.8)
    assert output.threshold == pytest.approx(0.5)
    assert output.overlay is not None
