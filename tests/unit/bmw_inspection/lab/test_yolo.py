# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Global resident YOLO tests for BMW six-view laboratory inspection."""

from __future__ import annotations

import csv
from pathlib import Path
import runpy
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from bmw_inspection.lab.contracts import BranchName, BranchStatus, ViewId
from bmw_inspection.lab.yolo import (
    DetectionBox,
    YoloBackend,
    YoloCalibrationSample,
    YoloEvidence,
    YoloScoreRow,
    fit_yolo_final_threshold,
    render_final_overlay,
    train_versioned_yolo,
)


REPO_ROOT = Path(__file__).parents[4]
ROIS = {
    ViewId.FRONT: (1, 1, 7, 5),
    ViewId.FRONT_LEFT: (0, 0, 6, 4),
    ViewId.FRONT_RIGHT: (2, 1, 8, 5),
    ViewId.BACK: (1, 0, 7, 4),
    ViewId.BACK_LEFT: (0, 1, 6, 5),
    ViewId.BACK_RIGHT: (2, 0, 8, 4),
}


def _checkpoint(path: Path, content: bytes = b"fake-best-checkpoint") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _result(
    confidences: list[float],
    classes: list[int] | None = None,
    coordinates: list[list[float]] | None = None,
) -> SimpleNamespace:
    class_ids = classes if classes is not None else [0] * len(confidences)
    boxes = np.asarray(
        coordinates
        if coordinates is not None
        else [[0.5 + index, 0.5, 3.5 + index, 2.5] for index in range(len(confidences))],
        dtype=np.float32,
    ).reshape((-1, 4))
    return SimpleNamespace(
        names={0: "defect", 1: "other"},
        boxes=SimpleNamespace(
            xyxy=boxes,
            conf=np.asarray(confidences, dtype=np.float32),
            cls=np.asarray(class_ids, dtype=np.float32),
        ),
    )


class _FakePredictor:
    def __init__(
        self,
        results: list[SimpleNamespace],
        *,
        names: dict[int, str] | None = None,
    ) -> None:
        self.names = names or {0: "defect"}
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        return [self._results.pop(0)]


def _backend(
    tmp_path: Path,
    predictor: _FakePredictor,
    *,
    candidate_conf: float = 0.01,
    final_threshold: float = 0.50,
    required_for_ok: bool = True,
) -> tuple[YoloBackend, list[Path]]:
    created: list[Path] = []

    def factory(path: Path) -> _FakePredictor:
        created.append(Path(path))
        return predictor

    backend = YoloBackend(
        _checkpoint(tmp_path / "best.pt"),
        part_rois=ROIS,
        candidate_conf=candidate_conf,
        final_threshold=final_threshold,
        required_for_ok=required_for_ok,
        model_factory=factory,
    )
    return backend, created


def _mono() -> np.ndarray:
    return np.arange(48, dtype=np.uint8).reshape(6, 8)


def test_subthreshold_candidate_is_saved_but_not_final_or_drawn_red(tmp_path: Path) -> None:
    predictor = _FakePredictor([_result([0.30])])
    backend, created = _backend(tmp_path, predictor)

    evidence = backend.predict(ViewId.FRONT, _mono())
    overlay = render_final_overlay(cv2.cvtColor(_mono(), cv2.COLOR_GRAY2BGR), evidence)

    assert len(created) == 1
    assert len(evidence.candidates) == 1
    assert evidence.final_boxes == ()
    assert evidence.status is BranchStatus.PASS
    assert evidence.score == pytest.approx(0.30)
    assert evidence.threshold == 0.50
    assert np.array_equal(overlay, cv2.cvtColor(_mono(), cv2.COLOR_GRAY2BGR))


def test_final_threshold_equality_is_ng_and_only_final_boxes_are_rendered(tmp_path: Path) -> None:
    predictor = _FakePredictor([_result([0.20, 0.50])])
    backend, _created = _backend(tmp_path, predictor, final_threshold=0.50)
    source = cv2.cvtColor(_mono(), cv2.COLOR_GRAY2BGR)

    evidence = backend.predict(ViewId.FRONT, source)
    overlay = render_final_overlay(source, evidence)

    assert len(evidence.candidates) == 2
    assert len(evidence.final_boxes) == 1
    assert evidence.final_boxes[0].confidence == pytest.approx(0.50)
    assert evidence.status is BranchStatus.NG
    assert np.any(overlay != source)


def test_one_resident_model_serves_six_views_and_mono_becomes_three_equal_channels(tmp_path: Path) -> None:
    predictor = _FakePredictor([_result([]) for _view in ViewId])
    backend, created = _backend(tmp_path, predictor)

    for view_id in ViewId:
        evidence = backend.predict(view_id, _mono())
        assert evidence.status is BranchStatus.PASS

    assert len(created) == 1
    assert len(predictor.calls) == 6
    for view_id, call in zip(ViewId, predictor.calls, strict=True):
        image = call["source"]
        assert isinstance(image, np.ndarray)
        x1, y1, x2, y2 = ROIS[view_id]
        assert image.shape == (y2 - y1, x2 - x1, 3)
        assert np.array_equal(image[..., 0], image[..., 1])
        assert np.array_equal(image[..., 1], image[..., 2])
        assert call["conf"] == 0.01
        assert call["imgsz"] == 1280


def test_boxes_are_shifted_from_roi_local_to_full_view_coordinates_and_model_is_identified(tmp_path: Path) -> None:
    predictor = _FakePredictor([_result([0.80])])
    backend, _created = _backend(tmp_path, predictor)

    evidence = backend.predict(ViewId.FRONT, _mono())

    box = evidence.final_boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((1.5, 1.5, 4.5, 3.5))
    assert box.class_name == "defect"
    assert evidence.model_id is not None and len(evidence.model_id) == 64
    assert evidence.artifact_paths["checkpoint"].endswith("best.pt")


def test_backend_rejects_non_defect_model_or_prediction_class(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one class named 'defect'"):
        _backend(tmp_path / "bad-model", _FakePredictor([], names={0: "scratch"}))

    predictor = _FakePredictor([_result([0.80], classes=[1])])
    backend, _created = _backend(tmp_path / "bad-result", predictor)
    with pytest.raises(ValueError, match="predicted class must be defect"):
        backend.predict(ViewId.FRONT, _mono())


def test_fully_clipped_degenerate_prediction_is_dropped(tmp_path: Path) -> None:
    predictor = _FakePredictor(
        [_result([0.90, 0.80], coordinates=[[-5.0, -5.0, -1.0, -1.0], [1.0, 1.0, 3.0, 3.0]])]
    )
    backend, _created = _backend(tmp_path, predictor)

    evidence = backend.predict(ViewId.FRONT, _mono())

    assert len(evidence.candidates) == 1
    assert evidence.candidates[0].confidence == pytest.approx(0.80)


def test_resident_predictor_calls_are_serialized(tmp_path: Path) -> None:
    class BlockingPredictor(_FakePredictor):
        def __init__(self) -> None:
            super().__init__([_result([]), _result([])])
            self.entered = threading.Event()
            self.release = threading.Event()
            self.second_entered = threading.Event()

        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            if self.entered.is_set():
                self.second_entered.set()
            else:
                self.entered.set()
                assert self.release.wait(timeout=2)
            return super().predict(**kwargs)

    predictor = BlockingPredictor()
    backend, _created = _backend(tmp_path, predictor)
    errors: list[BaseException] = []

    def run(view_id: ViewId) -> None:
        try:
            backend.predict(view_id, _mono())
        except BaseException as error:  # pragma: no cover - surfaced by the assertion below
            errors.append(error)

    first = threading.Thread(target=run, args=(ViewId.FRONT,))
    second = threading.Thread(target=run, args=(ViewId.BACK,))
    first.start()
    assert predictor.entered.wait(timeout=2)
    second.start()
    assert not predictor.second_entered.wait(timeout=0.1)
    predictor.release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not errors
    assert not first.is_alive() and not second.is_alive()
    assert len(predictor.calls) == 2


def test_yolo_evidence_rejects_inconsistent_business_fields() -> None:
    candidate = DetectionBox(0, 0, 1, 1, 0.8)
    common = {
        "branch": BranchName.YOLO,
        "view_id": ViewId.FRONT,
        "required_for_ok": True,
        "elapsed_ms": 1.0,
        "reason": "test",
        "model_id": "model",
        "artifact_paths": {},
        "candidates": (candidate,),
        "roi_xyxy": (0, 0, 2, 2),
    }
    with pytest.raises(ValueError, match="final_boxes must exactly match"):
        YoloEvidence(
            status=BranchStatus.PASS,
            score=0.8,
            threshold=0.5,
            final_boxes=(),
            **common,
        )


@pytest.mark.parametrize("confidence", [float("nan"), -0.1, 1.1])
def test_backend_rejects_nonfinite_or_out_of_range_confidence(tmp_path: Path, confidence: float) -> None:
    predictor = _FakePredictor([_result([confidence])])
    backend, _created = _backend(tmp_path, predictor)

    with pytest.raises(ValueError, match="confidence"):
        backend.predict(ViewId.FRONT, _mono())


def test_backend_reads_roi_thresholds_and_required_flag_from_profile(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "profile-best.pt")
    predictor = _FakePredictor([_result([])])
    profile = SimpleNamespace(
        yolo=SimpleNamespace(
            enabled=True,
            checkpoint=checkpoint,
            candidate_conf=0.02,
            final_threshold=0.60,
        ),
        part_rois=ROIS,
        required_for_ok=frozenset({BranchName.YOLO}),
    )
    backend = YoloBackend.from_experiment_config(profile, model_factory=lambda _path: predictor)

    evidence = backend.predict(ViewId.BACK_RIGHT, _mono())

    assert evidence.required_for_ok is True
    assert evidence.threshold == 0.60
    assert predictor.calls[0]["conf"] == 0.02


def test_calibration_f1_threshold_ignores_final_test_rows() -> None:
    calibration = [
        YoloScoreRow("normal-a", "calibration", "normal", 0.20),
        YoloScoreRow("normal-b", "calibration", "normal", 0.40),
        YoloScoreRow("defect-a", "calibration", "defect", 0.60),
        YoloScoreRow("defect-b", "calibration", "defect", 0.80),
    ]
    fit_a = fit_yolo_final_threshold(
        [*calibration, YoloScoreRow("final-a", "final_test", "normal", 0.99)],
        candidate_conf=0.01,
    )
    fit_b = fit_yolo_final_threshold(
        [*calibration, YoloScoreRow("final-b", "final_test", "defect", 0.01)],
        candidate_conf=0.01,
    )

    assert fit_a.threshold == pytest.approx(0.60)
    assert fit_b == fit_a
    assert fit_a.f1 == pytest.approx(1.0)


def test_training_cli_does_not_construct_final_test_samples(tmp_path: Path) -> None:
    image = tmp_path / "calibration-full.png"
    exported = tmp_path / "calibration-roi.png"
    assert cv2.imwrite(str(image), _mono())
    assert cv2.imwrite(str(exported), _mono()[1:5, 2:7])
    manifest = tmp_path / "export_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "view_id", "split", "label", "image_path", "exported_image_path"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sample_id": "cal",
                    "view_id": "front",
                    "split": "calibration",
                    "label": "normal",
                    "image_path": image.name,
                    "exported_image_path": exported.name,
                },
                {
                    "sample_id": "held-out",
                    "view_id": "not-a-view",
                    "split": "final_test",
                    "label": "defect",
                    "image_path": "must-not-be-read-full.png",
                    "exported_image_path": "must-not-be-read.png",
                },
            ]
        )
    namespace = runpy.run_path(str(REPO_ROOT / "pipeline/bmw_lab_train_yolo.py"))

    samples = namespace["_load_calibration_samples"](manifest)

    assert len(samples) == 1
    assert samples[0].sample_id == "cal"
    assert samples[0].image_path == image.resolve()


class _FakeTrainer:
    names = {0: "defect"}

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def train(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        best = _checkpoint(save_dir / "weights/best.pt", b"trained-best")
        training_args = {**kwargs, "batch": 16, "optimizer": "auto"}
        (save_dir / "args.yaml").write_text(yaml.safe_dump(training_args, sort_keys=True), encoding="utf-8")
        self.trainer = SimpleNamespace(best=best, save_dir=save_dir, args=SimpleNamespace(**training_args))
        return SimpleNamespace(save_dir=Path("/unstable-result-api-must-not-be-used"))


def test_versioned_fake_training_uses_seed42_imgsz1280_and_never_overwrites_or_activates(
    tmp_path: Path,
) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names:\n  0: defect\n", encoding="utf-8")
    base = _checkpoint(tmp_path / "base.pt", b"base")
    normal = tmp_path / "normal.png"
    defect = tmp_path / "defect.png"
    assert cv2.imwrite(str(normal), _mono())
    assert cv2.imwrite(str(defect), np.flipud(_mono()))
    trainer = _FakeTrainer()
    predictor = _FakePredictor([_result([0.20]), _result([0.80])])
    factory_calls: list[Path] = []
    factory_payloads: list[bytes] = []

    def factory(path: Path) -> object:
        factory_calls.append(Path(path))
        factory_payloads.append(Path(path).read_bytes())
        return trainer if Path(path) == base else predictor

    run = train_versioned_yolo(
        data_yaml=data_yaml,
        base_checkpoint=base,
        output_root=tmp_path / "runs",
        version="yolo-v1",
        calibration_samples=(
            YoloCalibrationSample("normal", ViewId.FRONT, normal, "calibration", "normal"),
            YoloCalibrationSample("defect", ViewId.FRONT, defect, "calibration", "defect"),
        ),
        part_rois={ViewId.FRONT: (0, 0, 8, 6)},
        candidate_conf=0.01,
        model_factory=factory,
    )

    assert run.output_dir == tmp_path / "runs/yolo-v1"
    assert run.best_checkpoint.read_bytes() == b"trained-best"
    assert (run.output_dir / "args.yaml").is_file()
    assert (run.output_dir / "calibration_predictions.csv").is_file()
    assert (run.output_dir / "metrics.json").is_file()
    assert trainer.calls[0]["seed"] == 42
    assert trainer.calls[0]["imgsz"] == 1280
    persisted_args = yaml.safe_load((run.output_dir / "args.yaml").read_text(encoding="utf-8"))
    assert persisted_args["batch"] == 16
    assert persisted_args["optimizer"] == "auto"
    assert persisted_args["ultralytics_version"] == "8.4.89"
    assert factory_calls[0] == base
    assert factory_calls[1].name == "best.pt"
    assert factory_payloads == [b"base", b"trained-best"]
    assert not (tmp_path / "active_profile.json").exists()
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        train_versioned_yolo(
            data_yaml=data_yaml,
            base_checkpoint=base,
            output_root=tmp_path / "runs",
            version="yolo-v1",
            calibration_samples=(),
            part_rois={},
            model_factory=factory,
        )


def test_ultralytics_optional_dependency_is_exactly_pinned() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'bmw-lab = ["ultralytics==8.4.89"]' in pyproject


def test_failed_staging_validation_preserves_concurrently_created_destination(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names:\n  0: defect\n", encoding="utf-8")
    base = _checkpoint(tmp_path / "base.pt")
    destination = tmp_path / "runs/yolo-race"
    trainer = _FakeTrainer()

    def factory(path: Path) -> object:
        if Path(path) == base:
            return trainer
        destination.mkdir(parents=True)
        (destination / "owner.txt").write_text("other publisher\n", encoding="utf-8")
        return SimpleNamespace(names={0: "wrong"})

    with pytest.raises(ValueError, match="exactly one class named 'defect'"):
        train_versioned_yolo(
            data_yaml=data_yaml,
            base_checkpoint=base,
            output_root=tmp_path / "runs",
            version="yolo-race",
            calibration_samples=(),
            part_rois={},
            model_factory=factory,
        )

    assert (destination / "owner.txt").read_text(encoding="utf-8") == "other publisher\n"


def test_ultralytics_pin_is_present_in_uv_lock() -> None:
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "ultralytics"' in lock
    assert 'version = "8.4.89"' in lock
