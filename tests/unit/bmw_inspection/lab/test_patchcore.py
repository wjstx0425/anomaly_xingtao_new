"""Tests for six resident BMW PatchCore models and calibration-only training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.contracts import BranchStatus, ViewId
from bmw_inspection.lab.patchcore import (
    PatchCoreBackend,
    PatchCoreTrainingProfile,
    RawPatchCorePrediction,
    fit_f1_threshold,
    train_patchcore_views,
    verify_prediction_equivalence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoints(root: Path) -> dict[ViewId, Path]:
    result: dict[ViewId, Path] = {}
    for view_id in ViewId:
        path = root / view_id.value / "model.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"checkpoint:{view_id.value}".encode())
        result[view_id] = path
    return result


class _FakePredictor:
    def __init__(self, checkpoint: Path, calls: list[tuple[Path, np.ndarray]]) -> None:
        self.checkpoint = checkpoint
        self.calls = calls
        self.score = 0.5
        self.anomaly_map = np.array([[0.25, 2.0], [1.0, 0.5]], dtype=np.float32)

    def predict(self, image: np.ndarray) -> RawPatchCorePrediction:
        self.calls.append((self.checkpoint, image.copy()))
        return RawPatchCorePrediction(score=self.score, anomaly_map=self.anomaly_map.copy())


class _FakeFactory:
    def __init__(self) -> None:
        self.loads: list[Path] = []
        self.calls: list[tuple[Path, np.ndarray]] = []
        self.predictors: dict[Path, _FakePredictor] = {}

    def __call__(self, checkpoint: Path) -> _FakePredictor:
        self.loads.append(checkpoint)
        predictor = _FakePredictor(checkpoint, self.calls)
        self.predictors[checkpoint] = predictor
        return predictor


def _backend(tmp_path: Path, factory: _FakeFactory, *, threshold: float | None = 0.5) -> PatchCoreBackend:
    checkpoints = _checkpoints(tmp_path / "checkpoints")
    return PatchCoreBackend(
        checkpoint_paths=checkpoints,
        thresholds={view_id: threshold for view_id in ViewId},
        rois={view_id: (1, 1, 4, 4) for view_id in ViewId},
        predictor_factory=factory,
        required_for_ok=True,
    )


def test_six_checkpoints_load_once_and_route_mono8_as_rgb_to_the_exact_view(tmp_path: Path) -> None:
    factory = _FakeFactory()
    backend = _backend(tmp_path, factory)
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)

    first = backend.predict(ViewId.FRONT_LEFT, image)
    second = backend.predict(ViewId.FRONT_LEFT, image)

    assert len(factory.loads) == 6
    assert set(factory.loads) == set(_checkpoints(tmp_path / "checkpoints").values())
    assert len(factory.calls) == 2
    checkpoint, received = factory.calls[0]
    assert checkpoint.name == "model.ckpt"
    assert checkpoint.parent.name == ViewId.FRONT_LEFT.value
    assert received.shape == (3, 3, 3)
    assert np.array_equal(received[:, :, 0], image[1:4, 1:4])
    assert np.array_equal(received[:, :, 0], received[:, :, 1])
    assert np.array_equal(received[:, :, 1], received[:, :, 2])
    assert first.status is BranchStatus.PASS
    assert second.status is BranchStatus.PASS


def test_threshold_equality_passes_and_raw_anomaly_map_is_preserved(tmp_path: Path) -> None:
    factory = _FakeFactory()
    backend = _backend(tmp_path, factory, threshold=0.5)
    expected = np.array([[0.25, 2.0], [1.0, 0.5]], dtype=np.float32)

    evidence = backend.predict(ViewId.FRONT, np.zeros((5, 5), dtype=np.uint8))
    decision = backend.decision_for(ViewId.FRONT)

    assert evidence.status is BranchStatus.PASS
    assert evidence.score == evidence.threshold == 0.5
    assert decision is not None
    assert np.array_equal(decision.raw_anomaly_map, expected)
    assert decision.raw_anomaly_map.flags.writeable is False
    assert decision.display_heatmap.dtype == np.uint8
    assert decision.display_heatmap.shape == (2, 2, 3)
    assert np.array_equal(factory.predictors[factory.loads[0]].anomaly_map, expected)


def test_raw_anomaly_map_preserves_predictor_dtype(tmp_path: Path) -> None:
    factory = _FakeFactory()
    backend = _backend(tmp_path, factory)
    predictor = factory.predictors[
        next(path for path in factory.loads if path.parent.name == ViewId.FRONT.value)
    ]
    predictor.anomaly_map = np.array([[0.25, 2.0], [1.0, 0.5]], dtype=np.float64)

    backend.predict(ViewId.FRONT, np.zeros((5, 5), dtype=np.uint8))
    decision = backend.decision_for(ViewId.FRONT)

    assert decision is not None
    assert decision.raw_anomaly_map.dtype == np.float64


@pytest.mark.parametrize("invalid_field", ("score", "map"))
def test_nonfinite_prediction_becomes_error_without_publishing_a_decision(
    tmp_path: Path,
    invalid_field: str,
) -> None:
    factory = _FakeFactory()
    backend = _backend(tmp_path, factory)
    predictor = factory.predictors[next(path for path in factory.loads if path.parent.name == ViewId.FRONT.value)]
    if invalid_field == "score":
        predictor.score = float("nan")
    else:
        predictor.anomaly_map[0, 0] = float("inf")

    evidence = backend.predict(ViewId.FRONT, np.zeros((5, 5), dtype=np.uint8))

    assert evidence.status is BranchStatus.ERROR
    assert evidence.score is None
    assert backend.decision_for(ViewId.FRONT) is None


def test_missing_calibration_threshold_is_review_and_cannot_be_required_ok(tmp_path: Path) -> None:
    factory = _FakeFactory()
    backend = _backend(tmp_path, factory, threshold=None)

    evidence = backend.predict(ViewId.BACK, np.zeros((5, 5), dtype=np.uint8))

    assert evidence.status is BranchStatus.REVIEW
    assert evidence.required_for_ok is True
    assert evidence.threshold is None
    assert "calibration defect" in evidence.reason


def test_checkpoint_load_error_names_the_failed_view(tmp_path: Path) -> None:
    checkpoints = _checkpoints(tmp_path / "checkpoints")

    def failing_factory(checkpoint: Path):
        if checkpoint.parent.name == ViewId.BACK_LEFT.value:
            raise OSError("corrupt checkpoint")
        return _FakePredictor(checkpoint, [])

    with pytest.raises(RuntimeError, match="back_left.*corrupt checkpoint"):
        PatchCoreBackend(
            checkpoint_paths=checkpoints,
            thresholds={view_id: 0.5 for view_id in ViewId},
            rois={view_id: (0, 0, 4, 4) for view_id in ViewId},
            predictor_factory=failing_factory,
            required_for_ok=True,
        )


def test_f1_threshold_uses_defect_when_score_is_strictly_greater() -> None:
    fit = fit_f1_threshold((
        ("normal", 0.1),
        ("normal", 0.2),
        ("defect", 0.8),
        ("defect", 0.9),
    ))

    assert fit.threshold == 0.2
    assert fit.f1 == 1.0
    assert fit.status is BranchStatus.PASS

    review = fit_f1_threshold((("normal", 0.1), ("normal", 0.2)))
    assert review.threshold is None
    assert review.status is BranchStatus.REVIEW


def test_f1_threshold_includes_a_boundary_below_the_minimum_score() -> None:
    fit = fit_f1_threshold((("defect", 0.1), ("normal", 0.2)))

    assert fit.threshold is not None
    assert fit.threshold < 0.1
    assert fit.f1 == pytest.approx(2 / 3)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((6, 8, 3), value, dtype=np.uint8))


def _patchcore_dataset(root: Path, *, missing_defect_view: ViewId | None = None) -> Path:
    for view_index, view_id in enumerate(ViewId):
        _write_image(root / view_id.value / "train/good/train.png", 10 + view_index)
        _write_image(root / view_id.value / "calibration/good/normal.png", 30 + view_index)
        if view_id is not missing_defect_view:
            _write_image(root / view_id.value / "calibration/defect/defect.png", 230 + view_index)
        _write_image(root / view_id.value / "final_test/good/ignored.png", 100 + view_index)
    return root


class _FakeTrainer:
    def __init__(self) -> None:
        self.calls: list[tuple[ViewId, Path, Path, PatchCoreTrainingProfile]] = []

    def train(
        self,
        view_id: ViewId,
        view_root: Path,
        run_dir: Path,
        profile: PatchCoreTrainingProfile,
    ) -> Path:
        self.calls.append((view_id, view_root, run_dir, profile))
        checkpoint = run_dir / "model.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"trained:{view_id.value}".encode())
        return checkpoint


class _CalibrationPredictor:
    def __init__(self, checkpoint: Path, seen: list[Path]) -> None:
        self.checkpoint = checkpoint
        self.seen = seen

    def predict_path(self, image_path: Path) -> RawPatchCorePrediction:
        self.seen.append(image_path)
        score = 0.9 if "defect" in image_path.parts else 0.1
        raw = np.array([[score, score + 0.01]], dtype=np.float32)
        return RawPatchCorePrediction(score=score, anomaly_map=raw)


class _RuntimePredictor:
    def __init__(self, checkpoint: Path, seen: list[np.ndarray], *, map_offset: float = 0.0) -> None:
        self.checkpoint = checkpoint
        self.seen = seen
        self.map_offset = map_offset

    def predict(self, image: np.ndarray) -> RawPatchCorePrediction:
        self.seen.append(image.copy())
        raw = np.array([[0.1, 0.11 + self.map_offset]], dtype=np.float32)
        return RawPatchCorePrediction(score=0.1, anomaly_map=raw)


def test_training_persists_defaults_and_calibrates_each_view_without_final_test(tmp_path: Path) -> None:
    dataset_root = _patchcore_dataset(tmp_path / "patchcore")
    trainer = _FakeTrainer()
    seen: list[Path] = []
    runtime_seen: list[np.ndarray] = []
    runtime_loads: list[Path] = []

    def runtime_factory(checkpoint: Path) -> _RuntimePredictor:
        runtime_loads.append(checkpoint)
        return _RuntimePredictor(checkpoint, runtime_seen)

    report = train_patchcore_views(
        dataset_root=dataset_root,
        output_root=tmp_path / "models-v1",
        trainer=trainer,
        calibration_predictor_factory=lambda checkpoint: _CalibrationPredictor(checkpoint, seen),
        runtime_predictor_factory=runtime_factory,
    )

    profile = PatchCoreTrainingProfile()
    assert profile.backbone == "wide_resnet50_2"
    assert profile.layers == ("layer2", "layer3")
    assert profile.image_size == (512, 512)
    assert profile.coreset_sampling_ratio == 0.1
    assert profile.num_neighbors == 9
    assert profile.seed == 42
    assert profile.max_epochs == 1
    assert len(trainer.calls) == 6
    assert all(call[3] == profile for call in trainer.calls)
    assert all("final_test" not in path.parts for path in seen)
    assert len(runtime_loads) == len(ViewId)
    assert len(runtime_seen) == len(ViewId)
    assert report["ok_profile_allowed"] is True
    assert report["review_views"] == []
    for view_id in ViewId:
        run_dir = tmp_path / "models-v1" / view_id.value
        assert (run_dir / "model.ckpt").is_file()
        config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
        assert config == profile.as_dict()
        threshold = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))
        assert threshold["threshold"] == 0.1
        score_rows = (run_dir / "calibration_scores.csv").read_text(encoding="utf-8")
        assert "normal.png" in score_rows and "defect.png" in score_rows
        raw_maps = sorted((run_dir / "calibration/raw_maps").glob("*.npy"))
        assert len(raw_maps) == 2
        assert any(
            np.array_equal(np.load(path), np.array([[0.1, 0.11]], dtype=np.float32))
            for path in raw_maps
        )
        receipt = json.loads((run_dir / "equivalence_receipt.json").read_text(encoding="utf-8"))
        assert receipt == {
            "atol": 1e-6,
            "checkpoint_sha256": _sha256(run_dir / "model.ckpt"),
            "map_equal": True,
            "sample": f"{view_id.value}/calibration/good/normal.png",
            "score_equal": True,
        }


def test_missing_per_view_calibration_defect_publishes_review_not_ok(tmp_path: Path) -> None:
    dataset_root = _patchcore_dataset(tmp_path / "patchcore", missing_defect_view=ViewId.BACK_RIGHT)
    trainer = _FakeTrainer()

    report = train_patchcore_views(
        dataset_root=dataset_root,
        output_root=tmp_path / "models-v1",
        trainer=trainer,
        calibration_predictor_factory=lambda checkpoint: _CalibrationPredictor(checkpoint, []),
        runtime_predictor_factory=lambda checkpoint: _RuntimePredictor(checkpoint, []),
    )

    assert report["ok_profile_allowed"] is False
    assert report["review_views"] == [ViewId.BACK_RIGHT.value]
    threshold = json.loads(
        (tmp_path / "models-v1/back_right/threshold.json").read_text(encoding="utf-8")
    )
    assert threshold["threshold"] is None
    assert threshold["status"] == BranchStatus.REVIEW.value


def test_training_rejects_publish_when_independent_runtime_prediction_differs(tmp_path: Path) -> None:
    dataset_root = _patchcore_dataset(tmp_path / "patchcore")

    with pytest.raises(ValueError, match="raw anomaly map differs"):
        train_patchcore_views(
            dataset_root=dataset_root,
            output_root=tmp_path / "models-v1",
            trainer=_FakeTrainer(),
            calibration_predictor_factory=lambda checkpoint: _CalibrationPredictor(checkpoint, []),
            runtime_predictor_factory=lambda checkpoint: _RuntimePredictor(
                checkpoint,
                [],
                map_offset=1e-3,
            ),
        )

    assert not (tmp_path / "models-v1").exists()


def test_same_checkpoint_score_and_raw_map_equivalence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    raw = np.array([[0.1, 0.2]], dtype=np.float32)

    receipt = verify_prediction_equivalence(
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=_sha256(checkpoint),
        training_prediction=RawPatchCorePrediction(0.2, raw),
        runtime_prediction=RawPatchCorePrediction(0.2, raw + 1e-7),
        atol=1e-6,
    )

    assert receipt.checkpoint_sha256 == _sha256(checkpoint)
    assert receipt.score_equal is True
    assert receipt.map_equal is True
    with pytest.raises(ValueError, match="raw anomaly map differs"):
        verify_prediction_equivalence(
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256=_sha256(checkpoint),
            training_prediction=RawPatchCorePrediction(0.2, raw),
            runtime_prediction=RawPatchCorePrediction(0.2, raw + 1e-3),
            atol=1e-6,
        )


def test_cli_defaults_match_the_frozen_training_profile() -> None:
    project_root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(str(project_root / "pipeline/bmw_lab_train_patchcore.py"))
    args = namespace["build_parser"]().parse_args(
        ["--dataset-root", "patchcore", "--output-root", "models-v1"]
    )

    assert args.backbone == "wide_resnet50_2"
    assert args.layers == ["layer2", "layer3"]
    assert args.image_size == [512, 512]
    assert args.coreset_sampling_ratio == 0.1
    assert args.num_neighbors == 9
    assert args.seed == 42
    assert args.max_epochs == 1
