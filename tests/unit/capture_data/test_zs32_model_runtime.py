# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the eight-view ZS32 PatchCore and YOLO runtime."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import cv2
import numpy as np
import pytest
from capture_data.zs32_inspection_orchestrator import InspectionRequest
from capture_data.zs32_model_runtime import (
    AnomalibPatchcoreBackend,
    ModelEvidence,
    PatchcoreArtifacts,
    UltralyticsYoloBackend,
    YoloSpec,
    ZS32ModelRuntime,
    _build_patchcore_mask,
    load_runtime_config,
)
from capture_data.zs32_patchcore_roi_dataset import VIEWS as PATCHCORE_ROI_VIEWS
from capture_data.zs32_patchcore_roi_dataset import load_patchcore_roi_config
from capture_data.zs32_view_roi_dataset import VIEWS as YOLO_ROI_VIEWS
from capture_data.zs32_view_roi_dataset import load_roi_config
from zs32_inspection.domain.views import CANONICAL_VIEWS

VIEWS = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    patchcore_roi = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": 12, "height": 10},
        "views": {view: {"roi": [0, 0, 6, 5]} for view in VIEWS},
    }
    yolo_roi = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": 12, "height": 10},
        "views": {view: {"roi": [6, 5, 12, 10]} for view in VIEWS},
    }
    patchcore_roi_path = tmp_path / "patchcore_roi.json"
    yolo_roi_path = tmp_path / "yolo_roi.json"
    patchcore_roi_path.write_text(json.dumps(patchcore_roi), encoding="utf-8")
    yolo_roi_path.write_text(json.dumps(yolo_roi), encoding="utf-8")

    checkpoints: dict[str, Path] = {}
    patchcore: dict[str, object] = {}
    for index, view in enumerate(VIEWS):
        checkpoint = tmp_path / f"{view}.ckpt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        checkpoints[view] = checkpoint
        patchcore[view] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "model_version": f"pc-{view}",
        }
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"yolo")
    config = {
        "schema_version": 1,
        "product": "ZS32",
        "profile": "zs32_right_eight_view_v1",
        "supported_hands": ["right"],
        "patchcore_roi_config": str(patchcore_roi_path),
        "yolo_roi_config": str(yolo_roi_path),
        "versions": {
            "threshold": "thresholds-v1",
            "patchcore_roi": "patchcore-roi-v1",
            "yolo_roi": "yolo-roi-v1",
            "template": "templates-v1",
        },
        "patchcore": patchcore,
        "yolo": {
            "weights": str(weights),
            "weights_sha256": _sha256(weights),
            "model_version": "yolo-v1",
            "imgsz": 640,
            "candidate_conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "class_map": {"0": "defect"},
        },
    }
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, checkpoints


def _write_images(tmp_path: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for index, view in enumerate(VIEWS):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        image[:5, :6] = (index + 1) * 10
        image[5:, 6:] = (index + 1) * 20
        path = tmp_path / f"{view}.png"
        assert cv2.imwrite(str(path), image)
        images[view] = path
    return images


def test_yolo_backend_rejects_boundary_collapsed_candidates(tmp_path: Path) -> None:
    """A zero-area candidate must fail closed instead of disappearing from strict evidence."""

    class _Xyxy:
        def detach(self) -> _Xyxy:
            return self

        def cpu(self) -> _Xyxy:
            return self

        def reshape(self, *_shape: int) -> np.ndarray:
            return np.asarray([0.0, 5.0, 12.0, 5.0])

    box = SimpleNamespace(cls=np.asarray([0]), conf=np.asarray([0.001]), xyxy=_Xyxy())
    predictions = [
        SimpleNamespace(boxes=[box] if view == "front" else [], names={0: "item"}, plot=lambda: np.zeros((8, 8, 3)))
        for view in VIEWS
    ]
    backend = object.__new__(UltralyticsYoloBackend)
    backend.model = SimpleNamespace(predict=lambda **_kwargs: predictions)
    backend.spec = YoloSpec(
        weights=tmp_path / "best.pt",
        weights_sha256="0" * 64,
        model_version="yolo-v1",
        imgsz=640,
        candidate_conf=0.001,
        iou=0.7,
        max_det=300,
        class_map={0: "defect"},
    )
    backend.device = "cpu"
    crops = {view: tmp_path / f"{view}.png" for view in VIEWS}

    with pytest.raises(ValueError, match="non-positive box"):
        backend.predict(crops, tmp_path / "evidence")


class _PatchcoreBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, int]]] = []

    def predict(
        self,
        view: str,
        crop_path: Path,
        evidence_path: Path,
        *,
        diagnostic_mask_threshold: float = 0.65,
    ) -> ModelEvidence:
        """Write a deterministic fake PatchCore overlay."""
        image = cv2.imread(str(crop_path))
        assert image is not None
        self.calls.append((view, image.shape[:2]))
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(evidence_path), image)
        raw_path = evidence_path.parent / "raw_maps" / f"{view}.npy"
        mask_path = evidence_path.parent / "masks" / f"{view}.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        raw = np.arange(4, dtype=np.float32).reshape(2, 2)
        np.save(raw_path, raw, allow_pickle=False)
        assert cv2.imwrite(str(mask_path), np.full(image.shape[:2], 255, dtype=np.uint8))
        return ModelEvidence(
            score=0.1 + len(self.calls) / 100,
            evidence_path=evidence_path,
            patchcore_artifacts=PatchcoreArtifacts(
                raw_anomaly_map_path=raw_path,
                mask_path=mask_path,
                mask_source="diagnostic_anomaly_map",
                diagnostic_mask_threshold=diagnostic_mask_threshold,
                raw_anomaly_map_shape=raw.shape,
                mask_shape=image.shape[:2],
            ),
        )


class _YoloBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Path]] = []

    def predict(self, crops: dict[str, Path], evidence_dir: Path) -> dict[str, ModelEvidence]:
        """Write deterministic explicit-empty YOLO evidence."""
        self.calls.append(dict(crops))
        results: dict[str, ModelEvidence] = {}
        for view, crop_path in crops.items():
            image = cv2.imread(str(crop_path))
            assert image is not None
            evidence_path = evidence_dir / f"{view}.png"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(evidence_path), image)
            results[view] = ModelEvidence(score=0.0, evidence_path=evidence_path, detections=())
        return results


def test_load_runtime_config_accepts_right_only_top_level_eight_view_roi(tmp_path: Path) -> None:
    """The Stage 29 top-level ROI schema must load as the right-hand runtime ROI."""
    config_path, _ = _write_fixture(tmp_path)

    config = load_runtime_config(config_path)

    assert tuple(config.patchcore) == VIEWS
    assert tuple(config.patchcore_rois) == ("right",)
    assert tuple(config.patchcore_rois["right"]) == VIEWS


def test_roi_loader_apis_accept_exact_right_eight_view_configs(tmp_path: Path) -> None:
    config_path, _ = _write_fixture(tmp_path)
    runtime_payload = json.loads(config_path.read_text(encoding="utf-8"))

    _, _, patchcore_rois, _ = load_patchcore_roi_config(
        Path(runtime_payload["patchcore_roi_config"]),
        hands=("right",),
    )
    _, _, yolo_rois, _ = load_roi_config(Path(runtime_payload["yolo_roi_config"]))

    assert tuple(patchcore_rois) == ("right",)
    assert tuple(patchcore_rois["right"]) == VIEWS
    assert tuple(yolo_rois) == VIEWS
    assert PATCHCORE_ROI_VIEWS is CANONICAL_VIEWS
    assert YOLO_ROI_VIEWS is CANONICAL_VIEWS


def test_load_runtime_config_requires_exactly_eight_patchcore_models(tmp_path: Path) -> None:
    """A missing view must invalidate the model bundle."""
    config_path, _ = _write_fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["patchcore"].pop("back_right")
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        load_runtime_config(config_path)


def test_load_runtime_config_requires_exactly_eight_yolo_rois(tmp_path: Path) -> None:
    config_path, _ = _write_fixture(tmp_path)
    runtime_payload = json.loads(config_path.read_text(encoding="utf-8"))
    yolo_roi_path = Path(runtime_payload["yolo_roi_config"])
    roi_payload = json.loads(yolo_roi_path.read_text(encoding="utf-8"))
    roi_payload["views"]["unexpected"] = {"roi": [0, 0, 6, 5]}
    yolo_roi_path.write_text(json.dumps(roi_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly the eight views"):
        load_runtime_config(config_path)

    with pytest.raises(ValueError, match="exactly"):
        load_roi_config(yolo_roi_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("product", "OTHER", "product"),
        ("supported_hands", ["left"], "right hand"),
        ("supported_hands", ["right", "left"], "right hand"),
    ],
)
def test_load_runtime_config_requires_zs32_right_only(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    config_path, _ = _write_fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload[field] = value
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_runtime_config(config_path)


def test_patchcore_mask_prefers_pred_mask() -> None:
    """Anomalib's explicit prediction mask must take precedence over fallback thresholding."""
    anomaly = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    pred_mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    mask, source, threshold = _build_patchcore_mask(
        anomaly,
        pred_mask,
        output_shape=(4, 4),
        diagnostic_mask_threshold=0.65,
    )

    assert source == "pred_mask"
    assert threshold is None
    assert mask.dtype == np.uint8
    assert mask.shape == (4, 4)
    assert set(np.unique(mask)) <= {0, 255}


def test_patchcore_mask_uses_normalized_diagnostic_fallback() -> None:
    """Missing pred_mask must use the normalized display-only diagnostic fallback."""
    anomaly = np.array([[10.0, 16.4], [16.5, 20.0]], dtype=np.float32)

    mask, source, threshold = _build_patchcore_mask(
        anomaly,
        None,
        output_shape=(2, 2),
        diagnostic_mask_threshold=0.65,
    )

    assert source == "diagnostic_anomaly_map"
    assert threshold == pytest.approx(0.65)
    assert mask.tolist() == [[0, 0], [255, 255]]


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_patchcore_mask_rejects_nonfinite_raw_map(invalid_value: float) -> None:
    """Lossless raw maps must never silently repair NaN or infinity values."""
    anomaly = np.array([[0.0, invalid_value]], dtype=np.float32)

    with pytest.raises(ValueError, match=r"anomaly_map.*finite"):
        _build_patchcore_mask(
            anomaly,
            None,
            output_shape=(1, 2),
            diagnostic_mask_threshold=0.65,
        )


@pytest.mark.parametrize(
    "pred_mask",
    [np.array([0.0, 1.0]), np.zeros((2, 2, 2)), np.array([[0.0, float("nan")]])],
)
def test_patchcore_mask_rejects_malformed_present_pred_mask(pred_mask: np.ndarray) -> None:
    """A present but malformed pred_mask must fail instead of selecting the fallback."""
    with pytest.raises(ValueError, match="pred_mask"):
        _build_patchcore_mask(
            np.zeros((2, 2), dtype=np.float32),
            pred_mask,
            output_shape=(2, 2),
            diagnostic_mask_threshold=0.65,
        )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.01, 1.01])
def test_patchcore_mask_rejects_invalid_diagnostic_threshold(threshold: float) -> None:
    """Display thresholds must be finite probabilities."""
    with pytest.raises(ValueError, match="diagnostic_mask_threshold"):
        _build_patchcore_mask(
            np.zeros((2, 2), dtype=np.float32),
            None,
            output_shape=(2, 2),
            diagnostic_mask_threshold=threshold,
        )


def test_anomalib_backend_persists_unscaled_raw_map_and_binary_mask(tmp_path: Path) -> None:
    """The backend must save float32 model output before display resizing or normalization."""
    crop = tmp_path / "crop.png"
    assert cv2.imwrite(str(crop), np.zeros((5, 6, 3), dtype=np.uint8))
    raw = np.array([[10.0, 16.4], [16.5, 20.0]], dtype=np.float64)

    class _Engine:
        @staticmethod
        def predict(**_kwargs: object) -> list[SimpleNamespace]:
            return [SimpleNamespace(image_path=crop, pred_score=0.4, anomaly_map=raw, pred_mask=None)]

    backend = AnomalibPatchcoreBackend.__new__(AnomalibPatchcoreBackend)
    backend.engines = {"front": _Engine()}
    backend.models = {"front": object()}
    evidence_path = tmp_path / "evidence" / "patchcore" / "front.png"

    evidence = backend.predict(
        "front",
        crop,
        evidence_path,
        diagnostic_mask_threshold=0.65,
    )

    assert evidence.patchcore_artifacts is not None
    artifacts = evidence.patchcore_artifacts
    saved_raw = np.load(artifacts.raw_anomaly_map_path, allow_pickle=False)
    saved_mask = cv2.imread(str(artifacts.mask_path), cv2.IMREAD_GRAYSCALE)
    assert saved_raw.dtype == np.float32
    np.testing.assert_array_equal(saved_raw, raw.astype(np.float32))
    assert artifacts.raw_anomaly_map_shape == (2, 2)
    assert artifacts.mask_shape == (5, 6)
    assert saved_mask is not None
    assert saved_mask.shape == (5, 6)
    assert set(np.unique(saved_mask)) <= {0, 255}


def test_load_runtime_config_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    """A changed checkpoint must invalidate the model bundle."""
    config_path, checkpoints = _write_fixture(tmp_path)
    checkpoints["front"].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_runtime_config(config_path)


def test_runtime_writes_continuous_evidence_and_never_returns_ok_without_thresholds(tmp_path: Path) -> None:
    """Evidence-only inference must retain scores while remaining REVIEW."""
    config_path, _ = _write_fixture(tmp_path)
    images = _write_images(tmp_path)
    patchcore = _PatchcoreBackend()
    yolo = _YoloBackend()
    runtime = ZS32ModelRuntime(load_runtime_config(config_path), patchcore_backend=patchcore, yolo_backend=yolo)
    output_dir = tmp_path / "output"

    result = runtime.run(
        InspectionRequest("part-001", "session-001", "group-001", "right", images),
        output_dir,
    )

    assert result.machine_status == "REVIEW"
    assert result.inspection_complete is False
    assert len(patchcore.calls) == len(VIEWS)
    assert len(yolo.calls) == 1
    assert tuple(yolo.calls[0]) == VIEWS
    with (output_dir / "patchcore.csv").open(encoding="utf-8", newline="") as file:
        patchcore_rows = list(csv.DictReader(file))
    with (output_dir / "yolo.csv").open(encoding="utf-8", newline="") as file:
        yolo_rows = list(csv.DictReader(file))
    assert [row["branch"] for row in patchcore_rows] == [f"anomaly_{view}" for view in VIEWS]
    assert all(row["score"] for row in patchcore_rows)
    assert all(row["low_threshold"] == "" and row["high_threshold"] == "" for row in patchcore_rows)
    assert all(row["status"] == "ERROR" for row in patchcore_rows + yolo_rows)
    assert all(row["detections"] == "[]" for row in yolo_rows)
    assert all(row["score"] == "0.0" for row in yolo_rows)
    for row in patchcore_rows + yolo_rows:
        assert Path(row["source_path"]).is_file()
        assert Path(row["evidence_path"]).is_file()
        assert _sha256(Path(row["source_path"])) == row["source_hash"]
        assert _sha256(Path(row["evidence_path"])) == row["evidence_hash"]
    manifest = json.loads((output_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["views"]) == set(VIEWS)
    assert manifest["views"]["front"]["model_supported"] is True
    assert manifest["views"]["front"]["patchcore"]["mask_source"] == "diagnostic_anomaly_map"
    assert manifest["views"]["front"]["patchcore"]["display_only"] is True
    assert manifest["views"]["front"]["patchcore"]["roi_xyxy"] == [0, 0, 6, 5]
    assert manifest["views"]["front"]["patchcore"]["raw_anomaly_map_shape"] == [2, 2]
    assert manifest["views"]["front"]["patchcore"]["mask_shape"] == [5, 6]
    assert Path(manifest["views"]["front"]["patchcore"]["raw_anomaly_map_path"]).is_file()
    assert Path(manifest["views"]["front"]["patchcore"]["mask_path"]).is_file()


def test_runtime_rejects_left_and_existing_output_before_backend_calls(tmp_path: Path) -> None:
    """Unsupported identity and mutable output targets must fail before loading."""
    config_path, _ = _write_fixture(tmp_path)
    images = _write_images(tmp_path)
    patchcore = _PatchcoreBackend()
    yolo = _YoloBackend()
    runtime = ZS32ModelRuntime(load_runtime_config(config_path), patchcore_backend=patchcore, yolo_backend=yolo)

    with pytest.raises(ValueError, match="unsupported hand"):
        runtime.run(InspectionRequest("p", "s", "g", "left", images), tmp_path / "left-output")
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        runtime.run(InspectionRequest("p", "s", "g", "right", images), tmp_path / "exists")

    assert patchcore.calls == []
    assert yolo.calls == []


def test_runtime_fails_closed_on_malformed_patchcore_artifact_but_retains_diagnostic_score(tmp_path: Path) -> None:
    """Invalid mask evidence must not reach Stage18 even when its model score is finite."""
    config_path, _ = _write_fixture(tmp_path)
    images = _write_images(tmp_path)

    class _MalformedMaskBackend(_PatchcoreBackend):
        def predict(
            self,
            view: str,
            crop_path: Path,
            evidence_path: Path,
            *,
            diagnostic_mask_threshold: float = 0.65,
        ) -> ModelEvidence:
            evidence = super().predict(
                view,
                crop_path,
                evidence_path,
                diagnostic_mask_threshold=diagnostic_mask_threshold,
            )
            if view == "front":
                assert evidence.patchcore_artifacts is not None
                assert cv2.imwrite(
                    str(evidence.patchcore_artifacts.mask_path),
                    np.full((5, 6), 127, dtype=np.uint8),
                )
            return evidence

    runtime = ZS32ModelRuntime(
        load_runtime_config(config_path),
        patchcore_backend=_MalformedMaskBackend(),
        yolo_backend=_YoloBackend(),
    )
    output_dir = tmp_path / "malformed-mask"

    runtime.run(InspectionRequest("p", "s", "g", "right", images), output_dir)

    with (output_dir / "patchcore.csv").open(encoding="utf-8", newline="") as file:
        front_row = next(row for row in csv.DictReader(file) if row["view"] == "front")
    manifest = json.loads((output_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    assert front_row["score"] == ""
    assert front_row["status"] == "ERROR"
    assert manifest["views"]["front"]["patchcore"]["score"] == pytest.approx(0.11)
    assert manifest["views"]["front"]["patchcore"]["status"] == "error"
    assert "binary" in manifest["views"]["front"]["patchcore"]["reason"]


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf"), "not-a-score"])
def test_runtime_omits_invalid_diagnostic_score_from_strict_manifest(
    tmp_path: Path,
    invalid_score: object,
) -> None:
    """Invalid model scores must fail closed without leaking non-standard JSON constants."""
    config_path, _ = _write_fixture(tmp_path)
    images = _write_images(tmp_path)

    class _InvalidScoreBackend(_PatchcoreBackend):
        def predict(
            self,
            view: str,
            crop_path: Path,
            evidence_path: Path,
            *,
            diagnostic_mask_threshold: float = 0.65,
        ) -> ModelEvidence:
            evidence = super().predict(
                view,
                crop_path,
                evidence_path,
                diagnostic_mask_threshold=diagnostic_mask_threshold,
            )
            if view != "front":
                return evidence
            return ModelEvidence(
                score=cast("float", invalid_score),
                evidence_path=evidence.evidence_path,
                patchcore_artifacts=evidence.patchcore_artifacts,
            )

    runtime = ZS32ModelRuntime(
        load_runtime_config(config_path),
        patchcore_backend=_InvalidScoreBackend(),
        yolo_backend=_YoloBackend(),
    )
    output_dir = tmp_path / "invalid-score"

    runtime.run(InspectionRequest("p", "s", "g", "right", images), output_dir)

    manifest_text = (output_dir / "runtime_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(
        manifest_text,
        parse_constant=lambda value: pytest.fail(f"non-standard JSON constant: {value}"),
    )
    with (output_dir / "patchcore.csv").open(encoding="utf-8", newline="") as file:
        front_row = next(row for row in csv.DictReader(file) if row["view"] == "front")
    assert manifest["views"]["front"]["patchcore"]["score"] is None
    assert manifest["views"]["front"]["patchcore"]["status"] == "error"
    assert front_row["score"] == ""
    assert front_row["status"] == "ERROR"


def test_runtime_rejects_duplicate_source_images_and_wrong_dimensions(tmp_path: Path) -> None:
    """Invalid eight-view captures must fail before model execution."""
    config_path, _ = _write_fixture(tmp_path)
    images = _write_images(tmp_path)
    runtime = ZS32ModelRuntime(
        load_runtime_config(config_path),
        patchcore_backend=_PatchcoreBackend(),
        yolo_backend=_YoloBackend(),
    )
    duplicate_images = dict(images)
    duplicate_images["back_right"] = duplicate_images["front"]

    with pytest.raises(ValueError, match="distinct"):
        runtime.run(InspectionRequest("p", "s", "g", "right", duplicate_images), tmp_path / "duplicate")

    assert cv2.imwrite(str(images["back_right"]), np.zeros((9, 12, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="dimensions"):
        runtime.run(InspectionRequest("p", "s", "g", "right", images), tmp_path / "wrong-size")


def test_runtime_publishes_review_diagnostics_when_model_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model-loader failure must still create a non-releasable evidence generation."""
    config_path, _ = _write_fixture(tmp_path)
    images = _write_images(tmp_path)
    runtime = ZS32ModelRuntime(load_runtime_config(config_path))

    def fail_initialization() -> None:
        msg = "synthetic loader failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(runtime, "_ensure_backends", fail_initialization)
    output_dir = tmp_path / "loader-failure"

    result = runtime.run(InspectionRequest("p", "s", "g", "right", images), output_dir)

    assert result.machine_status == "REVIEW"
    assert result.inspection_complete is False
    assert any("model initialization" in error for error in result.errors)
    assert len(list((output_dir / "evidence" / "patchcore").glob("*.error.json"))) == len(VIEWS)
    assert len(list((output_dir / "evidence" / "yolo").glob("*.error.json"))) == len(VIEWS)
