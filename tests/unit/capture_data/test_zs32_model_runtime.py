# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the six-view ZS32 PatchCore and YOLO runtime."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from capture_data.zs32_inspection_orchestrator import InspectionRequest
from capture_data.zs32_model_runtime import ModelEvidence, ZS32ModelRuntime, load_runtime_config

VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    patchcore_roi = {
        "schema_version": 1,
        "coordinate_system": "pixel_xyxy_half_open",
        "image_size": {"width": 12, "height": 10},
        "hands": {hand: {"views": {view: {"roi": [0, 0, 6, 5]} for view in VIEWS}} for hand in ("right", "left")},
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
        "profile": "zs32_right_six_view_v1",
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


class _PatchcoreBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, int]]] = []

    def predict(self, view: str, crop_path: Path, evidence_path: Path) -> ModelEvidence:
        """Write a deterministic fake PatchCore overlay."""
        image = cv2.imread(str(crop_path))
        assert image is not None
        self.calls.append((view, image.shape[:2]))
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(evidence_path), image)
        return ModelEvidence(score=0.1 + len(self.calls) / 100, evidence_path=evidence_path)


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


def test_load_runtime_config_requires_exactly_six_patchcore_models(tmp_path: Path) -> None:
    """A missing view must invalidate the model bundle."""
    config_path, _ = _write_fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["patchcore"].pop("back_right")
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        load_runtime_config(config_path)


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
    assert len(patchcore.calls) == 6
    assert len(yolo.calls) == 1
    assert set(yolo.calls[0]) == set(VIEWS)
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


def test_runtime_rejects_duplicate_source_images_and_wrong_dimensions(tmp_path: Path) -> None:
    """Invalid six-view captures must fail before model execution."""
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
    assert len(list((output_dir / "evidence" / "patchcore").glob("*.error.json"))) == 6
    assert len(list((output_dir / "evidence" / "yolo").glob("*.error.json"))) == 6
