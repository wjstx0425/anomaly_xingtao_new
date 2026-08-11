# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Atomic evidence publication tests for BMW laboratory inspection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.config import load_experiment_config
from bmw_inspection.lab.contracts import (
    BranchEvidence,
    BranchName,
    BranchStatus,
    CaptureSet,
    CapturedView,
    FinalStatus,
    InspectionResult,
    ViewId,
)
from bmw_inspection.lab.publisher import EvidencePublisher, PublicationArtifacts
from bmw_inspection.lab.yolo import DetectionBox, YoloEvidence


REPO_ROOT = Path(__file__).parents[4]


@dataclass(frozen=True, slots=True)
class _RawDecision:
    raw_anomaly_map: np.ndarray
    display_heatmap: np.ndarray


@dataclass(frozen=True, slots=True)
class _TemplateDecision:
    threshold: float
    best_template_path: Path
    offset_x: int = 0
    offset_y: int = 0


@dataclass(frozen=True, slots=True)
class _BrightDecision:
    roi_xyxy: tuple[int, int, int, int]
    roi: np.ndarray
    response: np.ndarray
    mask: np.ndarray
    mask_used_for_metrics: np.ndarray


def _capture_set() -> CaptureSet:
    now = datetime.now()
    return CaptureSet(
        capture_set_id="capture-publish",
        views={
            view_id: CapturedView(view_id, f"serial-{view_id.value}", np.full((8, 10), 30, np.uint8), now)
            for view_id in ViewId
        },
        created_at=now,
    )


def _result(capture_set: CaptureSet, evidence: tuple[BranchEvidence, ...]) -> InspectionResult:
    return InspectionResult(capture_set, evidence, FinalStatus.NG_YOLO, "yolo triggered", True)


def _config(tmp_path: Path):
    loaded = load_experiment_config(REPO_ROOT / "configs/bmw/experiments/bmw_lab_v1.json")
    return replace(loaded, result_root=tmp_path / "results")


def _row(branch: BranchName, view_id: ViewId, status: BranchStatus = BranchStatus.PASS) -> BranchEvidence:
    return BranchEvidence(branch, view_id, status, True, 0.1, 0.5, 1.0, status.value, "model-sha", {})


def test_publisher_stages_then_atomically_renames_complete_raw_evidence(tmp_path: Path) -> None:
    capture_set = _capture_set()
    candidate = DetectionBox(1, 1, 4, 4, 0.30)
    final = DetectionBox(2, 2, 5, 5, 0.80)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    yolo = YoloEvidence(
        branch=BranchName.YOLO,
        view_id=ViewId.FRONT,
        status=BranchStatus.NG,
        required_for_ok=True,
        score=0.80,
        threshold=0.50,
        elapsed_ms=2.0,
        reason="one final detection",
        model_id=checkpoint_sha,
        artifact_paths={"checkpoint": str(checkpoint)},
        candidates=(candidate, final),
        final_boxes=(final,),
        roi_xyxy=(0, 0, 10, 8),
    )
    raw_map = np.arange(12, dtype=np.float32).reshape(3, 4)
    heatmap = np.full((3, 4, 3), 90, np.uint8)
    artifacts = PublicationArtifacts(
        decisions={(BranchName.PATCHCORE, ViewId.BACK): _RawDecision(raw_map, heatmap)},
        crops={
            (BranchName.YOLO, ViewId.FRONT): np.full((3, 4), 55, np.uint8),
            (BranchName.PATCHCORE, ViewId.BACK): np.full((3, 4), 55, np.uint8),
        },
    )
    owned_decision = artifacts.decisions[(BranchName.PATCHCORE, ViewId.BACK)]
    assert isinstance(owned_decision, _RawDecision)
    with pytest.raises(ValueError):
        owned_decision.raw_anomaly_map[0, 0] = 99
    publisher = EvidencePublisher(_config(tmp_path).result_root)

    published = publisher.publish(
        _result(
            capture_set,
            (
                yolo,
                _row(BranchName.PATCHCORE, ViewId.BACK),
            ),
        ),
        config=_config(tmp_path),
        triggered_branches=(BranchName.YOLO,),
        artifacts=artifacts,
        run_id="run-atomic",
    )

    assert published.run_dir.is_dir()
    assert published.run_dir.parent.name == capture_set.created_at.strftime("%Y%m%d")
    assert not list(published.run_dir.parent.glob(".run-atomic.staging-*"))
    assert all((published.run_dir / "source" / f"{view_id.value}.png").is_file() for view_id in ViewId)
    assert np.array_equal(
        np.load(published.run_dir / "evidence/patchcore/back/anomaly_map.npy", allow_pickle=False),
        raw_map,
    )
    candidates = json.loads(
        (published.run_dir / "evidence/yolo/front/candidates.json").read_text(encoding="utf-8")
    )
    assert len(candidates["candidates"]) == 2
    assert len(candidates["final_boxes"]) == 1
    assert (published.run_dir / "crops/patchcore/back.png").is_file()
    assert json.loads((published.run_dir / "result.json").read_text(encoding="utf-8"))["triggered_branches"] == [
        "yolo"
    ]
    identities = json.loads((published.run_dir / "model_identities.json").read_text(encoding="utf-8"))
    yolo_identity = next(item for item in identities["evidence"] if item["branch"] == "yolo")
    assert yolo_identity["path"] == str(checkpoint)
    assert yolo_identity["sha256"] == checkpoint_sha
    assert any(item["branch"] == "bright_streak" for item in identities["config_assets"])
    with pytest.raises(Exception):
        published.run_dir = tmp_path  # type: ignore[misc]


def test_publication_failure_never_exposes_partial_final_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_set = _capture_set()
    result = InspectionResult(capture_set, (), FinalStatus.RETAKE, "blurred", False)
    publisher = EvidencePublisher(_config(tmp_path).result_root)
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="source image"):
        publisher.publish(
            result,
            config=_config(tmp_path),
            triggered_branches=(),
            artifacts=PublicationArtifacts(),
            run_id="run-failed",
        )

    run_parent = (
        _config(tmp_path).result_root
        / "bmw-lab-v1/runs"
        / capture_set.created_at.strftime("%Y%m%d")
    )
    assert not (run_parent / "run-failed").exists()
    assert not list(run_parent.glob(".run-failed.staging-*"))


def test_visual_evidence_is_decision_derived_and_fusion_is_annotated(tmp_path: Path) -> None:
    capture_set = _capture_set()
    template_path = tmp_path / "best-template.png"
    template = np.arange(12, dtype=np.uint8).reshape(3, 4) * 10
    assert cv2.imwrite(str(template_path), template)
    crop = np.flipud(template).copy()
    response = np.arange(12, dtype=np.float32).reshape(3, 4)
    mask = np.zeros((3, 4), np.uint8)
    mask[:, 1:3] = 255
    used = np.zeros((3, 4), np.uint8)
    used[1:, 1:3] = 255
    heatmap = np.zeros((3, 4, 3), np.uint8)
    heatmap[..., 2] = 255
    evidence = (
        _row(BranchName.TEMPLATE, ViewId.FRONT, BranchStatus.NG),
        _row(BranchName.BRIGHT_STREAK, ViewId.FRONT_LEFT),
        _row(BranchName.PATCHCORE, ViewId.BACK),
    )
    result = InspectionResult(capture_set, evidence, FinalStatus.NG_TEMPLATE, "template triggered", True)
    artifacts = PublicationArtifacts(
        decisions={
            (BranchName.TEMPLATE, ViewId.FRONT): _TemplateDecision(0.5, template_path),
            (BranchName.BRIGHT_STREAK, ViewId.FRONT_LEFT): _BrightDecision(
                (0, 0, 4, 3), crop, response, mask, used
            ),
            (BranchName.PATCHCORE, ViewId.BACK): _RawDecision(response, heatmap),
        },
        crops={
            (BranchName.TEMPLATE, ViewId.FRONT): crop,
            (BranchName.BRIGHT_STREAK, ViewId.FRONT_LEFT): crop,
            (BranchName.PATCHCORE, ViewId.BACK): crop,
        },
    )

    published = EvidencePublisher(_config(tmp_path).result_root).publish(
        result,
        config=_config(tmp_path),
        triggered_branches=(BranchName.TEMPLATE,),
        artifacts=artifacts,
        run_id="visual-evidence",
    )

    assert (published.run_dir / "evidence/template/front/overlay.png").is_file()
    difference = cv2.imread(str(published.run_dir / "evidence/template/front/difference.png"), cv2.IMREAD_GRAYSCALE)
    assert difference is not None and np.any(difference)
    bright_dir = published.run_dir / "evidence/bright_streak/front_left"
    bright_files = ("roi.png", "response.png", "mask.png", "mask_used_for_metrics.png", "overlay.png")
    assert all((bright_dir / name).is_file() for name in bright_files)
    patch_overlay = cv2.imread(str(published.run_dir / "evidence/patchcore/back/overlay.png"))
    assert patch_overlay is not None and patch_overlay.shape[:2] == capture_set.views[ViewId.BACK].image.shape[:2]
    assert not np.all(patch_overlay == 90)
    fusion = cv2.imread(str(published.run_dir / "evidence/fusion/front.png"))
    source = cv2.imread(str(published.run_dir / "source/front.png"))
    assert fusion is not None and source is not None and not np.array_equal(fusion, source)
    assert int(fusion[..., 2].max()) == 255


def test_publisher_rejects_trigger_and_artifact_mismatches_before_staging(tmp_path: Path) -> None:
    capture_set = _capture_set()
    yolo_ng = _row(BranchName.YOLO, ViewId.FRONT, BranchStatus.NG)
    result = _result(capture_set, (yolo_ng,))
    publisher = EvidencePublisher(_config(tmp_path).result_root)

    with pytest.raises(ValueError, match="triggered_branches"):
        publisher.publish(
            result,
            config=_config(tmp_path),
            triggered_branches=(),
            artifacts=PublicationArtifacts(),
            run_id="bad-trigger",
        )
    with pytest.raises(ValueError, match="artifact"):
        publisher.publish(
            result,
            config=_config(tmp_path),
            triggered_branches=(BranchName.YOLO,),
            artifacts=PublicationArtifacts(
                crops={(BranchName.PATCHCORE, ViewId.BACK): np.zeros((2, 2), np.uint8)}
            ),
            run_id="bad-artifact",
        )
    with pytest.raises(ValueError, match="threshold contradicts"):
        publisher.publish(
            result,
            config=_config(tmp_path),
            triggered_branches=(BranchName.YOLO,),
            artifacts=PublicationArtifacts(
                decisions={(BranchName.YOLO, ViewId.FRONT): _TemplateDecision(0.2, tmp_path / "unused")}
            ),
            run_id="bad-threshold",
        )

    run_parent = (
        _config(tmp_path).result_root
        / "bmw-lab-v1/runs"
        / capture_set.created_at.strftime("%Y%m%d")
    )
    assert not run_parent.exists()
