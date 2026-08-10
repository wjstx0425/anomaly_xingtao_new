"""Contracts for the independent BMW eight-view laboratory Demo."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    DemoFinalStatus,
    fuse_demo_status,
    load_capture_directory,
    load_demo_config,
    load_manifest_sample,
)


def _branch_result(status: BranchStatus) -> DemoBranchResult:
    return DemoBranchResult(
        branch=DemoBranch.TEMPLATE,
        view_id="front",
        status=status,
        score=0.1,
        threshold=0.2,
        elapsed_ms=1.0,
        reason="test",
        overlay=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def _write_demo_assets(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    roi = tmp_path / "roi.json"
    capture = tmp_path / "capture.json"
    run = tmp_path / "run"
    roi.write_text("{}", encoding="utf-8")
    capture.write_text("{}", encoding="utf-8")
    for view in VIEW_ORDER:
        model = run / "template" / view / "model.json"
        checkpoint = run / "efficientad" / view / "model.ckpt"
        model.parent.mkdir(parents=True)
        checkpoint.parent.mkdir(parents=True)
        model.write_text("{}", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
    bright = tmp_path / "bright_streak_asset/calibrated_config.json"
    yolo = run / "yolo/train/weights/best.pt"
    bright.parent.mkdir(parents=True)
    yolo.parent.mkdir(parents=True)
    bright.write_text("{}", encoding="utf-8")
    yolo.write_bytes(b"checkpoint")
    threshold_artifact = tmp_path / "part_thresholds.json"
    threshold_artifact.write_text(
        json.dumps(
            {
                "thresholds": {view: (index + 1) / 10 for index, view in enumerate(VIEW_ORDER)},
                "target_part_fpr": 0.05,
                "allowed_normal_false_positive_count": 1,
                "normal_part_count": 20,
                "normal_false_positive_count": 1,
                "observed_normal_part_fpr": 0.05,
                "defect_part_count": 6,
                "defect_detected_count": 6,
                "defect_image_hits": 10,
                "demo_only": True,
                "test_used_for_selection": True,
                "source_csv": "/retired-calibration-host/efficientad_scores.csv",
                "source_csv_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    return roi, capture, run, threshold_artifact


def _demo_payload(tmp_path: Path, *, threshold_artifact: Path) -> dict[str, object]:
    roi = tmp_path / "roi.json"
    capture = tmp_path / "capture.json"
    run = tmp_path / "run"
    return {
        "schema_version": 1,
        "demo_id": "bmw-eight-view-demo-test",
        "capture_config": str(capture),
        "roi_config": str(roi),
        "prepared_manifest": str(tmp_path / "manifest.csv"),
        "training_run": str(run),
        "result_root": str(tmp_path / "results"),
        "bright_streak": {"config": str(tmp_path / "bright_streak_asset/calibrated_config.json")},
        "efficientad": {"threshold_artifact": str(threshold_artifact)},
        "yolo": {"candidate_conf": 0.1, "final_threshold": 0.25, "imgsz": 640},
    }


def test_fusion_runs_as_all_pass_ng_or_error() -> None:
    assert fuse_demo_status((_branch_result(BranchStatus.PASS),)) is DemoFinalStatus.OK
    assert fuse_demo_status((_branch_result(BranchStatus.NG),)) is DemoFinalStatus.NG
    assert fuse_demo_status((_branch_result(BranchStatus.ERROR),)) is DemoFinalStatus.ERROR


def test_capture_directory_requires_exactly_eight_named_images(tmp_path: Path) -> None:
    for view in VIEW_ORDER:
        assert cv2.imwrite(str(tmp_path / f"{view}.png"), np.full((12, 16, 3), 20, dtype=np.uint8))

    images = load_capture_directory(tmp_path)

    assert tuple(images) == VIEW_ORDER
    assert all(image.shape == (12, 16, 3) for image in images.values())


def test_manifest_sample_loads_all_views_in_canonical_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = []
    for index, view in enumerate(reversed(VIEW_ORDER)):
        image_path = tmp_path / f"{view}.png"
        assert cv2.imwrite(str(image_path), np.full((8, 12, 3), index, dtype=np.uint8))
        rows.append({"sample_id": "sample-1", "view_id": view, "source_path": str(image_path)})
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "view_id", "source_path"))
        writer.writeheader()
        writer.writerows(rows)

    images = load_manifest_sample(manifest, "sample-1")

    assert tuple(images) == VIEW_ORDER
    assert all(image.shape == (8, 12, 3) for image in images.values())


def test_demo_config_resolves_current_model_assets(tmp_path: Path) -> None:
    roi, capture, run, threshold_artifact = _write_demo_assets(tmp_path)
    yolo = run / "yolo/train/weights/best.pt"
    bright = tmp_path / "bright_streak_asset/calibrated_config.json"
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )

    config = load_demo_config(config_path)

    assert config.views == VIEW_ORDER
    assert config.yolo_checkpoint == yolo.resolve()
    assert config.bright_streak_config == bright.resolve()
    assert tuple(config.template_models) == VIEW_ORDER
    assert tuple(config.efficientad_checkpoints) == VIEW_ORDER
    assert tuple(config.efficientad_thresholds) == VIEW_ORDER
    assert tuple(config.efficientad_thresholds.values()) == pytest.approx(
        tuple((index + 1) / 10 for index in range(len(VIEW_ORDER)))
    )
    assert config.efficientad_threshold_source_csv == "/retired-calibration-host/efficientad_scores.csv"
    assert config.efficientad_threshold_source_csv_sha256 == "a" * 64


@pytest.mark.parametrize("invalid", [None, "", "   ", 123])
def test_demo_config_rejects_missing_or_invalid_efficientad_source_csv(tmp_path: Path, invalid: object) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    payload = json.loads(threshold_artifact.read_text(encoding="utf-8"))
    if invalid is None:
        del payload["source_csv"]
    else:
        payload["source_csv"] = invalid
    threshold_artifact.write_text(json.dumps(payload), encoding="utf-8")
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_csv"):
        load_demo_config(config_path)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "a" * 63,
        "A" * 64,
        "g" * 64,
        123,
    ],
)
def test_demo_config_rejects_invalid_efficientad_source_sha256(tmp_path: Path, invalid: object) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    payload = json.loads(threshold_artifact.read_text(encoding="utf-8"))
    if invalid is None:
        del payload["source_csv_sha256"]
    else:
        payload["source_csv_sha256"] = invalid
    threshold_artifact.write_text(json.dumps(payload), encoding="utf-8")
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_csv_sha256"):
        load_demo_config(config_path)


@pytest.mark.parametrize(
    "bright_streak",
    [
        {},
        {"path": "calibrated_config.json"},
        {"config": "calibrated_config.json", "unexpected": True},
    ],
)
def test_demo_config_rejects_non_exact_bright_streak_asset_fields(
    tmp_path: Path,
    bright_streak: dict[str, object],
) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = bright_streak
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bright_streak配置字段"):
        load_demo_config(config_path)


def test_demo_config_fails_closed_when_bright_streak_asset_is_missing(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {"config": str(tmp_path / "missing_bright_streak.json")}
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bright_streak模型不存在"):
        load_demo_config(config_path)


def test_repository_demo_config_selects_the_bold_bright_streak_asset() -> None:
    config_path = Path(__file__).resolve().parents[4] / "configs/bmw/experiments/bmw_eight_view_demo_v1.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["bright_streak"] == {
        "config": (
            "../../../results/bmw_lab_one_click/bmw_lab_eight_view_v1/"
            "bright_streak_ridge_v4_bold/calibrated_config.json"
        )
    }


def test_demo_config_rejects_efficientad_threshold_artifact_missing_view(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    payload = json.loads(threshold_artifact.read_text(encoding="utf-8"))
    del payload["thresholds"]["back_secondary"]
    threshold_artifact.write_text(json.dumps(payload), encoding="utf-8")
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="八个视角"):
        load_demo_config(config_path)


@pytest.mark.parametrize("flag", ["demo_only", "test_used_for_selection"])
def test_demo_config_requires_explicit_efficientad_leakage_flags(tmp_path: Path, flag: str) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    payload = json.loads(threshold_artifact.read_text(encoding="utf-8"))
    payload[flag] = False
    threshold_artifact.write_text(json.dumps(payload), encoding="utf-8")
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=flag):
        load_demo_config(config_path)


@pytest.mark.parametrize("invalid", [True, "0.5", float("nan")])
def test_demo_config_rejects_non_numeric_efficientad_thresholds(tmp_path: Path, invalid: object) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    payload = json.loads(threshold_artifact.read_text(encoding="utf-8"))
    payload["thresholds"]["front"] = invalid
    threshold_artifact.write_text(json.dumps(payload), encoding="utf-8")
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="有限数值"):
        load_demo_config(config_path)


def test_demo_config_rejects_candidate_threshold_above_final(tmp_path: Path) -> None:
    path = tmp_path / "demo.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "demo_id": "bad",
                "capture_config": "missing.json",
                "roi_config": "missing.json",
                "prepared_manifest": "missing.csv",
                "training_run": "missing",
                "result_root": "results",
                "bright_streak": {"config": "missing.json"},
                "efficientad": {"threshold_artifact": "missing.json"},
                "yolo": {"candidate_conf": 0.5, "final_threshold": 0.25, "imgsz": 640},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate_conf"):
        load_demo_config(path)
