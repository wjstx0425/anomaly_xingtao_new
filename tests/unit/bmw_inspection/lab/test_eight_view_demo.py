"""Contracts for the independent BMW eight-view laboratory Demo."""

from __future__ import annotations

import json
import csv
import hashlib
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
    EightViewInspection,
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
                "checkpoint_sha256_by_view": {
                    view: hashlib.sha256((run / "efficientad" / view / "model.ckpt").read_bytes()).hexdigest()
                    for view in VIEW_ORDER
                },
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
        "efficientad": {
            "threshold_artifact": str(threshold_artifact),
            "threshold_artifact_sha256": hashlib.sha256(threshold_artifact.read_bytes()).hexdigest(),
        },
        "yolo": {"candidate_conf": 0.1, "final_threshold": 0.25, "imgsz": 640},
    }


def test_fusion_runs_as_all_pass_ng_or_error() -> None:
    assert fuse_demo_status((_branch_result(BranchStatus.PASS),)) is DemoFinalStatus.OK
    assert fuse_demo_status((_branch_result(BranchStatus.NG),)) is DemoFinalStatus.NG
    assert fuse_demo_status((_branch_result(BranchStatus.ERROR),)) is DemoFinalStatus.ERROR


def test_inspection_trusted_ok_diagnostics_default_to_empty_immutable_mappings() -> None:
    images = {view: np.zeros((8, 8, 3), dtype=np.uint8) for view in VIEW_ORDER}
    inspection = EightViewInspection(
        capture_id="sample",
        images=images,
        results=(_branch_result(BranchStatus.PASS),),
        final_status=DemoFinalStatus.OK,
        elapsed_ms=1.0,
    )

    assert dict(inspection.trusted_ok_by_comparison) == {}
    assert dict(inspection.roi_images) == {}
    assert dict(inspection.diagnostic_metadata) == {}
    with pytest.raises(TypeError):
        inspection.trusted_ok_by_comparison[("front", "roi")] = object()  # type: ignore[index]


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
    assert config.bright_streak_engine == "calibrated_rule_v1"
    assert tuple(config.template_models) == VIEW_ORDER
    assert tuple(config.efficientad_checkpoints) == VIEW_ORDER
    assert tuple(config.efficientad_thresholds) == VIEW_ORDER
    assert tuple(config.efficientad_thresholds.values()) == pytest.approx(
        tuple((index + 1) / 10 for index in range(len(VIEW_ORDER)))
    )
    assert config.efficientad_ignore_mask_index is None
    assert config.efficientad_ignore_mask_index_sha256 is None
    assert config.efficientad_threshold_source_csv == "/retired-calibration-host/efficientad_scores.csv"
    assert config.efficientad_threshold_source_csv_sha256 == "a" * 64
    assert config.efficientad_base_thresholds == config.efficientad_thresholds
    assert config.efficientad_threshold_margin == pytest.approx(0.0)
    assert config.trusted_ok_reference_index is None
    assert config.trusted_ok_reference_index_sha256 is None
    assert config.trusted_ok_reference_error is None


def test_demo_config_accepts_sha_bound_efficientad_ignore_mask_index(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    ignore_index = tmp_path / "ignore_masks/index.json"
    ignore_index.parent.mkdir()
    ignore_index.write_text('{"schema_version":"bmw.efficientad_manual_ignore_masks/1.0"}', encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["efficientad"] = {
        **payload["efficientad"],
        "ignore_mask_index": str(ignore_index),
        "ignore_mask_index_sha256": hashlib.sha256(ignore_index.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(config_path)

    assert config.efficientad_ignore_mask_index == ignore_index.resolve()
    assert config.efficientad_ignore_mask_index_sha256 == hashlib.sha256(ignore_index.read_bytes()).hexdigest()


def test_demo_config_accepts_sha_bound_template_mask_and_calibration_thresholds(tmp_path: Path) -> None:
    _roi, _capture, run, efficientad_thresholds = _write_demo_assets(tmp_path)
    mask_index = tmp_path / "mask_index.json"
    mask_index.write_text("{}", encoding="utf-8")
    mask_sha = hashlib.sha256(mask_index.read_bytes()).hexdigest()
    artifact = tmp_path / "template_masked_thresholds.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "bmw.template_manual_ignore_thresholds/1.0",
                "selection_split": "calibration",
                "final_test_used_for_selection": False,
                "manual_ignore_mask_index_sha256": mask_sha,
                "thresholds": {view: 0.01 for view in VIEW_ORDER},
                "views": {
                    view: {
                        "model_json_sha256": hashlib.sha256(
                            (run / "template" / view / "model.json").read_bytes()
                        ).hexdigest()
                    }
                    for view in VIEW_ORDER
                },
            }
        ),
        encoding="utf-8",
    )
    payload = _demo_payload(tmp_path, threshold_artifact=efficientad_thresholds)
    payload["efficientad"] = {
        **payload["efficientad"],
        "ignore_mask_index": str(mask_index),
        "ignore_mask_index_sha256": mask_sha,
    }
    payload["template"] = {
        "ignore_mask_index": str(mask_index),
        "ignore_mask_index_sha256": mask_sha,
        "threshold_artifact": str(artifact),
        "threshold_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(config_path)

    assert config.template_ignore_mask_index == mask_index.resolve()
    assert config.template_ignore_mask_index_sha256 == mask_sha
    assert config.template_masked_threshold_artifact == artifact.resolve()
    assert config.template_masked_threshold_artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert config.template_masked_thresholds == pytest.approx({view: 0.01 for view in VIEW_ORDER})


def test_demo_config_rejects_tampered_efficientad_ignore_mask_index(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    ignore_index = tmp_path / "ignore_masks/index.json"
    ignore_index.parent.mkdir()
    ignore_index.write_text("{}", encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["efficientad"] = {
        **payload["efficientad"],
        "ignore_mask_index": str(ignore_index),
        "ignore_mask_index_sha256": "0" * 64,
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ignore mask index SHA256不匹配"):
        load_demo_config(config_path)


def test_demo_config_loads_explicit_sha_bound_trusted_ok_index(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    release = tmp_path / "bmw_right_20260810_21_train_normal_approved_v2"
    release.mkdir()
    index = release / "reference_index.json"
    index.write_text('{"schema_version": 1}\n', encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["trusted_ok_reference"] = {
        "index": str(index),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(config_path)

    assert config.trusted_ok_reference_index == index.resolve()
    assert config.trusted_ok_reference_index_sha256 == hashlib.sha256(index.read_bytes()).hexdigest()
    assert config.trusted_ok_reference_error is None


def test_trusted_ok_relative_index_resolves_with_each_checkout_root(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    expected_paths = []
    for checkout_name in ("root-checkout", "worktree-checkout"):
        checkout = tmp_path / checkout_name
        config_dir = checkout / "configs/bmw/experiments"
        config_dir.mkdir(parents=True)
        release = checkout / "dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v2"
        release.mkdir(parents=True)
        index = release / "reference_index.json"
        index.write_text('{"schema_version": 1}\n', encoding="utf-8")
        payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
        payload["trusted_ok_reference"] = {
            "index": (
                "../../../dataset/bmw_trusted_ok_reference/"
                "bmw_right_20260810_21_train_normal_approved_v2/reference_index.json"
            ),
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        }
        config_path = config_dir / "demo.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        expected_paths.append(load_demo_config(config_path).trusted_ok_reference_index)

    assert expected_paths == [
        (tmp_path / "root-checkout/dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v2/reference_index.json").resolve(),
        (tmp_path / "worktree-checkout/dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v2/reference_index.json").resolve(),
    ]


def test_demo_config_records_trusted_ok_sha_failure_without_blocking_detector_config(
    tmp_path: Path,
) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    release = tmp_path / "bmw_right_20260810_21_train_normal_approved_v2"
    release.mkdir()
    index = release / "reference_index.json"
    index.write_text("{}\n", encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["trusted_ok_reference"] = {"index": str(index), "index_sha256": "a" * 64}
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(config_path)

    assert config.trusted_ok_reference_index == index.resolve()
    assert config.trusted_ok_reference_error == "可信OK参考索引SHA256不匹配"


@pytest.mark.parametrize(
    "trusted",
    [
        None,
        {},
        {"index": "reference_index.json"},
        {"index": "reference_index.json", "index_sha256": "A" * 64},
        {"index": "wrong-name.json", "index_sha256": "a" * 64},
        {"index": "reference_index.json", "index_sha256": "a" * 64, "extra": True},
    ],
)
def test_demo_config_rejects_invalid_trusted_ok_reference_block(
    tmp_path: Path,
    trusted: object,
) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["trusted_ok_reference"] = trusted
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="trusted_ok_reference"):
        load_demo_config(config_path)


def test_demo_config_loads_base_thresholds_and_deployment_margin(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    artifact = json.loads(threshold_artifact.read_text(encoding="utf-8"))
    artifact["base_thresholds"] = {
        view: artifact["thresholds"][view] - 0.05 for view in VIEW_ORDER
    }
    artifact["threshold_margin"] = 0.05
    threshold_artifact.write_text(json.dumps(artifact), encoding="utf-8")
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )

    config = load_demo_config(config_path)

    assert tuple(config.efficientad_base_thresholds) == VIEW_ORDER
    assert config.efficientad_threshold_margin == pytest.approx(0.05)
    for view in VIEW_ORDER:
        assert config.efficientad_thresholds[view] == pytest.approx(
            config.efficientad_base_thresholds[view] + 0.05
        )


def test_demo_config_rejects_inconsistent_base_threshold_margin(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    artifact = json.loads(threshold_artifact.read_text(encoding="utf-8"))
    artifact["base_thresholds"] = dict(artifact["thresholds"])
    artifact["threshold_margin"] = 0.05
    threshold_artifact.write_text(json.dumps(artifact), encoding="utf-8")
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="基础阈值.*部署阈值"):
        load_demo_config(config_path)


def test_demo_config_loads_sha_bound_raw_profile_v2_asset(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "raw_profile_report.json"
    report.write_text(
        json.dumps(
            {
                "roi_xyxy": [1792, 1180, 1873, 1793],
                "raw_profile_v2": {
                    "thresholds": {
                        "min_row_score": 30.0,
                        "min_presence_coverage_ratio": 0.1,
                        "min_longest_run_ratio": 0.1,
                        "max_gap_ratio": 0.02,
                        "max_gap_count": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {
        "engine": "raw_profile_v2",
        "config": str(report),
        "config_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(config_path)

    assert config.bright_streak_engine == "raw_profile_v2"
    assert config.bright_streak_config == report.resolve()


def test_demo_config_rejects_changed_raw_profile_v2_asset(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "raw_profile_report.json"
    report.write_text("{}", encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {
        "engine": "raw_profile_v2",
        "config": str(report),
        "config_sha256": "0" * 64,
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="光痕配置资产 SHA256"):
        load_demo_config(config_path)


def test_demo_config_loads_sha_bound_tracked_profile_v3_asset(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "tracked_profile_report.json"
    report.write_text('{"algorithm":"tracked_profile_v3"}', encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {
        "engine": "tracked_profile_v3",
        "config": str(report),
        "config_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(config_path)

    assert config.bright_streak_engine == "tracked_profile_v3"
    assert config.bright_streak_config == report.resolve()
    assert config.bright_streak_rotated_roi is None
    assert config.bright_streak_rotated_roi_sha256 is None


def test_demo_config_loads_sha_bound_manual_rotated_roi_v5_asset(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "tracked_profile_report.json"
    report.write_text('{"algorithm":"tracked_profile_v3"}', encoding="utf-8")
    rotated_roi = tmp_path / "bright_streak_rotated_roi.json"
    rotated_roi.write_text('{"schema":"bmw.bright_streak_rotated_roi/1.0"}', encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {
        "engine": "tracked_profile_v3_manual_rotated_roi",
        "config": str(report),
        "config_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "rotated_roi": str(rotated_roi),
        "rotated_roi_sha256": hashlib.sha256(rotated_roi.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(config_path)

    assert config.bright_streak_engine == "tracked_profile_v3_manual_rotated_roi"
    assert config.bright_streak_config == report.resolve()
    assert config.bright_streak_rotated_roi == rotated_roi.resolve()
    assert config.bright_streak_rotated_roi_sha256 == hashlib.sha256(rotated_roi.read_bytes()).hexdigest()


def test_demo_config_loads_manual_rotated_roi_weak_score_override(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "tracked_profile_report.json"
    report.write_text('{"algorithm":"tracked_profile_v3"}', encoding="utf-8")
    rotated_roi = tmp_path / "bright_streak_rotated_roi.json"
    rotated_roi.write_text('{"schema":"bmw.bright_streak_rotated_roi/1.0"}', encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {
        "engine": "tracked_profile_v3_manual_rotated_roi",
        "config": str(report),
        "config_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "rotated_roi": str(rotated_roi),
        "rotated_roi_sha256": hashlib.sha256(rotated_roi.read_bytes()).hexdigest(),
        "weak_row_score_override": 95.0,
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_demo_config(config_path)

    assert config.bright_streak_weak_row_score_override == 95.0


@pytest.mark.parametrize("override", [None, 94.0, 96.0])
def test_demo_config_rejects_unapproved_rotated_roi_weak_score_override(
    tmp_path: Path,
    override: float | None,
) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "tracked_profile_report.json"
    report.write_text('{"algorithm":"tracked_profile_v3"}', encoding="utf-8")
    rotated_roi = tmp_path / "bright_streak_rotated_roi.json"
    rotated_roi.write_text('{"schema":"bmw.bright_streak_rotated_roi/1.0"}', encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {
        "engine": "tracked_profile_v3_manual_rotated_roi",
        "config": str(report),
        "config_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "rotated_roi": str(rotated_roi),
        "rotated_roi_sha256": hashlib.sha256(rotated_roi.read_bytes()).hexdigest(),
        "weak_row_score_override": override,
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="只允许精确值95.0"):
        load_demo_config(config_path)


def test_real_v5_profile_binds_manual_rotated_tracked_v3_asset() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    profile_path = (
        repo_root
        / "configs/bmw/experiments/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    config = load_demo_config(profile_path)

    assert profile["bright_streak"] == {
        "engine": "tracked_profile_v3_manual_rotated_roi",
        "config": (
            "../../../results/bmw_lab_one_click/"
            "bmw_right_batch_20260810_21_bright_v3_tracked_v8/report.json"
        ),
        "config_sha256": "6b43690af67702333646fa7a88a2a5053a0afae6d19cb343bf6d3abb193c17d4",
        "rotated_roi": (
            "../../../results/bmw_bright_streak_rotated_roi/"
            "bmw_demo_20260813_164043_v3/roi.json"
        ),
        "rotated_roi_sha256": "6ea49dacfae8d7d090bded6f8d67187d13fe00246c502a35813b77ce28aba4c8",
        "weak_row_score_override": 95.0,
    }
    assert config.bright_streak_engine == "tracked_profile_v3_manual_rotated_roi"
    assert config.bright_streak_rotated_roi == (
        repo_root
        / "results/bmw_bright_streak_rotated_roi/bmw_demo_20260813_164043_v3/roi.json"
    ).resolve()
    assert config.template_ignore_mask_index is not None
    assert config.efficientad_ignore_mask_index == config.template_ignore_mask_index
    assert config.bright_streak_weak_row_score_override == 95.0


@pytest.mark.parametrize("missing_field", ["rotated_roi", "rotated_roi_sha256"])
def test_demo_config_rejects_partial_manual_rotated_roi_fields(
    tmp_path: Path,
    missing_field: str,
) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "tracked_profile_report.json"
    report.write_text('{"algorithm":"tracked_profile_v3"}', encoding="utf-8")
    rotated_roi = tmp_path / "bright_streak_rotated_roi.json"
    rotated_roi.write_text("{}", encoding="utf-8")
    bright_streak = {
        "engine": "tracked_profile_v3_manual_rotated_roi",
        "config": str(report),
        "config_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "rotated_roi": str(rotated_roi),
        "rotated_roi_sha256": hashlib.sha256(rotated_roi.read_bytes()).hexdigest(),
    }
    del bright_streak[missing_field]
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = bright_streak
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bright_streak配置字段"):
        load_demo_config(config_path)


def test_demo_config_rejects_changed_manual_rotated_roi_asset(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "tracked_profile_report.json"
    report.write_text('{"algorithm":"tracked_profile_v3"}', encoding="utf-8")
    rotated_roi = tmp_path / "bright_streak_rotated_roi.json"
    rotated_roi.write_text("{}", encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {
        "engine": "tracked_profile_v3_manual_rotated_roi",
        "config": str(report),
        "config_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "rotated_roi": str(rotated_roi),
        "rotated_roi_sha256": hashlib.sha256(rotated_roi.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    rotated_roi.write_text('{"changed":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="倾斜光痕ROI SHA256不匹配"):
        load_demo_config(config_path)


def test_demo_config_rejects_changed_tracked_profile_v3_asset(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "tracked_profile_report.json"
    report.write_text('{"algorithm":"tracked_profile_v3"}', encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {
        "engine": "tracked_profile_v3",
        "config": str(report),
        "config_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="光痕配置资产 SHA256"):
        load_demo_config(config_path)


def test_demo_config_rejects_unknown_sha_bound_bright_streak_engine(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    report = tmp_path / "unknown.json"
    report.write_text("{}", encoding="utf-8")
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    payload["bright_streak"] = {
        "engine": "unknown_v9",
        "config": str(report),
        "config_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bright_streak.engine"):
        load_demo_config(config_path)


def test_demo_config_fails_closed_when_efficientad_checkpoint_changes(tmp_path: Path) -> None:
    _roi, _capture, run, threshold_artifact = _write_demo_assets(tmp_path)
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )
    (run / "efficientad" / VIEW_ORDER[0] / "model.ckpt").write_bytes(b"replaced-checkpoint")

    with pytest.raises(ValueError, match="checkpoint SHA256"):
        load_demo_config(config_path)


def test_demo_config_accepts_efficientad_asset_key_order_and_normalizes_views(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    artifact = json.loads(threshold_artifact.read_text(encoding="utf-8"))
    artifact["thresholds"] = {
        view: artifact["thresholds"][view] for view in sorted(VIEW_ORDER)
    }
    artifact["checkpoint_sha256_by_view"] = {
        view: artifact["checkpoint_sha256_by_view"][view] for view in sorted(VIEW_ORDER)
    }
    threshold_artifact.write_text(json.dumps(artifact), encoding="utf-8")
    config_path = tmp_path / "demo.json"
    config_path.write_text(
        json.dumps(_demo_payload(tmp_path, threshold_artifact=threshold_artifact)),
        encoding="utf-8",
    )

    config = load_demo_config(config_path)

    assert tuple(config.efficientad_thresholds) == VIEW_ORDER


def test_demo_config_fails_closed_when_threshold_artifact_changes(tmp_path: Path) -> None:
    _roi, _capture, _run, threshold_artifact = _write_demo_assets(tmp_path)
    payload = _demo_payload(tmp_path, threshold_artifact=threshold_artifact)
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    threshold_artifact.write_text(threshold_artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="threshold artifact SHA256"):
        load_demo_config(config_path)


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


def test_repository_demo_config_selects_the_compatible_right_ridge_asset() -> None:
    config_path = Path(__file__).resolve().parents[4] / "configs/bmw/experiments/bmw_eight_view_demo_v1.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["bright_streak"] == {
        "config": (
            "../../../results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/"
            "bright_streak_right_ridge_v1_bold/calibrated_config.json"
        )
    }


def test_repository_demo_config_selects_right_multisource_models_and_thresholds() -> None:
    config_path = Path(__file__).resolve().parents[4] / "configs/bmw/experiments/bmw_eight_view_demo_v1.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["demo_id"] == "bmw-eight-view-right-multisource-demo-v1"
    assert payload["roi_config"] == "../rois/bmw_right_hdr_eight_view_v1.json"
    assert payload["prepared_manifest"] == (
        "../../../dataset/bmw_lab_prepared/bmw_right_batch_20260810_21_v1/manifests/dataset_manifest.csv"
    )
    assert payload["training_run"] == "../../../results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1"
    assert payload["efficientad"]["threshold_artifact"] == (
        "../../../results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/"
        "efficientad/score_analysis/part_thresholds.json"
    )
    assert len(payload["efficientad"]["threshold_artifact_sha256"]) == 64


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
