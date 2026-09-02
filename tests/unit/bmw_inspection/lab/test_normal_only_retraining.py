"""Contracts for the minimal BMW normal-only selective retrainer."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bmw_inspection.lab import normal_only_retraining as retraining
from bmw_inspection.lab.normal_only_retraining import (
    NormalOnlyRetrainingConfig,
    STEP_ORDER,
    build_bright_streak_manifest,
    derive_fixed_setup_roi,
    materialize_normal_only_training_data,
    run_normal_only_retraining,
)
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_roi import EightViewRoiConfig, save_roi_config


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_normal_only_retraining_cli_exists() -> None:
    assert (REPO_ROOT / "pipeline/bmw_lab_retrain_normal_only.py").is_file()


def _write_bright_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "sample_id", "physical_part_id", "session_id", "group_id", "view_id",
        "camera_serial", "source_path", "source_sha256", "source_class",
        "business_label", "split", "expected_status", "review_reason",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(sample: str, source_class: str, split: str) -> dict[str, str]:
    return {
        "sample_id": sample,
        "physical_part_id": sample,
        "session_id": "session",
        "group_id": sample,
        "view_id": "front_left",
        "camera_serial": "DA9625347",
        "source_path": f"/{sample}.png",
        "source_sha256": "a" * 64,
        "source_class": source_class,
        "business_label": "OK" if source_class == "normal" else "NG",
        "split": split,
        "expected_status": "OK" if source_class == "normal" else "NG_NO_STREAK",
        "review_reason": "",
    }


def test_plan_never_contains_yolo() -> None:
    assert STEP_ORDER == (
        "derive_roi", "materialize", "prepare_bright_streak", "template",
        "bright_streak", "efficientad", "score_normal_test", "calibrate_normal_thresholds",
    )
    assert "yolo" not in STEP_ORDER


def test_bright_manifest_uses_new_normals_and_only_historical_no_streak(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    old_prepared = tmp_path / "old-prepared"
    old_release = tmp_path / "old-release"
    _write_bright_manifest(
        prepared / "manifests/bright_streak.csv",
        [_row("new-normal", "normal", "calibration")],
    )
    _write_bright_manifest(
        old_prepared / "manifests/bright_streak.csv",
        [
            _row("old-normal", "normal", "calibration"),
            _row("old-missing-cal", "no_streak", "calibration"),
            _row("old-missing-test", "no_streak", "final_test"),
        ],
    )
    old_release.mkdir()
    (old_release / "report.json").write_text(
        json.dumps({"prepared_manifest": str(old_prepared / "manifests/dataset_manifest.csv")}),
        encoding="utf-8",
    )

    report = build_bright_streak_manifest(prepared, old_release, tmp_path / "combined.csv")

    assert report["normal_count"] == 1
    assert report["no_streak_count"] == 2
    with Path(report["manifest"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["sample_id"] for row in rows} == {
        "new-normal", "historical::old-missing-cal", "historical::old-missing-test",
    }


def _config(tmp_path: Path) -> NormalOnlyRetrainingConfig:
    prepared = tmp_path / "prepared"
    manifest = prepared / "manifests/dataset_manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("sample_id\nnew-sample\n", encoding="utf-8")
    (prepared / "report.json").write_text(
        json.dumps({"capture_scope": "right", "source_class_part_counts": {"normal": 10}}),
        encoding="utf-8",
    )
    reference = tmp_path / "reference.csv"
    reference.write_text("reference\n", encoding="utf-8")
    reuse_roi = tmp_path / "reuse.json"
    save_roi_config(
        reuse_roi,
        EightViewRoiConfig(
            dataset_id="reuse",
            source_manifest=reference,
            source_manifest_sha256=__import__("hashlib").sha256(reference.read_bytes()).hexdigest(),
            representative_sample_id="old",
            image_width=12,
            image_height=10,
            part_rois={view: (1, 1, 11, 9) for view in VIEW_ORDER},
            binding_mode="fixed_setup",
            capture_scope="right",
        ),
    )
    return NormalOnlyRetrainingConfig(
        repo_root=tmp_path,
        prepared_root=prepared,
        reuse_roi=reuse_roi,
        old_no_streak_release=tmp_path / "old",
        training_root=tmp_path / "training",
        training_id="new-training",
        output_root=tmp_path / "results",
        run_id="new-run",
        derived_roi=tmp_path / "derived.json",
        bright_base_config=tmp_path / "bright.json",
        imagenette_dir=tmp_path / "imagenette",
    )


def test_derive_roi_reuses_an_identical_completed_asset(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert derive_fixed_setup_roi(config)["status"] == "complete"
    assert derive_fixed_setup_roi(config)["status"] == "reused"


def test_runner_resumes_an_existing_failed_run_directory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.run_dir.mkdir(parents=True)
    (config.run_dir / "run_report.json").write_text(
        json.dumps({"status": "failed", "failed_step": "derive_roi"}), encoding="utf-8"
    )
    calls: list[str] = []

    def handler(name: str):
        def run() -> dict[str, str]:
            calls.append(name)
            return {"status": "complete", "scores_csv": "scores.csv"}
        return run

    report = run_normal_only_retraining(
        config,
        stage_handlers={name: handler(name) for name in STEP_ORDER},
    )

    assert report["status"] == "complete"
    assert calls == list(STEP_ORDER)


def test_materialize_reuses_a_matching_completed_release(tmp_path: Path) -> None:
    config = _config(tmp_path)
    derive_fixed_setup_roi(config)
    manifest = config.prepared_root / "manifests/dataset_manifest.csv"
    release = config.training_release
    release.mkdir(parents=True)
    (release / "report.json").write_text(
        json.dumps(
            {
                "release_status": "published",
                "prepared_manifest": str(manifest.resolve()),
                "prepared_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "roi_config": str(config.derived_roi.resolve()),
                "roi_config_sha256": hashlib.sha256(config.derived_roi.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert materialize_normal_only_training_data(config)["status"] == "reused"


def test_normal_only_template_stage_reuses_baseline_thresholds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train = getattr(retraining, "train_normal_only_templates", None)
    assert train is not None
    config = _config(tmp_path)
    lab = retraining._lab_config(config)
    baseline = tmp_path / "baseline-template"
    expected = {view: 0.01 + index / 1000 for index, view in enumerate(VIEW_ORDER)}
    for view, threshold in expected.items():
        model = baseline / view / "model.json"
        model.parent.mkdir(parents=True)
        model.write_text(json.dumps({"threshold": threshold}), encoding="utf-8")
    groups = {view: [SimpleNamespace(image_path=tmp_path / f"{view}.png")] for view in VIEW_ORDER}
    monkeypatch.setattr(retraining, "_load_eight_view_template_rows", lambda _path: groups)
    monkeypatch.setattr(retraining.cv2, "imread", lambda *_args, **_kwargs: np.zeros((8, 10), dtype=np.uint8))
    observed: dict[str, float] = {}

    def fixed_trainer(_rows, _roi, output_dir, *, threshold, **_kwargs):
        view = Path(output_dir).name
        observed[view] = threshold
        Path(output_dir).mkdir(parents=True)
        model = Path(output_dir) / "model.json"
        model.write_text("{}", encoding="utf-8")
        (Path(output_dir) / "model.sha256").write_text("a" * 64, encoding="ascii")
        return model

    report = train(lab, baseline, trainer=fixed_trainer)

    assert observed == expected
    assert report["model_count"] == 8


def test_bright_streak_stage_reuses_an_identical_completed_result(tmp_path: Path) -> None:
    reuse = getattr(retraining, "recalibrate_bright_streak_or_reuse", None)
    assert callable(reuse)
    manifest = tmp_path / "bright.csv"
    base_config = tmp_path / "base.json"
    manifest.write_text("sample_id\n", encoding="utf-8")
    base_config.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "bright-output"
    output.mkdir()
    calibrated = output / "calibrated_config.json"
    calibrated.write_text("{}\n", encoding="utf-8")
    report = {
        "status": "complete",
        "manifest": str(manifest.resolve()),
        "base_config": str(base_config.resolve()),
        "config": str(calibrated.resolve()),
        "identities": {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "base_config_sha256": hashlib.sha256(base_config.read_bytes()).hexdigest(),
            "calibrated_config_sha256": hashlib.sha256(calibrated.read_bytes()).hexdigest(),
        },
    }
    (output / "report.json").write_text(json.dumps(report), encoding="utf-8")

    result = reuse(manifest, base_config, output)

    assert result["status"] == "reused"
    assert result["manifest"] == str(manifest.resolve())
