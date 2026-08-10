"""Regression tests for standalone BMW bright-streak recalibration."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.contracts import load_config
from bmw_inspection.lab.bright_streak_recalibration import recalibrate_bright_streak


REPO_ROOT = Path(__file__).resolve().parents[4]


def _write_synthetic_config(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "mode": "demo",
        "camera_serial": "DA9625347",
        "image_width": 160,
        "image_height": 120,
        "exposure": 4000.0,
        "gain": 0.0,
        "timeout_ms": 3000,
        "warmup_frames": 1,
        "roi_xyxy": [60, 10, 100, 110],
        "result_root": "results/test-only",
        "thresholds": {
            "min_mean_intensity": 20.0,
            "max_mean_intensity": 245.0,
            "max_dark_clip_ratio": 0.25,
            "max_bright_clip_ratio": 0.10,
            "min_laplacian_variance": 1.0,
            "background_kernel_px": 15,
            "response_mad_scale": 4.0,
            "min_component_area_px": 8,
            "min_component_width_px": 1.0,
            "max_component_width_px": 12.0,
            "center_tolerance_px": 8.0,
            "micro_gap_close_px": 2,
            "min_contrast_snr": 4.0,
            "min_coverage_ratio": 0.40,
            "min_longest_run_ratio": 0.45,
            "max_gap_ratio": 0.10,
            "max_gap_count": 1,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _streak_image(*, present: bool) -> np.ndarray:
    yy, xx = np.indices((120, 160))
    image = (76 + ((3 * xx + 5 * yy) % 9)).astype(np.uint8)
    if present:
        image[15:105, 78:82] = 210
    return image


def _broken_but_present_streak_image() -> np.ndarray:
    image = _streak_image(present=False)
    image[15:55, 78:82] = 210
    image[60:105, 78:82] = 210
    return image


def _synthetic_recalibration_inputs(
    root: Path,
    *,
    sample_prefix: str = "sample",
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    config_path = root / "base_config.json"
    _write_synthetic_config(config_path)
    rows = (
        (f"{sample_prefix}-cal-id-says-no-streak", "calibration", "OK", True),
        (f"{sample_prefix}-cal-id-says-ok", "calibration", "NG_NO_STREAK", False),
        (f"{sample_prefix}-final-id-says-ok", "final_test", "OK", False),
        (f"{sample_prefix}-final-id-says-no-streak", "final_test", "NG_NO_STREAK", True),
    )
    manifest_path = root / "bright_streak.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "split", "expected_status", "source_path"),
        )
        writer.writeheader()
        for index, (sample_id, split, expected_status, present) in enumerate(rows):
            image_path = root / f"image_{index}.png"
            assert cv2.imwrite(str(image_path), _streak_image(present=present))
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "expected_status": expected_status,
                    "source_path": image_path,
                }
            )
    return manifest_path, config_path


def _bold_continuity_inputs(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    config_path = root / "base_config.json"
    _write_synthetic_config(config_path)
    rows = (
        ("calibration-normal", "calibration", "OK", _streak_image(present=True)),
        ("calibration-no-streak", "calibration", "NG_NO_STREAK", _streak_image(present=False)),
        ("final-normal-with-gap", "final_test", "OK", _broken_but_present_streak_image()),
        ("final-no-streak", "final_test", "NG_NO_STREAK", _streak_image(present=False)),
    )
    manifest_path = root / "bright_streak.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "split", "expected_status", "source_path"),
        )
        writer.writeheader()
        for index, (sample_id, split, expected_status, image) in enumerate(rows):
            image_path = root / f"image_{index}.png"
            assert cv2.imwrite(str(image_path), image)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "expected_status": expected_status,
                    "source_path": image_path,
                }
            )
    return manifest_path, config_path


def test_recalibration_fits_calibration_and_reports_final_test(tmp_path: Path) -> None:
    manifest, config = _synthetic_recalibration_inputs(tmp_path / "inputs")

    report = recalibrate_bright_streak(manifest, config, tmp_path / "out")

    assert report["fit_split"] == "calibration"
    assert report["final_test_used_for_fit"] is False
    assert report["demo_only"] is False
    assert report["presence_fit_split"] == "calibration"
    assert report["final_test_used_for_presence_fit"] is False
    assert report["continuity_fit_split"] == "calibration"
    assert report["final_test_normal_used_for_continuity_fit"] is False
    calibrated_path = Path(report["config"])
    assert calibrated_path.is_file()
    assert load_config(calibrated_path).roi_xyxy is not None
    assert Path(report["metrics_csv"]).is_file()
    assert Path(report["report_json"]).is_file()
    metrics = list(csv.DictReader(Path(report["metrics_csv"]).open(encoding="utf-8")))
    assert [row["split"] for row in metrics] == [
        "calibration",
        "calibration",
        "final_test",
        "final_test",
    ]
    assert len(report["final_test_outcomes"]) == 2
    assert len(report["final_test_errors"]) == 2
    assert all(Path(row["evidence_path"]).is_file() for row in report["final_test_errors"])
    assert json.loads(Path(report["report_json"]).read_text(encoding="utf-8")) == report


def test_sample_ids_do_not_change_thresholds_or_decisions(tmp_path: Path) -> None:
    first_manifest, first_config = _synthetic_recalibration_inputs(
        tmp_path / "first",
        sample_prefix="misleading-first",
    )
    second_manifest, second_config = _synthetic_recalibration_inputs(
        tmp_path / "second",
        sample_prefix="unrelated-second",
    )

    first = recalibrate_bright_streak(first_manifest, first_config, tmp_path / "out-first")
    second = recalibrate_bright_streak(second_manifest, second_config, tmp_path / "out-second")

    assert first["fitted_thresholds"] == second["fitted_thresholds"]
    assert [row["predicted_status"] for row in first["final_test_outcomes"]] == [
        row["predicted_status"] for row in second["final_test_outcomes"]
    ]


def test_bold_continuity_relaxes_only_continuity_with_final_normals(tmp_path: Path) -> None:
    manifest, config = _bold_continuity_inputs(tmp_path / "inputs")

    default = recalibrate_bright_streak(manifest, config, tmp_path / "default")
    bold = recalibrate_bright_streak(
        manifest,
        config,
        tmp_path / "bold",
        bold_continuity=True,
    )

    assert bold["demo_only"] is True
    assert bold["presence_fit_split"] == "calibration"
    assert bold["final_test_used_for_presence_fit"] is False
    assert bold["continuity_fit_split"] == "calibration+final_test_normal"
    assert bold["final_test_normal_used_for_continuity_fit"] is True
    for name in ("min_contrast_snr", "min_coverage_ratio"):
        assert bold["fitted_thresholds"][name] == default["fitted_thresholds"][name]
    assert bold["fitted_thresholds"]["min_longest_run_ratio"] < (
        default["fitted_thresholds"]["min_longest_run_ratio"]
    )
    assert bold["fitted_thresholds"]["max_gap_ratio"] > (
        default["fitted_thresholds"]["max_gap_ratio"]
    )
    assert bold["fitted_thresholds"]["max_gap_count"] > (
        default["fitted_thresholds"]["max_gap_count"]
    )
    assert [row["predicted_status"] for row in default["final_test_outcomes"]] == [
        "NG_BROKEN",
        "NG_NO_STREAK",
    ]
    assert [row["predicted_status"] for row in bold["final_test_outcomes"]] == [
        "OK",
        "NG_NO_STREAK",
    ]
    assert json.loads(Path(bold["report_json"]).read_text(encoding="utf-8")) == bold


def test_cli_defaults_match_the_eight_view_handoff() -> None:
    script = REPO_ROOT / "pipeline/bmw_lab_recalibrate_bright_streak.py"
    spec = importlib.util.spec_from_file_location("bmw_lab_recalibrate_bright_streak_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([])

    assert args.manifest == (
        REPO_ROOT / "dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1/manifests/bright_streak.csv"
    )
    assert args.base_config == (
        REPO_ROOT
        / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak/calibrated_config.json"
    )
    assert args.output_dir == (
        REPO_ROOT / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak_ridge_v2"
    )
    assert args.bold_continuity is False
    assert module.build_parser().parse_args(["--bold-continuity"]).bold_continuity is True
