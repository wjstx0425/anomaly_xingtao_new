"""Synthetic regression coverage for the offline BMW raw-profile candidate."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.bright_streak_raw_profile import (
    analyze_raw_profile,
    classify_raw_profile,
    evaluate_bright_streak_raw_profile,
    fit_raw_profile_thresholds,
)


def _roi(*, runs: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Create the fixed 613x81 grayscale ROI with optional centre streak runs."""
    image = np.full((613, 81), 70, dtype=np.uint8)
    for start, stop in runs:
        image[start:stop, 36:45] = 220
    return image


def test_raw_profile_reports_a_continuous_central_streak() -> None:
    metrics = analyze_raw_profile(_roi(runs=((0, 613),)), min_row_score=30.0)

    assert metrics.row_scores.shape == (613,)
    assert metrics.mask.shape == (613,)
    assert metrics.mask.dtype == np.bool_
    assert metrics.coverage_ratio == 1.0
    assert metrics.longest_run_px == 613
    assert metrics.longest_run_ratio == 1.0
    assert metrics.max_gap_px == 0
    assert metrics.max_gap_ratio == 0.0
    assert metrics.gap_count == 0
    assert np.all(metrics.row_scores > 30.0)


def test_raw_profile_reports_an_internal_break_in_anotherwise_visible_streak() -> None:
    metrics = analyze_raw_profile(
        _roi(runs=((50, 300), (380, 560))),
        min_row_score=30.0,
    )

    assert metrics.coverage_ratio == 430 / 613
    assert metrics.longest_run_px == 250
    assert metrics.max_gap_px == 80
    assert metrics.max_gap_ratio == 80 / 613
    assert metrics.gap_count == 1
    assert not metrics.mask[340]


def test_raw_profile_reports_no_streak_without_inventing_a_gap() -> None:
    metrics = analyze_raw_profile(_roi(runs=()), min_row_score=30.0)

    assert metrics.coverage_ratio == 0.0
    assert metrics.longest_run_px == 0
    assert metrics.longest_run_ratio == 0.0
    assert metrics.max_gap_px == 0
    assert metrics.max_gap_ratio == 0.0
    assert metrics.gap_count == 0
    assert not metrics.mask.any()


def test_raw_profile_thresholds_fit_only_calibration_and_classify_a_break() -> None:
    continuous = analyze_raw_profile(_roi(runs=((0, 613),)), min_row_score=0.0)
    missing = analyze_raw_profile(_roi(runs=()), min_row_score=0.0)
    thresholds = fit_raw_profile_thresholds(
        [
            {"label": "normal", "row_scores": continuous.row_scores},
            {"label": "no_streak", "row_scores": missing.row_scores},
        ]
    )
    broken = analyze_raw_profile(
        _roi(runs=((50, 300), (380, 560))),
        min_row_score=thresholds.min_row_score,
    )

    assert thresholds.min_row_score > 0.0
    assert thresholds.min_presence_coverage_ratio > 0.0
    assert classify_raw_profile(broken, thresholds) == "NG_BROKEN"
    assert classify_raw_profile(
        analyze_raw_profile(_roi(runs=()), min_row_score=thresholds.min_row_score),
        thresholds,
    ) == "NG_NO_STREAK"


def _write_evaluation_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "demo",
                "camera_serial": "test-camera",
                "image_width": 81,
                "image_height": 613,
                "exposure": 4000.0,
                "gain": 0.0,
                "timeout_ms": 3000,
                "warmup_frames": 1,
                "roi_xyxy": [0, 0, 81, 613],
                "result_root": "results/test-only",
                "thresholds": {
                    "min_mean_intensity": 20.0,
                    "max_mean_intensity": 245.0,
                    "max_dark_clip_ratio": 0.55,
                    "max_bright_clip_ratio": 0.1,
                    "min_laplacian_variance": 1.0,
                    "background_kernel_px": 21,
                    "response_mad_scale": 4.0,
                    "min_component_area_px": 8,
                    "min_component_width_px": 1.0,
                    "max_component_width_px": 20.0,
                    "center_tolerance_px": 18.0,
                    "micro_gap_close_px": 3,
                    "min_contrast_snr": 3.0,
                    "min_coverage_ratio": 0.6,
                    "min_longest_run_ratio": 0.5,
                    "max_gap_ratio": 0.05,
                    "max_gap_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_evaluation_manifest(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    config = root / "config.json"
    _write_evaluation_config(config)
    rows = (
        ("cal-normal", "calibration", "OK", ((0, 613),)),
        ("cal-no-streak", "calibration", "NG_NO_STREAK", ()),
        ("final-broken", "final_test", "OK", ((50, 300), (380, 560))),
        ("final-no-streak", "final_test", "NG_NO_STREAK", ()),
    )
    manifest = root / "bright_streak.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "split", "expected_status", "source_path"),
        )
        writer.writeheader()
        for index, (sample_id, split, expected_status, runs) in enumerate(rows):
            image_path = root / f"image-{index}.png"
            assert cv2.imwrite(str(image_path), _roi(runs=runs))
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "expected_status": expected_status,
                    "source_path": image_path,
                }
            )
    return manifest, config


def test_raw_profile_evaluation_fits_calibration_then_compares_final_test_without_overwrite(
    tmp_path: Path,
) -> None:
    manifest, config = _write_evaluation_manifest(tmp_path / "inputs")
    output_dir = tmp_path / "raw-profile-v2"

    report = evaluate_bright_streak_raw_profile(
        manifest,
        config,
        output_dir,
        roi_xyxy=(0, 0, 81, 613),
    )

    assert report["fit_split"] == "calibration"
    assert report["final_test_used_for_fit"] is False
    assert report["roi_xyxy"] == [0, 0, 81, 613]
    assert report["base_roi_xyxy"] == [0, 0, 81, 613]
    assert report["raw_profile_v2"]["final_test_count"] == 2
    assert report["current_algorithm"]["final_test_count"] == 2
    assert report["raw_profile_v2"]["outcomes"][0]["predicted_status"] == "NG_BROKEN"
    assert Path(report["metrics_csv"]).is_file()
    metrics_rows = list(csv.DictReader(Path(report["metrics_csv"]).open(encoding="utf-8")))
    profile_path = Path(metrics_rows[0]["raw_profile_npz"])
    assert profile_path.is_file()
    with np.load(profile_path) as profile:
        assert profile["row_scores"].shape == (613,)
        assert profile["mask"].shape == (613,)
    assert json.loads(Path(report["report_json"]).read_text(encoding="utf-8")) == report

    with pytest.raises(FileExistsError, match="output directory already exists"):
        evaluate_bright_streak_raw_profile(
            manifest,
            config,
            output_dir,
            roi_xyxy=(0, 0, 81, 613),
        )


def test_raw_profile_cli_defaults_to_the_21_point_candidate_without_overwrite_flag() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = repo_root / "pipeline/bmw_lab_evaluate_bright_streak_raw_profile.py"
    spec = importlib.util.spec_from_file_location(
        "bmw_lab_evaluate_bright_streak_raw_profile",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args([])

    assert args.manifest == (
        repo_root
        / "dataset/bmw_lab_prepared/bmw_right_batch_20260810_21_v1/manifests/bright_streak.csv"
    )
    assert args.base_config == (
        repo_root
        / "results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/"
        "bright_streak_right_ridge_v1_bold/calibrated_config.json"
    )
    assert args.output_dir == (
        repo_root
        / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v2_roi_corrected"
    )
    assert args.roi_xyxy == [1792, 1180, 1873, 1793]
    assert not hasattr(args, "overwrite")
