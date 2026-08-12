"""Synthetic safeguards for BMW tracked bright-streak profiles."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from bmw_inspection.lab.bright_streak_tracked_profile import (
    TrackedProfileGeometry,
    TrackedProfileMetrics,
    TrackedProfileThresholds,
    analyze_tracked_profile,
    classify_tracked_profile,
    fit_tracked_profile_thresholds,
)


def _thresholds() -> TrackedProfileThresholds:
    return TrackedProfileThresholds(
        strong_row_score=80.0,
        weak_row_score=30.0,
        min_presence_coverage_ratio=0.5,
        min_longest_run_ratio=0.5,
        max_gap_ratio=10 / 613,
        max_gap_count=1,
    )


def _diagonal_roi() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((613, 81), 40, dtype=np.uint8)
    centres = 20 + np.arange(613, dtype=np.int64) // 16
    for row, centre in enumerate(centres):
        image[row, centre - 3 : centre + 4] = 220
    return image, centres


def _score_roi(path_scores: np.ndarray) -> np.ndarray:
    image = np.full((613, 81), 40, dtype=np.float64)
    for row, score in enumerate(path_scores):
        image[row, 37:44] = 40.0 + score
    return image


def _metrics_for_scores(path_scores: np.ndarray) -> TrackedProfileMetrics:
    return analyze_tracked_profile(
        _score_roi(path_scores),
        TrackedProfileGeometry(smooth_window=1),
        _thresholds(),
    )


def test_tracked_profile_follows_a_slow_diagonal_without_mutating_the_roi() -> None:
    image, expected_x = _diagonal_roi()
    before = image.copy()

    metrics = analyze_tracked_profile(image, TrackedProfileGeometry(), _thresholds())

    assert np.array_equal(image, before)
    assert metrics.response_map.shape == (613, 49)
    assert metrics.path_x.shape == (613,)
    assert metrics.path_scores.shape == (613,)
    assert np.all(np.abs(metrics.path_x - expected_x) <= 2)
    assert metrics.active_start_row == 0
    assert metrics.active_stop_row == 613
    expected_total = metrics.path_scores.sum() - np.abs(np.diff(metrics.path_x)).sum()
    assert metrics.path_total_score == pytest.approx(expected_total)


def test_tracked_profile_returns_owned_read_only_arrays() -> None:
    image, _ = _diagonal_roi()

    metrics = analyze_tracked_profile(image, TrackedProfileGeometry(), _thresholds())

    for array in (
        metrics.response_map,
        metrics.path_x,
        metrics.path_scores,
        metrics.strong_mask,
        metrics.accepted_mask,
        metrics.bridged_mask,
    ):
        assert array.flags.owndata
        assert not array.flags.writeable


def test_tracked_profile_rejects_non_fixed_roi_shape_with_exact_message() -> None:
    message = "tracked bright-streak profile requires a grayscale ROI with shape (613, 81)"

    with pytest.raises(ValueError, match=re.escape(message)):
        analyze_tracked_profile(
            np.zeros((612, 81), dtype=np.uint8),
            TrackedProfileGeometry(),
            _thresholds(),
        )


def test_tracked_profile_rejects_even_smoothing_window_with_exact_message() -> None:
    message = "tracked-profile smooth_window must be a positive odd integer"

    with pytest.raises(ValueError, match=re.escape(message)):
        TrackedProfileGeometry(smooth_window=4)


def test_tracked_profile_rejects_reversed_hysteresis_thresholds() -> None:
    with pytest.raises(
        ValueError,
        match="tracked-profile weak_row_score must not exceed strong_row_score",
    ):
        TrackedProfileThresholds(
            strong_row_score=30.0,
            weak_row_score=80.0,
            min_presence_coverage_ratio=0.5,
            min_longest_run_ratio=0.5,
            max_gap_ratio=0.1,
            max_gap_count=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strong_row_score", np.inf),
        ("weak_row_score", np.nan),
        ("max_gap_ratio", np.nan),
    ],
)
def test_tracked_profile_rejects_non_finite_thresholds(field: str, value: float) -> None:
    values = {
        "strong_row_score": 80.0,
        "weak_row_score": 30.0,
        "min_presence_coverage_ratio": 0.5,
        "min_longest_run_ratio": 0.5,
        "max_gap_ratio": 0.1,
        "max_gap_count": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match="tracked-profile thresholds must be finite"):
        TrackedProfileThresholds(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_presence_coverage_ratio", -0.1),
        ("min_presence_coverage_ratio", 1.1),
        ("min_longest_run_ratio", -0.1),
        ("min_longest_run_ratio", 1.1),
        ("max_gap_ratio", -0.1),
        ("max_gap_ratio", 1.1),
    ],
)
def test_tracked_profile_rejects_out_of_range_ratios(field: str, value: float) -> None:
    values = {
        "strong_row_score": 80.0,
        "weak_row_score": 30.0,
        "min_presence_coverage_ratio": 0.5,
        "min_longest_run_ratio": 0.5,
        "max_gap_ratio": 0.1,
        "max_gap_count": 1,
    }
    values[field] = value

    with pytest.raises(
        ValueError,
        match=r"tracked-profile ratio thresholds must be within \[0, 1\]",
    ):
        TrackedProfileThresholds(**values)


@pytest.mark.parametrize("max_gap_count", [-1, True, 1.0])
def test_tracked_profile_rejects_invalid_gap_count(max_gap_count: object) -> None:
    with pytest.raises(
        ValueError,
        match="tracked-profile max_gap_count must be a non-negative integer",
    ):
        TrackedProfileThresholds(
            strong_row_score=80.0,
            weak_row_score=30.0,
            min_presence_coverage_ratio=0.5,
            min_longest_run_ratio=0.5,
            max_gap_ratio=0.1,
            max_gap_count=max_gap_count,
        )


def test_tracked_profile_classifies_a_continuous_diagonal_as_ok() -> None:
    continuous_diagonal = _metrics_for_scores(np.full(613, 120.0))

    assert classify_tracked_profile(continuous_diagonal, _thresholds()) == "OK"


def test_tracked_profile_classifies_a_weak_path_without_a_strong_seed_as_absent() -> None:
    no_strong_seed = _metrics_for_scores(np.full(613, 60.0))

    assert not no_strong_seed.strong_mask.any()
    assert not no_strong_seed.accepted_mask.any()
    assert no_strong_seed.active_start_row is None
    assert no_strong_seed.active_stop_row is None
    assert classify_tracked_profile(no_strong_seed, _thresholds()) == "NG_NO_STREAK"
    permissive = replace(
        _thresholds(),
        min_presence_coverage_ratio=0.0,
        min_longest_run_ratio=0.0,
    )
    assert classify_tracked_profile(no_strong_seed, permissive) == "NG_NO_STREAK"


def test_tracked_profile_keeps_a_diagnostic_path_but_no_active_segment_on_background() -> None:
    background = _metrics_for_scores(np.zeros(613))

    assert background.path_x.shape == (613,)
    assert background.path_scores.shape == (613,)
    assert np.all(background.path_x == 16)
    assert background.active_start_row is None
    assert background.active_stop_row is None
    assert classify_tracked_profile(background, _thresholds()) == "NG_NO_STREAK"


def test_tracked_profile_bridges_five_soft_rows_between_strong_runs() -> None:
    scores = np.full(613, 120.0)
    scores[300:305] = 60.0

    five_soft_rows_between_strong_runs = _metrics_for_scores(scores)

    assert five_soft_rows_between_strong_runs.strong_mask.sum() == 608
    assert five_soft_rows_between_strong_runs.accepted_mask.sum() == 613
    assert five_soft_rows_between_strong_runs.bridged_mask.sum() == 5
    assert five_soft_rows_between_strong_runs.coverage_ratio == 1.0
    assert five_soft_rows_between_strong_runs.longest_run_px == 613
    assert five_soft_rows_between_strong_runs.max_gap_px == 0
    assert five_soft_rows_between_strong_runs.gap_count == 0
    assert classify_tracked_profile(five_soft_rows_between_strong_runs, _thresholds()) == "OK"


def test_tracked_profile_classifies_twelve_background_rows_as_broken() -> None:
    scores = np.full(613, 120.0)
    scores[300:312] = 0.0

    twelve_background_rows_between_strong_runs = _metrics_for_scores(scores)

    assert twelve_background_rows_between_strong_runs.strong_mask.sum() == 601
    assert twelve_background_rows_between_strong_runs.accepted_mask.sum() == 601
    assert not twelve_background_rows_between_strong_runs.bridged_mask.any()
    assert twelve_background_rows_between_strong_runs.coverage_ratio == 601 / 613
    assert twelve_background_rows_between_strong_runs.longest_run_px == 301
    assert twelve_background_rows_between_strong_runs.longest_run_ratio == 301 / 613
    assert twelve_background_rows_between_strong_runs.max_gap_px == 12
    assert twelve_background_rows_between_strong_runs.max_gap_ratio == 12 / 613
    assert twelve_background_rows_between_strong_runs.gap_count == 1
    assert (
        classify_tracked_profile(twelve_background_rows_between_strong_runs, _thresholds())
        == "NG_BROKEN"
    )


def test_tracked_profile_discards_weak_components_without_a_strong_row() -> None:
    scores = np.zeros(613)
    scores[0:10] = 60.0
    scores[20:30] = 120.0

    metrics = _metrics_for_scores(scores)

    assert metrics.strong_mask.sum() == 10
    assert metrics.accepted_mask.sum() == 10
    assert not metrics.accepted_mask[0:10].any()
    assert metrics.accepted_mask[20:30].all()
    assert not metrics.bridged_mask.any()
    assert metrics.active_start_row == 20
    assert metrics.active_stop_row == 30
    assert metrics.coverage_ratio == 10 / 613
    assert metrics.longest_run_px == 10
    assert metrics.max_gap_px == 0
    assert metrics.gap_count == 0


def test_tracked_profile_fits_separated_peak_thresholds_and_normal_envelopes() -> None:
    normal_continuous = np.full(613, 110.0)
    normal_bridged = np.full(613, 100.0)
    normal_bridged[:10] = 0.0
    normal_bridged[300:305] = 40.0
    normal_bridged[-20:] = 0.0
    no_streak_low = np.full(613, 10.0)
    no_streak_peak = np.full(613, 20.0)
    records = [
        {"label": "normal", "path_scores": normal_continuous},
        {"label": "normal", "path_scores": normal_bridged},
        {"label": "no_streak", "path_scores": no_streak_low},
        {"label": "no_streak", "path_scores": no_streak_peak},
    ]

    thresholds = fit_tracked_profile_thresholds(records, TrackedProfileGeometry())

    assert thresholds.strong_row_score == 60.0
    assert thresholds.weak_row_score == 40.0
    assert thresholds.min_presence_coverage_ratio == 583 / 613
    assert thresholds.min_longest_run_ratio == 583 / 613
    assert thresholds.max_gap_ratio == 0.0
    assert thresholds.max_gap_count == 0
    assert all(
        np.array_equal(record["path_scores"], expected)
        for record, expected in zip(
            records,
            (normal_continuous, normal_bridged, no_streak_low, no_streak_peak),
            strict=True,
        )
    )


@pytest.mark.parametrize("label", ["broken", "final_test", "OK", "NG_NO_STREAK"])
def test_tracked_profile_fitting_rejects_non_calibration_labels(label: str) -> None:
    with pytest.raises(
        ValueError,
        match="tracked-profile calibration labels must be normal or no_streak",
    ):
        fit_tracked_profile_thresholds(
            [{"label": label, "path_scores": np.zeros(613)}],
            TrackedProfileGeometry(),
        )


@pytest.mark.parametrize("final_test", [True, 1, 0, "false", None])
def test_tracked_profile_fitting_rejects_invalid_final_test_marker(final_test: object) -> None:
    with pytest.raises(
        ValueError,
        match="tracked-profile fitting final_test must be the boolean false when provided",
    ):
        fit_tracked_profile_thresholds(
            [
                {"label": "normal", "path_scores": np.full(613, 100.0)},
                {"label": "no_streak", "path_scores": np.zeros(613), "final_test": final_test},
            ],
            TrackedProfileGeometry(),
        )


def test_tracked_profile_fitting_allows_explicit_calibration_and_false_marker() -> None:
    thresholds = fit_tracked_profile_thresholds(
        [
            {
                "label": "normal",
                "path_scores": np.full(613, 100.0),
                "split": "calibration",
                "final_test": False,
            },
            {"label": "no_streak", "path_scores": np.zeros(613)},
        ],
        TrackedProfileGeometry(),
    )

    assert thresholds.strong_row_score == 50.0


@pytest.mark.parametrize("split", ["final_test", "test", "validation", None])
def test_tracked_profile_fitting_rejects_non_calibration_splits(split: str | None) -> None:
    with pytest.raises(
        ValueError,
        match="tracked-profile threshold fitting accepts only split='calibration'",
    ):
        fit_tracked_profile_thresholds(
            [
                {"label": "normal", "path_scores": np.full(613, 100.0)},
                {"label": "no_streak", "path_scores": np.zeros(613), "split": split},
            ],
            TrackedProfileGeometry(),
        )


def test_tracked_profile_fitting_requires_both_calibration_classes() -> None:
    with pytest.raises(
        ValueError,
        match="tracked-profile calibration needs both normal and no_streak samples",
    ):
        fit_tracked_profile_thresholds(
            [{"label": "normal", "path_scores": np.full(613, 100.0)}],
            TrackedProfileGeometry(),
        )


def test_tracked_profile_fitting_requires_strict_peak_separation() -> None:
    with pytest.raises(
        ValueError,
        match="tracked-profile calibration peak envelopes must be strictly separated",
    ):
        fit_tracked_profile_thresholds(
            [
                {"label": "normal", "path_scores": np.full(613, 50.0)},
                {"label": "no_streak", "path_scores": np.full(613, 50.0)},
            ],
            TrackedProfileGeometry(),
        )
