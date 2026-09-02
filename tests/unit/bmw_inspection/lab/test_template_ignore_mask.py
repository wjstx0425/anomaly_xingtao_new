from __future__ import annotations

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.template_ignore_mask import (
    masked_ccoeff_normed_map,
    prepare_template_inspect_mask,
    select_masked_template_match,
)


def _brute_masked_ccoeff(query: np.ndarray, template: np.ndarray, inspect: np.ndarray) -> float:
    valid = inspect.astype(bool)
    query_values = query[valid].astype(np.float64)
    template_values = template[valid].astype(np.float64)
    query_values -= query_values.mean()
    template_values -= template_values.mean()
    return float(
        np.dot(query_values, template_values)
        / np.sqrt(np.dot(query_values, query_values) * np.dot(template_values, template_values))
    )


def test_all_inspect_mask_matches_opencv_ccoeff_normed() -> None:
    rng = np.random.default_rng(7)
    query = rng.integers(0, 256, size=(9, 10), dtype=np.uint8)
    template = rng.integers(0, 256, size=(6, 7), dtype=np.uint8)
    inspect = np.ones(query.shape, dtype=np.uint8)

    actual = masked_ccoeff_normed_map(query, template, inspect, minimum_valid_pixels=3)
    expected = cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)

    assert actual == pytest.approx(expected, abs=2e-5)


def test_each_shift_uses_its_own_aligned_inspect_mask_patch() -> None:
    rng = np.random.default_rng(11)
    query = rng.integers(0, 256, size=(7, 8), dtype=np.uint8)
    template = rng.integers(0, 256, size=(5, 6), dtype=np.uint8)
    inspect = np.ones(query.shape, dtype=np.uint8)
    inspect[0:2, 1:4] = 0
    inspect[5:7, 5:8] = 0

    actual = masked_ccoeff_normed_map(query, template, inspect, minimum_valid_pixels=3)

    for y in range(actual.shape[0]):
        for x in range(actual.shape[1]):
            expected = _brute_masked_ccoeff(
                query[y : y + template.shape[0], x : x + template.shape[1]],
                template,
                inspect[y : y + template.shape[0], x : x + template.shape[1]],
            )
            assert actual[y, x] == pytest.approx(expected, abs=2e-5)


def test_ignored_corruption_does_not_change_selected_similarity() -> None:
    template = np.arange(64, dtype=np.uint8).reshape(8, 8)
    clean = cv2.copyMakeBorder(template, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    corrupted = clean.copy()
    corrupted[3:6, 3:6] = 255
    inspect = np.ones(clean.shape, dtype=np.uint8)
    inspect[3:6, 3:6] = 0

    clean_match = select_masked_template_match(
        clean,
        (template,),
        inspect,
        minimum_valid_pixels=3,
    )
    corrupted_match = select_masked_template_match(
        corrupted,
        (template,),
        inspect,
        minimum_valid_pixels=3,
    )

    assert corrupted_match.location == clean_match.location == (1, 1)
    assert corrupted_match.similarity == pytest.approx(clean_match.similarity, abs=1e-6)
    assert corrupted_match.similarity == pytest.approx(1.0, abs=1e-6)


def test_prepare_template_inspect_mask_uses_nearest_resize_and_reflected_fit_padding() -> None:
    ignore = np.zeros((4, 2), dtype=np.uint8)
    ignore[1:3, 0] = 255

    inspect = prepare_template_inspect_mask(ignore, (8, 8))

    assert inspect.dtype == np.uint8
    assert inspect.shape == (8, 8)
    assert set(np.unique(inspect).tolist()) <= {0, 1}
    assert np.count_nonzero(inspect == 0) > 0
    resized = cv2.resize((ignore == 0).astype(np.uint8), (4, 8), interpolation=cv2.INTER_NEAREST)
    expected = cv2.copyMakeBorder(resized, 0, 0, 2, 2, cv2.BORDER_REFLECT_101)
    assert np.array_equal(inspect, expected)


def test_select_masked_template_match_fails_closed_when_no_candidate_has_variance() -> None:
    query = np.full((6, 6), 20, dtype=np.uint8)
    template = np.full((4, 4), 20, dtype=np.uint8)
    inspect = np.ones(query.shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="no valid masked Template candidate"):
        select_masked_template_match(
            query,
            (template,),
            inspect,
            minimum_valid_pixels=3,
        )


def test_masked_ccoeff_marks_too_small_valid_regions_invalid() -> None:
    query = np.arange(36, dtype=np.uint8).reshape(6, 6)
    template = query[:4, :4].copy()
    inspect = np.zeros(query.shape, dtype=np.uint8)
    inspect[0, :2] = 1

    response = masked_ccoeff_normed_map(query, template, inspect, minimum_valid_pixels=3)

    assert np.isnan(response).all()
