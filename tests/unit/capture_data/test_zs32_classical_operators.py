# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for standalone ZS32 classical diagnostic operators."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from capture_data.zs32_classical_operators import (
    OperatorParameters,
    build_material_mask,
    detect_pits_spots,
    detect_thin_lines,
    render_overlay,
)


def synthetic_metal_roi() -> np.ndarray:
    """Create a mildly textured metal rectangle on a black background."""
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    gradient = np.linspace(132, 172, 88, dtype=np.uint8)
    material = np.repeat(gradient[np.newaxis, :], 64, axis=0)
    image[16:80, 20:108] = np.repeat(material[:, :, np.newaxis], 3, axis=2)
    return image


def test_material_mask_rejects_black_background_and_image_border() -> None:
    """The material mask should retain the metal interior but remove its edge."""
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[16:80, 20:108] = 150

    mask = build_material_mask(image, OperatorParameters())

    assert mask.dtype == np.uint8
    assert mask[48, 64] == 255
    assert mask[0, 0] == 0
    assert mask[16, 20] == 0


@pytest.mark.parametrize("shape", [(10,), (10, 10), (10, 10, 4)])
def test_operators_reject_non_bgr_inputs(shape: tuple[int, ...]) -> None:
    """Operators should reject arrays outside the uint8 BGR image contract."""
    image = np.zeros(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="BGR"):
        detect_thin_lines(image, OperatorParameters())


def test_thin_line_score_increases_for_interior_dark_scratch() -> None:
    """An interior elongated dark mark should increase thin-line evidence."""
    normal = synthetic_metal_roi()
    scratch = normal.copy()
    cv2.line(scratch, (30, 48), (98, 48), (35, 35, 35), 2)
    parameters = OperatorParameters()

    normal_result = detect_thin_lines(normal, parameters)
    scratch_result = detect_thin_lines(scratch, parameters)

    assert scratch_result.score > normal_result.score
    assert np.count_nonzero(scratch_result.mask) > 0
    assert scratch_result.operator == "thin_line"


def test_pit_spot_score_increases_for_round_dark_spot() -> None:
    """An interior compact dark mark should increase pit/spot evidence."""
    normal = synthetic_metal_roi()
    spotted = normal.copy()
    cv2.circle(spotted, (64, 48), 6, (25, 25, 25), -1)
    parameters = OperatorParameters()

    normal_result = detect_pits_spots(normal, parameters)
    spotted_result = detect_pits_spots(spotted, parameters)

    assert spotted_result.score > normal_result.score
    assert np.count_nonzero(spotted_result.mask) > 0
    assert spotted_result.components[0].area > 0
    assert spotted_result.operator == "pit_spot"


def test_fixed_outer_edge_is_excluded_from_both_masks() -> None:
    """The material silhouette must not become diagnostic evidence."""
    image = synthetic_metal_roi()

    line = detect_thin_lines(image, OperatorParameters())
    spot = detect_pits_spots(image, OperatorParameters())

    assert np.count_nonzero(line.mask[:24]) == 0
    assert np.count_nonzero(spot.mask[:24]) == 0
    assert line.material_mask[48, 20] == 0
    assert line.material_mask[48, 107] == 0
    assert line.material_mask[16, 64] == 0
    assert line.material_mask[79, 64] == 0


def test_material_mask_preserves_large_interior_void() -> None:
    """A designed opening must not be filled back into the material surface."""
    image = synthetic_metal_roi()
    cv2.circle(image, (64, 48), 14, (0, 0, 0), -1)

    mask = build_material_mask(image, OperatorParameters())

    assert mask[48, 64] == 0


@pytest.mark.parametrize(
    ("detector", "expected_color"),
    [
        (detect_thin_lines, (0, 0, 255)),
        (detect_pits_spots, (0, 165, 255)),
    ],
)
def test_render_overlay_preserves_input_and_marks_evidence(
    detector: object,
    expected_color: tuple[int, int, int],
) -> None:
    """Overlays should use the branch color without mutating the image."""
    image = synthetic_metal_roi()
    if detector is detect_thin_lines:
        cv2.line(image, (30, 48), (98, 48), (35, 35, 35), 2)
    else:
        cv2.circle(image, (64, 48), 6, (25, 25, 25), -1)
    original = image.copy()
    result = detector(image, OperatorParameters())  # type: ignore[operator]

    overlay = render_overlay(image, result)

    assert overlay.shape == image.shape
    assert overlay.dtype == image.dtype
    assert np.array_equal(image, original)
    assert np.any(np.all(overlay == expected_color, axis=2))


def test_results_are_json_safe_without_embedding_arrays() -> None:
    """Result metadata should be serializable and omit response arrays."""
    image = synthetic_metal_roi()
    cv2.circle(image, (64, 48), 6, (25, 25, 25), -1)

    result = detect_pits_spots(image, OperatorParameters()).to_dict()
    parameters = OperatorParameters().to_dict()

    assert result["operator"] == "pit_spot"
    assert "mask" not in result
    assert parameters["spot_kernel_sizes"] == [9, 17, 25]
    json.dumps(result, allow_nan=False)
    json.dumps(parameters, allow_nan=False)
