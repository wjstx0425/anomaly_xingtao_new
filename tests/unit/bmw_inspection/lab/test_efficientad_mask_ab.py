"""Focused contracts for the BMW EfficientAD foreground-mask A/B experiment."""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

from bmw_inspection.lab.efficientad_mask_ab import (
    aggregate_anomaly_map,
    apply_fixed_fill,
    build_candidate_mask,
    load_foreground_mask_asset,
    validate_binary_mask,
)


def test_validate_binary_mask_rejects_non_binary_and_empty_masks() -> None:
    with pytest.raises(ValueError, match="binary"):
        validate_binary_mask(np.array([[0, 1], [2, 255]], dtype=np.uint8), (2, 2))
    with pytest.raises(ValueError, match="foreground"):
        validate_binary_mask(np.zeros((2, 2), dtype=np.uint8), (2, 2))
    with pytest.raises(ValueError, match="shape"):
        validate_binary_mask(np.full((2, 3), 255, dtype=np.uint8), (2, 2))


def test_apply_fixed_fill_preserves_foreground_for_gray_and_bgr() -> None:
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    bgr = np.repeat(gray[:, :, None], 3, axis=2)

    assert np.array_equal(apply_fixed_fill(gray, mask, 7), np.array([[10, 7], [7, 40]], dtype=np.uint8))
    assert np.array_equal(
        apply_fixed_fill(bgr, mask, 7),
        np.array([[[10, 10, 10], [7, 7, 7]], [[7, 7, 7], [40, 40, 40]]], dtype=np.uint8),
    )


def test_map_aggregates_separate_background_peak_and_are_deterministic() -> None:
    anomaly_map = np.array(
        [
            [9.0, 0.1, 0.2, 0.3],
            [0.1, 0.6, 0.7, 0.2],
            [0.1, 0.8, 0.9, 0.2],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=np.float32,
    )
    roi_mask = np.zeros((8, 8), dtype=np.uint8)
    roi_mask[2:6, 2:6] = 255

    result = aggregate_anomaly_map(
        anomaly_map,
        roi_mask,
        quantile=0.75,
        top_k_fraction=0.5,
        component_threshold=0.5,
        minimum_component_area=3,
    )

    assert result.global_max == pytest.approx(9.0)
    assert result.foreground_max == pytest.approx(0.9)
    assert result.background_max == pytest.approx(9.0)
    assert result.hotspot_region == "background"
    assert result.background_max_advantage == pytest.approx(8.1)
    assert result.foreground_top_k_mean == pytest.approx((0.9 + 0.8) / 2)
    assert result.foreground_component_score == pytest.approx((0.6 + 0.7 + 0.8 + 0.9) / 4)


def test_minimum_component_area_rejects_isolated_foreground_peak() -> None:
    anomaly_map = np.zeros((5, 5), dtype=np.float32)
    anomaly_map[1, 1] = 1.0
    anomaly_map[3, 2:5] = 0.7
    mask = np.full((5, 5), 255, dtype=np.uint8)

    result = aggregate_anomaly_map(
        anomaly_map,
        mask,
        quantile=0.9,
        top_k_fraction=0.1,
        component_threshold=0.6,
        minimum_component_area=3,
    )

    assert result.foreground_max == pytest.approx(1.0)
    assert result.foreground_component_score == pytest.approx(0.7)


@pytest.mark.parametrize("bad", [np.array([[np.nan]], dtype=np.float32), np.array([[np.inf]], dtype=np.float32)])
def test_map_aggregates_fail_closed_on_non_finite_values(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        aggregate_anomaly_map(bad, np.full((1, 1), 255, dtype=np.uint8))


def test_candidate_mask_is_binary_and_keeps_bright_part() -> None:
    image = np.zeros((64, 80, 3), dtype=np.uint8)
    image[8:58, 15:70] = 150
    image[20:45, 28:55] = 80

    mask, median = build_candidate_mask([image, image.copy(), image.copy()], working_size=64, erosion_px=1)

    assert mask.shape == image.shape[:2]
    assert median.shape == image.shape
    assert set(np.unique(mask)) == {0, 255}
    assert mask[32, 40] == 255
    assert mask[2, 2] == 0


def test_mask_asset_verifies_mask_sha_and_roi_binding(tmp_path) -> None:
    mask_path = tmp_path / "front.png"
    assert cv2.imwrite(str(mask_path), np.array([[0, 255], [255, 255]], dtype=np.uint8))
    digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    index = tmp_path / "index.json"
    payload = {
        "schema_version": "bmw.efficientad_foreground_masks/1.0",
        "candidate_only": True,
        "public_roi_config_sha256": "a" * 64,
        "fixed_fill_value": 0,
        "views": {
            "front": {
                "mask_path": "front.png",
                "mask_sha256": digest,
                "roi_height": 2,
                "roi_width": 2,
            }
        },
    }
    index.write_text(json.dumps(payload), encoding="utf-8")

    asset = load_foreground_mask_asset(index, expected_views=("front",), expected_roi_sha256="a" * 64)
    assert asset.fixed_fill_value == 0
    assert np.array_equal(asset.masks["front"], np.array([[0, 255], [255, 255]], dtype=np.uint8))

    payload["views"]["front"]["mask_sha256"] = "b" * 64
    index.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        load_foreground_mask_asset(index, expected_views=("front",), expected_roi_sha256="a" * 64)
