"""Tests for deterministic EfficientAD anomaly-map component scoring."""

from __future__ import annotations

import numpy as np
import pytest

from bmw_inspection.lab.efficientad_component_filter import (
    ComponentFilterPolicy,
    score_anomaly_components,
)


@pytest.fixture
def policy() -> ComponentFilterPolicy:
    """Return gates that distinguish noise, points, lines, and broad regions."""
    return ComponentFilterPolicy(
        low_threshold=0.30,
        seed_threshold=0.50,
        p95_threshold=0.65,
        minimum_area=8,
        hard_peak_threshold=0.90,
        line_minimum_length=5,
        line_minimum_area=4,
    )


def test_isolated_shallow_component_is_rejected(policy: ComponentFilterPolicy) -> None:
    anomaly_map = np.zeros((9, 9), dtype=np.float32)
    anomaly_map[4, 4] = 0.55

    result = score_anomaly_components(anomaly_map, None, policy)

    assert result.score == pytest.approx(0.0)
    assert result.accepted_components == ()
    assert len(result.rejected_components) == 1
    assert result.rejected_components[0].acceptance_reason == "below_component_gates"
    assert not np.any(result.accepted_mask)
    assert result.hotspot is None


def test_tiny_hard_point_is_accepted(policy: ComponentFilterPolicy) -> None:
    anomaly_map = np.zeros((9, 9), dtype=np.float32)
    anomaly_map[3, 7] = 0.95

    result = score_anomaly_components(anomaly_map, None, policy)

    assert result.score == pytest.approx(0.95)
    assert len(result.accepted_components) == 1
    component = result.accepted_components[0]
    assert component.area == 1
    assert component.peak == pytest.approx(0.95)
    assert component.mean == pytest.approx(0.95)
    assert component.p95 == pytest.approx(0.95)
    assert component.bounding_box_xyxy == (7, 3, 8, 4)
    assert component.acceptance_reason == "hard_peak"
    assert result.hotspot == (7, 3)


def test_thin_long_component_is_accepted_by_line_gate(policy: ComponentFilterPolicy) -> None:
    anomaly_map = np.zeros((9, 9), dtype=np.float32)
    anomaly_map[5, 1:6] = 0.72

    result = score_anomaly_components(anomaly_map, None, policy)

    assert result.score == pytest.approx(0.72)
    assert len(result.accepted_components) == 1
    component = result.accepted_components[0]
    assert component.area == 5
    assert component.bounding_box_xyxy == (1, 5, 6, 6)
    assert component.acceptance_reason == "line"
    assert np.count_nonzero(result.accepted_mask) == 5


def test_broad_component_is_accepted_by_area_and_p95(policy: ComponentFilterPolicy) -> None:
    anomaly_map = np.zeros((9, 9), dtype=np.float32)
    anomaly_map[2:5, 3:6] = 0.70

    result = score_anomaly_components(anomaly_map, None, policy)

    assert result.score == pytest.approx(0.70)
    assert len(result.accepted_components) == 1
    component = result.accepted_components[0]
    assert component.area == 9
    assert component.acceptance_reason == "area_p95"
    assert component.bounding_box_xyxy == (3, 2, 6, 5)


def test_ignore_mask_excludes_anomaly_before_component_scoring(
    policy: ComponentFilterPolicy,
) -> None:
    anomaly_map = np.zeros((8, 8), dtype=np.float32)
    anomaly_map[0:2, 0:2] = 0.99
    ignore_mask = np.zeros((4, 4), dtype=np.uint8)
    ignore_mask[0, 0] = 255

    result = score_anomaly_components(anomaly_map, ignore_mask, policy)

    assert result.score == pytest.approx(0.0)
    assert result.accepted_components == ()
    assert result.rejected_components == ()
    assert result.ignored_pixel_count == 4
    assert not np.any(result.accepted_mask)
    assert result.hotspot is None
