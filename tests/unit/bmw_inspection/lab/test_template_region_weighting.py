from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from bmw_inspection.lab.eight_view_demo import VIEW_ORDER
from bmw_inspection.lab.template_region_weighting import (
    TemplateWeightedRegion,
    build_aligned_template_weights,
    load_template_weighted_regions,
    normal_envelope_threshold,
    weighted_ccoeff,
)


def test_regions_allow_zero_or_many(tmp_path: Path) -> None:
    payload = {"views": {view: [] for view in VIEW_ORDER}}
    payload["views"]["front"] = [
        {"id": "foot", "roi_xyxy": [1, 2, 5, 7]},
        {"id": "edge", "roi_xyxy": [6, 1, 10, 4]},
    ]
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    regions = load_template_weighted_regions(
        path,
        expected_shapes={view: (10, 12) for view in VIEW_ORDER},
    )

    assert [region.region_id for region in regions["front"]] == ["foot", "edge"]
    assert regions["front"][0].roi_xyxy == (1, 2, 5, 7)
    assert regions["back"] == ()


@pytest.mark.parametrize(
    "roi_xyxy",
    ([1, 2, 13, 7], [1, 2, 1, 7], [1.5, 2, 5, 7]),
)
def test_regions_reject_invalid_public_roi_local_coordinates(
    tmp_path: Path,
    roi_xyxy: list[float],
) -> None:
    payload = {"views": {view: [] for view in VIEW_ORDER}}
    payload["views"]["front"] = [{"id": "foot", "roi_xyxy": roi_xyxy}]
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="front"):
        load_template_weighted_regions(
            path,
            expected_shapes={view: (10, 12) for view in VIEW_ORDER},
        )


def test_weight_map_uses_max_overlap_and_ignore_mask_wins() -> None:
    regions = (
        TemplateWeightedRegion("a", (0, 0, 6, 6)),
        TemplateWeightedRegion("b", (3, 3, 8, 8)),
    )
    ignore = np.zeros((8, 8), dtype=np.uint8)
    ignore[4:6, 4:6] = 255

    weights = build_aligned_template_weights(
        input_shape=(8, 8),
        target_size=(8, 8),
        max_shift=0,
        best_location=(0, 0),
        regions=regions,
        region_weight=3.0,
        outside_weight=0.5,
        ignore_mask=ignore,
    )

    assert weights[3, 3] == 3.0
    assert weights[7, 0] == 0.5
    assert weights[4, 4] == 0.0
    assert weights.max() == 3.0


def test_weight_map_follows_aspect_fit_and_alignment_slice() -> None:
    weights = build_aligned_template_weights(
        input_shape=(4, 8),
        target_size=(8, 8),
        max_shift=1,
        best_location=(2, 1),
        regions=(TemplateWeightedRegion("left", (0, 0, 2, 4)),),
        region_weight=3.0,
        outside_weight=1.0,
        ignore_mask=None,
    )

    assert weights.shape == (8, 8)
    assert np.all(weights[2:6, 0] == 3.0)
    assert np.all(weights[:, 2:] == 1.0)


def test_weight_three_emphasizes_difference_inside_region() -> None:
    template = np.arange(64, dtype=np.float64).reshape(8, 8)
    query = template.copy()
    query[1:3, 1:3] += 50
    base_similarity = weighted_ccoeff(query, template, np.ones((8, 8)))
    weights = np.ones((8, 8))
    weights[1:3, 1:3] = 3.0
    weighted_similarity = weighted_ccoeff(query, template, weights)

    assert 1.0 - weighted_similarity > 1.0 - base_similarity


def test_weighted_ccoeff_rejects_empty_or_constant_valid_pixels() -> None:
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    with pytest.raises(ValueError, match="有效权重为空"):
        weighted_ccoeff(image, image, np.zeros_like(image))
    with pytest.raises(ValueError, match="分母为零"):
        weighted_ccoeff(np.ones_like(image), np.ones_like(image), np.ones_like(image))


def test_normal_envelope_adds_ten_percent_and_handles_zero() -> None:
    assert normal_envelope_threshold([0.1, 0.2]) == pytest.approx(0.22)
    assert normal_envelope_threshold([0.0, 0.0]) == math.nextafter(0.0, math.inf)
