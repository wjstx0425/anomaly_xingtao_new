"""A wrong automatic material boundary cannot change reference edge identity."""

import cv2
import numpy as np
import pytest

from bmw_inspection.checks.contour_compare import extraction


@pytest.mark.parametrize("edge_x", [30, 1])
def test_self_comparison_does_not_promote_wrong_automatic_internal_edge(monkeypatch, edge_x):
    image = np.full((100, 120, 3), 20, np.uint8)
    image[20:80, edge_x:90] = 80
    image[20:80, edge_x + 10:90] = 200  # Internal same-polarity reflection ten pixels inward.
    mask = np.zeros((100, 120), np.uint8)
    mask[20:80, edge_x:90] = 1
    automatic = mask.copy()
    automatic[:, :edge_x + 10] = 0  # Automatic segmentation mistakes reflection for material.

    def contour(binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours[0][:, 0].astype(float)

    def segment(*args, **kwargs):
        return {"mask": automatic.copy(), "dense_xy": contour(automatic), "diagnostics": {}}

    monkeypatch.setattr(extraction, "coarse_segment", segment)
    reference = {
        "mask": mask, "image": image.copy(), "dense_xy": contour(mask),
        "sample_xy": np.array([[float(edge_x), 40], [float(edge_x), 41], [float(edge_x), 42]]),
        "inward_normal_xy": np.tile([1.0, 0], (3, 1)),
    }
    config = {"extraction": {
        "edge_selection_mode": "continuous_v3", "min_edge_amplitude": 10.0,
        "min_candidate_margin": 0.2, "search_inward_px": 20.0, "search_outward_px": 20.0,
    }}
    result = extraction.extract_full_outline(image, reference, {"T_test_to_reference": np.eye(3)}, config)
    if edge_x == 30:
        assert all(item["valid"] for item in result["reference_edge_evidence"])
    # Both same-polarity image signatures are plausible; a wrong automatic
    # segmentation must not preferentially choose the inner edge and report 10px.
    # At x=1, border clipping must not remove only the current true edge while
    # allowing reference calibration and selection of its interior reflection.
    assert not result["valid"].any()
    assert np.isnan(result["sample_xy"]).all()
    assert np.isnan(result["normal_offset_u_px"]).all()
