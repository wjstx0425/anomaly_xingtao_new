"""Regression coverage found while visually reviewing generated BMW masks."""

import numpy as np

from bmw_inspection.lab.efficientad_mask_ab import build_candidate_mask


def test_candidate_mask_keeps_dim_image_border_as_background() -> None:
    image = np.full((80, 100, 3), 20, dtype=np.uint8)
    image[10:72, 18:88] = 150

    mask, _median = build_candidate_mask([image] * 3, working_size=80, erosion_px=1)

    assert not np.any(mask[0] == 255)
    assert not np.any(mask[-1] == 255)
    assert not np.any(mask[:, 0] == 255)
    assert not np.any(mask[:, -1] == 255)
