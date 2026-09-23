"""Reference calibration requires observed edges, not a nominal-mask shortcut."""

import numpy as np
import pytest

from bmw_inspection.checks.contour_compare.extraction import _candidate_transitions
from bmw_inspection.checks.contour_compare.reference_edges import select_reference_edge


def profiles(edge=0.0, polarity=1):
    offsets = np.arange(-15.0, 15.01, 0.25)
    values = 20 + 160 * np.clip((offsets - edge) + 0.5, 0, 1)
    if polarity < 0:
        values = 220 - values
    material = (offsets >= 0).astype(float)
    return offsets, values, material


def transitions(offsets, values):
    radius = 4
    contrasts = values[2 * radius:] - values[:-2 * radius]
    return _candidate_transitions(values, contrasts, np.abs(contrasts) >= 10, offsets[radius:-radius], offsets)


@pytest.mark.parametrize("edge", [0.0, 0.25, -0.5, 3.0, -3.0])
@pytest.mark.parametrize("polarity", [1, -1])
def test_clear_image_step_preserves_measured_subpixel_position(edge, polarity):
    offsets, values, material = profiles(edge, polarity)
    result = select_reference_edge(transitions(offsets, values), offsets, values, material)
    assert result["valid"]
    assert result["u_px"] == pytest.approx(edge, abs=1e-10)
    assert result["confidence"] > 0


def test_two_persistent_edges_remain_unknown_even_if_one_is_stronger_or_nearer():
    offsets, _, material = profiles()
    values = 20 + 60 * (offsets >= -2.0) + 120 * (offsets >= 2.5)
    result = select_reference_edge([(-2.0, 60, 60), (2.5, 120, 120)], offsets, values, material)
    assert not result["valid"]
    assert result["u_px"] is None
    assert result["reason"] == "ambiguous_reference_edges"
    assert sum(item["supported"] for item in result["candidates"]) == 2


def test_reflection_with_reversed_context_polarity_is_rejected():
    offsets, values, material = profiles()
    result = select_reference_edge([(0.0, 80, -80)], offsets, values, material)
    assert not result["valid"]
    assert result["candidates"][0]["reason"] == "image_context_polarity_conflict"


def test_narrow_highlight_cannot_supply_a_material_edge():
    offsets, _, material = profiles()
    values = 20 + 160 * (np.abs(offsets) < 0.6)
    result = select_reference_edge([(-0.6, 160, 160), (0.6, 160, -160)], offsets, values, material)
    assert not result["valid"]
    assert not any(item["supported"] for item in result["candidates"])


@pytest.mark.parametrize("case", ["no_transitions", "no_mask_boundary", "outside_window", "short_profile"])
def test_absent_or_insufficient_evidence_remains_unknown(case):
    offsets, values, material = profiles()
    edges = [(0.0, 160, 160)]
    if case == "no_transitions":
        edges = []
    elif case == "no_mask_boundary":
        material[:] = 1
    elif case == "outside_window":
        edges = [(6.5, 160, 160)]
    else:
        selected = np.abs(offsets) <= 2
        offsets, values, material = offsets[selected], values[selected], material[selected]
    result = select_reference_edge(edges, offsets, values, material)
    assert not result["valid"]
    assert result["u_px"] is None


def test_conflicting_material_direction_stays_unknown():
    offsets, values, material = profiles()
    material[(offsets >= 1.75) & (offsets <= 2.25)] = 0
    material[(offsets >= -2.25) & (offsets <= -1.75)] = 1
    result = select_reference_edge([(0.0, 160, 160)], offsets, values, material)
    assert not result["valid"]
    assert result["candidates"][0]["reason"] == "conflicting_material_direction"


def test_profile_arrays_are_not_modified():
    arrays = profiles(3)
    originals = [item.copy() for item in arrays]
    select_reference_edge([(3.0, 160, 160)], *arrays)
    for before, after in zip(originals, arrays, strict=True):
        np.testing.assert_array_equal(before, after)


def test_nonfinite_profile_is_rejected():
    offsets, values, material = profiles()
    values[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_reference_edge([], offsets, values, material)
