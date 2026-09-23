"""Evidence-preserving candidate selection, independent of image extraction."""

from itertools import product

import numpy as np
import pytest

from bmw_inspection.checks.contour_compare.path_selection import select_candidate_path


def candidate(u, cost=0):
    return {"u_px": u, "cost": cost}


def test_strong_zigzag_and_real_twelve_pixel_notch_are_not_smoothed_away():
    offsets = [0, 12, 0, 12, 12, 0]
    rows = [[candidate(u), candidate(0 if u else 12, 1.2)] for u in offsets]
    result = select_candidate_path(rows, closed=True)
    np.testing.assert_array_equal(result["selected_indices"], np.zeros(len(rows), int))


def test_single_point_texture_is_rejected_using_both_neighbors():
    rows = [[candidate(0)], [candidate(12, 0), candidate(0, .2)], [candidate(0)]]
    result = select_candidate_path(rows)
    np.testing.assert_array_equal(result["selected_indices"], [0, 1, 0])
    assert result["margins"][1] == pytest.approx(.6)


def test_equal_parallel_edges_are_unknown_including_zero_margin():
    rows = [[candidate(0), candidate(12)] for _ in range(8)]
    for closed in (False, True):
        result = select_candidate_path(rows, closed=closed, ambiguity_margin=0)
        assert (result["selected_indices"] == -1).all()
        np.testing.assert_allclose(result["margins"], 0)


def test_missing_rows_are_not_filled_or_bridged():
    rows = [[candidate(0)], [], [candidate(0), candidate(12)], [candidate(0), candidate(12)]]
    result = select_candidate_path(rows)
    np.testing.assert_array_equal(result["selected_indices"], [0, -1, -1, -1])
    assert np.isnan(result["margins"][1])
    assert result["diagnostics"]["block_count"] == 2


def test_whole_shape_offset_does_not_change_selection():
    rows = [[candidate(0)], [candidate(1, .1), candidate(8)], [candidate(0)]]
    shifted = [[candidate(item["u_px"] + 57, item["cost"]) for item in row] for row in rows]
    original = select_candidate_path(rows)
    result = select_candidate_path(shifted)
    np.testing.assert_array_equal(result["selected_indices"], original["selected_indices"])
    np.testing.assert_allclose(result["margins"], original["margins"])


def test_closed_loop_and_marginals_match_brute_force():
    rows = [[candidate(0, .1), candidate(5)], [candidate(2)], [candidate(4, .2), candidate(-4, .1)], [candidate(0, .3), candidate(6)]]
    paths = list(product(*(range(len(row)) for row in rows)))
    energies = []
    for path in paths:
        chosen = [rows[i][j] for i, j in enumerate(path)]
        energies.append(sum(c["cost"] for c in chosen) + sum(.4 * min(abs(chosen[i]["u_px"] - chosen[(i+1) % len(rows)]["u_px"]) / 4, 1) for i in range(len(rows))))
    result = select_candidate_path(rows, closed=True)
    for i, row in enumerate(rows):
        constrained = [min(energy for path, energy in zip(paths, energies) if path[i] == j) for j in range(len(row))]
        ordered = sorted(constrained)
        margin = ordered[1] - ordered[0] if len(row) > 1 else np.inf
        assert result["margins"][i] == pytest.approx(margin)
        assert result["selected_indices"][i] == (int(np.argmin(constrained)) if margin > .15 else -1)


def test_closed_gap_keeps_seam_adjacency_but_never_bridges_missing():
    rows = [[candidate(0), candidate(12)], [], [candidate(12)]]
    result = select_candidate_path(rows, closed=True)
    np.testing.assert_array_equal(result["selected_indices"], [1, -1, 0])
    assert result["diagnostics"]["block_count"] == 1
    assert not result["diagnostics"]["cycle_solved"]


def test_closed_loop_is_rotation_equivariant():
    rows = [[candidate(0), candidate(12, .2)], [candidate(12)], [candidate(0, .1), candidate(12)], [candidate(0, .5), candidate(12)]]
    result = select_candidate_path(rows, closed=True)
    shifted = select_candidate_path(rows[2:] + rows[:2], closed=True)
    np.testing.assert_array_equal(shifted["selected_indices"], np.roll(result["selected_indices"], -2))
    np.testing.assert_allclose(shifted["margins"], np.roll(result["margins"], -2))


def test_empty_and_single_candidate_inputs():
    assert select_candidate_path([])["selected_indices"].size == 0
    result = select_candidate_path([[], []], closed=True)
    assert result["diagnostics"]["block_count"] == 0
    assert np.isnan(result["margins"]).all()
    assert select_candidate_path([[candidate(32)]], closed=True)["selected_indices"][0] == 0


@pytest.mark.parametrize("kwargs", [{"continuity_weight": -1}, {"ambiguity_margin": float("nan")}, {"jump_cap_px": 0}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError, match="Path weights"):
        select_candidate_path([[candidate(0)]], **kwargs)


def test_nonfinite_candidate_evidence_rejected():
    with pytest.raises(ValueError, match="finite u_px"):
        select_candidate_path([[candidate(float("nan"))]])
