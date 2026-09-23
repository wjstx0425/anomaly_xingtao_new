"""Synthetic geometry contracts; these do not validate industrial imaging."""
import cv2
import numpy as np
import pytest

from bmw_inspection.checks.contour_compare.comparison import compare_outline
from bmw_inspection.checks.contour_compare.geometry import point_to_segments, resample_closed, rigid_fit, transform_points
from bmw_inspection.checks.contour_compare.registration import estimate_rigid_pose


def fixture():
    dense = np.array([[20., 20.], [120., 20.], [120., 120.], [20., 120.]])
    ref = resample_closed(dense, 1.)
    ref.update(dense_xy=dense, required_mask=np.ones(len(ref["sample_xy"]), bool), arc_id=np.zeros(len(ref["sample_xy"]), int))
    obs = {"sample_xy": ref["sample_xy"].copy(), "valid": np.ones(400, bool),
           "dense_xy": ref["sample_xy"].copy(), "dense_valid": np.ones(400, bool),
           "invalid_reason": np.full(400, "", dtype=object),
           "registration": {"valid": True, "T_reference_to_test": np.eye(3)}}
    config = {"comparison": {"default_inward_tolerance_px": 2., "default_outward_tolerance_px": 2., "min_exceedance_arc_px": 8.}}
    return ref, obs, config


def test_segment_distance_interior_and_degenerate():
    distances, feet, indices = point_to_segments([[3, 4], [9, 0]], [[0, 0], [9, 0]], [[10, 0], [9, 0]])
    np.testing.assert_allclose(distances, [4, 0])
    np.testing.assert_allclose(feet, [[3, 0], [9, 0]])
    assert indices.tolist() == [0, 0]


def test_rigid_direction_no_scale_or_reflection():
    points = np.array([[0, 0], [30, 0], [0, 50]], float)
    angle = .17
    matrix = np.array([[np.cos(angle), -np.sin(angle), 9], [np.sin(angle), np.cos(angle), -4], [0, 0, 1]])
    test = transform_points(points, np.linalg.inv(matrix))
    fitted = rigid_fit(test, points)
    np.testing.assert_allclose(fitted, matrix, atol=1e-12)
    np.testing.assert_allclose(transform_points(transform_points(test, fitted), np.linalg.inv(fitted)), test, atol=1e-12)
    scaled = rigid_fit(points * 1.1, points)
    assert np.linalg.norm(transform_points(points * 1.1, scaled)-points) > 1
    assert np.linalg.det(scaled[:2, :2]) == pytest.approx(1.)


def test_self_and_fixed_perimeter():
    ref, obs, config = fixture()
    out = compare_outline(ref, obs, config)
    assert out["status"] == "PASS"
    assert out["full_perimeter_pass"]
    assert out["reference_perimeter_px"] == pytest.approx(400)
    assert out["test_to_reference"]["max_px"] < 1e-10


@pytest.mark.parametrize("start", [25, 125, 225, 325])
@pytest.mark.parametrize("displacement", [6., -6.])
def test_every_side_inward_and_outward(start, displacement):
    ref, obs, config = fixture()
    run = np.arange(start, start + 20)
    obs["sample_xy"][run] += displacement * ref["inward_normal_xy"][run]
    obs["dense_xy"] = obs["sample_xy"].copy()
    out = compare_outline(ref, obs, config)
    assert out["status"] == "NG"
    assert any(set(run).intersection(e["sample_ids"]) for e in out["events"])


def test_periodic_event_and_unknown_ng_priority():
    ref, obs, config = fixture()
    run = np.r_[np.arange(390, 400), np.arange(0, 10)]
    obs["sample_xy"][run] += 10 * ref["inward_normal_xy"][run]
    obs["dense_xy"] = obs["sample_xy"].copy()
    out = compare_outline(ref, obs, config)
    main = [e for e in out["events"] if e["source"] == "bidirectional"]
    assert len(main) == 1
    assert main[0]["exceedance_arc_px"] <= len(run)
    obs["valid"][200:204] = False
    obs["sample_xy"][200:204] = np.nan
    out = compare_outline(ref, obs, config)
    assert out["status"] == "NG"
    assert out["unobserved_required_length_px"] == 4
    assert not out["required_coverage_complete"]
    assert not out["full_perimeter_pass"]
    assert np.isnan(out["signed_ref_to_test_px"]).all()
    starts, ends = out["test_segment_starts_xy"], out["test_segment_ends_xy"]
    assert np.all(np.linalg.norm(ends-starts, axis=1) < 20)


def test_short_peak_unknown_and_exclusions_never_full_pass():
    ref, obs, config = fixture()
    obs["sample_xy"][50] += [0, 8]
    obs["dense_xy"] = obs["sample_xy"].copy()
    assert compare_outline(ref, obs, config)["status"] == "REVIEW"
    ref, obs, config = fixture()
    obs["valid"][10:14] = False
    out = compare_outline(ref, obs, config)
    assert out["observed_required_fraction"] == .99
    assert out["status"] == "REVIEW"
    ref["required_mask"][10:14] = False
    out = compare_outline(ref, obs, config)
    assert out["status"] == "PASS"
    assert not out["full_perimeter_pass"]
    assert out["required_fraction_of_perimeter"] == .99


def test_coarse_escape_is_not_lost_and_unsupported_is_review():
    ref, obs, config = fixture()
    obs["dense_xy"][25:55] += [0, -40]
    out = compare_outline(ref, obs, config)
    assert out["status"] == "NG"
    assert any(e["source"] == "whole_coarse_outline" for e in out["events"])
    obs["dense_valid"][25:55] = False
    out = compare_outline(ref, obs, config)
    assert out["status"] == "REVIEW"


def test_invalid_registration_never_uses_identity():
    ref, obs, config = fixture()
    obs["registration"] = {"valid": False, "reason_codes": ["anchor_missing"]}
    out = compare_outline(ref, obs, config)
    assert out["status"] == "REVIEW"
    assert out["observed_required_fraction"] == 0


def test_anchor_registration_translation_and_missing_anchor():
    reference = np.full((180, 240), 220, np.uint8)
    cv2.circle(reference, (60, 60), 14, 20, -1)
    cv2.circle(reference, (170, 120), 18, 20, -1)
    test = cv2.warpAffine(reference, np.float32([[1, 0, 4], [0, 1, -3]]), (240, 180), borderValue=220)
    config = {"anchor_rois_xyxy": [[30, 30, 90, 90], [140, 90, 200, 150]],
              "max_rotation_deg": 2, "max_anchor_motion_px": 10,
              "max_anchor_spacing_change_px": 1, "max_rms_residual_px": 1}
    out = estimate_rigid_pose(test, reference, config)
    assert out["valid"], out
    np.testing.assert_allclose(out["T_test_to_reference"][:2, 2], [-4, 3], atol=.01)
    blank = np.full_like(reference, 220)
    out = estimate_rigid_pose(blank, reference, config)
    assert not out["valid"]
    assert out["T_test_to_reference"] is None


def test_nonconvex_perimeter_and_inward_normals():
    dense = np.array([[0, 0], [30, 0], [30, 30], [20, 30], [20, 10], [10, 10], [10, 30], [0, 30]], float)
    for curve in (dense, dense[::-1]):
        sampled = resample_closed(curve, 1.)
        assert sampled["cell_length_px"].sum() == pytest.approx(160.)
        for point, normal, corner in zip(sampled["sample_xy"], sampled["inward_normal_xy"], sampled["corner_mask"], strict=True):
            if not corner:
                interior = point + normal * .25
                assert cv2.pointPolygonTest(dense.astype(np.float32), tuple(interior), False) >= 0


def test_no_interpolated_line_across_unknown_gap():
    ref, obs, config = fixture()
    obs["valid"][40:60] = False
    # Plausible display-only points may remain in the observation but have no
    # authority to produce a measured segment or to increase coverage.
    out = compare_outline(ref, obs, config)
    assert out["unobserved_required_length_px"] == 20
    assert len(out["test_segment_starts_xy"]) == 379
    assert np.isnan(out["ref_to_test_px"][40:60]).all()


def test_global_diagnostics_prevent_false_pass():
    ref, obs, config = fixture()
    obs["diagnostics"] = {"multiple_components": True, "image_clipped": True}
    out = compare_outline(ref, obs, config)
    assert out["status"] == "REVIEW"
    assert not out["full_perimeter_pass"]


@pytest.mark.parametrize("value", [None, 0, -1, np.nan, np.inf, True, "2"])
@pytest.mark.parametrize("field", ["inward_tolerance_px", "outward_tolerance_px", "min_exceedance_arc_px"])
def test_pure_api_rejects_invalid_arc_override(value, field):
    ref, obs, config = fixture()
    obs["sample_xy"][25:55] += [0, 10]
    config["comparison"]["arc_overrides"] = [{"arc_id": 0, field: value}]
    with pytest.raises(ValueError):
        compare_outline(ref, obs, config)


@pytest.mark.parametrize("field", ["default_inward_tolerance_px", "default_outward_tolerance_px", "min_exceedance_arc_px"])
@pytest.mark.parametrize("value", [None, 0, -1, np.nan, np.inf])
def test_pure_api_rejects_invalid_defaults(field, value):
    ref, obs, config = fixture()
    config["comparison"][field] = value
    with pytest.raises(ValueError):
        compare_outline(ref, obs, config)


def test_pure_api_rejects_bad_arc_ids_and_merge():
    ref, obs, config = fixture()
    for overrides in ([{"arc_id": 9}], [{"arc_id": 0}, {"arc_id": 0}]):
        config["comparison"]["arc_overrides"] = overrides
        with pytest.raises(ValueError):
            compare_outline(ref, obs, config)
    config["comparison"]["arc_overrides"] = []
    config["comparison"]["merge_valid_gap_px"] = 1
    with pytest.raises(ValueError):
        compare_outline(ref, obs, config)


@pytest.mark.parametrize("field,value", [("required_mask", False), ("cell_length_px", 0),
                                            ("cell_length_px", np.nan), ("sample_xy", np.nan),
                                            ("dense_xy", np.inf), ("inward_normal_xy", 0)])
def test_pure_api_rejects_invalid_reference(field, value):
    ref, obs, config = fixture()
    ref[field][...] = value
    with pytest.raises(ValueError):
        compare_outline(ref, obs, config)


def test_coarse_component_events_do_not_wrap_across_nan_separator():
    ref, obs, config = fixture()
    # Each component has a short outward event. They must not become one event
    # spanning the first and last components through array wraparound.
    first = np.array([[50., 10.], [51., 10.], [52., 10.]])
    second = np.array([[55., 10.], [56., 10.], [57., 10.]])
    obs["dense_xy"] = np.vstack([first, [[np.nan, np.nan]], second])
    obs["dense_valid"] = np.array([1, 1, 1, 0, 1, 1, 1], bool)
    out = compare_outline(ref, obs, config)
    events = [e for e in out["events"] if e["source"] == "whole_coarse_outline"]
    assert len(events) == 2
    assert all(e["status"] == "REVIEW" for e in events)


def test_partial_overlap_fuses_coarse_and_profile_support():
    ref, obs, config = fixture()
    obs["sample_xy"][55:85] += [0, 10]
    obs["dense_xy"][53:85] += [0, 12]
    out = compare_outline(ref, obs, config)
    assert len(out["events"]) == 1
    event = out["events"][0]
    assert event["status"] == "NG"
    assert event["sources"] == ["bidirectional", "whole_coarse_outline"]
    assert len(event["supports"]) == 2
    assert event["peak_distance_px"] == pytest.approx(12.)
    assert event["exceedance_arc_px"] == len(set(event["sample_ids"]))
    assert event["exceedance_arc_px"] < sum(item["exceedance_arc_px"] for item in event["supports"])
    assert event["bbox_reference_xyxy"][3] == pytest.approx(32.)


def test_coarse_support_does_not_merge_through_unknown_cells():
    ref, obs, config = fixture()
    obs["sample_xy"][45:85] += [0, 10]
    obs["valid"][60:65] = False
    obs["sample_xy"][60:65] = np.nan
    obs["dense_xy"][45:85] += [0, 12]
    out = compare_outline(ref, obs, config)
    profiles = [e for e in out["events"] if e["source"] == "bidirectional"]
    assert len(profiles) == 2
    assert all(not set(range(60, 65)).intersection(e["sample_ids"]) for e in profiles)
    for event in out["events"]:
        ids = np.asarray(event["sample_ids"])
        assert event["exceedance_arc_px"] == ref["cell_length_px"][ids[out["valid"][ids]]].sum()


def test_isolated_unchanged_edge_does_not_measure_distance_to_distant_segment():
    ref, obs, config = fixture()
    obs['valid'][:] = False
    obs['valid'][0] = True
    obs['valid'][6:17] = True
    result = compare_outline(ref, obs, config)
    assert result['status'] == 'REVIEW'
    assert result['events'] == []
    assert not result['valid'][0]
    assert result['invalid_reason'][0] == 'isolated_observation_without_segment'


def test_unconfirmed_reference_cells_are_not_counted_as_observed():
    ref, obs, config = fixture()
    ref['reference_valid'] = np.ones(len(ref['sample_xy']), bool)
    ref['reference_valid'][20:25] = False
    result = compare_outline(ref, obs, config)
    assert result['status'] == 'REVIEW'
    assert result['unobserved_required_length_px'] == 5


def test_unconfirmed_coarse_shape_change_keeps_localization_and_blocks_pass():
    ref, obs, config = fixture()
    obs['dense_xy'][25:45] += 12 * ref['inward_normal_xy'][25:45]
    obs['dense_candidate_eligible'] = np.zeros(len(obs['dense_xy']), bool)
    obs['valid'][20:50] = False
    result = compare_outline(ref, obs, config)
    assert result['status'] == 'REVIEW'
    assert result['events'] == []
    assert len(result['unconfirmed_coarse_events']) == 1
    assert result['unconfirmed_coarse_events'][0]['peak_candidate_distance_px'] == 12
