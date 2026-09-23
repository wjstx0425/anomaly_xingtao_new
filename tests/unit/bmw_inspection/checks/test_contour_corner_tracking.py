"""Image-contour corner-chain contracts, independent of classification tolerances."""
import cv2
import numpy as np
import pytest

from bmw_inspection.checks.contour_compare.corner_tracking import track_corner_runs


def scene():
    mask = np.zeros((90, 90), np.uint8)
    mask[20:61, 20:61] = 1
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    nominal = contours[0][:, 0].astype(float)
    corners = np.linalg.norm(nominal - [60, 20], axis=1) <= 8
    observed = nominal + [6, 0]
    valid = np.ones(len(nominal), bool)
    return nominal, corners, observed, valid, observed.copy(), valid.copy()


def test_six_pixel_corner_motion_preserves_actual_chain_and_inputs():
    args = scene()
    before = [a.copy() for a in args]
    out = track_corner_runs(*args)
    assert out['valid'].all()
    np.testing.assert_allclose(out['sample_xy'], args[2])
    for a, b in zip(args, before):
        np.testing.assert_array_equal(a, b)
    assert out['diagnostics']['runs'][0]['measured_path_length_px'] > 0


def test_broken_image_support_cannot_be_bridged():
    args = list(scene())
    corner = np.flatnonzero(np.all(args[0] == [60, 20], axis=1))[0]
    args[5][corner] = False
    out = track_corner_runs(*args)
    assert not out['valid'][args[1]].any()
    assert np.isnan(out['sample_xy'][args[1]]).all()


def test_missing_profile_endpoint_stays_unknown():
    args = list(scene())
    start = np.flatnonzero(args[1] & ~np.roll(args[1], 1))[0]
    args[3][:] = False
    out = track_corner_runs(*args)
    assert not out['valid'][args[1]].any()
    assert out['diagnostics']['runs'][0]['reason'] == 'missing_profile_anchor'


def test_no_jumping_across_nan_component_separator():
    args = list(scene())
    corner = np.flatnonzero(np.all(args[0] == [60, 20], axis=1))[0]
    args[4] = np.insert(args[4], corner, [np.nan, np.nan], axis=0)
    args[5] = np.insert(args[5], corner, False)
    out = track_corner_runs(*args)
    assert not out['valid'][args[1]].any()


def test_ambiguous_components_do_not_choose_first():
    args = list(scene())
    args[4] = np.concatenate([args[4], [[np.nan, np.nan]], args[4] + [0, .5]])
    args[5] = np.r_[args[5], False, args[5]]
    out = track_corner_runs(*args)
    assert not out['valid'][args[1]].any()


@pytest.mark.parametrize('distance', [float('nan'), float('inf'), 0, -1])
def test_invalid_tracking_distance(distance):
    with pytest.raises(ValueError):
        track_corner_runs(*scene(), max_distance_px=distance)


def test_missing_adjacent_anchor_searches_supported_flank_without_straightening():
    args = list(scene())
    start = np.flatnonzero(args[1] & ~np.roll(args[1], 1))[0]
    args[3][start - 1] = False
    out = track_corner_runs(*args)
    assert out['valid'][args[1]].all()
    np.testing.assert_allclose(out['sample_xy'][args[1]], args[2][args[1]])


def test_persistent_corner_filters_staircase_diagonal():
    from bmw_inspection.checks.contour_compare.corner_tracking import persistent_corner_mask
    stairs = np.asarray([[k//2, (k+1)//2] for k in range(81)], float)
    xy = np.vstack([stairs, [0, 45]])
    lengths = np.linalg.norm(np.roll(xy, -1, axis=0)-xy, axis=1)
    raw = np.ones(len(xy), bool)
    out = persistent_corner_mask(xy, raw, lengths)
    assert not out[12:69].any()
    assert raw.all()


def test_persistent_corner_keeps_true_right_angle_and_geometry():
    from bmw_inspection.checks.contour_compare.corner_tracking import persistent_corner_mask
    xy, raw, *_ = scene()
    lengths = np.ones(len(xy))
    original = xy.copy()
    out = persistent_corner_mask(xy, raw, lengths)
    assert out[np.all(xy == [60, 20], axis=1)].all()
    assert not out[~raw].any()
    np.testing.assert_array_equal(xy, original)
    np.testing.assert_array_equal(lengths, np.ones(len(xy)))


def test_persistent_corner_keeps_narrow_foot_without_full_wrap():
    from bmw_inspection.checks.contour_compare.corner_tracking import persistent_corner_mask
    xy = np.array([[0.,0.], [2.,0.], [2.,12.], [0.,12.]])
    lengths = np.linalg.norm(np.roll(xy, -1, axis=0)-xy, axis=1)
    assert persistent_corner_mask(xy, np.ones(4,bool), lengths).all()
    tiny = xy / 10
    assert persistent_corner_mask(tiny, np.ones(4,bool), lengths/10).all()


@pytest.mark.parametrize('value', [0., -1., float('nan')])
def test_persistent_corner_rejects_bad_lengths(value):
    from bmw_inspection.checks.contour_compare.corner_tracking import persistent_corner_mask
    xy, raw, *_ = scene()
    lengths = np.ones(len(xy)); lengths[0] = value
    with pytest.raises(ValueError):
        persistent_corner_mask(xy, raw, lengths)
