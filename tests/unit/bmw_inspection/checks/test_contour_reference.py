"""Reference geometry corruption, material support and structural arc contracts."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.checks.contour_compare.contracts import ContourInputError, read_json, write_json
from bmw_inspection.checks.contour_compare.reference import _normal_support, load_reference, save_reference, teach_reference


def reference_inputs():
    config = read_json(Path(__file__).resolve().parents[4] / "configs/bmw/checks/contour/left_front.draft.json")
    config["image"].update(width=120, height=100)
    config["part_roi_xyxy"] = [5, 5, 115, 95]
    config["registration"]["anchor_rois_xyxy"] = [[30, 30, 40, 40], [70, 60, 80, 70]]
    mask = np.zeros((100, 120), np.uint8)
    mask[20:80, 25:95] = 255
    image = np.repeat(np.where(mask[..., None] > 0, 200, 20).astype(np.uint8), 3, axis=2)
    return image, config, mask


def teach(**annotations):
    image, config, mask = reference_inputs()
    return teach_reference(image, config, {"mask": mask, "confirmed": True, **annotations})


def corrupt_bundle(path, update):
    geometry_path = path / "contour.npz"
    with np.load(geometry_path, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    update(data)
    np.savez_compressed(geometry_path, **data)
    config = read_json(path / "recipe.json")
    config["asset_sha256"]["contour.npz"] = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    write_json(path / "recipe.json", config)


def test_round_trip_has_fixed_perimeter_and_immutable_arrays(tmp_path):
    original = teach()
    save_reference(original, tmp_path / "bundle")
    loaded = load_reference(tmp_path / "bundle/recipe.json")
    np.testing.assert_array_equal(original["sample_xy"], loaded["sample_xy"])
    assert not loaded["dense_xy"].flags.writeable
    assert loaded["s_px"][0] == 0
    assert np.all(np.diff(loaded["s_px"]) > 0)
    assert loaded["cell_length_px"].sum() == pytest.approx(256)


@pytest.mark.parametrize("key,mutate", [
    ("dense_xy", lambda a: a.__setitem__((0, 0), -1)),
    ("sample_xy", lambda a: a.__setitem__((10, 0), a[10, 0] + 2)),
    ("s_px", lambda a: a.__setitem__(0, 1)),
    ("cell_length_px", lambda a: a.__imul__(2)),
    ("inward_normal_xy", lambda a: a.__imul__(-1)),
    ("inward_normal_xy", lambda a: a.__imul__(2)),
    ("arc_id", lambda a: a.__setitem__(10, 99)),
    ("required_mask", lambda a: a.__setitem__(10, False)),
    ("corner_mask", lambda a: a.__setitem__(10, not a[10])),
])
def test_geometry_corruption_rejected_even_with_updated_checksum(tmp_path, key, mutate):
    save_reference(teach(), tmp_path / "bundle")
    corrupt_bundle(tmp_path / "bundle", lambda data: mutate(data[key]))
    with pytest.raises(ContourInputError):
        load_reference(tmp_path / "bundle/recipe.json")


def test_narrow_unverifiable_material_stays_draft():
    image, config, mask = reference_inputs()
    # One-pixel protrusion: the normal probes cannot establish material polarity.
    mask[49, 10:25] = 255
    image[mask > 0] = 200
    model = teach_reference(image, config, {"mask": mask, "confirmed": True})
    assert not model["reference_valid"].all()
    assert model["config"]["reference_confirmed"] is False
    assert model["config"]["reference_normal_unconfirmed_sample_ids"]


@pytest.mark.parametrize("reverse", [False, True])
def test_normal_support_recovers_one_pixel_background_gap(reverse):
    mask = np.zeros((9, 11), np.uint8)
    mask[:, :6] = 255
    mask[:, 7] = 255  # Exterior sliver makes both two-pixel probes foreground.
    normal = np.array([[1.0 if reverse else -1.0, 0.0]])
    forward, backward = _normal_support(mask, np.array([[5.0, 4.0]]), normal)
    assert bool(forward[0]) is not reverse
    assert bool(backward[0]) is reverse


def test_normal_support_rejects_conflicting_probe_directions():
    mask = np.zeros((9, 11), np.uint8)
    mask[:, [3, 5, 6]] = 255
    # +1 is material and -1 background, but +2 background and -2 material.
    forward, backward = _normal_support(mask, np.array([[5.0, 4.0]]), np.array([[1.0, 0.0]]))
    assert forward[0] == backward[0]


def test_normal_support_does_not_invent_material_for_one_pixel_structure():
    mask = np.zeros((9, 11), np.uint8)
    mask[:, 5] = 255
    forward, backward = _normal_support(mask, np.array([[5.0, 4.0]]), np.array([[1.0, 0.0]]))
    assert not forward[0] and not backward[0]


def test_normal_recovery_preserves_full_draft_geometry(tmp_path):
    image, config, mask = reference_inputs()
    mask[30:71, 96] = 255
    mask[30, 95] = 255  # Connect a thin exterior strip at one end.
    image[mask > 0] = 200
    model = teach_reference(image, config, {"mask": mask, "confirmed": False})
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    np.testing.assert_array_equal(model["dense_xy"], contours[0][:, 0])
    assert model["required_mask"].all()
    assert not model["config"]["reference_confirmed"]
    save_reference(model, tmp_path / "draft")
    with np.load(tmp_path / "draft/contour.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(model["reference_valid"], saved["reference_valid"])
        np.testing.assert_array_equal(model["required_mask"], saved["required_mask"])
    with pytest.raises(ContourInputError, match="remains draft"):
        load_reference(tmp_path / "draft/recipe.json")


def test_structural_ranges_and_override_round_trip(tmp_path):
    image, config, mask = reference_inputs()
    count = len(teach()["sample_xy"])
    ranges = [{"arc_id": 4, "start_sample": 0, "end_sample": 100}, {"arc_id": 9, "start_sample": 100, "end_sample": count}]
    config["comparison"]["arc_overrides"] = [{"arc_id": 9, "inward_tolerance_px": 3}]
    model = teach_reference(image, config, {"mask": mask, "confirmed": True, "arc_ranges": ranges})
    save_reference(model, tmp_path / "bundle")
    loaded = load_reference(tmp_path / "bundle/recipe.json")
    assert np.all(loaded["arc_id"][:100] == 4)
    assert np.all(loaded["arc_id"][100:] == 9)


@pytest.mark.parametrize("ranges", [[], [{"arc_id": 1, "start_sample": 0, "end_sample": 10}], [{"arc_id": 1, "start_sample": 0, "end_sample": 256}, {"arc_id": 2, "start_sample": 1, "end_sample": 2}]])
def test_arc_ranges_must_be_complete_and_nonoverlapping(ranges):
    with pytest.raises(ContourInputError):
        teach(arc_ranges=ranges)


def test_unknown_override_rejected_in_teach_and_load(tmp_path):
    image, config, mask = reference_inputs()
    config["comparison"]["arc_overrides"] = [{"arc_id": 77, "inward_tolerance_px": 3}]
    with pytest.raises(ContourInputError, match="existing"):
        teach_reference(image, config, {"mask": mask, "confirmed": True})
    save_reference(teach(), tmp_path / "bundle")
    recipe = read_json(tmp_path / "bundle/recipe.json")
    recipe["comparison"]["arc_overrides"] = config["comparison"]["arc_overrides"]
    write_json(tmp_path / "bundle/recipe.json", recipe)
    with pytest.raises(ContourInputError, match="existing"):
        load_reference(tmp_path / "bundle/recipe.json")
