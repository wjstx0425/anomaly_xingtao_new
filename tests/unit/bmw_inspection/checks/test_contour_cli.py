"""End-to-end synthetic offline CLI and atomic evidence contracts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bmw_inspection.cli.contour import main
from bmw_inspection.checks.contour_compare.contracts import read_json, validate_config
from bmw_inspection.checks.contour_compare.reference import atomic_directory


@pytest.fixture
def taught(tmp_path):
    draft = read_json(Path(__file__).resolve().parents[4] / "configs/bmw/checks/contour/left_front.draft.json")
    draft["image"].update(width=220, height=220)
    draft["part_roi_xyxy"] = [10, 10, 210, 210]
    draft["registration"]["anchor_rois_xyxy"] = [[55, 55, 85, 85], [135, 135, 165, 165]]
    draft["extraction"].update(search_inward_px=15, search_outward_px=15)
    image = np.full((220, 220, 3), 20, np.uint8)
    mask = np.zeros((220, 220), np.uint8)
    cv2.rectangle(image, (30, 30), (190, 190), (200, 200, 200), -1)
    cv2.rectangle(mask, (30, 30), (190, 190), 255, -1)
    for center in ((70, 70), (150, 150)):
        cv2.circle(image, center, 8, (20, 20, 20), -1)
    cv2.imwrite(str(tmp_path / "reference.png"), image)
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    (tmp_path / "draft.json").write_text(json.dumps(draft))
    code = main(["teach", "--image", str(tmp_path / "reference.png"), "--draft-config", str(tmp_path / "draft.json"), "--mask", str(tmp_path / "mask.png"), "--confirm-reference", "--output-dir", str(tmp_path / "bundle")])
    assert code == 0
    return tmp_path, image


def test_teach_inspect_evaluate_evidence(taught):
    root, image = taught
    image[85:115, 30:42] = 20
    cv2.imwrite(str(root / "test.png"), image)
    args = ["inspect", "--config", str(root / "bundle/recipe.json"), "--image", str(root / "test.png"), "--capture-id", "synthetic-notch", "--source-kind", "fused_only", "--output-dir", str(root / "run")]
    assert main(args) == 10
    result = read_json(root / "run/result.json")
    assert result["status"] == "NG"
    assert result["independent_source_count"] == 1
    assert result["whole_part_release"] is None
    assert result["events"]
    assert read_json(root / "run/distance_samples.json")["test_to_ref_px"]
    with (root / "run/contour_samples.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert any(r["signed_test_to_ref_px"] for r in rows)
    before = (root / "run/result.json").read_bytes()
    assert main(args) == 2
    assert (root / "run/result.json").read_bytes() == before
    (root / "manifest.csv").write_text("sample_id,physical_part_id,placement_group,split,hand,view_id,image_path,channel,ground_truth\na,,,development,left,front,test.png,fused,defect\nb,,,development,left,front,reference.png,fused,normal\n")
    assert main(["evaluate", "--config", str(root / "bundle/recipe.json"), "--manifest", str(root / "manifest.csv"), "--output-dir", str(root / "batch")]) == 0
    report = read_json(root / "batch/report.json")
    assert report["counts"]["NG"] >= 1
    assert report["label_metrics"]["defect"]["NG_fraction"] == 1


def test_draft_error_and_no_overwrite(taught):
    root, _ = taught
    assert main(["inspect", "--config", str(root / "draft.json"), "--image", str(root / "reference.png"), "--capture-id", "draft", "--output-dir", str(root / "error")]) == 2
    assert read_json(root / "error/result.json")["status"] == "ERROR"
    with pytest.raises(RuntimeError), atomic_directory(root / "failed") as tmp:
        (tmp / "half.txt").write_text("incomplete")
        raise RuntimeError("disk failure simulation")
    assert not (root / "failed").exists()
    assert not (root / "failed.lock").exists()


def test_invalid_override_rejected(taught):
    root, _ = taught
    config = read_json(root / "bundle/recipe.json")
    config["comparison"]["arc_overrides"] = [{"arc_id": 0, "inward_tolerance_px": None}]
    with pytest.raises(ValueError, match="positive finite"):
        validate_config(config)


def test_filename_labels_do_not_change_inference_and_arrays_unchanged(taught):
    from bmw_inspection.cli.contour import inspect_image
    from bmw_inspection.checks.contour_compare.reference import load_reference
    root, image = taught
    reference = load_reference(root / "bundle/recipe.json")
    before = image.copy()
    image.flags.writeable = False
    _, first = inspect_image(image, reference, capture_id="normal", source="normal.png")
    _, second = inspect_image(image, reference, capture_id="defect", source="defect.png")
    assert first["status"] == second["status"]
    np.testing.assert_allclose(first["test_to_ref_px"], second["test_to_ref_px"], equal_nan=True)
    np.testing.assert_array_equal(image, before)


def test_size_and_dtype_errors_remain_errors(taught):
    from bmw_inspection.cli.contour import inspect_image
    from bmw_inspection.checks.contour_compare.reference import load_reference
    root, image = taught
    reference = load_reference(root / "bundle/recipe.json")
    with pytest.raises(ValueError, match="size"):
        inspect_image(image[:-1], reference, capture_id="bad-size")
    with pytest.raises(ValueError, match="uint8"):
        inspect_image(image.astype(np.uint16), reference, capture_id="bad-dtype")


def test_failed_image_write_does_not_publish_evidence(taught, monkeypatch):
    from bmw_inspection.cli.contour import inspect_image
    from bmw_inspection.checks.contour_compare.reference import load_reference
    from bmw_inspection.checks.contour_compare.evidence import write_evidence
    root, image = taught
    reference = load_reference(root / "bundle/recipe.json")
    observation, result = inspect_image(image, reference, capture_id="save-failure")
    monkeypatch.setattr(cv2, "imwrite", lambda *_: False)
    with pytest.raises(OSError, match="write failed"):
        write_evidence(image, reference, observation, result, root / "incomplete")
    assert not (root / "incomplete").exists()


def test_missing_pose_preserves_full_unknown_overlay(taught):
    root, image = taught
    image[55:85, 55:85] = 200
    cv2.imwrite(str(root / "missing-hole.png"), image)
    code = main(["inspect", "--config", str(root / "bundle/recipe.json"), "--image", str(root / "missing-hole.png"), "--capture-id", "missing-hole", "--output-dir", str(root / "unknown")])
    assert code == 20
    result = read_json(root / "unknown/result.json")
    assert result["observed_required_fraction"] == 0
    assert not result["full_perimeter_pass"]
    overlay = cv2.imread(str(root / "unknown/comparison_overlay.png"))
    assert np.any(np.all(overlay == (0, 140, 255), axis=2))
