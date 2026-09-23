"""Replay explicitly unapproved real contour references without issuing a pass."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from bmw_inspection.checks.contour_compare import compare_outline, estimate_rigid_pose, extract_full_outline, save_reference, teach_reference
from bmw_inspection.checks.contour_compare.contracts import read_json, write_json
from bmw_inspection.checks.contour_compare.evidence import write_evidence
from bmw_inspection.checks.contour_compare.extraction import coarse_segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--reference-mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", help="Select rows while retaining the manifest reference")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = read_json(args.manifest)
    rows = manifest["samples"]
    ref_row = next(row for row in rows if row["sample_id"] == manifest["reference_sample_id"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(args.draft_config)
    reference_image = cv2.imread(str(root / ref_row["image_path"]), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(args.reference_mask, cv2.IMREAD_GRAYSCALE)
    reference = teach_reference(reference_image, config, {"mask": mask, "confirmed": False, "reference_version": Path(args.reference_mask).stem + "_diagnostic"})
    save_reference(reference, output / "reference_draft")
    config = reference["config"]
    print("Reference:", len(reference["sample_xy"]), "samples;", int((~reference["reference_valid"]).sum()), "unsupported normals", flush=True)
    t = time.perf_counter()
    baseline = coarse_segment(reference_image, mask, config, {"T_test_to_reference": np.eye(3)})
    reference["automatic_reference_segmentation"] = baseline
    np.savez_compressed(output / "automatic_reference_segmentation.npz", mask=baseline["mask"], dense_xy=baseline["dense_xy"])
    write_json(output / "automatic_reference_segmentation.json", baseline["diagnostics"])
    preprocessing_ms = (time.perf_counter()-t)*1000
    code_root = root / "src/bmw_inspection/checks/contour_compare"
    code_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in code_root.glob("*.py")}
    summary = []
    selected = [row for row in rows if not args.sample_id or row["sample_id"] in args.sample_id]
    if not selected:
        raise ValueError("no selected samples")
    for row in selected[:args.limit]:
        path = root / row["image_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"input checksum mismatch: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        start = time.perf_counter()
        registration = estimate_rigid_pose(image, reference_image, config)
        if not registration["valid"]:
            raise ValueError(f"diagnostic pose failed: {row['sample_id']}: {registration['reason_codes']}")
        observation = extract_full_outline(image, reference, registration, config)
        observation["valid"] &= reference["reference_valid"]
        observation["sample_xy"][~observation["valid"]] = np.nan
        observation["invalid_reason"][~reference["reference_valid"]] = "reference_normal_unknown"
        result = compare_outline(reference, observation, config)
        result.update(status="REVIEW", algorithm_candidate_status=result["status"], diagnostic_only=True, full_perimeter_pass=False, reference_confirmed=False, validation_level="development", whole_part_release=None, capture_id=row["sample_id"], hand=config["hand"], view_id=config["view_id"], image_source=str(path), image_sha256=digest, source_kind="fused_only", independent_source_count=1, parameter_snapshot=config, scope="single_view_outer_contour", code_sha256=code_hashes, timing_ms={"algorithm_total": (time.perf_counter()-start)*1000, "reference_preprocessing_once": preprocessing_ms})
        result["reason_codes"].append("draft_reference_not_approved")
        result = write_evidence(image, reference, observation, result, output / "samples" / row["sample_id"])
        summary.append({"sample_id": row["sample_id"], "status": result["status"], "algorithm_candidate_status": result["algorithm_candidate_status"], "observed_required_fraction": result["observed_required_fraction"], "unobserved_required_length_px": result["unobserved_required_length_px"], "ng_events": sum(e["status"] == "NG" for e in result["events"]), "review_events": sum(e["status"] == "REVIEW" for e in result["events"]), "test_to_reference": result["test_to_reference"], "timing_ms": result["timing_ms"], "diagnostics": observation["diagnostics"]})
        write_json(output / "summary.json", summary)
        print({key: value for key, value in summary[-1].items() if key != "diagnostics"}, flush=True)


if __name__ == "__main__":
    main()
