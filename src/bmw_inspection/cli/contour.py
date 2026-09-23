"""Offline teach/inspect/evaluate command for visible outer contour inspection."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from bmw_inspection.checks.contour_compare import compare_outline, estimate_rigid_pose, extract_full_outline, load_reference, save_reference, teach_reference
from bmw_inspection.checks.contour_compare.contracts import ContourInputError, json_ready, read_image, read_json, validate_config, write_json
from bmw_inspection.checks.contour_compare.evidence import write_evidence
from bmw_inspection.checks.contour_compare.reference import atomic_directory


EXIT_CODES = {"PASS": 0, "NG": 10, "REVIEW": 20, "ERROR": 2}


def inspect_image(image, reference, *, capture_id, source_kind="unknown", source=None):
    """Run read-only measurements; do not consume evaluation labels."""
    from bmw_inspection.checks.contour_compare.contracts import check_image
    config = validate_config(reference["config"])
    reference_valid = np.asarray(reference.get("reference_valid", []), dtype=bool)
    if reference_valid.shape != (len(reference["sample_xy"]),) or not reference_valid.all():
        raise ContourInputError("reference contains unconfirmed cells")
    check_image(image, config)
    if not isinstance(capture_id, str) or not capture_id.strip():
        raise ContourInputError("nonempty capture_id required")
    if source_kind not in ("unknown", "fused_only", "real_exposure"):
        raise ContourInputError("invalid source_kind")
    start = time.perf_counter()
    registration = estimate_rigid_pose(image, reference["image"], config)
    pose_end = time.perf_counter()
    if registration["valid"]:
        observation = extract_full_outline(image, reference, registration, config)
    else:
        n = len(reference["sample_xy"])
        observation = {"sample_xy": np.full((n, 2), np.nan), "valid": np.zeros(n, bool), "invalid_reason": np.full(n, "registration_failed"), "dense_xy": np.empty((0, 2)), "dense_valid": np.zeros(0, bool), "registration": registration}
    observation["registration"] = registration
    extraction_end = time.perf_counter()
    result = compare_outline(reference, observation, config)
    end = time.perf_counter()
    result.update(schema_version=1, check_id="contour_compare", capture_id=capture_id, hand=config["hand"], view_id=config["view_id"], reference_version=config["reference_version"], config_version=config["config_version"], scope="single_view_outer_contour", related_requirements=[5, 10], covered_subscope="visible_projected_outer_perimeter", validation_level="development", whole_part_release=None, image_source=source, source_kind=source_kind, independent_source_count=1, image_size_wh=[image.shape[1], image.shape[0]], channel=config["image"]["channel"], parameter_snapshot=copy.deepcopy(config), timing_ms={"registration": (pose_end-start)*1000, "extraction": (extraction_end-pose_end)*1000, "comparison": (end-extraction_end)*1000, "algorithm_total": (end-start)*1000})
    return observation, result


def _inspect_file(path, reference, capture_id, source_kind, output):
    if Path(output).exists():
        raise FileExistsError(output)
    image = read_image(path, reference["config"])
    observation, result = inspect_image(image, reference, capture_id=capture_id, source_kind=source_kind, source=str(Path(path).resolve()))
    result["image_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return write_evidence(image, reference, observation, result, output)


def _teach(args):
    config = read_json(args.draft_config)
    if args.parameters:
        overrides = read_json(args.parameters)
        for section, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(section), dict):
                config[section].update(value)
            else:
                config[section] = value
    if args.roi:
        config["part_roi_xyxy"] = args.roi
    image = read_image(args.image, validate_config(config, draft=True))
    if args.mask:
        mask = cv2.imread(str(args.mask), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ContourInputError("mask unreadable")
        confirmed = args.confirm_reference
    elif args.headless:
        from bmw_inspection.checks.contour_compare.extraction import coarse_segment
        mask = coarse_segment(image, None, config)["mask"].astype(np.uint8)*255
        # Automatic segmentation alone is never an operator-confirmed reference.
        confirmed = False
    else:
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise ContourInputError("GUI display unavailable; use --mask or --headless")
        from bmw_inspection.checks.contour_compare.teaching_ui import edit_reference
        mask, config["part_roi_xyxy"] = edit_reference(image, config)
        confirmed = True
    annotations = read_json(args.annotations) if args.annotations else {}
    annotations.update(mask=mask, confirmed=confirmed, reference_version=args.reference_version)
    reference = teach_reference(image, config, annotations)
    save_reference(reference, args.output_dir)
    print(f"Reference saved: {args.output_dir}/recipe.json; confirmed={reference['config']['reference_confirmed']}; development only")
    return 0


def _evaluate(args):
    reference = load_reference(args.config)
    manifest = Path(args.manifest)
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ContourInputError("empty evaluation manifest")
    identifiers = [r.get("sample_id", "") for r in rows]
    if any(not i or Path(i).name != i or i in (".", "..") or "\\" in i for i in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ContourInputError("sample_id must be unique safe directory names")
    report = []
    with atomic_directory(args.output_dir) as temporary:
        for row in rows:
            sid = row["sample_id"]
            try:
                for key in ("hand", "view_id"):
                    if row.get(key) != reference["config"][key]:
                        raise ContourInputError(f"manifest {key} differs from recipe")
                if row.get("channel") != reference["config"]["image"]["channel"]:
                    raise ContourInputError("manifest channel differs from recipe")
                path = manifest.parent / row["image_path"]
                kind = "fused_only" if row["channel"] == "fused" else "real_exposure"
                result = _inspect_file(path, reference, sid, kind, temporary / sid)
            except (ValueError, TypeError, OSError, KeyError, cv2.error) as exc:
                result = {"status": "ERROR", "capture_id": sid, "reason_codes": [type(exc).__name__], "error": str(exc), "whole_part_release": None}
                with atomic_directory(temporary / sid) as error_dir:
                    write_json(error_dir / "result.json", result)
            report.append({"sample_id": sid, "ground_truth": row.get("ground_truth"), "physical_part_id": row.get("physical_part_id") or None, "placement_group": row.get("placement_group") or None, "split": row.get("split") or None, "status": result["status"], "observed_required_fraction": result.get("observed_required_fraction"), "required_coverage_complete": result.get("required_coverage_complete", False), "unknown_arcs": result.get("unknown_arcs", []), "timing_ms": result.get("timing_ms"), "evidence": f"{sid}/result.json"})
        counts = {s: sum(r["status"] == s for r in report) for s in EXIT_CODES}
        strata = {}
        for labels, name in (({"normal", "OK", "PASS"}, "normal"), ({"defect", "NG"}, "defect")):
            selected = [r for r in report if r["ground_truth"] in labels]
            count = len(selected)
            strata[name] = {"count": count, **{s + "_fraction": sum(r["status"] == s for r in selected)/count if count else None for s in EXIT_CODES}, "non_PASS_fraction": sum(r["status"] != "PASS" for r in selected)/count if count else None}
        times = [r["timing_ms"]["algorithm_total"] for r in report if r["timing_ms"]]
        groups = {}
        for row in rows:
            for key in ("physical_part_id", "placement_group"):
                identity = row.get(key)
                if identity:
                    groups.setdefault((key, identity), set()).add(row.get("split", ""))
        leakage = [{"kind": k, "id": ident, "splits": sorted(splits)} for (k, ident), splits in groups.items() if len(splits) > 1]
        write_json(temporary / "report.json", {"validation_level": "development", "counts": counts, "label_metrics": strata, "coverage_complete_fraction": sum(r["required_coverage_complete"] for r in report)/len(report), "algorithm_ms_p50": float(np.percentile(times, 50)) if times else None, "algorithm_ms_p95": float(np.percentile(times, 95)) if times else None, "split_leakage": leakage, "independent_validation_supported": False, "samples": report})
    print(f"Evaluation completed: {args.output_dir}/report.json; {counts}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    teach = commands.add_parser("teach", help="Reference-only interactive editor or headless mask input")
    teach.add_argument("--image", required=True)
    teach.add_argument("--draft-config", required=True)
    teach.add_argument("--output-dir", required=True)
    teach.add_argument("--mask")
    teach.add_argument("--annotations", help="JSON with unknown_sample_ids and optional structural annotations")
    teach.add_argument("--parameters", help="Explicit JSON configuration overrides")
    teach.add_argument("--roi", type=int, nargs=4)
    teach.add_argument("--headless", action="store_true")
    teach.add_argument("--confirm-reference", action="store_true", help="Operator confirms supplied mask depicts all visible real edges")
    teach.add_argument("--reference-version", default="v001")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--image", required=True)
    inspect.add_argument("--capture-id", required=True)
    inspect.add_argument("--source-kind", choices=("unknown", "fused_only", "real_exposure"), default="unknown")
    inspect.add_argument("--output-dir", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "teach":
            return _teach(args)
        if args.command == "evaluate":
            return _evaluate(args)
        result = _inspect_file(args.image, load_reference(args.config), args.capture_id, args.source_kind, args.output_dir)
        print(f"{result['status']}: full_perimeter_pass={result['full_perimeter_pass']}; {args.output_dir}/result.json")
        return EXIT_CODES[result["status"]]
    except (ValueError, TypeError, OSError, KeyError, cv2.error) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        # Never overwrite an existing run, including partial/error evidence.
        if args.command == "inspect" and not Path(args.output_dir).exists():
            try:
                with atomic_directory(args.output_dir) as temporary:
                    write_json(temporary / "result.json", {"schema_version": 1, "status": "ERROR", "capture_id": args.capture_id, "reason_codes": [type(exc).__name__], "error": str(exc), "whole_part_release": None, "full_perimeter_pass": False})
            except OSError:
                pass  # Original failure remains explicit on stderr and exit code.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
