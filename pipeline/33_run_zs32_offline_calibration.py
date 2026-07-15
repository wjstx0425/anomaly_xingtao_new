# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: EM102, TRY003

"""Run resumable ZS32 template/PatchCore/YOLO commissioning calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from capture_data.zs32_offline_calibration import OfflineCalibrationCase

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.fusion_calibration import run_calibration  # noqa: E402
from capture_data.zs32_model_runtime import ZS32ModelRuntime, load_runtime_config  # noqa: E402
from capture_data.zs32_offline_calibration import (  # noqa: E402
    load_offline_cases,
    merge_calibration_rows,
    run_offline_batch,
    validate_yolo_annotation_dataset,
    validate_yolo_runtime_crop_identity,
    write_case_index,
    write_yolo_annotation_calibration_rows,
)
from capture_data.zs32_template_gate import load_model  # noqa: E402
from capture_data.zs32_yolo_auxiliary_calibration import run_yolo_auxiliary_calibration  # noqa: E402

from zs32_inspection.domain.views import VIEW_ORDER  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage33 commissioning parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--template-calibration-csv", type=Path, required=True)
    parser.add_argument("--template-model-dir", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--yolo-dataset-root", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hand", choices=("right",), default="right")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--target-recall", type=float, default=1.0)
    parser.add_argument("--normal-quantile", type=float, default=0.995)
    parser.add_argument(
        "--yolo-aux-min-image-precision",
        type=float,
        default=None,
        help="Publish an optional val-selected per-view YOLO image-presence high-precision report.",
    )
    parser.add_argument(
        "--yolo-aux-use-test-for-selection",
        action="store_true",
        help="TEMPORARY: merge test into YOLO auxiliary threshold selection and mark the result as leaked.",
    )
    parser.add_argument(
        "--reuse-existing-inference",
        action="store_true",
        help="Reuse complete case outputs under --output-dir and run only aggregation/calibration publication.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--commissioning-only",
        action="store_true",
        required=True,
        help="Acknowledge that current splits/labels are not a production-independent calibration contract.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_contract(cases: Sequence[OfflineCalibrationCase]) -> list[dict[str, Any]]:
    """Bind every selected physical identity to the bytes scored by the models."""
    return [
        {
            "part_id": case.part_id,
            "gt_label": case.gt_label,
            "split": case.split,
            "session_id": case.session_id,
            "group_id": case.group_id,
            "sources": {view: {"path": str(path), "sha256": _sha256(path)} for view, path in case.images.items()},
        }
        for case in cases
    ]


def _yolo_dataset_contract(root: Path) -> dict[str, Any]:
    """Hash the YOLO split manifests and labels that define per-view targets."""
    root = root.expanduser().resolve()
    records = []
    for split in ("train", "val", "test"):
        label_dir = root / "labels" / split
        image_dir = root / "images" / split
        if not label_dir.is_dir() or not image_dir.is_dir():
            raise FileNotFoundError(f"YOLO split is incomplete: {split}")
        for label_path in sorted(label_dir.glob("*.txt")):
            image_matches = [path for path in image_dir.glob(f"{label_path.stem}.*") if path.is_file()]
            if len(image_matches) != 1:
                raise ValueError(f"YOLO label must match exactly one image: {label_path}")
            records.append(
                {
                    "split": split,
                    "stem": label_path.stem,
                    "label_sha256": _sha256(label_path),
                    "image_path": str(image_matches[0]),
                    "image_size": image_matches[0].stat().st_size,
                    "image_sha256": _sha256(image_matches[0]),
                },
            )
    data_yaml = root / "data.yaml"
    return {
        "root": str(root),
        "data_yaml_sha256": _sha256(data_yaml) if data_yaml.is_file() else None,
        "record_count": len(records),
        "records_sha256": _canonical_sha256(records),
    }


def build_run_contract(
    args: argparse.Namespace,
    cases: Sequence[OfflineCalibrationCase],
) -> dict[str, Any]:
    """Build immutable provenance for inference, labels, splits, and fit parameters."""
    template_model = args.template_model_dir.expanduser().resolve() / "model.json"
    if not template_model.is_file():
        raise FileNotFoundError(f"template model does not exist: {template_model}")
    template_payload = load_model(template_model.parent)
    config = load_runtime_config(args.runtime_config.resolve())
    template_versions = template_payload["versions"]
    runtime_roi_versions = {config.versions.patchcore_roi, config.versions.yolo_roi}
    if runtime_roi_versions != {template_versions["roi"]}:
        raise ValueError(
            "runtime/Template ROI version mismatch: "
            f"runtime={sorted(runtime_roi_versions)}, template={template_versions['roi']!r}",
        )
    if config.versions.template != template_versions["template"]:
        raise ValueError(
            "runtime/Template generation mismatch: "
            f"runtime={config.versions.template!r}, template={template_versions['template']!r}",
        )
    validate_yolo_annotation_dataset(cases, args.yolo_dataset_root)
    source_records = _source_contract(cases)
    return {
        "schema_version": "1.0",
        "commissioning_only": True,
        "hand": args.hand,
        "target_recall": args.target_recall,
        "normal_quantile": args.normal_quantile,
        "crop_manifest": str(args.crop_manifest.resolve()),
        "crop_manifest_sha256": _sha256(args.crop_manifest.resolve()),
        "template_calibration_csv": str(args.template_calibration_csv.resolve()),
        "template_calibration_csv_sha256": _sha256(args.template_calibration_csv.resolve()),
        "template_model": str(template_model),
        "template_model_sha256": _sha256(template_model),
        "runtime_config": str(args.runtime_config.resolve()),
        "runtime_config_sha256": _sha256(args.runtime_config.resolve()),
        "source_record_count": len(source_records),
        "source_records_sha256": _canonical_sha256(source_records),
        "yolo_dataset": _yolo_dataset_contract(args.yolo_dataset_root),
    }


def prepare_output_contract(
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    """Create or validate the run contract before any case is reused."""
    output_dir = output_dir.expanduser().resolve()
    contract_path = output_dir / "commissioning_run_contract.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists; pass --resume: {output_dir}")
        if not contract_path.is_file():
            raise ValueError(f"existing output has no commissioning run contract: {contract_path}")
        if json.loads(contract_path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"existing output run contract differs from current inputs: {contract_path}")
        return contract_path
    output_dir.mkdir(parents=True)
    temporary = contract_path.with_name(f".{contract_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(contract_path)
    return contract_path


def _write_metadata(
    args: argparse.Namespace,
    cases: Sequence[OfflineCalibrationCase],
    metrics: dict[str, object],
) -> Path:
    """Publish explicit commissioning provenance and safety limitations."""
    output_dir = args.output_dir.expanduser().resolve()
    path = output_dir / "commissioning_metadata.json"
    split_counts = Counter((case.split, case.gt_label) for case in cases)
    payload = {
        "schema_version": "1.0",
        "commissioning_only": True,
        "production_locked_thresholds": False,
        "data_leakage_risk": True,
        "view_routing": "source basename semantic view; manifest resolved_view is trace-only",
        "model_branch_count": 3,
        "versioned_threshold_group_count": 18,
        "strict_zs32_right_required_group_count": 36,
        "limitations": [
            "PatchCore and template training splits are not aligned with one shared held-out calibration contract.",
            "YOLO labels are physical-part labels, not approved per-view branch targets or bounding-box ground truth.",
            "Quality, registration, and geometry threshold groups are not included.",
            "The held-out test split contains only four defect physical parts.",
            "This artifact is for offline commissioning and must not be presented as production locked thresholds.",
        ],
        "inputs": {
            "crop_manifest": str(args.crop_manifest.resolve()),
            "crop_manifest_sha256": _sha256(args.crop_manifest.resolve()),
            "template_calibration_csv": str(args.template_calibration_csv.resolve()),
            "template_calibration_csv_sha256": _sha256(args.template_calibration_csv.resolve()),
            "template_model_dir": str(args.template_model_dir.resolve()),
            "runtime_config": str(args.runtime_config.resolve()),
            "runtime_config_sha256": _sha256(args.runtime_config.resolve()),
            "yolo_dataset": str(args.yolo_dataset_root.resolve()),
            "template_model_sha256": _sha256(args.template_model_dir.resolve() / "model.json"),
        },
        "fit_parameters": {
            "target_recall": args.target_recall,
            "normal_quantile": args.normal_quantile,
        },
        "case_counts": {
            "total": len(cases),
            "calibration_normal": split_counts["calibration", 0],
            "calibration_defect": split_counts["calibration", 1],
            "test_normal": split_counts["test", 0],
            "test_defect": split_counts["test", 1],
        },
        "calibration_valid_for_observed_18_groups": bool(metrics["overall"]["calibration_valid"]),
        "threshold_artifact": str(output_dir / "threshold_calibration" / "thresholds.json"),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def build_threshold_quality_report(
    threshold_payload: dict[str, object],
    metrics: dict[str, object],
    *,
    target_recall: float = 1.0,
) -> dict[str, object]:
    """Reject mathematically valid but operationally degenerate commissioning thresholds."""
    records = threshold_payload.get("thresholds", [])
    yolo_records = [record for record in records if isinstance(record, dict) and record.get("branch") == "yolo"]
    invalid_thresholds = [
        record["view"]
        for record in yolo_records
        if _finite_float(record.get("low_threshold")) is None or _finite_float(record.get("high_threshold")) is None
    ]
    zero_low = [
        record["view"]
        for record in yolo_records
        if (value := _finite_float(record.get("low_threshold"))) is not None and value <= 0
    ]
    zero_high = [
        record["view"]
        for record in yolo_records
        if (value := _finite_float(record.get("high_threshold"))) is not None and value <= 0
    ]
    overall = metrics["overall"]
    reasons = [
        "commissioning artifact contains only 18/36 strict right-profile groups",
        "combined YOLO branch uses physical-part labels instead of per-view bbox targets",
    ]
    if not overall["calibration_valid"]:
        reasons.append("Stage31 algorithm calibration is invalid")
    recall = overall["non_clear_recall"]
    if recall is None or float(recall) < target_recall:
        reasons.append(f"held-out non-clear recall is below target {target_recall}: {recall}")
    if invalid_thresholds:
        reasons.append(f"YOLO null or non-finite threshold groups: {invalid_thresholds}")
    if zero_low:
        reasons.append(f"YOLO zero low threshold groups: {zero_low}")
    if zero_high:
        reasons.append(f"YOLO zero high threshold groups: {zero_high}")
    if float(overall["normal_reject_rate"] or 0) > 0.2:
        reasons.append(f"held-out normal strong-reject rate is {overall['normal_reject_rate']}")
    if int(overall["defect_part_count"]) < 10:
        reasons.append(f"held-out defect part count is only {overall['defect_part_count']}")
    return {
        "schema_version": "1.0",
        "commissioning_only": True,
        "algorithm_calibration_valid": bool(overall["calibration_valid"]),
        "usable_for_runtime_threshold_injection": not reasons,
        "yolo_zero_low_views": zero_low,
        "yolo_zero_high_views": zero_high,
        "yolo_invalid_threshold_views": invalid_thresholds,
        "heldout_normal_reject_rate": overall["normal_reject_rate"],
        "heldout_non_clear_recall": overall["non_clear_recall"],
        "heldout_defect_part_count": overall["defect_part_count"],
        "target_recall": target_recall,
        "rejection_reasons": reasons,
    }


def _finite_float(value: object) -> float | None:
    """Return one finite float or None for missing/invalid threshold values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _write_quality_report(output_dir: Path, metrics: dict[str, object], *, target_recall: float) -> Path:
    """Publish the operational threshold acceptance result separately from fit metrics."""
    thresholds_path = output_dir / "threshold_calibration" / "thresholds.json"
    threshold_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    report = build_threshold_quality_report(threshold_payload, metrics, target_recall=target_recall)
    path = output_dir / "threshold_quality_report.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _write_filtered_rows(
    source: Path,
    output: Path,
    predicate: Callable[[dict[str, str]], bool],
) -> int:
    """Publish a deterministic Stage31 subset without changing score values."""
    with source.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if predicate(row)]
    if not fieldnames or not rows:
        raise ValueError(f"filtered calibration rows are empty: {output}")
    if output.exists():
        with output.open(encoding="utf-8", newline="") as file:
            if list(csv.DictReader(file)) != rows:
                raise ValueError(f"existing filtered calibration rows differ: {output}")
        return len(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return len(rows)


def _publication_payload(
    input_csv: Path,
    report_dir: Path,
    args: argparse.Namespace,
    required_views: Sequence[str],
) -> dict[str, Any]:
    """Bind one Stage31-derived directory to its exact input and parameters."""
    return {
        "schema_version": "1.0",
        "commissioning_only": True,
        "input_csv": str(input_csv.resolve()),
        "input_csv_sha256": _sha256(input_csv),
        "target_recall": args.target_recall,
        "normal_quantile": args.normal_quantile,
        "required_views": list(required_views),
        "thresholds_sha256": _sha256(report_dir / "thresholds.json"),
        "metrics_sha256": _sha256(report_dir / "calibration_metrics.json"),
    }


def _run_or_validate_calibration(
    input_csv: Path,
    report_dir: Path,
    args: argparse.Namespace,
    *,
    required_views: tuple[str, ...] = VIEW_ORDER,
) -> dict[str, object]:
    """Run calibration once or validate every reusable artifact against a sidecar."""
    with input_csv.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for view in required_views:
        fit_labels = {
            row.get("gt_label", "").strip()
            for row in rows
            if row.get("split", "").strip() == "calibration" and row.get("view", "").strip() == view
        }
        if fit_labels != {"0", "1"}:
            raise ValueError(
                f"view {view!r} requires both normal and defect calibration rows; found labels {sorted(fit_labels)}",
            )
    sidecar = report_dir / "stage33_publication.json"
    if report_dir.exists():
        if not args.resume:
            raise FileExistsError(f"threshold publication already exists: {report_dir}")
        if not sidecar.is_file():
            raise ValueError(f"threshold publication has no Stage33 provenance sidecar: {sidecar}")
        payload = _publication_payload(input_csv, report_dir, args, required_views)
        if json.loads(sidecar.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"threshold publication differs from current input or parameters: {report_dir}")
        return json.loads((report_dir / "calibration_metrics.json").read_text(encoding="utf-8"))
    metrics = run_calibration(
        input_csv,
        report_dir,
        target_recall=args.target_recall,
        normal_quantile=args.normal_quantile,
        required_views=required_views,
    )
    payload = _publication_payload(input_csv, report_dir, args, required_views)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _run_yolo_per_view_calibration(output_dir: Path, yolo_rows: Path, args: argparse.Namespace) -> Path:
    """Calibrate YOLO image targets independently so part/view labels stay valid."""
    base = output_dir / "yolo_annotation_thresholds_by_view"
    summaries = {}
    for view in VIEW_ORDER:
        view_csv = output_dir / "yolo_annotation_by_view" / f"{view}.csv"
        _write_filtered_rows(yolo_rows, view_csv, lambda row, expected=view: row["view"] == expected)
        report_dir = base / view
        metrics = _run_or_validate_calibration(view_csv, report_dir, args, required_views=(view,))
        thresholds = json.loads((report_dir / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
        if len(thresholds) != 1:
            raise ValueError(f"expected one YOLO threshold record for {view}")
        record = thresholds[0]
        recall = metrics["overall"]["non_clear_recall"]
        normal_reject_rate = metrics["overall"]["normal_reject_rate"]
        low_threshold = _finite_float(record.get("low_threshold"))
        high_threshold = _finite_float(record.get("high_threshold"))
        rejection_reasons = []
        if not metrics["overall"]["calibration_valid"]:
            rejection_reasons.append("calibration_invalid")
        if (
            low_threshold is None
            or high_threshold is None
            or low_threshold <= 0
            or high_threshold <= 0
            or low_threshold == high_threshold
        ):
            rejection_reasons.append("degenerate_thresholds")
        if int(record["defect_count"]) < 5:
            rejection_reasons.append("fit_positive_count_below_5")
        if int(metrics["overall"]["defect_part_count"]) < 5:
            rejection_reasons.append("test_positive_count_below_5")
        if recall is None or float(recall) < args.target_recall:
            rejection_reasons.append("heldout_recall_below_target")
        if normal_reject_rate is None or float(normal_reject_rate) > 0.2:
            rejection_reasons.append("heldout_normal_reject_rate_above_0.2")
        summaries[view] = {
            "calibration_valid": metrics["overall"]["calibration_valid"],
            "fit_normal_count": record["normal_count"],
            "fit_defect_count": record["defect_count"],
            "test_normal_count": metrics["overall"]["normal_part_count"],
            "test_defect_count": metrics["overall"]["defect_part_count"],
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "test_non_clear_recall": recall,
            "test_normal_reject_rate": normal_reject_rate,
            "test_review_rate": metrics["overall"]["review_rate"],
            "operationally_usable": not rejection_reasons,
            "operationally_degenerate": "degenerate_thresholds" in rejection_reasons,
            "rejection_reasons": rejection_reasons,
        }
    summary_path = base / "summary.json"
    payload = {
        "schema_version": "1.0",
        "commissioning_only": True,
        "label_semantics": "per-view YOLO box presence",
        "fit_source_split": "val",
        "evaluation_source_split": "test",
        "acceptance_contract": {
            "minimum_fit_positive_count": 5,
            "minimum_test_positive_count": 5,
            "minimum_heldout_non_clear_recall": args.target_recall,
            "maximum_heldout_normal_reject_rate": 0.2,
            "requires_distinct_positive_thresholds": True,
        },
        "all_views_operationally_usable": all(item["operationally_usable"] for item in summaries.values()),
        "views": summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary_path


def _auxiliary_publication_payload(
    input_csv: Path,
    output_dir: Path,
    commissioning_root: Path,
    minimum_fit_precision: float,
    *,
    use_test_for_selection: bool,
) -> dict[str, Any]:
    """Bind the optional auxiliary policy to inputs, parameters, and outputs."""
    payload = {
        "schema_version": "1.0",
        "commissioning_only": True,
        "algorithm": (
            "per_view_high_precision_auxiliary_test_leakage_v1"
            if use_test_for_selection
            else "per_view_high_precision_auxiliary_v1"
        ),
        "input_csv": str(input_csv.resolve()),
        "input_csv_sha256": _sha256(input_csv),
        "commissioning_run_contract_sha256": _sha256(commissioning_root / "commissioning_run_contract.json"),
        "yolo_runtime_crop_identity_sha256": _sha256(commissioning_root / "yolo_runtime_crop_identity.json"),
        "minimum_fit_image_presence_precision": minimum_fit_precision,
        "threshold_floor_exclusive": 0.0,
        "fit_split": "calibration",
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "thresholds_sha256": _sha256(output_dir / "thresholds.json"),
        "summary_sha256": _sha256(output_dir / "summary.json"),
    }
    if use_test_for_selection:
        payload.update(
            {
                "fit_split": "calibration+test",
                "evaluation_split": "test_reused_for_selection",
                "test_used_for_selection": True,
                "data_leakage": True,
            },
        )
    return payload


def _run_or_validate_yolo_auxiliary(
    input_csv: Path,
    commissioning_root: Path,
    args: argparse.Namespace,
) -> Path | None:
    """Publish or validate the explicitly requested YOLO auxiliary policy."""
    minimum_precision = args.yolo_aux_min_image_precision
    if minimum_precision is None:
        return None
    if not 0 < minimum_precision <= 1:
        message = "--yolo-aux-min-image-precision must be within (0, 1]"
        raise ValueError(message)
    use_test_for_selection = args.yolo_aux_use_test_for_selection
    output_name = (
        "yolo_high_precision_auxiliary_test_leakage"
        if use_test_for_selection
        else "yolo_high_precision_auxiliary"
    )
    output_dir = commissioning_root / output_name
    sidecar = output_dir / "stage33_publication.json"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"YOLO auxiliary publication already exists: {output_dir}")
        if not sidecar.is_file():
            raise ValueError(f"YOLO auxiliary publication has no provenance sidecar: {sidecar}")
        payload = _auxiliary_publication_payload(
            input_csv,
            output_dir,
            commissioning_root,
            minimum_precision,
            use_test_for_selection=use_test_for_selection,
        )
        if json.loads(sidecar.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"YOLO auxiliary publication differs from current inputs: {output_dir}")
        return output_dir / "summary.json"
    publication_dir = output_dir.with_name(f".{output_dir.name}.publication.tmp")
    if publication_dir.exists():
        raise FileExistsError(f"stale YOLO auxiliary publication staging exists: {publication_dir}")
    try:
        run_yolo_auxiliary_calibration(
            input_csv,
            publication_dir,
            minimum_fit_image_precision=minimum_precision,
            use_test_for_selection=use_test_for_selection,
        )
        payload = _auxiliary_publication_payload(
            input_csv,
            publication_dir,
            commissioning_root,
            minimum_precision,
            use_test_for_selection=use_test_for_selection,
        )
        publication_sidecar = publication_dir / sidecar.name
        publication_sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        publication_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(publication_dir, ignore_errors=True)
        raise
    return output_dir / "summary.json"


def main() -> None:
    """Run preflight, persistent inference, aggregation, and provisional calibration."""
    args = build_parser().parse_args()
    if args.yolo_aux_min_image_precision is not None and not 0 < args.yolo_aux_min_image_precision <= 1:
        message = "--yolo-aux-min-image-precision must be within (0, 1]"
        raise ValueError(message)
    cases = load_offline_cases(
        args.crop_manifest.resolve(),
        args.template_calibration_csv.resolve(),
        path_root=args.path_root.resolve(),
        hand=args.hand,
    )
    run_contract = build_run_contract(args, cases)
    config = load_runtime_config(args.runtime_config.resolve())
    counts = Counter((case.split, case.gt_label) for case in cases)
    print(
        "cases: "
        f"{len(cases)} "
        f"calibration(normal={counts['calibration', 0]}, defect={counts['calibration', 1]}) "
        f"test(normal={counts['test', 0]}, defect={counts['test', 1]})",
    )
    if args.preflight_only:
        print("preflight: ok")
        return

    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_contract(output_dir, run_contract, resume=args.resume)
    if args.reuse_existing_inference:
        stats = {"case_count": len(cases), "completed": 0, "resumed": len(cases)}
    else:
        runtime = ZS32ModelRuntime(
            config,
            accelerator=args.accelerator,
            devices=args.devices,
            yolo_device=args.yolo_device,
        )
        stats = run_offline_batch(
            cases,
            runtime,
            args.template_model_dir.resolve(),
            output_dir,
            resume=True,
        )
    matched_yolo_crops = validate_yolo_runtime_crop_identity(
        cases,
        output_dir / "cases",
        args.yolo_dataset_root,
    )
    crop_identity_report = output_dir / "yolo_runtime_crop_identity.json"
    crop_identity_payload = {
        "schema_version": "1.0",
        "commissioning_only": True,
        "pixel_identity_valid": True,
        "matched_crop_count": matched_yolo_crops,
        "yolo_dataset": str(args.yolo_dataset_root.resolve()),
        "runtime_config_sha256": _sha256(args.runtime_config.resolve()),
    }
    crop_identity_temporary = crop_identity_report.with_name(f".{crop_identity_report.name}.tmp")
    crop_identity_temporary.write_text(
        json.dumps(crop_identity_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    crop_identity_temporary.replace(crop_identity_report)
    write_case_index(cases, output_dir / "case_index.csv")
    aggregate = output_dir / "calibration_rows.csv"
    merged_count = merge_calibration_rows(cases, output_dir / "cases", aggregate)
    threshold_dir = output_dir / "threshold_calibration"
    metrics = _run_or_validate_calibration(aggregate, threshold_dir, args)
    metadata = _write_metadata(args, cases, metrics)
    quality_report = _write_quality_report(output_dir, metrics, target_recall=args.target_recall)
    template_patchcore_rows = output_dir / "template_patchcore_calibration_rows.csv"
    _write_filtered_rows(aggregate, template_patchcore_rows, lambda row: row["branch"] != "yolo")
    template_patchcore_dir = output_dir / "template_patchcore_threshold_calibration"
    _run_or_validate_calibration(template_patchcore_rows, template_patchcore_dir, args)
    yolo_rows = output_dir / "yolo_annotation_calibration_rows.csv"
    write_yolo_annotation_calibration_rows(cases, aggregate, args.yolo_dataset_root, yolo_rows)
    yolo_threshold_dir = output_dir / "yolo_annotation_threshold_calibration"
    _run_or_validate_calibration(yolo_rows, yolo_threshold_dir, args)
    yolo_by_view_summary = _run_yolo_per_view_calibration(output_dir, yolo_rows, args)
    yolo_auxiliary_summary = _run_or_validate_yolo_auxiliary(yolo_rows, output_dir, args)
    print(f"batch: {stats}")
    print(f"yolo_runtime_crop_identity: {crop_identity_report} ({matched_yolo_crops})")
    print(f"calibration_rows: {aggregate} ({merged_count})")
    print(f"thresholds: {threshold_dir / 'thresholds.json'}")
    print(f"commissioning_metadata: {metadata}")
    print(f"threshold_quality_report: {quality_report}")
    print(f"yolo_annotation_rows: {yolo_rows}")
    print(f"yolo_annotation_thresholds: {yolo_threshold_dir / 'thresholds.json'}")
    print(f"template_patchcore_thresholds: {template_patchcore_dir / 'thresholds.json'}")
    print(f"yolo_by_view_summary: {yolo_by_view_summary}")
    if yolo_auxiliary_summary is not None:
        print(f"yolo_high_precision_auxiliary: {yolo_auxiliary_summary}")
    print(f"heldout_metrics: {threshold_dir / 'calibration_metrics.json'}")


if __name__ == "__main__":
    main()
