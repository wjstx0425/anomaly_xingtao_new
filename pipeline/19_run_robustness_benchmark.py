# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 19: summarize robust fused inspection benchmark results."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data import fusion_engine as fusion  # noqa: E402


IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")
DEFECT_TYPES = {"less", "more", "corner", "surface", "crack"}
SLOT_PATTERN = re.compile(r"(?:^|[_/\-])slot(?P<slot>[0-9]+)(?:$|[_/\-.])")


def build_parser() -> argparse.ArgumentParser:
    """Build the robustness benchmark CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--clean-normal-root", type=Path, help="Clean normal/normal_test image root.")
    parser.add_argument("--stress-normal-root", type=Path, help="Locked stress-normal image root.")
    parser.add_argument("--defect-root", type=Path, help="Defect image root.")
    parser.add_argument("--invalid-root", type=Path, help="Optional invalid-input image root.")
    parser.add_argument("--geometry-template-dir", type=Path, help="Accepted for future geometry execution.")
    parser.add_argument("--geometry-thresholds", type=Path, help="Accepted for future geometry execution.")
    parser.add_argument("--geometry-csv", type=Path, help="Existing geometry_predictions.csv report for MVP-1.")
    parser.add_argument("--quality-csv", type=Path, help="Optional quality_gate.csv report.")
    parser.add_argument("--registration-csv", type=Path, help="Optional registration_results.csv report.")
    parser.add_argument("--anomaly-predictions", type=Path, help="Existing AnomalyDINO predictions.csv report.")
    parser.add_argument("--efficientad-predictions", type=Path, help="Optional EfficientAD predictions.csv report.")
    parser.add_argument("--crack-predictions", type=Path, help="Optional crack branch predictions.csv report.")
    parser.add_argument("--fusion-config", type=Path, help="YAML/JSON fusion config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for robustness benchmark reports.")
    return parser


def _load_predictions(args: argparse.Namespace, warnings: list[str]) -> list[fusion.BranchPrediction]:
    """Load all branch predictions accepted by the benchmark wrapper."""
    if args.geometry_csv is None and args.geometry_template_dir is not None:
        warnings.append(
            "geometry-template-dir is recorded but not executed in MVP-1; pass --geometry-csv to fuse geometry.",
        )
    if args.geometry_csv is None and args.geometry_thresholds is not None:
        warnings.append(
            "geometry-thresholds is recorded but not executed in MVP-1; pass --geometry-csv to fuse geometry.",
        )

    return [
        *fusion.load_optional_branch_predictions_csv(args.quality_csv, branch="quality", warnings=warnings),
        *fusion.load_optional_branch_predictions_csv(args.registration_csv, branch="registration", warnings=warnings),
        *fusion.load_optional_branch_predictions_csv(args.geometry_csv, branch="geometry", warnings=warnings),
        *fusion.load_optional_branch_predictions_csv(
            args.anomaly_predictions,
            branch="anomaly_dino",
            warnings=warnings,
        ),
        *fusion.load_optional_branch_predictions_csv(
            args.efficientad_predictions,
            branch="efficient_ad",
            warnings=warnings,
        ),
        *fusion.load_optional_branch_predictions_csv(args.crack_predictions, branch="crack", warnings=warnings),
    ]


def _iter_images(root: Path | None) -> list[Path]:
    """Return image-like files from a benchmark input root."""
    if root is None:
        return []
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [root]
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _aliases_for_path(path: Path) -> set[str]:
    """Return strong path aliases used to detect whether an input already has predictions."""
    aliases = {str(path)}
    try:
        aliases.add(str(path.resolve(strict=False)))
    except OSError:
        pass
    return aliases


def _weak_aliases_for_path(path: Path) -> set[str]:
    """Return basename/stem aliases that are only safe when unique."""
    return {path.name, path.stem}


def _add_unique_alias(index: dict[str, str | None], alias: str, part_id: str) -> None:
    """Add a weak alias to an index, marking conflicting aliases as ambiguous."""
    existing = index.get(alias)
    if existing is None and alias in index:
        return
    if existing is not None and existing != part_id:
        index[alias] = None
        return
    index[alias] = part_id


def _prediction_alias_indices(
    grouped: dict[str, list[fusion.BranchPrediction]],
) -> tuple[dict[str, str], dict[str, str | None]]:
    """Build strong and weak alias-to-part indices from grouped branch predictions."""
    strong_index: dict[str, str] = {}
    weak_index: dict[str, str | None] = {}
    for part_id, predictions in grouped.items():
        strong_index[part_id] = part_id
        for prediction in predictions:
            if prediction.part_id not in prediction.weak_aliases:
                strong_index[prediction.part_id] = part_id
            for alias in prediction.aliases:
                strong_index[alias] = part_id
            for alias in prediction.weak_aliases:
                _add_unique_alias(weak_index, alias, part_id)
            if prediction.source_path:
                for alias in _aliases_for_path(Path(prediction.source_path)):
                    strong_index[alias] = part_id
                for alias in _weak_aliases_for_path(Path(prediction.source_path)):
                    _add_unique_alias(weak_index, alias, part_id)
    return strong_index, weak_index


def _input_weak_alias_counts(roots: tuple[tuple[Path | None, str], ...]) -> Counter[str]:
    """Count basename/stem aliases across all benchmark input roots."""
    counts: Counter[str] = Counter()
    for root, _ in roots:
        for image_path in _iter_images(root):
            counts.update(_weak_aliases_for_path(image_path))
    return counts


def _slot_from_path(path: Path) -> str | None:
    """Infer a slot id from a benchmark input filename when possible."""
    match = SLOT_PATTERN.search(str(path))
    if match is None:
        return None
    return f"slot{int(match.group('slot')):02d}"


def _defect_type_from_path(path: Path, case_type: str) -> str | None:
    """Infer a known defect type from a defect benchmark filename or folder."""
    if case_type != "defect":
        return None
    tokens = re.split(r"[_\-\s.]+", " ".join([path.stem, *path.parts]).lower())
    return next((token for token in tokens if token in DEFECT_TYPES), None)


def _add_unpredicted_inputs(
    grouped: dict[str, list[fusion.BranchPrediction]],
    args: argparse.Namespace,
    warnings: list[str],
) -> dict[str, list[str]]:
    """Add benchmark inputs that have no branch prediction so metrics keep honest denominators."""
    missing_required: dict[str, list[str]] = {}
    roots = (
        (args.clean_normal_root, "clean_normal"),
        (args.stress_normal_root, "stress_normal"),
        (args.defect_root, "defect"),
        (args.invalid_root, "invalid"),
    )
    strong_alias_index, weak_alias_index = _prediction_alias_indices(grouped)
    input_weak_counts = _input_weak_alias_counts(roots)
    missing_count = 0
    for root, case_type in roots:
        for image_path in _iter_images(root):
            strong_aliases = _aliases_for_path(image_path)
            weak_aliases = _weak_aliases_for_path(image_path)
            unique_weak_aliases = {alias for alias in weak_aliases if input_weak_counts[alias] == 1}
            if any(alias in strong_alias_index for alias in strong_aliases):
                continue
            if any(weak_alias_index.get(alias) is not None for alias in unique_weak_aliases):
                continue
            part_id = image_path.stem
            if part_id in grouped:
                part_id = f"{image_path.parent.name}_{part_id}"
            branch = "missing_prediction" if case_type == "invalid" else "benchmark_input"
            reason = (
                "not_evaluated/missing_prediction: invalid input without branch prediction"
                if case_type == "invalid"
                else f"{case_type} input without branch prediction"
            )
            grouped[part_id] = [
                fusion.BranchPrediction(
                    part_id=part_id,
                    side="unknown",
                    view=None,
                    slot_id=_slot_from_path(image_path),
                    branch=branch,
                    pred_label=0,
                    score=None,
                    threshold=None,
                    defect_type=_defect_type_from_path(image_path, case_type),
                    reason=reason,
                    source_path=str(image_path),
                    aliases=tuple(sorted(strong_aliases)),
                    weak_aliases=tuple(sorted(weak_aliases)),
                    gt_defect_type=_defect_type_from_path(image_path, case_type),
                ),
            ]
            missing_required[part_id] = [
                "not_evaluated/missing_prediction" if case_type == "invalid" else "branch_prediction",
            ]
            for alias in strong_aliases:
                strong_alias_index[alias] = part_id
            for alias in unique_weak_aliases:
                _add_unique_alias(weak_alias_index, alias, part_id)
            missing_count += 1
    if missing_count:
        warnings.append(f"{missing_count} benchmark inputs had no branch prediction and were marked INVALID_CAPTURE.")
    return missing_required


def _aggregate_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Aggregate benchmark rows by defect type or slot."""
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "defect_total": 0, "detected": 0, "normal_total": 0, "false_positives": 0},
    )
    for row in rows:
        value = str(row.get(key) or "unknown")
        item = grouped[value]
        item["total"] += 1
        status = str(row.get("final_status") or "")
        case_type = str(row.get("case_type") or "")
        if case_type == "defect":
            item["defect_total"] += 1
            if status.startswith("NG_"):
                item["detected"] += 1
        if case_type in {"clean_normal", "stress_normal"}:
            item["normal_total"] += 1
            if status.startswith("NG_"):
                item["false_positives"] += 1

    output = []
    for value, counts in sorted(grouped.items()):
        output.append(
            {
                key: value,
                "total": counts["total"],
                "defect_total": counts["defect_total"],
                "detected": counts["detected"],
                "normal_total": counts["normal_total"],
                "false_positives": counts["false_positives"],
                "recall": (
                    counts["detected"] / counts["defect_total"] if counts["defect_total"] else 0.0
                ),
            },
        )
    return output


def run_benchmark(args: argparse.Namespace) -> dict[str, float | int]:
    """Run CSV-only robustness benchmarking and write reports."""
    config = fusion.load_fusion_config(args.fusion_config)
    warnings: list[str] = []
    predictions = _load_predictions(args, warnings)
    grouped = fusion.group_predictions_by_part(predictions)
    missing_required = _add_unpredicted_inputs(grouped, args, warnings)
    for part_id, part_predictions in grouped.items():
        configured_missing = fusion.missing_required_views(part_predictions, config)
        if configured_missing:
            missing_required.setdefault(part_id, [])
            missing_required[part_id].extend(configured_missing)
    for part_id in missing_required:
        missing_required[part_id] = sorted(set(missing_required[part_id]))
    if not grouped:
        msg = "No branch predictions were available for robustness benchmarking."
        raise RuntimeError(msg)

    decisions = fusion.fuse_grouped_predictions(grouped, missing_required_by_part=missing_required, config=config)
    branch_trace_predictions = [prediction for part_predictions in grouped.values() for prediction in part_predictions]
    summary = fusion.compute_benchmark_summary(
        decisions,
        grouped,
        clean_normal_root=args.clean_normal_root,
        stress_normal_root=args.stress_normal_root,
        defect_root=args.defect_root,
        invalid_root=args.invalid_root,
    )
    summary["missing_prediction_count"] = len(missing_required)
    detail_rows = fusion.benchmark_detail_rows(
        decisions,
        grouped,
        clean_normal_root=args.clean_normal_root,
        stress_normal_root=args.stress_normal_root,
        defect_root=args.defect_root,
        invalid_root=args.invalid_root,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fusion.write_branch_predictions_csv(branch_trace_predictions, args.output_dir / "branch_predictions.csv")
    fusion.write_fused_decisions_csv(decisions, args.output_dir / "fused_predictions.csv")
    fusion.write_dict_rows(
        args.output_dir / "robustness_summary.csv",
        [summary],
        list(summary),
    )
    detail_fieldnames = [
        "part_id",
        "case_type",
        "final_status",
        "final_label",
        "defect_type",
        "gt_defect_type",
        "evidence_type",
        "slot_id",
        "triggered_branch",
        "reason",
    ]
    fusion.write_dict_rows(args.output_dir / "robustness_details.csv", detail_rows, detail_fieldnames)
    fusion.write_dict_rows(args.output_dir / "by_defect_type.csv", _aggregate_rows(detail_rows, "defect_type"), [
        "defect_type",
        "total",
        "defect_total",
        "detected",
        "normal_total",
        "false_positives",
        "recall",
    ])
    fusion.write_dict_rows(args.output_dir / "by_slot.csv", _aggregate_rows(detail_rows, "slot_id"), [
        "slot_id",
        "total",
        "defect_total",
        "detected",
        "normal_total",
        "false_positives",
        "recall",
    ])
    fusion.write_dict_rows(
        args.output_dir / "misses.csv",
        [row for row in detail_rows if row["case_type"] == "defect" and not str(row["final_status"]).startswith("NG_")],
        detail_fieldnames,
    )
    fusion.write_dict_rows(
        args.output_dir / "false_positives.csv",
        [
            row
            for row in detail_rows
            if row["case_type"] in {"clean_normal", "stress_normal", "unknown"}
            and str(row["final_status"]).startswith("NG_")
        ],
        detail_fieldnames,
    )
    fusion.write_dict_rows(
        args.output_dir / "retake_cases.csv",
        [row for row in detail_rows if row["final_status"] == "RETAKE"],
        detail_fieldnames,
    )
    fusion.write_summary_markdown(
        args.output_dir / "robustness_summary.md",
        "Robustness Benchmark Summary",
        summary,
        warnings,
    )
    return summary


def main() -> None:
    """Run robustness benchmark summary generation."""
    args = build_parser().parse_args()
    summary = run_benchmark(args)
    print(f"Benchmarked {summary['total_parts']} parts")
    print(f"Wrote robustness summary: {args.output_dir / 'robustness_summary.md'}")


if __name__ == "__main__":
    main()
