# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 18: fuse quality, geometry, and anomaly inspection results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data import fusion_engine as fusion  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the fusion CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, help="Optional multiview manifest CSV with part_id/side/view rows.")
    parser.add_argument("--quality-csv", type=Path, help="Optional quality_gate.csv report.")
    parser.add_argument("--registration-csv", type=Path, help="Optional registration_results.csv report.")
    parser.add_argument("--geometry-csv", type=Path, help="Optional geometry_predictions.csv report.")
    parser.add_argument("--anomaly-csv", type=Path, help="Optional AnomalyDINO predictions.csv report.")
    parser.add_argument("--efficientad-csv", type=Path, help="Optional EfficientAD predictions.csv report.")
    parser.add_argument("--crack-csv", type=Path, help="Optional crack branch predictions.csv report.")
    parser.add_argument(
        "--branch-csv",
        action="append",
        default=[],
        help="Additional branch CSV as BRANCH=PATH, e.g. yolo=results/yolo_predictions.csv.",
    )
    parser.add_argument("--fusion-config", type=Path, help="YAML/JSON fusion config.")
    parser.add_argument(
        "--required-view",
        action="append",
        default=[],
        help="Required side:view key, e.g. top:uniform. Can be provided multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for fused CSV reports.")
    return parser


def _load_custom_branch_predictions(
    values: list[str],
    warnings: list[str],
) -> list[fusion.BranchPrediction]:
    """Load arbitrary branch CSVs passed as ``BRANCH=PATH`` values."""
    predictions: list[fusion.BranchPrediction] = []
    for value in values:
        if "=" not in value:
            msg = f"--branch-csv must be BRANCH=PATH, got: {value}"
            raise ValueError(msg)
        branch, path_text = value.split("=", maxsplit=1)
        branch = branch.strip()
        path_text = path_text.strip()
        if not branch or not path_text:
            msg = f"--branch-csv must include non-empty BRANCH and PATH, got: {value}"
            raise ValueError(msg)
        predictions.extend(
            fusion.load_optional_branch_predictions_csv(Path(path_text), branch=branch, warnings=warnings),
        )
    return predictions


def run_fusion(args: argparse.Namespace) -> list[fusion.FusedDecision]:
    """Run CSV-based fusion and write branch/fused reports."""
    config = fusion.load_fusion_config(args.fusion_config)
    warnings: list[str] = []
    predictions = [
        *fusion.load_optional_branch_predictions_csv(args.quality_csv, branch="quality", warnings=warnings),
        *fusion.load_optional_branch_predictions_csv(args.registration_csv, branch="registration", warnings=warnings),
        *fusion.load_optional_branch_predictions_csv(args.geometry_csv, branch="geometry", warnings=warnings),
        *fusion.load_optional_branch_predictions_csv(args.anomaly_csv, branch="anomaly_dino", warnings=warnings),
        *fusion.load_optional_branch_predictions_csv(args.efficientad_csv, branch="efficient_ad", warnings=warnings),
        *fusion.load_optional_branch_predictions_csv(args.crack_csv, branch="crack", warnings=warnings),
        *_load_custom_branch_predictions(args.branch_csv, warnings),
    ]

    missing_required = fusion.load_manifest_missing_required(
        args.manifest,
        config,
        explicit_required_views=args.required_view,
    )
    grouped = fusion.group_predictions_by_part(predictions)
    for part_id, part_predictions in grouped.items():
        configured_missing = fusion.missing_required_views(part_predictions, config, args.required_view)
        if configured_missing:
            missing_required.setdefault(part_id, [])
            missing_required[part_id].extend(configured_missing)
    for part_id in missing_required:
        missing_required[part_id] = sorted(set(missing_required[part_id]))
        grouped.setdefault(part_id, [])
    if not grouped:
        msg = "No branch predictions or manifest records were available to fuse."
        raise RuntimeError(msg)

    decisions = fusion.fuse_grouped_predictions(grouped, missing_required_by_part=missing_required, config=config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fusion.write_branch_predictions_csv(predictions, args.output_dir / "branch_predictions.csv")
    fusion.write_fused_decisions_csv(decisions, args.output_dir / "fused_predictions.csv")
    summary = {
        "parts": len(decisions),
        "branches": len(predictions),
        "ok": sum(decision.final_status == "OK" for decision in decisions),
        "ng": sum(decision.final_status.startswith("NG_") for decision in decisions),
        "retake": sum(decision.final_status == "RETAKE" for decision in decisions),
        "invalid_capture": sum(decision.final_status == "INVALID_CAPTURE" for decision in decisions),
        "suspect": sum(decision.final_status == "SUSPECT" for decision in decisions),
    }
    fusion.write_summary_markdown(args.output_dir / "summary.md", "Inspection Fusion Summary", summary, warnings)
    return decisions


def main() -> None:
    """Run inspection result fusion."""
    args = build_parser().parse_args()
    decisions = run_fusion(args)
    print(f"Fused {len(decisions)} parts")
    print(f"Wrote fused predictions: {args.output_dir / 'fused_predictions.csv'}")


if __name__ == "__main__":
    main()
