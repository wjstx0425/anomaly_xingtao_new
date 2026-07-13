# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 18: fuse quality, geometry, and anomaly inspection results."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data import fusion_engine as fusion  # noqa: E402
from capture_data.inspection_audit import build_part_audit, write_part_audit  # noqa: E402

ZS32_PROFILE_PATH = REPO_ROOT / "config/fusion/zs32_six_view.json"


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
    parser.add_argument("--profile", choices=("zs32",), help="Named strict fusion profile.")
    parser.add_argument("--audit-dir", type=Path, help="ZS32 per-part audit JSON directory.")
    parser.add_argument(
        "--require-complete-evidence",
        action="store_true",
        help="Prevent OK release when source or evidence artifacts are missing.",
    )
    parser.add_argument(
        "--required-view",
        action="append",
        default=[],
        help="Required side:view key, e.g. top:uniform. Can be provided multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for fused CSV reports.")
    return parser


def _fusion_config_path(args: argparse.Namespace) -> Path | None:
    """Resolve a named profile without silently merging another config."""
    profile_path = ZS32_PROFILE_PATH if args.profile == "zs32" else None
    if (
        profile_path is not None
        and args.fusion_config is not None
        and profile_path.resolve() != args.fusion_config.resolve()
    ):
        msg = "incompatible --profile and --fusion-config values"
        raise ValueError(msg)
    return profile_path or args.fusion_config


def _missing_artifacts(predictions: list[fusion.BranchPrediction]) -> list[str]:
    """Describe absent source and evidence files for one physical part."""
    missing: list[str] = []
    for prediction in predictions:
        evidence_id = f"{prediction.view or 'unknown'}:{prediction.branch}"
        for artifact_name, path_value in (
            ("source", prediction.source_path),
            ("evidence", prediction.evidence_path),
        ):
            if path_value is None or not Path(path_value).is_file():
                missing.append(f"missing {artifact_name} for {evidence_id}")
    return sorted(set(missing))


def _inspection_is_complete(
    predictions: list[fusion.BranchPrediction],
    config: dict[str, object],
    explicit_required_views: list[str],
) -> bool:
    """Return whether required views, branches, and artifacts are all present."""
    return not (
        _missing_artifacts(predictions)
        or fusion.missing_required_views(predictions, config, explicit_required_views)
        or fusion.missing_required_branch_keys(predictions, config)
    )


def _audit_output_path(audit_dir: Path, part_id: str) -> Path:
    """Build an audit path while rejecting identities that can escape its root."""
    if not part_id or Path(part_id).is_absolute() or part_id in {".", ".."} or "/" in part_id or "\\" in part_id:
        msg = f"unsafe part_id for audit filename: {part_id!r}"
        raise ValueError(msg)
    return audit_dir / f"{part_id}.json"


def _enforce_complete_evidence(
    decisions: list[fusion.FusedDecision],
    grouped: dict[str, list[fusion.BranchPrediction]],
) -> list[fusion.FusedDecision]:
    """Add a system trigger and prevent OK release for incomplete evidence."""
    enforced: list[fusion.FusedDecision] = []
    for decision in decisions:
        predictions = grouped.get(decision.part_id, [])
        missing = _missing_artifacts(predictions)
        if not missing:
            enforced.append(decision)
            continue
        reason = "; ".join(missing)
        trigger = fusion.TriggerEvidence(
            evidence_id="system:incomplete_evidence",
            branch="system",
            side=predictions[0].side if predictions else "zs32",
            view=None,
            level=fusion.EvidenceLevel.GRAY.value,
            score=None,
            low_threshold=None,
            high_threshold=None,
            reason=reason,
        )
        enforced.append(
            replace(
                decision,
                final_status="REVIEW",
                final_label=None,
                triggered_branch="system",
                reason=f"{decision.reason}; {reason}",
                triggered_evidence=(*decision.triggered_evidence, trigger),
            ),
        )
    return enforced


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
    config = fusion.load_fusion_config(_fusion_config_path(args))
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
    if args.require_complete_evidence:
        decisions = _enforce_complete_evidence(decisions, grouped)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fusion.write_branch_predictions_csv(predictions, args.output_dir / "branch_predictions.csv")
    fusion.write_fused_decisions_csv(decisions, args.output_dir / "fused_predictions.csv")
    if args.profile == "zs32":
        audit_dir = args.audit_dir or args.output_dir / "audit"
        decision_by_part = {decision.part_id: decision for decision in decisions}
        for part_id, part_predictions in grouped.items():
            decision = decision_by_part[part_id]
            audit = build_part_audit(
                part_id,
                part_predictions,
                machine_status=decision.final_status,
                triggered_evidence=decision.triggered_evidence,
            )
            write_part_audit(audit, _audit_output_path(audit_dir, part_id))
    complete_inspections = sum(
        _inspection_is_complete(grouped.get(decision.part_id, []), config, args.required_view) for decision in decisions
    )
    summary = {
        "parts": len(decisions),
        "branches": len(predictions),
        "ok": sum(decision.final_status == "OK" for decision in decisions),
        "ng": sum(decision.final_status.startswith("NG_") for decision in decisions),
        "retake": sum(decision.final_status == "RETAKE" for decision in decisions),
        "invalid_capture": sum(decision.final_status == "INVALID_CAPTURE" for decision in decisions),
        "review": sum(decision.final_status == "REVIEW" for decision in decisions),
        "suspect": sum(decision.final_status == "SUSPECT" for decision in decisions),
        "complete_inspections": complete_inspections,
        "incomplete_inspections": len(decisions) - complete_inspections,
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
