# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 18: fuse quality, geometry, and anomaly inspection results."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data import fusion_engine as fusion  # noqa: E402
from capture_data.inspection_audit import build_part_audit, sha256_file, write_part_audit  # noqa: E402

ZS32_PROFILE_PATH = REPO_ROOT / "config/fusion/zs32_six_view.json"


class DiagnosticGenerationError(RuntimeError):
    """Raised after a strict fail-closed diagnostic generation is published."""


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
        immutable_ng = decision.final_status.startswith("NG_")
        enforced.append(
            replace(
                decision,
                final_status=decision.final_status if immutable_ng else "REVIEW",
                final_label=decision.final_label if immutable_ng else None,
                triggered_branch=decision.triggered_branch if immutable_ng else "system",
                reason=f"{decision.reason}; {reason}",
                triggered_evidence=(*decision.triggered_evidence, trigger),
            ),
        )
    return enforced


def _system_trigger(evidence_id: str, reason: str) -> fusion.TriggerEvidence:
    """Create one stable system-level audit trigger."""
    return fusion.TriggerEvidence(
        evidence_id=evidence_id,
        branch="system",
        side="zs32",
        view=None,
        level=fusion.EvidenceLevel.GRAY.value,
        score=None,
        low_threshold=None,
        high_threshold=None,
        reason=reason,
    )


def _diagnostic_decision(part_id: str, reason: str) -> fusion.FusedDecision:
    """Create a non-releasable decision for malformed strict input."""
    trigger = _system_trigger("system:malformed_input", reason)
    return fusion.FusedDecision(
        part_id=part_id,
        final_status="INVALID_CAPTURE",
        final_label=None,
        defect_side=None,
        defect_view=None,
        defect_slot=None,
        defect_type=None,
        triggered_branch="system",
        reason=reason,
        triggered_evidence=(trigger,),
    )


def _strict_stage_faults(predictions: list[fusion.BranchPrediction]) -> tuple[list[str], list[str]]:
    """Return invalid-capture and review faults owned by the stage boundary."""
    hashes_by_digest: dict[str, set[str]] = {}
    invalid: list[str] = []
    review: list[str] = []
    for prediction in predictions:
        source_path = Path(prediction.source_path) if prediction.source_path else None
        if source_path is not None and source_path.is_file():
            digest = sha256_file(source_path)
            hashes_by_digest.setdefault(digest, set()).add(prediction.view or "unknown")
            declared_hash = getattr(prediction, "source_hash", None)
            if declared_hash and declared_hash.lower() != digest:
                invalid.append(f"source hash mismatch: {prediction.view or 'unknown'}:{prediction.branch}")
        status = (prediction.status or "").strip().upper()
        if prediction.branch not in fusion.GATE_BRANCHES and status in {
            "DRIFT",
            "ERROR",
            "FAIL",
            "FAILED",
            "ROI_FAILURE",
            "TIMEOUT",
        }:
            review.append(f"{prediction.view or 'unknown'}:{prediction.branch} status={status}")
    for digest, views in hashes_by_digest.items():
        if len(views) > 1:
            invalid.append(f"duplicate source hash across required views: {digest} ({', '.join(sorted(views))})")
    return sorted(set(invalid)), sorted(set(review))


def _apply_stage_faults(
    decision: fusion.FusedDecision,
    *,
    invalid_faults: list[str],
    review_faults: list[str],
) -> fusion.FusedDecision:
    """Apply strict stage faults without downgrading a machine NG to REVIEW."""
    if invalid_faults:
        reason = "; ".join(invalid_faults)
        return replace(
            decision,
            final_status="INVALID_CAPTURE",
            final_label=None,
            triggered_branch="system",
            reason=f"{decision.reason}; {reason}",
            triggered_evidence=(*decision.triggered_evidence, _system_trigger("system:identity_fault", reason)),
        )
    if not review_faults:
        return decision
    reason = "; ".join(review_faults)
    immutable_ng = decision.final_status.startswith("NG_")
    return replace(
        decision,
        final_status=decision.final_status if immutable_ng else "REVIEW",
        final_label=decision.final_label if immutable_ng else None,
        triggered_branch=decision.triggered_branch if immutable_ng else "system",
        reason=f"{decision.reason}; {reason}",
        triggered_evidence=(*decision.triggered_evidence, _system_trigger("system:runtime_fault", reason)),
    )


def _fuse_predictions(
    grouped: dict[str, list[fusion.BranchPrediction]],
    missing_required: dict[str, list[str]],
    config: dict[str, object],
    *,
    strict_zs32: bool,
) -> tuple[list[fusion.FusedDecision], list[str]]:
    """Fuse each part, converting strict malformed rows into diagnostics."""
    decisions: list[fusion.FusedDecision] = []
    malformed: list[str] = []
    for part_id in sorted(set(grouped) | set(missing_required)):
        predictions = grouped.get(part_id, [])
        try:
            decision = fusion.fuse_part_predictions(
                part_id,
                predictions,
                missing_required=missing_required.get(part_id, ()),
                config=config,
            )
        except (TypeError, ValueError) as error:
            if not strict_zs32:
                raise
            reason = f"malformed strict input for {part_id}: {error}"
            malformed.append(reason)
            decision = _diagnostic_decision(part_id, reason)
        if strict_zs32:
            invalid_faults, review_faults = _strict_stage_faults(predictions)
            decision = _apply_stage_faults(
                decision,
                invalid_faults=invalid_faults,
                review_faults=review_faults,
            )
            malformed_reasons = [
                trigger.reason
                for trigger in decision.triggered_evidence
                if "dual thresholds" in trigger.reason
                or "EvidenceLevel" in trigger.reason
                or "evidence level" in trigger.reason.lower()
            ]
            malformed.extend(f"malformed strict input for {part_id}: {reason}" for reason in malformed_reasons)
        decisions.append(decision)
    return decisions, malformed


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
        path = Path(path_text)
        loaded = fusion.load_optional_branch_predictions_csv(path, branch=branch, warnings=warnings)
        _attach_audit_metadata(loaded, path)
        predictions.extend(loaded)
    return predictions


def _attach_audit_metadata(predictions: list[fusion.BranchPrediction], path: Path | None) -> None:
    """Retain audit-only CSV identity fields not owned by the fusion dataclass."""
    if path is None or not path.is_file():
        return
    rows = fusion.read_csv_rows(path)
    for prediction, row in zip(predictions, rows, strict=False):
        for attribute, aliases in (
            ("session_id", ("session_id", "capture_session", "session")),
            ("timestamp", ("timestamp", "captured_at", "inspection_timestamp")),
        ):
            value = next(
                (
                    text
                    for name in aliases
                    if (raw_value := row.get(name)) is not None and (text := str(raw_value).strip())
                ),
                None,
            )
            if value is not None:
                object.__setattr__(prediction, attribute, value)  # noqa: PLC2801


def _load_optional_predictions(
    path: Path | None,
    *,
    branch: str,
    warnings: list[str],
) -> list[fusion.BranchPrediction]:
    """Load an optional branch CSV while retaining audit-only metadata."""
    predictions = fusion.load_optional_branch_predictions_csv(path, branch=branch, warnings=warnings)
    _attach_audit_metadata(predictions, path)
    return predictions


def _validate_generation_target(args: argparse.Namespace, *, strict_zs32: bool) -> None:
    """Reject overwrite and split-publication targets before doing work."""
    if args.output_dir.exists():
        msg = f"final output generation already exists: {args.output_dir}"
        raise FileExistsError(msg)
    if strict_zs32 and args.audit_dir is not None:
        expected = args.output_dir / "audit"
        if args.audit_dir.resolve(strict=False) != expected.resolve(strict=False):
            msg = f"strict ZS32 audits must be inside the atomic generation at {expected}"
            raise ValueError(msg)


def _summary_counts(
    decisions: list[fusion.FusedDecision],
    grouped: dict[str, list[fusion.BranchPrediction]],
    config: dict[str, object],
    explicit_required_views: list[str],
) -> tuple[dict[str, int], dict[str, bool]]:
    """Build summary counts and the per-part inspection-complete state."""
    completion = {
        decision.part_id: (
            _inspection_is_complete(
                grouped.get(decision.part_id, []),
                config,
                explicit_required_views,
            )
            and decision.final_status not in {"INVALID_CAPTURE", "RETAKE"}
            and not any(trigger.branch == "system" for trigger in decision.triggered_evidence)
        )
        for decision in decisions
    }
    complete_count = sum(completion.values())
    summary = {
        "parts": len(decisions),
        "branches": sum(len(rows) for rows in grouped.values()),
        "ok": sum(decision.final_status == "OK" for decision in decisions),
        "ng": sum(decision.final_status.startswith("NG_") for decision in decisions),
        "retake": sum(decision.final_status == "RETAKE" for decision in decisions),
        "invalid_capture": sum(decision.final_status == "INVALID_CAPTURE" for decision in decisions),
        "review": sum(decision.final_status == "REVIEW" for decision in decisions),
        "suspect": sum(decision.final_status == "SUSPECT" for decision in decisions),
        "complete_inspections": complete_count,
        "incomplete_inspections": len(decisions) - complete_count,
    }
    return summary, completion


def _publish_generation(
    args: argparse.Namespace,
    predictions: list[fusion.BranchPrediction],
    grouped: dict[str, list[fusion.BranchPrediction]],
    decisions: list[fusion.FusedDecision],
    summary: dict[str, int],
    completion: dict[str, bool],
    warnings: list[str],
    *,
    strict_zs32: bool,
) -> None:
    """Publish CSV, summary, and every audit through one sibling directory rename."""
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.staging-",
            dir=args.output_dir.parent,
        ),
    )
    try:
        fusion.write_branch_predictions_csv(predictions, staging / "branch_predictions.csv")
        fusion.write_fused_decisions_csv(decisions, staging / "fused_predictions.csv")
        fusion.write_summary_markdown(staging / "summary.md", "Inspection Fusion Summary", summary, warnings)
        if strict_zs32:
            decision_by_part = {decision.part_id: decision for decision in decisions}
            for part_id, part_predictions in grouped.items():
                decision = decision_by_part[part_id]
                audit = build_part_audit(
                    part_id,
                    part_predictions,
                    machine_status=decision.final_status,
                    triggered_evidence=decision.triggered_evidence,
                    inspection_complete=completion.get(part_id, False),
                )
                write_part_audit(audit, _audit_output_path(staging / "audit", part_id))
        staging.replace(args.output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def run_fusion(args: argparse.Namespace) -> list[fusion.FusedDecision]:
    """Run CSV-based fusion and write branch/fused reports."""
    config = fusion.load_fusion_config(_fusion_config_path(args))
    strict_zs32 = args.profile == "zs32"
    _validate_generation_target(args, strict_zs32=strict_zs32)
    warnings: list[str] = []
    predictions = [
        *_load_optional_predictions(args.quality_csv, branch="quality", warnings=warnings),
        *_load_optional_predictions(args.registration_csv, branch="registration", warnings=warnings),
        *_load_optional_predictions(args.geometry_csv, branch="geometry", warnings=warnings),
        *_load_optional_predictions(args.anomaly_csv, branch="anomaly_dino", warnings=warnings),
        *_load_optional_predictions(args.efficientad_csv, branch="efficient_ad", warnings=warnings),
        *_load_optional_predictions(args.crack_csv, branch="crack", warnings=warnings),
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

    decisions, malformed = _fuse_predictions(
        grouped,
        missing_required,
        config,
        strict_zs32=strict_zs32,
    )
    if strict_zs32 or args.require_complete_evidence:
        decisions = _enforce_complete_evidence(decisions, grouped)
    summary, completion = _summary_counts(decisions, grouped, config, args.required_view)
    _publish_generation(
        args,
        predictions,
        grouped,
        decisions,
        summary,
        completion,
        warnings,
        strict_zs32=strict_zs32,
    )
    if malformed:
        msg = "strict diagnostic generation published: " + "; ".join(malformed)
        raise DiagnosticGenerationError(msg)
    return decisions


def main() -> None:
    """Run inspection result fusion."""
    args = build_parser().parse_args()
    decisions = run_fusion(args)
    print(f"Fused {len(decisions)} parts")
    print(f"Wrote fused predictions: {args.output_dir / 'fused_predictions.csv'}")


if __name__ == "__main__":
    main()
