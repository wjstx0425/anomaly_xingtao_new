"""Fit template/model thresholds and evaluate the isolated held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.calibration import (
    CalibrationProvenance,
    FitParameters,
    GroundTruth,
    ScoreBranch,
    ScoreGroup,
    ScoreRow,
    SplitRole,
    TemplateCalibrationGroup,
    build_calibration_artifact,
    evaluate_heldout,
    fit_dual_thresholds,
    fit_template_thresholds,
)
from zs32_inspection.config.loaders import load_recipe, load_roi_config, load_topology
from zs32_inspection.config.guards import require_bound_capture_gate_policy
from zs32_inspection.data.dataset_release import verify_dataset_release
from zs32_inspection.data.calibration_targets import (
    CalibrationTargetsSnapshot,
    CalibrationTargetValue,
    load_calibration_targets,
)
from zs32_inspection.domain.contracts import AnomalyFamily
from zs32_inspection.models import ModelSlot
from zs32_inspection.models.base import (
    SHA256_PATTERN,
    ModelContractError,
    YoloDetection,
    sha256_file,
)
from zs32_inspection.runtime.execution_receipt import ExecutionReceipt
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    VerifiedAtomicPublication,
    canonical_json_bytes,
    verify_atomic_publication,
)

from ._common import (
    array_value,
    command_error,
    command_result,
    load_object,
    object_value,
    require_keys,
)
from ._deployment_assets import parse_candidate
from ._environment import require_linux_nvidia
from .score_calibration import _load_canonical_rows, _selected_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit and evaluate one immutable ZS32 calibration artifact")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--score-run", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-id", required=True)
    return parser


_SCORE_RUN_FILES = frozenset({"scores.jsonl", "score_audit.jsonl", "score_run.json"})
_SCORE_RUN_FIELDS = (
    "schema", "schema_version", "score_run_id", "dataset_release_id",
    "dataset_manifest_sha256", "canonical_manifest_sha256", "calibration_targets_sha256",
    "calibration_target_count", "split_assignments_sha256",
    "calibration_split_id", "test_split_id", "candidate_id", "candidate_digest",
    "candidate_descriptor_sha256", "candidate_status", "recipe_digest", "anomaly_family",
    "template_assets_descriptor_sha256", "topology_id", "topology_sha256", "roi_version",
    "roi_sha256", "enabled_hands", "required_slots", "model_digests", "template_models",
    "anomaly_models", "yolo_model_digest", "yolo_bundle_digest", "device", "branch_order",
    "canonical_row_count", "score_row_count", "score_audit_row_count",
    "excluded_score_count", "scores_sha256", "score_audit_sha256",
    "execution_receipt",
)


def _canonical_json_object(content: bytes, context: str) -> dict[str, object]:
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {context} JSON: {error}") from error
    item = object_value(payload, context)
    if canonical_json_bytes(item) != content:
        raise ValueError(f"{context} must use canonical JSON serialization")
    return item


def _canonical_jsonl(content: bytes, context: str) -> tuple[dict[str, object], ...]:
    if not content or not content.endswith(b"\n"):
        raise ValueError(f"{context} must be non-empty canonical JSONL ending with newline")
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(content.splitlines(keepends=True), start=1):
        if raw_line in {b"\n", b"\r\n"}:
            raise ValueError(f"{context} contains a blank row at line {line_number}")
        try:
            payload = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {context} JSON at line {line_number}: {error}") from error
        row = object_value(payload, f"{context}[{line_number}]")
        if canonical_json_bytes(row) != raw_line:
            raise ValueError(f"{context}[{line_number}] is not canonical JSON")
        rows.append(row)
    if not rows:
        raise ValueError(f"{context} contains no rows")
    return tuple(rows)


def _sha256_value(value: object, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a 64-character lowercase SHA256")
    return value


_AUDIT_COMMON_FIELDS = {
    "sequence_index", "score_sequence_index", "included_in_calibration",
    "dataset_release_id", "part_instance_id", "capture_set_id", "hand", "view",
    "split_role", "split_id", "roi_version", "roi_digest", "part_ground_truth",
    "calibration_target", "target_reason", "model_digest", "inspection_id", "branch",
    "source_sha256", "crop_sha256", "raw_score",
}
_AUDIT_BRANCH_FIELDS = {
    "template": {"similarity", "best_template_sha256", "offset_xy"},
    "anomaly": {"model_family", "heatmap_sha256"},
    "yolo": {"detections"},
}


def _audit_text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"{context} must be a canonical non-empty string")
    return value


def _audit_number(value: object, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _validate_score_audit_row(row: dict[str, object], index: int) -> None:
    branch = row.get("branch")
    if not isinstance(branch, str) or branch not in _AUDIT_BRANCH_FIELDS:
        raise ValueError(f"score_audit[{index}].branch is invalid")
    require_keys(
        row,
        required=tuple(_AUDIT_COMMON_FIELDS | _AUDIT_BRANCH_FIELDS[branch]),
        context=f"score_audit[{index}]",
    )
    if isinstance(row["sequence_index"], bool) or not isinstance(row["sequence_index"], int):
        raise ValueError(f"score_audit[{index}].sequence_index must be an integer")
    for field in (
        "dataset_release_id", "part_instance_id", "capture_set_id", "hand", "view",
        "split_id", "roi_version", "inspection_id",
    ):
        _audit_text(row[field], f"score_audit[{index}].{field}")
    if row["split_role"] not in {"calibration", "test"}:
        raise ValueError(f"score_audit[{index}].split_role is invalid")
    if row["part_ground_truth"] not in {"normal", "defect"}:
        raise ValueError(f"score_audit[{index}].part_ground_truth is invalid")
    if row["calibration_target"] not in {"normal", "defect", "exclude"}:
        raise ValueError(f"score_audit[{index}].calibration_target is invalid")
    reason = row["target_reason"]
    if reason is not None:
        _audit_text(reason, f"score_audit[{index}].target_reason")
    for field in (
        "roi_digest", "model_digest", "source_sha256", "crop_sha256",
    ):
        _sha256_value(row[field], f"score_audit[{index}].{field}")
    raw_score = _audit_number(row["raw_score"], f"score_audit[{index}].raw_score")

    if branch == "template":
        similarity = _audit_number(
            row["similarity"],
            f"score_audit[{index}].similarity",
        )
        if (
            not -1 <= similarity <= 1
            or not 0 <= raw_score <= 2
            or not math.isclose(raw_score, 1 - similarity, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(f"score_audit[{index}] template score relation is invalid")
        _sha256_value(
            row["best_template_sha256"],
            f"score_audit[{index}].best_template_sha256",
        )
        offset = row["offset_xy"]
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offset)
        ):
            raise ValueError(f"score_audit[{index}].offset_xy must contain two integers")
    elif branch == "anomaly":
        if row["model_family"] not in {item.value for item in AnomalyFamily}:
            raise ValueError(f"score_audit[{index}].model_family is invalid")
        heatmap = row["heatmap_sha256"]
        if heatmap is not None:
            _sha256_value(heatmap, f"score_audit[{index}].heatmap_sha256")
    else:
        detections = row["detections"]
        if not isinstance(detections, list):
            raise ValueError(f"score_audit[{index}].detections must be an array")
        try:
            parsed = tuple(
                YoloDetection.from_mapping(object_value(item, f"score_audit[{index}].detections"))
                for item in detections
            )
        except ModelContractError as error:
            raise ValueError(f"score_audit[{index}] contains an invalid YOLO detection") from error
        expected = max((item.confidence for item in parsed), default=0.0)
        if not math.isclose(raw_score, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"score_audit[{index}] YOLO raw_score differs from detections")


def _score_run_model_bindings(
    run: dict[str, object],
) -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str], frozenset[str]],
    dict[tuple[str, str], str],
]:
    template_bindings: dict[tuple[str, str], str] = {}
    template_references: dict[tuple[str, str], frozenset[str]] = {}
    for index, raw in enumerate(array_value(run["template_models"], "template_models")):
        item = object_value(raw, f"template_models[{index}]")
        require_keys(
            item,
            required=("hand", "view", "model_digest", "reference_digests"),
            context=f"template_models[{index}]",
        )
        key = (
            _audit_text(item["hand"], f"template_models[{index}].hand"),
            _audit_text(item["view"], f"template_models[{index}].view"),
        )
        if key in template_bindings:
            raise ValueError(f"template_models contains duplicate slot {key}")
        template_bindings[key] = _sha256_value(
            item["model_digest"],
            f"template_models[{index}].model_digest",
        )
        references = array_value(
            item["reference_digests"],
            f"template_models[{index}].reference_digests",
        )
        if not references:
            raise ValueError(f"template_models[{index}] has no reference digests")
        parsed_references = tuple(
            _sha256_value(
                digest,
                f"template_models[{index}].reference_digests[{reference_index}]",
            )
            for reference_index, digest in enumerate(references)
        )
        if len(parsed_references) != len(set(parsed_references)):
            raise ValueError(f"template_models[{index}] contains duplicate references")
        template_references[key] = frozenset(parsed_references)

    anomaly_bindings: dict[tuple[str, str], str] = {}
    for index, raw in enumerate(array_value(run["anomaly_models"], "anomaly_models")):
        item = object_value(raw, f"anomaly_models[{index}]")
        require_keys(
            item,
            required=("hand", "view", "model_digest"),
            context=f"anomaly_models[{index}]",
        )
        key = (
            _audit_text(item["hand"], f"anomaly_models[{index}].hand"),
            _audit_text(item["view"], f"anomaly_models[{index}].view"),
        )
        if key in anomaly_bindings:
            raise ValueError(f"anomaly_models contains duplicate slot {key}")
        anomaly_bindings[key] = _sha256_value(
            item["model_digest"],
            f"anomaly_models[{index}].model_digest",
        )
    if not template_bindings or set(template_bindings) != set(anomaly_bindings):
        raise ValueError("score run template/anomaly model slots must be identical and non-empty")
    return template_bindings, template_references, anomaly_bindings


def _load_verified_score_run(
    root: Path,
) -> tuple[VerifiedAtomicPublication, dict[str, object], tuple[dict[str, object], ...]]:
    publication = verify_atomic_publication(
        root,
        required_paths=_SCORE_RUN_FILES,
        allowed_paths=_SCORE_RUN_FILES,
    )
    run_bytes = publication.read_bytes("score_run.json")
    scores_bytes = publication.read_bytes("scores.jsonl")
    audit_bytes = publication.read_bytes("score_audit.jsonl")
    run = _canonical_json_object(run_bytes, "score_run.json")
    require_keys(run, required=_SCORE_RUN_FIELDS, context="score_run.json")
    if run["schema"] != "zs32.calibration_score_run" or run["schema_version"] != 3:
        raise ValueError("score_run.json schema must be zs32.calibration_score_run version 3")
    if run["score_run_id"] != publication.publication_id:
        raise ValueError("score_run_id differs from the atomic publication identity")
    for field in (
        "dataset_manifest_sha256", "canonical_manifest_sha256",
        "calibration_targets_sha256", "split_assignments_sha256",
        "candidate_digest", "candidate_descriptor_sha256", "recipe_digest",
        "template_assets_descriptor_sha256", "topology_sha256", "roi_sha256",
        "yolo_model_digest", "yolo_bundle_digest", "scores_sha256", "score_audit_sha256",
    ):
        _sha256_value(run[field], f"score_run.json.{field}")
    if hashlib.sha256(scores_bytes).hexdigest() != run["scores_sha256"]:
        raise ValueError("scores.jsonl digest differs from score_run.json")
    if hashlib.sha256(audit_bytes).hexdigest() != run["score_audit_sha256"]:
        raise ValueError("score_audit.jsonl digest differs from score_run.json")
    if run["anomaly_family"] not in {item.value for item in AnomalyFamily}:
        raise ValueError("score_run.json anomaly_family is invalid")
    (
        template_model_by_slot,
        template_references_by_slot,
        anomaly_model_by_slot,
    ) = _score_run_model_bindings(run)
    receipt = ExecutionReceipt.from_mapping(
        object_value(run["execution_receipt"], "score_run.json.execution_receipt")
    )
    if receipt.operation != "score_calibration":
        raise ValueError("score_run.json execution receipt operation must be score_calibration")
    rows = _canonical_jsonl(scores_bytes, "scores.jsonl")
    audit = _canonical_jsonl(audit_bytes, "score_audit.jsonl")
    for index, row in enumerate(rows):
        _score_row(row, index)
    if (
        isinstance(run["score_row_count"], bool)
        or not isinstance(run["score_row_count"], int)
        or run["score_row_count"] != len(rows)
        or isinstance(run["canonical_row_count"], bool)
        or not isinstance(run["canonical_row_count"], int)
        or run["canonical_row_count"] <= 0
        or isinstance(run["score_audit_row_count"], bool)
        or not isinstance(run["score_audit_row_count"], int)
        or run["score_audit_row_count"] != 3 * run["canonical_row_count"]
        or isinstance(run["excluded_score_count"], bool)
        or not isinstance(run["excluded_score_count"], int)
        or run["excluded_score_count"] != len(audit) - len(rows)
        or run["excluded_score_count"] < 0
        or isinstance(run["calibration_target_count"], bool)
        or not isinstance(run["calibration_target_count"], int)
        or run["calibration_target_count"] < len(audit)
    ):
        raise ValueError("score run row counts are inconsistent")
    referenced_score_indices: list[int] = []
    for index, audit_row in enumerate(audit):
        _validate_score_audit_row(audit_row, index)
        if audit_row["inspection_id"] != (
            f"{run['score_run_id']}:{audit_row['capture_set_id']}"
        ):
            raise ValueError(f"score audit inspection identity differs at row {index}")
        if audit_row.get("sequence_index") != index:
            raise ValueError("score audit sequence is not contiguous")
        included = audit_row.get("included_in_calibration")
        score_index = audit_row.get("score_sequence_index")
        target = audit_row.get("calibration_target")
        if included is True:
            if isinstance(score_index, bool) or not isinstance(score_index, int):
                raise ValueError("included score audit requires an integer score index")
            if score_index < 0 or score_index >= len(rows):
                raise ValueError("score audit index is outside scores.jsonl")
            score = rows[score_index]
            referenced_score_indices.append(score_index)
            for audit_field, score_field in (
                ("dataset_release_id", "dataset_release_id"),
                ("part_instance_id", "part_instance_id"),
                ("capture_set_id", "capture_set_id"),
                ("hand", "hand"), ("view", "view"), ("branch", "branch"),
                ("raw_score", "score"), ("model_digest", "model_digest"),
                ("part_ground_truth", "part_ground_truth"),
                ("calibration_target", "ground_truth"),
                ("split_role", "split_role"), ("split_id", "split_id"),
                ("roi_version", "roi_version"), ("roi_digest", "roi_digest"),
            ):
                if audit_row.get(audit_field) != score.get(score_field):
                    raise ValueError(
                        f"score audit differs from scores.jsonl at audit row {index}: {audit_field}"
                    )
            if target == "exclude":
                raise ValueError("excluded target cannot enter calibration scores")
        elif included is False:
            if score_index is not None or target != "exclude":
                raise ValueError("excluded score audit must have null index and exclude target")
        else:
            raise ValueError("score audit included_in_calibration must be a boolean")
        slot_key = (audit_row["hand"], audit_row["view"])
        expected_model_digest = (
            template_model_by_slot.get(slot_key)
            if audit_row["branch"] == "template"
            else anomaly_model_by_slot.get(slot_key)
            if audit_row["branch"] == "anomaly"
            else run["yolo_model_digest"]
        )
        if audit_row["model_digest"] != expected_model_digest:
            raise ValueError(f"score audit model binding differs at row {index}")
        if (
            audit_row["branch"] == "template"
            and audit_row["best_template_sha256"]
            not in template_references_by_slot.get(slot_key, frozenset())
        ):
            raise ValueError(f"score audit template reference is not approved at row {index}")
        if (
            audit_row["branch"] == "anomaly"
            and audit_row["model_family"] != run["anomaly_family"]
        ):
            raise ValueError(f"score audit anomaly family differs at row {index}")
    if referenced_score_indices != list(range(len(rows))):
        raise ValueError("score audit must reference scores.jsonl exactly once in order")
    sample_identity_fields = (
        "dataset_release_id", "part_instance_id", "capture_set_id", "hand", "view",
        "part_ground_truth", "split_role", "split_id", "roi_version", "roi_digest",
        "inspection_id", "source_sha256", "crop_sha256",
    )
    observed_samples: set[tuple[object, ...]] = set()
    for offset in range(0, len(audit), 3):
        triple = audit[offset : offset + 3]
        if [row.get("branch") for row in triple] != ["template", "anomaly", "yolo"]:
            raise ValueError("audit rows must contain one ordered branch triple per crop")
        identities = {
            tuple(row.get(field) for field in sample_identity_fields)
            for row in triple
        }
        if len(identities) != 1:
            raise ValueError("score audit branch triple mixes canonical crop identities")
        identity = identities.pop()
        if identity in observed_samples:
            raise ValueError("score run contains a duplicate canonical crop audit triple")
        observed_samples.add(identity)
    if len(observed_samples) != run["canonical_row_count"]:
        raise ValueError("canonical score audit triple count differs from score_run.json")
    return publication, run, rows


def _score_row(payload: dict[str, object], index: int) -> ScoreRow:
    fields = (
        "dataset_release_id", "part_instance_id", "capture_set_id", "hand", "view",
        "branch", "score", "ground_truth", "split_role", "split_id", "model_digest",
        "roi_version", "roi_digest", "part_ground_truth",
    )
    require_keys(payload, required=fields, context=f"scores[{index}]")
    return ScoreRow(
        dataset_release_id=payload["dataset_release_id"],
        part_instance_id=payload["part_instance_id"],
        capture_set_id=payload["capture_set_id"],
        hand=payload["hand"],
        view=payload["view"],
        branch=ScoreBranch(payload["branch"]),
        score=payload["score"],
        ground_truth=GroundTruth(payload["ground_truth"]),
        split_role=SplitRole(payload["split_role"]),
        split_id=payload["split_id"],
        model_digest=payload["model_digest"],
        roi_version=payload["roi_version"],
        roi_digest=payload["roi_digest"],
        part_ground_truth=GroundTruth(payload["part_ground_truth"]),
    )


def _validate_score_target_replay(
    *,
    audit: Sequence[dict[str, object]],
    targets: CalibrationTargetsSnapshot,
    canonical_rows,
    calibration_split_id: str,
    test_split_id: str,
) -> None:
    """Rejoin every audited score to the approved dataset target contract."""
    expected = []
    for row in canonical_rows:
        for branch in ("template", "anomaly", "yolo"):
            key = (
                row.capture_set_id,
                row.part_instance_id,
                row.hand,
                row.view,
                branch,
            )
            target = targets.by_key.get(key)
            if target is None:
                raise ValueError(f"approved calibration target is missing during replay: {key}")
            if row.split == SplitRole.TEST.value and target.target is CalibrationTargetValue.EXCLUDE:
                raise ValueError(
                    f"held-out test target cannot be exclude because full-system evaluation "
                    f"would be incomplete: {key}"
                )
            expected.append((row, target))
    if len(audit) != len(expected):
        raise ValueError("score audit does not exactly cover selected canonical target rows")
    for index, (audit_row, (row, target)) in enumerate(zip(audit, expected, strict=True)):
        actual_key = (
            audit_row.get("capture_set_id"),
            audit_row.get("part_instance_id"),
            audit_row.get("hand"),
            audit_row.get("view"),
            audit_row.get("branch"),
        )
        if actual_key != target.key:
            raise ValueError(f"score audit target order/identity differs at row {index}")
        expected_split_role = SplitRole(row.split)
        expected_split_id = (
            calibration_split_id
            if expected_split_role is SplitRole.CALIBRATION
            else test_split_id
        )
        if (
            audit_row.get("dataset_release_id") != row.dataset_release_id
            or audit_row.get("part_ground_truth") != target.part_ground_truth
            or audit_row.get("calibration_target") != target.target.value
            or audit_row.get("target_reason") != target.reason
            or audit_row.get("included_in_calibration")
            is not (target.target is not CalibrationTargetValue.EXCLUDE)
            or audit_row.get("split_role") != expected_split_role.value
            or audit_row.get("split_id") != expected_split_id
            or audit_row.get("roi_version") != row.roi_version
            or audit_row.get("roi_digest") != row.roi_sha256
            or audit_row.get("source_sha256") != row.source_sha256
            or audit_row.get("crop_sha256") != row.crop_sha256
        ):
            raise ValueError(f"score audit differs from approved target/canonical row {index}")


def _dual_group(payload: object, index: int) -> ScoreGroup:
    item = object_value(payload, f"required_dual_groups[{index}]")
    require_keys(
        item,
        required=("hand", "view", "branch", "model_digest", "roi_version"),
        context=f"required_dual_groups[{index}]",
    )
    group = ScoreGroup(
        hand=item["hand"],
        view=item["view"],
        branch=ScoreBranch(item["branch"]),
        model_digest=item["model_digest"],
        roi_version=item["roi_version"],
    )
    if group.branch not in {ScoreBranch.ANOMALY, ScoreBranch.YOLO}:
        raise ValueError(f"required_dual_groups[{index}] branch must be anomaly or yolo")
    return group


def _template_group(payload: object, index: int) -> TemplateCalibrationGroup:
    item = object_value(payload, f"required_template_groups[{index}]")
    require_keys(
        item,
        required=("hand", "view", "model_digest", "roi_version", "roi_digest"),
        context=f"required_template_groups[{index}]",
    )
    return TemplateCalibrationGroup(
        hand=item["hand"],
        view=item["view"],
        model_digest=item["model_digest"],
        roi_version=item["roi_version"],
        roi_digest=item["roi_digest"],
    )


def _provenance(payload: object) -> CalibrationProvenance:
    item = object_value(payload, "provenance")
    require_keys(
        item,
        required=(
            "recipe_digest", "profile_digest", "topology_digest", "roi_digest",
            "dataset_release_id", "dataset_manifest_digest", "calibration_split_id",
            "test_split_id", "model_digests",
        ),
        context="provenance",
    )
    return CalibrationProvenance(
        recipe_digest=item["recipe_digest"],
        profile_digest=item["profile_digest"],
        topology_digest=item["topology_digest"],
        roi_digest=item["roi_digest"],
        dataset_release_id=item["dataset_release_id"],
        dataset_manifest_digest=item["dataset_manifest_digest"],
        calibration_split_id=item["calibration_split_id"],
        test_split_id=item["test_split_id"],
        model_digests=tuple(array_value(item["model_digests"], "provenance.model_digests")),
    )


def _fit_parameters(payload: object) -> FitParameters:
    item = object_value(payload, "fit_parameters")
    require_keys(
        item,
        required=("target_defect_recall", "normal_quantile", "min_normal_parts", "min_defect_parts"),
        context="fit_parameters",
    )
    return FitParameters(
        target_defect_recall=item["target_defect_recall"],
        normal_quantile=item["normal_quantile"],
        min_normal_parts=item["min_normal_parts"],
        min_defect_parts=item["min_defect_parts"],
    )


def _validate_score_run_provenance(
    run: dict[str, object],
    *,
    recipe,
    topology,
    roi,
    dataset,
    dataset_manifest_digest: str,
    candidate,
    candidate_descriptor_digest: str,
    provenance: CalibrationProvenance,
) -> None:
    """Bind one immutable score publication to every calibration input identity."""
    required_slots = {
        ModelSlot(hand.value, view)
        for hand in recipe.allowed_hands
        for view in topology.required_views
    }
    candidate.validate_slots(tuple(required_slots))
    if candidate.status.value != "registered":
        raise ValueError("calibration requires candidate status=registered")
    expected_dataset = (
        dataset.dataset_release_id,
        dataset_manifest_digest,
        dataset.canonical_manifest_sha256,
        dataset.calibration_targets_sha256,
        dataset.calibration_target_count,
        dataset.split_assignments_sha256,
    )
    actual_dataset = (
        run["dataset_release_id"],
        run["dataset_manifest_sha256"],
        run["canonical_manifest_sha256"],
        run["calibration_targets_sha256"],
        run["calibration_target_count"],
        run["split_assignments_sha256"],
    )
    if actual_dataset != expected_dataset:
        raise ValueError("score run dataset provenance differs from the verified dataset release")
    if (
        run["calibration_split_id"] != provenance.calibration_split_id
        or run["test_split_id"] != provenance.test_split_id
    ):
        raise ValueError("score run split IDs differ from the calibration specification")
    if (
        run["candidate_id"] != candidate.candidate_id
        or run["candidate_digest"] != candidate.digest
        or run["candidate_descriptor_sha256"] != candidate_descriptor_digest
        or run["candidate_status"] != candidate.status.value
    ):
        raise ValueError("score run candidate identity differs from the explicit candidate descriptor")
    if (
        candidate.recipe_digest != recipe.recipe_sha256
        or candidate.topology_digest != topology.topology_sha256
        or candidate.roi_version != roi.roi_config_id
        or candidate.roi_digest != roi.roi_sha256
        or candidate.dataset_release_id != dataset.dataset_release_id
        or candidate.dataset_manifest_digest != dataset_manifest_digest
        or candidate.anomaly_family is not recipe.anomaly_family
    ):
        raise ValueError("candidate provenance differs from recipe/topology/ROI/dataset")
    if (
        run["recipe_digest"] != recipe.recipe_sha256
        or run["anomaly_family"] != recipe.anomaly_family.value
        or run["topology_id"] != topology.topology_id
        or run["topology_sha256"] != topology.topology_sha256
        or run["roi_version"] != roi.roi_config_id
        or run["roi_sha256"] != roi.roi_sha256
    ):
        raise ValueError("score run recipe/topology/ROI provenance mismatch")

    enabled_hands = array_value(run["enabled_hands"], "score_run.enabled_hands")
    if (
        len(enabled_hands) != len(set(enabled_hands))
        or set(enabled_hands) != {hand.value for hand in recipe.allowed_hands}
    ):
        raise ValueError("score run enabled_hands differ from recipe")
    slot_keys = array_value(run["required_slots"], "score_run.required_slots")
    if len(slot_keys) != len(set(slot_keys)) or set(slot_keys) != {slot.key for slot in required_slots}:
        raise ValueError("score run required_slots differ from recipe/topology")

    anomaly_by_slot = {
        (binding.hand.value, binding.view_id): binding.asset.sha256
        for binding in recipe.anomaly_bindings
    }
    template_by_slot = {
        (binding.hand.value, binding.view_id): binding.asset.sha256
        for binding in recipe.template_bindings
    }
    if set(anomaly_by_slot) != {(slot.hand, slot.view) for slot in required_slots}:
        raise ValueError("recipe anomaly bindings do not cover the score run slots")
    if set(template_by_slot) != {(slot.hand, slot.view) for slot in required_slots}:
        raise ValueError("recipe template bindings do not cover the score run slots")
    if any(
        candidate.anomaly_artifacts[slot].model_digest
        != anomaly_by_slot[(slot.hand, slot.view)]
        for slot in required_slots
    ):
        raise ValueError("candidate anomaly models differ from recipe bindings")
    if candidate.yolo.model_digest != recipe.yolo_model.sha256:
        raise ValueError("candidate YOLO model differs from recipe binding")

    expected_model_digests = {
        recipe.yolo_model.sha256,
        *anomaly_by_slot.values(),
        *template_by_slot.values(),
    }
    model_digests = array_value(run["model_digests"], "score_run.model_digests")
    if (
        len(model_digests) != len(set(model_digests))
        or any(SHA256_PATTERN.fullmatch(item) is None for item in model_digests if isinstance(item, str))
        or any(not isinstance(item, str) for item in model_digests)
        or set(model_digests) != expected_model_digests
    ):
        raise ValueError("score run model_digests differ from recipe/candidate")

    anomaly_models: dict[tuple[str, str], str] = {}
    for index, raw in enumerate(array_value(run["anomaly_models"], "score_run.anomaly_models")):
        item = object_value(raw, f"score_run.anomaly_models[{index}]")
        require_keys(
            item,
            required=("hand", "view", "model_digest"),
            context=f"score_run.anomaly_models[{index}]",
        )
        key = (item["hand"], item["view"])
        if key in anomaly_models:
            raise ValueError(f"score run contains duplicate anomaly model slot {key}")
        anomaly_models[key] = _sha256_value(
            item["model_digest"], f"score_run.anomaly_models[{index}].model_digest"
        )
    if anomaly_models != anomaly_by_slot:
        raise ValueError("score run anomaly model map differs from recipe/candidate")

    template_models: dict[tuple[str, str], str] = {}
    for index, raw in enumerate(array_value(run["template_models"], "score_run.template_models")):
        item = object_value(raw, f"score_run.template_models[{index}]")
        require_keys(
            item,
            required=("hand", "view", "model_digest", "reference_digests"),
            context=f"score_run.template_models[{index}]",
        )
        key = (item["hand"], item["view"])
        if key in template_models:
            raise ValueError(f"score run contains duplicate template model slot {key}")
        template_models[key] = _sha256_value(
            item["model_digest"], f"score_run.template_models[{index}].model_digest"
        )
        references = array_value(
            item["reference_digests"],
            f"score_run.template_models[{index}].reference_digests",
        )
        if not references or len(references) != len(set(references)):
            raise ValueError("score run template reference digests must be non-empty and unique")
        for reference_index, digest in enumerate(references):
            _sha256_value(
                digest,
                f"score_run.template_models[{index}].reference_digests[{reference_index}]",
            )
    if template_models != template_by_slot:
        raise ValueError("score run template model map differs from recipe")
    if (
        run["yolo_model_digest"] != candidate.yolo.model_digest
        or run["yolo_bundle_digest"] != candidate.yolo.bundle_digest
    ):
        raise ValueError("score run YOLO identity differs from the explicit candidate")
    device = object_value(run["device"], "score_run.device")
    require_keys(device, required=("accelerator", "device"), context="score_run.device")
    if device["accelerator"] != "gpu" or not isinstance(device["device"], str) or not device["device"].strip():
        raise ValueError("score run device must record a non-empty GPU device")
    if array_value(run["branch_order"], "score_run.branch_order") != ["template", "anomaly", "yolo"]:
        raise ValueError("score run branch_order differs from the frozen scoring contract")
    receipt = ExecutionReceipt.from_mapping(
        object_value(run["execution_receipt"], "score_run.execution_receipt")
    )
    expected_execution_inputs = {
        "dataset_manifest": run["dataset_manifest_sha256"],
        "canonical_manifest": run["canonical_manifest_sha256"],
        "calibration_targets": run["calibration_targets_sha256"],
        "candidate_descriptor": run["candidate_descriptor_sha256"],
        "template_assets_descriptor": run["template_assets_descriptor_sha256"],
    }
    if dict(receipt.input_sha256_by_role) != expected_execution_inputs:
        raise ValueError("score execution receipt inputs differ from score_run provenance")
    expected_parameters_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "calibration_split_id": run["calibration_split_id"],
                "test_split_id": run["test_split_id"],
                "device": device["device"],
            }
        )
    ).hexdigest()
    if receipt.parameters_sha256 != expected_parameters_sha256:
        raise ValueError("score execution receipt parameters differ from score_run parameters")
    if receipt.environment.device_mapping.requested_device != device["device"]:
        raise ValueError("score execution receipt GPU device differs from score_run device")


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    recipe = load_recipe(args.recipe)
    require_bound_capture_gate_policy(recipe)
    topology = load_topology(args.topology)
    roi = load_roi_config(args.roi)
    dataset_root = args.dataset_release.expanduser()
    dataset = verify_dataset_release(dataset_root)
    dataset_manifest_digest = sha256_file(dataset_root / "dataset_release.json")
    candidate_path = args.candidate.expanduser()
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise ValueError("candidate descriptor must be a regular non-symlink file")
    candidate_metadata = candidate_path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(candidate_metadata.st_mode)
        or candidate_metadata.st_nlink != 1
        or candidate_metadata.st_uid != os.geteuid()
    ):
        raise ValueError("candidate descriptor must be a private regular file owned by this user")
    candidate_descriptor_digest = sha256_file(candidate_path)
    candidate = parse_candidate(
        load_object(candidate_path, "model candidate"),
        asset_root=args.asset_root,
    )
    score_publication, score_run, raw_score_rows = _load_verified_score_run(args.score_run)
    spec = load_object(args.spec, "calibration spec")
    require_keys(
        spec,
        required=(
            "schema_version", "provenance", "fit_parameters",
            "required_dual_groups", "required_template_groups",
        ),
        context="calibration spec",
    )
    if spec["schema_version"] != 1:
        raise ValueError("calibration spec schema_version must be 1")
    provenance = _provenance(spec["provenance"])
    parameters = _fit_parameters(spec["fit_parameters"])
    if (
        provenance.recipe_digest != recipe.recipe_sha256
        or provenance.profile_digest != recipe.fusion_policy_sha256
        or provenance.topology_digest != topology.topology_sha256
        or provenance.roi_digest != roi.roi_sha256
    ):
        raise ValueError("calibration provenance differs from recipe/topology/ROI")
    if recipe.topology_id != topology.topology_id or recipe.roi_config_id != roi.roi_config_id:
        raise ValueError("recipe identity differs from topology/ROI")
    if (
        provenance.dataset_release_id != dataset.dataset_release_id
        or provenance.dataset_manifest_digest
        != dataset_manifest_digest
        or dataset.topology_sha256 != topology.topology_sha256
        or dataset.roi_sha256 != roi.roi_sha256
        or not {hand.value for hand in recipe.allowed_hands}.issubset(set(dataset.hands))
    ):
        raise ValueError("calibration dataset release provenance mismatch")
    for hand in recipe.allowed_hands:
        roi.require_ready(hand, topology.required_views)
    _validate_score_run_provenance(
        score_run,
        recipe=recipe,
        topology=topology,
        roi=roi,
        dataset=dataset,
        dataset_manifest_digest=dataset_manifest_digest,
        candidate=candidate,
        candidate_descriptor_digest=candidate_descriptor_digest,
        provenance=provenance,
    )
    calibration_targets = load_calibration_targets(
        dataset_root / "calibration_targets.json"
    )
    if calibration_targets.sha256 != dataset.calibration_targets_sha256:
        raise ValueError("dataset calibration target bytes changed before calibration")
    required_score_slots = tuple(
        ModelSlot(hand.value, view)
        for hand in recipe.allowed_hands
        for view in topology.required_views
    )
    selected_canonical_rows = _selected_rows(
        _load_canonical_rows(dataset_root),
        required_slots=required_score_slots,
        required_views=topology.required_views,
    )
    score_audit_rows = _canonical_jsonl(
        score_publication.read_bytes("score_audit.jsonl"),
        "score_audit.jsonl",
    )
    _validate_score_target_replay(
        audit=score_audit_rows,
        targets=calibration_targets,
        canonical_rows=selected_canonical_rows,
        calibration_split_id=provenance.calibration_split_id,
        test_split_id=provenance.test_split_id,
    )
    rows = tuple(
        _score_row(row, index)
        for index, row in enumerate(raw_score_rows)
    )
    if any(row.dataset_release_id != provenance.dataset_release_id for row in rows):
        raise ValueError("score rows contain a dataset release outside calibration provenance")
    if any(row.hand not in {hand.value for hand in recipe.allowed_hands} for row in rows):
        raise ValueError("score rows contain a hand outside the selected recipe")
    if any(
        row.roi_digest != provenance.roi_digest or row.roi_version != roi.roi_config_id
        for row in rows
    ):
        raise ValueError("score rows contain an ROI identity outside calibration provenance")
    if any(row.model_digest not in set(provenance.model_digests) for row in rows):
        raise ValueError("score rows contain a model digest outside calibration provenance")
    if any(
        (row.split_role is SplitRole.CALIBRATION and row.split_id != provenance.calibration_split_id)
        or (row.split_role is SplitRole.TEST and row.split_id != provenance.test_split_id)
        for row in rows
    ):
        raise ValueError("score split IDs differ from calibration/test provenance")
    dual_groups = tuple(
        _dual_group(item, index)
        for index, item in enumerate(array_value(spec["required_dual_groups"], "required_dual_groups"))
    )
    template_groups = tuple(
        _template_group(item, index)
        for index, item in enumerate(
            array_value(spec["required_template_groups"], "required_template_groups")
        )
    )
    required_slots = {
        (hand.value, view)
        for hand in recipe.allowed_hands
        for view in topology.required_views
    }
    anomaly_binding = {
        (item.hand.value, item.view_id): item for item in recipe.anomaly_bindings
    }
    template_binding = {
        (item.hand.value, item.view_id): item for item in recipe.template_bindings
    }
    if set(anomaly_binding) != required_slots or set(template_binding) != required_slots:
        raise ValueError("recipe model/template slots do not cover every enabled hand and required view")
    expected_dual = {
        ScoreGroup(
            hand=hand,
            view=view,
            branch=branch,
            model_digest=(
                anomaly_binding[(hand, view)].asset.sha256
                if branch is ScoreBranch.ANOMALY
                else recipe.yolo_model.sha256
            ),
            roi_version=roi.roi_config_id,
        )
        for hand, view in required_slots
        for branch in (ScoreBranch.ANOMALY, ScoreBranch.YOLO)
    }
    expected_template = {
        TemplateCalibrationGroup(
            hand=hand,
            view=view,
            model_digest=template_binding[(hand, view)].asset.sha256,
            roi_version=roi.roi_config_id,
            roi_digest=roi.roi_sha256,
        )
        for hand, view in required_slots
    }
    expected_model_digests = {
        recipe.yolo_model.sha256,
        *(item.asset.sha256 for item in recipe.anomaly_bindings),
        *(item.asset.sha256 for item in recipe.template_bindings),
    }
    if set(provenance.model_digests) != expected_model_digests:
        raise ValueError("calibration model_digests differ from recipe assets")
    if len(dual_groups) != len(set(dual_groups)) or set(dual_groups) != expected_dual:
        raise ValueError("calibration dual groups differ from dynamic recipe/topology requirements")
    if len(template_groups) != len(set(template_groups)) or set(template_groups) != expected_template:
        raise ValueError("calibration template groups differ from dynamic recipe/topology requirements")
    candidate_after = candidate_path.stat(follow_symlinks=False)
    candidate_identity_before = (
        candidate_metadata.st_dev,
        candidate_metadata.st_ino,
        candidate_metadata.st_size,
        candidate_metadata.st_mtime_ns,
        candidate_metadata.st_nlink,
        candidate_metadata.st_uid,
    )
    candidate_identity_after = (
        candidate_after.st_dev,
        candidate_after.st_ino,
        candidate_after.st_size,
        candidate_after.st_mtime_ns,
        candidate_after.st_nlink,
        candidate_after.st_uid,
    )
    if (
        candidate_path.is_symlink()
        or not stat.S_ISREG(candidate_after.st_mode)
        or candidate_identity_after != candidate_identity_before
        or sha256_file(candidate_path) != candidate_descriptor_digest
    ):
        raise ValueError("candidate descriptor changed while calibration inputs were verified")
    reverified_score_publication = verify_atomic_publication(
        args.score_run,
        required_paths=_SCORE_RUN_FILES,
        allowed_paths=_SCORE_RUN_FILES,
    )
    if reverified_score_publication.root_sha256 != score_publication.root_sha256:
        raise ValueError("score run publication root changed during calibration input verification")
    dual = fit_dual_thresholds(
        rows,
        required_groups=dual_groups,
        calibration_split_id=provenance.calibration_split_id,
        parameters=parameters,
    )
    template = fit_template_thresholds(
        rows,
        required_groups=template_groups,
        calibration_split_id=provenance.calibration_split_id,
        parameters=parameters,
    )
    metrics = evaluate_heldout(
        rows,
        dual_thresholds=dual,
        template_thresholds=template,
        required_dual_groups=dual_groups,
        required_template_groups=template_groups,
        test_split_id=provenance.test_split_id,
    )
    artifact = build_calibration_artifact(
        provenance=provenance,
        parameters=parameters,
        required_dual_groups=dual_groups,
        required_template_groups=template_groups,
        dual_thresholds=dual,
        template_thresholds=template,
        heldout_metrics=metrics,
    )
    artifact_bytes = canonical_json_bytes(artifact.payload())
    if hashlib.sha256(artifact_bytes).hexdigest() != artifact.artifact_sha256:
        raise RuntimeError("calibration artifact serialization differs from its canonical digest")
    template_payload = {
        "schema_version": 1,
        "calibration_artifact_sha256": artifact.artifact_sha256,
        "thresholds": [item.to_dict() for item in artifact.template_thresholds],
    }
    model_payload = {
        "schema_version": 1,
        "calibration_artifact_sha256": artifact.artifact_sha256,
        "thresholds": [item.to_dict() for item in artifact.dual_thresholds],
    }
    input_provenance_payload = {
        "schema": "zs32.calibration_inputs",
        "schema_version": 3,
        "calibration_id": args.calibration_id,
        "score_run_id": score_publication.publication_id,
        "score_run_root_sha256": score_publication.root_sha256,
        "scores_sha256": score_run["scores_sha256"],
        "score_audit_sha256": score_run["score_audit_sha256"],
        "score_execution_receipt_sha256": object_value(
            score_run["execution_receipt"],
            "score_run.execution_receipt",
        )["receipt_sha256"],
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.digest,
        "candidate_descriptor_sha256": candidate_descriptor_digest,
        "recipe_digest": recipe.recipe_sha256,
        "dataset_release_id": dataset.dataset_release_id,
        "dataset_manifest_sha256": dataset_manifest_digest,
        "calibration_targets_sha256": dataset.calibration_targets_sha256,
        "calibration_target_count": dataset.calibration_target_count,
        "calibration_split_id": provenance.calibration_split_id,
        "test_split_id": provenance.test_split_id,
    }

    def validate(staging: Path) -> None:
        staged_artifact_digest = hashlib.sha256(
            (staging / "calibration_artifact.json").read_bytes()
        ).hexdigest()
        if staged_artifact_digest != artifact.artifact_sha256:
            raise PublicationError("published calibration artifact digest mismatch")
        if not (staging / "template_thresholds.json").is_file() or not (staging / "model_thresholds.json").is_file():
            raise PublicationError("published calibration threshold files are missing")

    with AtomicDirectoryPublisher(args.output_root, args.calibration_id) as publisher:
        publisher.write_bytes("calibration_artifact.json", artifact_bytes)
        publisher.write_json("template_thresholds.json", template_payload)
        publisher.write_json("model_thresholds.json", model_payload)
        publisher.write_json("metrics.json", artifact.heldout_metrics.to_dict())
        publisher.write_json("input_provenance.json", input_provenance_payload)
        published = publisher.finalize(
            validator=validate,
            required_paths=frozenset(
                {
                    "calibration_artifact.json",
                    "template_thresholds.json",
                    "model_thresholds.json",
                    "metrics.json",
                    "input_provenance.json",
                }
            ),
        )
    command_result(
        "zs32-calibrate",
        {
            "status": "ok" if artifact.calibration_valid else "non_deployable",
            "artifact_sha256": artifact.artifact_sha256,
            "calibration_valid": artifact.calibration_valid,
            "candidate_digest": candidate.digest,
            "score_run_id": score_publication.publication_id,
            "score_run_root_sha256": score_publication.root_sha256,
            "published_path": str(published),
        },
    )
    return 0 if artifact.calibration_valid else 3


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-calibrate", error)


if __name__ == "__main__":
    raise SystemExit(main())
