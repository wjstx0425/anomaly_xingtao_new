# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rule-based fusion for industrial defect inspection branch predictions."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

PASS_STATUS = "PASS"
FAIL_STATUS = "FAIL"
DEFAULT_BRANCH_ORDER = ("geometry", "crack", "anomaly_dino", "efficient_ad", "surface_texture", "yolo")
DEFAULT_STATUS_BY_BRANCH = {
    "geometry": "NG_GEOMETRY",
    "crack": "NG_CRACK",
    "anomaly_dino": "NG_ANOMALY",
    "efficient_ad": "NG_GLOBAL",
    "feature_presence": "NG_GEOMETRY",
    "surface_texture": "SUSPECT",
    "yolo": "NG_YOLO",
}
GATE_BRANCHES = {"quality", "quality_gate", "registration"}
BRANCH_FIELDNAMES = [
    "part_id",
    "side",
    "view",
    "slot_id",
    "branch",
    "pred_label",
    "score",
    "threshold",
    "defect_type",
    "evidence_type",
    "gt_defect_type",
    "reason",
    "source_path",
    "evidence_path",
    "status",
    "low_threshold",
    "high_threshold",
    "evidence_level",
    "model_version",
    "threshold_version",
    "roi_version",
    "template_version",
]
KNOWN_DEFECT_TYPES = {"corner", "crack", "deform", "less", "more", "surface"}
FUSED_FIELDNAMES = [
    "part_id",
    "final_status",
    "final_label",
    "defect_side",
    "defect_view",
    "defect_slot",
    "defect_type",
    "triggered_branch",
    "reason",
]


class EvidenceLevel(str, Enum):
    """Strength of evidence emitted by a prediction branch."""

    CLEAR = "CLEAR"
    GRAY = "GRAY"
    STRONG = "STRONG"


@dataclass(frozen=True)
class BranchPrediction:
    """One normalized prediction from a quality, geometry, model, or registration branch."""

    part_id: str
    side: str
    view: str | None
    slot_id: str | None
    branch: str
    pred_label: int
    score: float | None
    threshold: float | None
    defect_type: str | None
    reason: str | None
    source_path: str | None
    aliases: tuple[str, ...] = ()
    weak_aliases: tuple[str, ...] = ()
    status: str | None = None
    evidence_path: str | None = None
    evidence_type: str | None = None
    gt_defect_type: str | None = None
    low_threshold: float | None = None
    high_threshold: float | None = None
    evidence_level: str | None = None
    model_version: str | None = None
    threshold_version: str | None = None
    roi_version: str | None = None
    template_version: str | None = None


def classify_evidence(prediction: BranchPrediction) -> EvidenceLevel:
    """Classify a branch prediction using explicit, dual, or legacy evidence semantics."""
    if prediction.evidence_level is not None:
        return EvidenceLevel(prediction.evidence_level.upper())
    low, high, score = prediction.low_threshold, prediction.high_threshold, prediction.score
    if low is None and high is None:
        return EvidenceLevel.STRONG if prediction.pred_label == 1 else EvidenceLevel.CLEAR
    if score is None or low is None or high is None:
        msg = f"incomplete dual thresholds for {prediction.part_id}:{prediction.branch}"
        raise ValueError(msg)
    if not all(isfinite(value) for value in (score, low, high)) or low > high:
        msg = f"invalid dual thresholds for {prediction.part_id}:{prediction.branch}"
        raise ValueError(msg)
    if score >= high:
        return EvidenceLevel.STRONG
    if score >= low:
        return EvidenceLevel.GRAY
    return EvidenceLevel.CLEAR


@dataclass(frozen=True)
class FusedDecision:
    """Final fail-closed inspection decision for one part."""

    part_id: str
    final_status: str
    final_label: int | None
    defect_side: str | None
    defect_view: str | None
    defect_slot: str | None
    defect_type: str | None
    triggered_branch: str | None
    reason: str


def _clean_text(value: Any) -> str | None:
    """Return a stripped string or ``None`` for empty CSV values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_value(row: Mapping[str, Any], names: Sequence[str]) -> str | None:
    """Return the first non-empty CSV value from a list of column aliases."""
    for name in names:
        value = _clean_text(row.get(name))
        if value is not None:
            return value
    return None


def _string_sequence(value: Any) -> tuple[str, ...]:
    """Normalize a config scalar or sequence into stripped string values."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = _clean_text(value)
        return () if text is None else (text,)
    if isinstance(value, Sequence):
        return tuple(text for item in value if (text := _clean_text(item)) is not None)
    text = _clean_text(value)
    return () if text is None else (text,)


def _float_value(value: Any) -> float | None:
    """Convert a CSV scalar to a float when present."""
    text = _clean_text(value)
    if text is None:
        return None
    return float(text)


def _int_label(value: Any, *, default: int = 0) -> int:
    """Convert a CSV label or status-like value to ``0`` or ``1``."""
    text = _clean_text(value)
    if text is None:
        return default
    lowered = text.lower()
    if lowered in {"1", "true", "yes", "y", "ng", "anomaly", "anomalous", "defect", FAIL_STATUS.lower()}:
        return 1
    if lowered in {"0", "false", "no", "n", "ok", "normal", PASS_STATUS.lower(), "warn"}:
        return 0
    return int(float(text))


def _source_path(row: Mapping[str, Any]) -> str | None:
    """Return the most useful source path-like column."""
    return _first_value(row, ("source_path", "processed_path", "image_path", "path", "file_path"))


def _path_aliases_for_value(value: str | None) -> tuple[set[str], set[str]]:
    """Return strong and weak aliases for a path-like value."""
    if value is None:
        return set(), set()
    path = Path(value)
    strong_aliases = {value}
    weak_aliases = {path.name, path.stem}
    try:
        strong_aliases.add(str(path.resolve(strict=False)))
    except OSError:
        pass
    return {alias for alias in strong_aliases if alias}, {alias for alias in weak_aliases if alias}


def _row_aliases(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return strong matching aliases from IDs and all path-like columns in a branch row."""
    aliases: set[str] = set()
    for name in ("part_id", "sample_id", "id"):
        value = _clean_text(row.get(name))
        if value is not None:
            aliases.add(value)
    for name in ("source_path", "processed_path", "image_path", "path", "file_path"):
        strong_aliases, _ = _path_aliases_for_value(_clean_text(row.get(name)))
        aliases.update(strong_aliases)
    return tuple(sorted(aliases))


def _row_weak_aliases(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return basename/stem aliases from path-like columns in a branch row."""
    aliases: set[str] = set()
    for name in ("source_path", "processed_path", "image_path", "path", "file_path"):
        _, weak_aliases = _path_aliases_for_value(_clean_text(row.get(name)))
        aliases.update(weak_aliases)
    return tuple(sorted(aliases))


def _part_id_from_row(row: Mapping[str, Any], source_path: str | None) -> str:
    """Resolve a stable part identifier from CSV fields or path stem."""
    part_id = _first_value(row, ("part_id", "sample_id", "id"))
    if part_id is not None:
        return part_id
    if source_path is not None:
        return Path(source_path).stem
    return "unknown"


def _side_from_row(row: Mapping[str, Any], source_path: str | None) -> str:
    """Resolve side from CSV fields or path parts."""
    side = _first_value(row, ("side", "face", "position"))
    if side is not None:
        return side
    if source_path is not None:
        parts = {part.lower() for part in Path(source_path).parts}
        if "top" in parts:
            return "top"
        if "bottom" in parts or "bottom_zs32" in parts:
            return "bottom"
    return "unknown"


def _slot_from_row(row: Mapping[str, Any], source_path: str | None) -> str | None:
    """Resolve slot id from CSV fields or filename."""
    slot = _first_value(row, ("slot_id", "slot"))
    if slot is not None:
        return slot
    if source_path is None:
        return None
    match = re.search(r"(?:^|[_/\-])slot(?P<slot>[0-9]+)(?:$|[_/\-.])", source_path)
    if match is None:
        return None
    return f"slot{int(match.group('slot')):02d}"


def _gt_defect_type_from_path(source_path: str | None) -> str | None:
    """Infer a human GT defect type from a source path when possible."""
    if source_path is None:
        return None
    path = Path(source_path)
    tokens = re.split(r"[_\-\s.]+", " ".join([path.stem, *path.parts]).lower())
    return next((token for token in tokens if token in KNOWN_DEFECT_TYPES), None)


def _status_from_branch(config: Mapping[str, Any], branch: str) -> str:
    """Resolve the NG status emitted by a positive branch."""
    rules = config.get("rules", {})
    if isinstance(rules, Mapping):
        branch_rules = rules.get(branch, {})
        if isinstance(branch_rules, Mapping):
            configured = _clean_text(branch_rules.get("status_on_positive"))
            if configured is not None:
                return configured
    return DEFAULT_STATUS_BY_BRANCH.get(branch, "NG_GLOBAL")


def _branch_reason(prediction: BranchPrediction) -> str:
    """Return a compact explanation for a branch trigger."""
    if prediction.reason:
        return prediction.reason
    score = "" if prediction.score is None else f" score={prediction.score:g}"
    threshold = "" if prediction.threshold is None else f" threshold={prediction.threshold:g}"
    return f"{prediction.branch} positive{score}{threshold}".strip()


def _gate_missing_decision(part_id: str, branch: str) -> FusedDecision:
    """Return a retake decision for a configured PASS prerequisite missing its branch row."""
    return FusedDecision(
        part_id=part_id,
        final_status="RETAKE",
        final_label=None,
        defect_side=None,
        defect_view=None,
        defect_slot=None,
        defect_type=None,
        triggered_branch=branch,
        reason=f"missing required {branch} PASS",
    )


def _gate_not_pass_decision(part_id: str, prediction: BranchPrediction) -> FusedDecision:
    """Return a retake decision for a gate row that is present but not PASS."""
    return FusedDecision(
        part_id=part_id,
        final_status="RETAKE",
        final_label=None,
        defect_side=prediction.side,
        defect_view=prediction.view,
        defect_slot=prediction.slot_id,
        defect_type=prediction.defect_type,
        triggered_branch=prediction.branch,
        reason=f"required {prediction.branch} PASS but got {prediction.status}",
    )


def _requires_pass(config: Mapping[str, Any], name: str) -> bool:
    """Return whether the fusion config requires a PASS row for a gate."""
    ok_requires = config.get("ok_requires", {})
    if not isinstance(ok_requires, Mapping):
        return False
    value = _clean_text(ok_requires.get(name))
    return value is not None and value.upper() == PASS_STATUS


def _gate_has_pass(predictions: Sequence[BranchPrediction], branches: set[str]) -> bool:
    """Return whether a configured gate has an explicit PASS row."""
    for prediction in predictions:
        if prediction.branch not in branches:
            continue
        if prediction.status is None:
            return prediction.pred_label == 0
        if prediction.status.upper() == PASS_STATUS:
            return True
    return False


def _first_nonpass_gate(predictions: Sequence[BranchPrediction], branches: set[str]) -> BranchPrediction | None:
    """Return the first present gate row that is not PASS."""
    for prediction in predictions:
        if prediction.branch not in branches:
            continue
        if prediction.status is not None and prediction.status.upper() != PASS_STATUS:
            return prediction
    return None


def required_view_keys(config: Mapping[str, Any], explicit_required_views: Sequence[str] = ()) -> set[str]:
    """Return required side:view keys from fusion config and explicit CLI values."""
    required = {item for item in explicit_required_views if item}
    ok_requires = config.get("ok_requires", {})
    if isinstance(ok_requires, Mapping):
        sides = _string_sequence(ok_requires.get("required_sides"))
        views = _string_sequence(ok_requires.get("required_views"))
        required.update(f"{side}:{view}" for side in sides for view in views)
    return required


def missing_required_views(
    predictions: Sequence[BranchPrediction],
    config: Mapping[str, Any],
    explicit_required_views: Sequence[str] = (),
) -> list[str]:
    """Return configured required side:view keys absent from branch predictions."""
    required = required_view_keys(config, explicit_required_views)
    if not required:
        return []
    observed = {
        f"{prediction.side}:{prediction.view}"
        for prediction in predictions
        if prediction.side not in {"", "unknown"} and prediction.view
    }
    return sorted(required - observed)


def _is_near_threshold(prediction: BranchPrediction, ratio: float) -> bool:
    """Return whether a negative branch is close enough to its threshold for review."""
    if prediction.pred_label != 0 or prediction.score is None or prediction.threshold is None:
        return False
    if prediction.threshold <= 0:
        return prediction.score > 0
    return prediction.score >= prediction.threshold * ratio


def _decision_from_prediction(
    part_id: str,
    status: str,
    final_label: int | None,
    prediction: BranchPrediction,
    reason: str,
) -> FusedDecision:
    """Create a fused decision from one triggering branch prediction."""
    return FusedDecision(
        part_id=part_id,
        final_status=status,
        final_label=final_label,
        defect_side=prediction.side,
        defect_view=prediction.view,
        defect_slot=prediction.slot_id,
        defect_type=prediction.defect_type,
        triggered_branch=prediction.branch,
        reason=reason,
    )


def fuse_part_predictions(
    part_id: str,
    predictions: Sequence[BranchPrediction],
    *,
    missing_required: Sequence[str] = (),
    config: Mapping[str, Any] | None = None,
) -> FusedDecision:
    """Fuse branch predictions for one part into OK/NG/SUSPECT/RETAKE/INVALID_CAPTURE."""
    fusion_config = config or {}
    missing = [item for item in missing_required if item]
    if missing:
        return FusedDecision(
            part_id=part_id,
            final_status="INVALID_CAPTURE",
            final_label=None,
            defect_side=None,
            defect_view=None,
            defect_slot=None,
            defect_type=None,
            triggered_branch=None,
            reason="missing required input: " + ", ".join(sorted(missing)),
        )

    for branch in ("quality", "quality_gate", "registration"):
        for prediction in predictions:
            if prediction.branch == branch and prediction.pred_label == 1:
                return _decision_from_prediction(part_id, "RETAKE", None, prediction, _branch_reason(prediction))

    if _requires_pass(fusion_config, "quality_gate") and not any(
        prediction.branch in {"quality", "quality_gate"} for prediction in predictions
    ):
        return _gate_missing_decision(part_id, "quality_gate")
    if _requires_pass(fusion_config, "quality_gate"):
        nonpass_quality = _first_nonpass_gate(predictions, {"quality", "quality_gate"})
        if nonpass_quality is not None:
            return _gate_not_pass_decision(part_id, nonpass_quality)
        if not _gate_has_pass(predictions, {"quality", "quality_gate"}):
            return _gate_missing_decision(part_id, "quality_gate")
    if _requires_pass(fusion_config, "registration") and not any(
        prediction.branch == "registration" for prediction in predictions
    ):
        return _gate_missing_decision(part_id, "registration")
    if _requires_pass(fusion_config, "registration"):
        nonpass_registration = _first_nonpass_gate(predictions, {"registration"})
        if nonpass_registration is not None:
            return _gate_not_pass_decision(part_id, nonpass_registration)
        if not _gate_has_pass(predictions, {"registration"}):
            return _gate_missing_decision(part_id, "registration")

    branch_order = _string_sequence(fusion_config.get("branch_order")) or DEFAULT_BRANCH_ORDER
    for branch in branch_order:
        for prediction in predictions:
            if prediction.branch == branch and prediction.pred_label == 1:
                status = _status_from_branch(fusion_config, branch)
                if prediction.status is not None and prediction.status.upper() == "SUSPECT":
                    status = "SUSPECT"
                return _decision_from_prediction(
                    part_id,
                    status,
                    None if status == "SUSPECT" else 1,
                    prediction,
                    _branch_reason(prediction),
                )

    suspect_policy = fusion_config.get("suspect_policy", {})
    suspect_enabled = bool(suspect_policy.get("enable", False)) if isinstance(suspect_policy, Mapping) else False
    near_threshold_ratio = (
        float(suspect_policy.get("near_threshold_ratio", 0.9)) if isinstance(suspect_policy, Mapping) else 0.9
    )
    if suspect_enabled:
        for prediction in predictions:
            if prediction.branch not in GATE_BRANCHES and _is_near_threshold(prediction, near_threshold_ratio):
                reason = (
                    f"{prediction.branch} near threshold"
                    f" score={prediction.score:g} threshold={prediction.threshold:g}"
                )
                return _decision_from_prediction(part_id, "SUSPECT", None, prediction, reason)

    return FusedDecision(
        part_id=part_id,
        final_status="OK",
        final_label=0,
        defect_side=None,
        defect_view=None,
        defect_slot=None,
        defect_type=None,
        triggered_branch=None,
        reason="all available reliable branches below threshold",
    )


def _prediction_aliases(prediction: BranchPrediction) -> set[str]:
    """Return strong aliases used to merge rows from different branch CSV schemas."""
    aliases = set(prediction.aliases)
    if prediction.part_id not in prediction.weak_aliases:
        aliases.add(prediction.part_id)
    strong_aliases, _ = _path_aliases_for_value(prediction.source_path)
    aliases.update(strong_aliases)
    return {alias for alias in aliases if alias}


def _add_weak_alias(index: dict[str, str | None], alias: str, part_id: str) -> None:
    """Add a weak alias, marking it ambiguous once it points at multiple parts."""
    existing = index.get(alias)
    if existing is None and alias in index:
        return
    if existing is not None and existing != part_id:
        index[alias] = None
        return
    index[alias] = part_id


def _case_tokens(predictions: Sequence[BranchPrediction]) -> set[str]:
    """Return coarse benchmark case tokens found in source paths."""
    tokens: set[str] = set()
    for prediction in predictions:
        source_path = prediction.source_path or ""
        parts = {part.lower() for part in Path(source_path).parts}
        text = source_path.lower()
        if "defect" in parts or "defect" in text:
            tokens.add("defect")
        if "invalid" in parts or "invalid" in text:
            tokens.add("invalid")
        if "stress" in parts or "stress" in text:
            tokens.add("stress")
        if parts & {"normal", "normal_test", "clean_normal"} or "normal" in text:
            tokens.add("normal")
    return tokens


def _weak_alias_can_match(
    prediction: BranchPrediction,
    candidate_predictions: Sequence[BranchPrediction],
) -> bool:
    """Return whether a weak basename/stem match is not contradicted by case tokens."""
    prediction_tokens = _case_tokens((prediction,))
    candidate_tokens = _case_tokens(candidate_predictions)
    return not prediction_tokens or not candidate_tokens or bool(prediction_tokens & candidate_tokens)


def _group_part_id(prediction: BranchPrediction, grouped: Mapping[str, Sequence[BranchPrediction]]) -> str:
    """Return a stable group id without merging ambiguous path-stem part ids."""
    if prediction.part_id not in prediction.weak_aliases or prediction.part_id not in grouped:
        return prediction.part_id
    if prediction.source_path:
        parent_name = Path(prediction.source_path).parent.name
        if parent_name:
            candidate = f"{parent_name}_{prediction.part_id}"
            if candidate not in grouped:
                return candidate
    index = 2
    while f"{prediction.part_id}_{index}" in grouped:
        index += 1
    return f"{prediction.part_id}_{index}"


def group_predictions_by_part(predictions: Iterable[BranchPrediction]) -> dict[str, list[BranchPrediction]]:
    """Group branch predictions by part id and path aliases in stable insertion order."""
    grouped: dict[str, list[BranchPrediction]] = {}
    alias_to_part: dict[str, str] = {}
    weak_alias_to_part: dict[str, str | None] = {}
    for prediction in predictions:
        aliases = _prediction_aliases(prediction)
        matched_parts = [alias_to_part[alias] for alias in aliases if alias in alias_to_part]
        if matched_parts:
            part_id = matched_parts[0]
        else:
            weak_matches = {
                mapped_part_id
                for alias in prediction.weak_aliases
                if (mapped_part_id := weak_alias_to_part.get(alias)) is not None
                and _weak_alias_can_match(prediction, grouped[mapped_part_id])
            }
            part_id = next(iter(weak_matches)) if len(weak_matches) == 1 else _group_part_id(prediction, grouped)
        grouped.setdefault(part_id, [])
        for other_part_id in matched_parts[1:]:
            if other_part_id == part_id or other_part_id not in grouped:
                continue
            grouped[part_id].extend(grouped.pop(other_part_id))
            for alias, mapped_part_id in list(alias_to_part.items()):
                if mapped_part_id == other_part_id:
                    alias_to_part[alias] = part_id
        grouped[part_id].append(prediction)
        for alias in aliases:
            alias_to_part[alias] = part_id
        for alias in prediction.weak_aliases:
            _add_weak_alias(weak_alias_to_part, alias, part_id)
    return grouped


def fuse_grouped_predictions(
    grouped_predictions: Mapping[str, Sequence[BranchPrediction]],
    *,
    missing_required_by_part: Mapping[str, Sequence[str]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[FusedDecision]:
    """Fuse every part in a grouped prediction mapping."""
    missing_map = missing_required_by_part or {}
    part_ids = sorted(set(grouped_predictions) | set(missing_map))
    return [
        fuse_part_predictions(
            part_id,
            grouped_predictions.get(part_id, ()),
            missing_required=missing_map.get(part_id, ()),
            config=config,
        )
        for part_id in part_ids
    ]


def _prediction_from_row(row: Mapping[str, Any], branch: str) -> BranchPrediction:
    """Normalize one branch-specific CSV row."""
    source_path = _source_path(row)
    part_id = _part_id_from_row(row, source_path)
    side = _side_from_row(row, source_path)
    view = _first_value(row, ("view", "light", "capture_view"))
    slot_id = _slot_from_row(row, source_path)
    normalized_branch = _clean_text(row.get("branch")) or branch
    status = _first_value(row, ("status", "quality_status", "registration_status"))

    if normalized_branch == "geometry":
        label = _int_label(_first_value(row, ("geometry_pred_label", "pred_label", "final_pred_label")))
        score = _float_value(_first_value(row, ("raw_score", "geometry_score", "score", "final_score")))
        threshold = _float_value(_first_value(row, ("geometry_threshold", "threshold")))
        defect_type = _first_value(row, ("geometry_type", "defect_type"))
    elif normalized_branch in {"quality", "quality_gate", "registration"}:
        label = _int_label(_first_value(row, ("pred_label", "fail_label", "label", "status")), default=0)
        if status is not None and status.lower() == FAIL_STATUS.lower():
            label = 1
        score = _float_value(_first_value(row, ("raw_score", "score", "quality_score", "registration_score")))
        threshold = _float_value(_first_value(row, ("threshold",)))
        defect_type = _first_value(row, ("defect_type",))
    else:
        label = _int_label(
            _first_value(row, ("deploy_pred_label", "pred_label", "review_pred_label", "anomalib_pred_label")),
        )
        score = _float_value(_first_value(row, ("raw_score", "pred_score", "score", "anomaly_score", "deploy_score")))
        threshold = _float_value(_first_value(row, ("deploy_threshold", "threshold", "anomaly_threshold")))
        defect_type = _first_value(row, ("defect_type",))

    return BranchPrediction(
        part_id=part_id,
        side=side,
        view=view,
        slot_id=slot_id,
        branch=normalized_branch,
        pred_label=label,
        score=score,
        threshold=threshold,
        defect_type=defect_type,
        reason=_first_value(row, ("reason", "quality_reason", "registration_reason", "threshold_source")),
        source_path=source_path,
        aliases=_row_aliases(row),
        weak_aliases=_row_weak_aliases(row),
        status=status,
        evidence_path=_first_value(row, ("evidence_path", "overlay_path", "artifact_path")),
        evidence_type=_first_value(row, ("evidence_type", "operator_evidence", "evidence")),
        gt_defect_type=(
            _first_value(row, ("gt_defect_type", "ground_truth_defect_type", "label_defect_type"))
            or _gt_defect_type_from_path(source_path)
        ),
        low_threshold=_float_value(row.get("low_threshold")),
        high_threshold=_float_value(row.get("high_threshold")),
        evidence_level=_clean_text(row.get("evidence_level")),
        model_version=_clean_text(row.get("model_version")),
        threshold_version=_clean_text(row.get("threshold_version")),
        roi_version=_clean_text(row.get("roi_version")),
        template_version=_clean_text(row.get("template_version")),
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_branch_predictions_csv(path: Path, *, branch: str) -> list[BranchPrediction]:
    """Load a branch-specific CSV report into normalized branch predictions."""
    return [_prediction_from_row(row, branch) for row in read_csv_rows(path)]


def load_optional_branch_predictions_csv(
    path: Path | None,
    *,
    branch: str,
    warnings: list[str],
) -> list[BranchPrediction]:
    """Load a branch CSV when present, recording warnings instead of failing for missing optional files."""
    if path is None:
        return []
    if not path.is_file():
        warnings.append(f"missing {branch} csv: {path}")
        return []
    return load_branch_predictions_csv(path, branch=branch)


def write_branch_predictions_csv(predictions: Sequence[BranchPrediction], output_csv: Path) -> None:
    """Write normalized branch predictions to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BRANCH_FIELDNAMES)
        writer.writeheader()
        for prediction in predictions:
            row = asdict(prediction)
            writer.writerow({field: row.get(field) for field in BRANCH_FIELDNAMES})


def write_fused_decisions_csv(decisions: Sequence[FusedDecision], output_csv: Path) -> None:
    """Write fused inspection decisions to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FUSED_FIELDNAMES)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(asdict(decision))


def load_fusion_config(path: Path | None) -> dict[str, Any]:
    """Load a JSON/YAML fusion config, returning an empty config when omitted."""
    if path is None:
        return {}
    if not path.is_file():
        msg = f"Missing fusion config: {path}"
        raise FileNotFoundError(msg)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ModuleNotFoundError as error:
            msg = "YAML fusion configs require PyYAML; use JSON or install PyYAML."
            raise RuntimeError(msg) from error
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        msg = f"Fusion config must contain a mapping: {path}"
        raise ValueError(msg)
    return data


def load_manifest_missing_required(
    manifest_csv: Path | None,
    config: Mapping[str, Any],
    *,
    explicit_required_views: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Return missing required side/view keys by part from a manifest CSV."""
    if manifest_csv is None:
        return {}
    rows = read_csv_rows(manifest_csv)
    required = sorted(required_view_keys(config, explicit_required_views))
    if not required:
        return {}

    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_path = _source_path(row)
        part_id = _part_id_from_row(row, source_path)
        side = _side_from_row(row, source_path)
        view = _first_value(row, ("view", "light", "capture_view"))
        if view is not None:
            observed[part_id].add(f"{side}:{view}")
    return {part_id: sorted(set(required) - views) for part_id, views in observed.items() if set(required) - views}


def _path_matches_root(path_text: str | None, root: Path | None) -> bool:
    """Return whether a path is inside or textually under a root."""
    if path_text is None or root is None:
        return False
    path = Path(path_text)
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return str(root) in str(path)


def _infer_case_type(
    part_id: str,
    predictions: Sequence[BranchPrediction],
    *,
    clean_normal_root: Path | None = None,
    stress_normal_root: Path | None = None,
    defect_root: Path | None = None,
    invalid_root: Path | None = None,
) -> str:
    """Infer benchmark case type from configured roots or source path tokens."""
    paths = [prediction.source_path for prediction in predictions if prediction.source_path]
    if any(_path_matches_root(path, defect_root) for path in paths):
        return "defect"
    if any(_path_matches_root(path, stress_normal_root) for path in paths):
        return "stress_normal"
    if any(_path_matches_root(path, clean_normal_root) for path in paths):
        return "clean_normal"
    if any(_path_matches_root(path, invalid_root) for path in paths):
        return "invalid"

    text = " ".join([part_id, *[str(path) for path in paths]]).lower()
    if "invalid" in text:
        return "invalid"
    if "defect" in text:
        return "defect"
    if "stress" in text:
        return "stress_normal"
    if "normal" in text:
        return "clean_normal"
    return "unknown"


def _is_ng(decision: FusedDecision) -> bool:
    """Return whether a decision is a defect-positive NG status."""
    return decision.final_label == 1 or decision.final_status.startswith("NG_")


def _is_rejected(decision: FusedDecision) -> bool:
    """Return whether a decision rejects an input from automatic OK."""
    return decision.final_status != "OK"


def _gt_defect_type_for_predictions(predictions: Sequence[BranchPrediction]) -> str | None:
    """Return the best available human GT defect type for benchmark grouping."""
    for prediction in predictions:
        if prediction.gt_defect_type:
            return prediction.gt_defect_type
    for prediction in predictions:
        path_type = _gt_defect_type_from_path(prediction.source_path)
        if path_type:
            return path_type
    for prediction in predictions:
        if prediction.defect_type and prediction.defect_type in KNOWN_DEFECT_TYPES:
            return prediction.defect_type
    return None


def _is_not_evaluated_missing_prediction(
    decision: FusedDecision,
    predictions: Sequence[BranchPrediction],
) -> bool:
    """Return whether a decision represents an input that was not evaluated by any branch."""
    marker = "not_evaluated/missing_prediction"
    return marker in decision.reason or any(
        prediction.branch == "missing_prediction" or marker in (prediction.reason or "")
        for prediction in predictions
    )


def compute_benchmark_summary(
    decisions: Sequence[FusedDecision],
    grouped_predictions: Mapping[str, Sequence[BranchPrediction]],
    *,
    clean_normal_root: Path | None = None,
    stress_normal_root: Path | None = None,
    defect_root: Path | None = None,
    invalid_root: Path | None = None,
) -> dict[str, float | int]:
    """Compute compact robustness metrics from fused decisions."""
    counts = {
        "total_parts": len(decisions),
        "clean_normal_total": 0,
        "stress_normal_total": 0,
        "normal_total": 0,
        "unknown_total": 0,
        "defect_total": 0,
        "invalid_total": 0,
        "clean_normal_false_positives": 0,
        "normal_false_positives": 0,
        "stress_normal_false_positives": 0,
        "defect_detected": 0,
        "geometry_detected": 0,
        "anomaly_dino_detected": 0,
        "invalid_rejected": 0,
        "not_evaluated_missing_prediction_count": 0,
        "retake_cases": 0,
    }
    defect_type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "detected": 0})
    for decision in decisions:
        predictions = grouped_predictions.get(decision.part_id, ())
        case_type = _infer_case_type(
            decision.part_id,
            predictions,
            clean_normal_root=clean_normal_root,
            stress_normal_root=stress_normal_root,
            defect_root=defect_root,
            invalid_root=invalid_root,
        )
        if decision.final_status == "RETAKE":
            counts["retake_cases"] += 1
        if case_type == "defect":
            counts["defect_total"] += 1
            if _is_ng(decision):
                counts["defect_detected"] += 1
            if any(prediction.branch == "geometry" and prediction.pred_label == 1 for prediction in predictions):
                counts["geometry_detected"] += 1
            if any(prediction.branch == "anomaly_dino" and prediction.pred_label == 1 for prediction in predictions):
                counts["anomaly_dino_detected"] += 1
            defect_type = _gt_defect_type_for_predictions(predictions) or decision.defect_type or "unknown"
            defect_type_counts[defect_type]["total"] += 1
            if _is_ng(decision):
                defect_type_counts[defect_type]["detected"] += 1
        elif case_type == "invalid":
            counts["invalid_total"] += 1
            if _is_not_evaluated_missing_prediction(decision, predictions):
                counts["not_evaluated_missing_prediction_count"] += 1
            elif _is_rejected(decision):
                counts["invalid_rejected"] += 1
        elif case_type == "stress_normal":
            counts["stress_normal_total"] += 1
            counts["normal_total"] += 1
            if _is_ng(decision):
                counts["normal_false_positives"] += 1
                counts["stress_normal_false_positives"] += 1
        elif case_type in {"clean_normal", "unknown"}:
            if case_type == "unknown":
                counts["unknown_total"] += 1
                continue
            counts["clean_normal_total"] += 1
            counts["normal_total"] += 1
            if _is_ng(decision):
                counts["clean_normal_false_positives"] += 1
                counts["normal_false_positives"] += 1

    defect_total = counts["defect_total"]
    invalid_total = counts["invalid_total"]
    normal_total = counts["normal_total"]
    counts["defect_recall"] = counts["defect_detected"] / defect_total if defect_total else 0.0
    counts["fused_recall"] = counts["defect_recall"]
    counts["geometry_recall"] = counts["geometry_detected"] / defect_total if defect_total else 0.0
    counts["anomaly_dino_recall"] = counts["anomaly_dino_detected"] / defect_total if defect_total else 0.0
    counts["invalid_reject_rate"] = counts["invalid_rejected"] / invalid_total if invalid_total else 0.0
    counts["normal_fp_rate"] = counts["normal_false_positives"] / normal_total if normal_total else 0.0
    for defect_type in ("less", "more", "corner", "surface", "crack"):
        type_count = defect_type_counts[defect_type]
        counts[f"{defect_type}_recall"] = (
            type_count["detected"] / type_count["total"] if type_count["total"] else 0.0
        )
    return counts


def benchmark_detail_rows(
    decisions: Sequence[FusedDecision],
    grouped_predictions: Mapping[str, Sequence[BranchPrediction]],
    *,
    clean_normal_root: Path | None = None,
    stress_normal_root: Path | None = None,
    defect_root: Path | None = None,
    invalid_root: Path | None = None,
) -> list[dict[str, str | int | float | None]]:
    """Build per-part benchmark rows used for CSV reports."""
    rows = []
    for decision in decisions:
        predictions = grouped_predictions.get(decision.part_id, ())
        case_type = _infer_case_type(
            decision.part_id,
            predictions,
            clean_normal_root=clean_normal_root,
            stress_normal_root=stress_normal_root,
            defect_root=defect_root,
            invalid_root=invalid_root,
        )
        first_prediction = predictions[0] if predictions else None
        gt_defect_type = _gt_defect_type_for_predictions(predictions)
        rows.append(
            {
                "part_id": decision.part_id,
                "case_type": case_type,
                "final_status": decision.final_status,
                "final_label": "" if decision.final_label is None else decision.final_label,
                "defect_type": gt_defect_type or decision.defect_type or (first_prediction.defect_type if first_prediction else ""),
                "evidence_type": first_prediction.evidence_type if first_prediction else "",
                "gt_defect_type": gt_defect_type or "",
                "slot_id": decision.defect_slot or (first_prediction.slot_id if first_prediction else ""),
                "triggered_branch": decision.triggered_branch or "",
                "reason": decision.reason,
            },
        )
    return rows


def write_dict_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write arbitrary dictionary rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, title: str, summary: Mapping[str, Any], warnings: Sequence[str] = ()) -> None:
    """Write a compact Markdown summary report."""
    lines = [f"# {title}", "", "| metric | value |", "| --- | ---: |"]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
