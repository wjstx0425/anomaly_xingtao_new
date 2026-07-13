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
from numbers import Real
from pathlib import Path
from typing import Any

PASS_STATUS = "PASS"
FAIL_STATUS = "FAIL"
DEFAULT_BRANCH_ORDER = (
    "template_match",
    "geometry",
    "crack",
    "anomaly_dino",
    "efficient_ad",
    "surface_texture",
    "yolo",
)
DEFAULT_STATUS_BY_BRANCH = {
    "template_match": "NG_TEMPLATE",
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
    "hand",
    "product",
    "profile",
    "capture_session",
    "group_id",
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
    "source_hash",
    "evidence_hash",
    "manifest_identity",
    "detections",
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
    hand: str | None = None
    product: str | None = None
    profile: str | None = None
    source_hash: str | None = None
    evidence_hash: str | None = None
    manifest_identity: str | None = None
    capture_session: str | None = None
    group_id: str | None = None
    detections: tuple[dict[str, Any], ...] | None = None


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
class TriggerEvidence:
    """One retained GRAY or STRONG branch trigger."""

    evidence_id: str
    branch: str
    side: str
    view: str | None
    level: str
    score: float | None
    low_threshold: float | None
    high_threshold: float | None
    reason: str


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
    triggered_evidence: tuple[TriggerEvidence, ...] = ()


@dataclass(frozen=True)
class FaceDecision:
    """Staged fusion result for one physical face of a part."""

    part_id: str
    face: str
    stage_status: str
    final_status: str | None
    inspection_complete: bool
    triggered_evidence: tuple[TriggerEvidence, ...]
    reason: str
    defect_side: str | None = None
    defect_view: str | None = None
    defect_slot: str | None = None
    defect_type: str | None = None
    triggered_branch: str | None = None


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


def _structured_detections(value: object) -> tuple[dict[str, Any], ...] | None:
    """Parse one JSON-list detection summary while preserving every box field."""
    text = _clean_text(value)
    if text is None:
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        msg = "detections must be a JSON list"
        raise TypeError(msg)
    detections: list[dict[str, Any]] = []
    for index, detection in enumerate(parsed):
        if not isinstance(detection, Mapping):
            msg = f"detection {index} must be a JSON object"
            raise TypeError(msg)
        detections.append(dict(detection))
    return tuple(detections)


def _finite_real(value: object) -> bool:
    """Return whether a value is a finite real number but not a boolean."""
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _yolo_detection_faults(prediction: BranchPrediction) -> list[str]:
    """Validate the strict per-box YOLO evidence contract."""
    identity = ":".join(
        (prediction.hand or "", prediction.side, prediction.view or "", prediction.branch),
    )
    if prediction.detections is None:
        return [f"missing detections for strict YOLO summary: {identity}"]
    if not isinstance(prediction.detections, Sequence) or isinstance(prediction.detections, (str, bytes)):
        return [f"detections must be a sequence for strict YOLO summary: {identity}"]

    faults: list[str] = []
    for index, detection in enumerate(prediction.detections):
        prefix = f"invalid YOLO detection {identity}:{index}"
        if not isinstance(detection, Mapping):
            faults.append(f"{prefix}: box must be an object")
            continue
        faults.extend(
            f"{prefix}: missing {field}" for field in ("class", "confidence", "xyxy", "area") if field not in detection
        )

        class_value = detection.get("class")
        valid_class = (
            isinstance(class_value, str)
            and bool(class_value.strip())
            or isinstance(class_value, int)
            and not isinstance(class_value, bool)
            and class_value >= 0
        )
        if "class" in detection and not valid_class:
            faults.append(f"{prefix}: class must be a nonempty name or nonnegative integer")

        confidence = detection.get("confidence")
        if "confidence" in detection and (not _finite_real(confidence) or not 0 <= confidence <= 1):
            faults.append(f"{prefix}: confidence must be finite and within [0, 1]")

        xyxy = detection.get("xyxy")
        valid_xyxy = (
            isinstance(xyxy, Sequence)
            and not isinstance(xyxy, (str, bytes))
            and len(xyxy) == 4
            and all(_finite_real(coordinate) and coordinate >= 0 for coordinate in xyxy)
        )
        if "xyxy" in detection and not valid_xyxy:
            faults.append(f"{prefix}: xyxy must contain four finite nonnegative coordinates")
        elif valid_xyxy and (xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]):
            faults.append(f"{prefix}: xyxy must have increasing coordinates")

        area = detection.get("area")
        if "area" in detection and (not _finite_real(area) or area <= 0):
            faults.append(f"{prefix}: area must be finite and positive")

        faults.extend(
            f"{prefix}: {flag} must be boolean"
            for flag in ("in_roi", "border", "touches_border")
            if flag in detection and not isinstance(detection[flag], bool)
        )
    return faults


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


def _gate_missing_decision(
    part_id: str,
    branch: str,
    triggered_evidence: tuple[TriggerEvidence, ...] = (),
) -> FusedDecision:
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
        triggered_evidence=triggered_evidence,
    )


def _gate_not_pass_decision(
    part_id: str,
    prediction: BranchPrediction,
    triggered_evidence: tuple[TriggerEvidence, ...] = (),
) -> FusedDecision:
    """Return a retake decision for a gate row that is present but not PASS."""
    status = _clean_text(prediction.status)
    observed = status if status is not None else "missing status (explicit PASS required)"
    return FusedDecision(
        part_id=part_id,
        final_status="RETAKE",
        final_label=None,
        defect_side=prediction.side,
        defect_view=prediction.view,
        defect_slot=prediction.slot_id,
        defect_type=prediction.defect_type,
        triggered_branch=prediction.branch,
        reason=f"required {prediction.branch} PASS but got {observed}",
        triggered_evidence=triggered_evidence,
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
        required.update(_string_sequence(ok_requires.get("required_view_keys")))
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


def missing_required_branch_keys(
    predictions: Sequence[BranchPrediction],
    config: Mapping[str, Any],
) -> list[str]:
    """Return required view:branch keys absent from normalized predictions."""
    configured = config.get("required_branches_by_view", {})
    if not isinstance(configured, Mapping):
        return []
    observed = {(prediction.view, prediction.branch) for prediction in predictions if prediction.view}
    missing: list[str] = []
    for view in sorted(str(item) for item in configured):
        branches = _string_sequence(configured.get(view))
        missing.extend(f"{view}:{branch}" for branch in branches if (view, branch) not in observed)
    return missing


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
    triggered_evidence: tuple[TriggerEvidence, ...] = (),
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
        triggered_evidence=triggered_evidence,
    )


def _evidence_reason(prediction: BranchPrediction, level: EvidenceLevel) -> str:
    """Return a compact explanation for retained evidence."""
    if prediction.reason:
        return prediction.reason
    score = "" if prediction.score is None else f" score={prediction.score:g}"
    low = "" if prediction.low_threshold is None else f" low_threshold={prediction.low_threshold:g}"
    high = "" if prediction.high_threshold is None else f" high_threshold={prediction.high_threshold:g}"
    return f"{prediction.branch} {level.value.lower()} evidence{score}{low}{high}"


def _trigger_evidence(
    prediction: BranchPrediction,
    level: EvidenceLevel,
    reason: str | None = None,
    evidence_id: str | None = None,
) -> TriggerEvidence:
    """Convert one classified prediction into immutable trigger evidence."""
    view = prediction.view or "unknown"
    return TriggerEvidence(
        evidence_id=evidence_id or f"{view}:{prediction.branch}",
        branch=prediction.branch,
        side=prediction.side,
        view=prediction.view,
        level=level.value,
        score=prediction.score,
        low_threshold=prediction.low_threshold,
        high_threshold=prediction.high_threshold,
        reason=reason or _evidence_reason(prediction, level),
    )


def _classified_triggers(
    predictions: Sequence[BranchPrediction],
    config: Mapping[str, Any],
    *,
    strict_zs32: bool = False,
) -> tuple[tuple[TriggerEvidence, BranchPrediction], ...]:
    """Classify non-gate rows once and retain all GRAY/STRONG evidence in policy order."""
    branch_order = _string_sequence(config.get("branch_order")) or DEFAULT_BRANCH_ORDER
    branch_rank = {branch: index for index, branch in enumerate(branch_order)}
    raw_classified: list[tuple[BranchPrediction, EvidenceLevel, str | None]] = []
    suspect_policy = config.get("suspect_policy", {})
    suspect_enabled = bool(suspect_policy.get("enable", False)) if isinstance(suspect_policy, Mapping) else False
    near_threshold_ratio = (
        float(suspect_policy.get("near_threshold_ratio", 0.9)) if isinstance(suspect_policy, Mapping) else 0.9
    )
    for prediction in predictions:
        if prediction.branch in GATE_BRANCHES:
            continue
        reason = None
        if strict_zs32 and prediction.status is not None and prediction.status.upper() == "SUSPECT":
            level = EvidenceLevel.GRAY
        else:
            try:
                level = classify_evidence(prediction)
            except (ValueError, TypeError):
                if strict_zs32:
                    continue
                raise
            if (
                level is EvidenceLevel.CLEAR
                and suspect_enabled
                and _is_near_threshold(prediction, near_threshold_ratio)
            ):
                level = EvidenceLevel.GRAY
                reason = (
                    f"{prediction.branch} near threshold score={prediction.score:g} threshold={prediction.threshold:g}"
                )
        if level is not EvidenceLevel.CLEAR:
            raw_classified.append((prediction, level, reason))
    base_id_counts: dict[str, int] = defaultdict(int)
    for prediction, _, _ in raw_classified:
        view = prediction.view or "unknown"
        base_id_counts[f"{view}:{prediction.branch}"] += 1
    used_ids: set[str] = set()
    base_id_ordinals: dict[str, int] = defaultdict(int)
    classified: list[tuple[TriggerEvidence, BranchPrediction]] = []
    for prediction, level, reason in raw_classified:
        view = prediction.view or "unknown"
        base_id = f"{view}:{prediction.branch}"
        base_id_ordinals[base_id] += 1
        evidence_id = base_id
        if base_id_counts[base_id] > 1:
            slot_id = _clean_text(prediction.slot_id)
            suffix = slot_id or str(base_id_ordinals[base_id])
            evidence_id = f"{base_id}:{suffix}"
            collision_index = 2
            while evidence_id in used_ids:
                evidence_id = f"{base_id}:{suffix}:{collision_index}"
                collision_index += 1
        used_ids.add(evidence_id)
        classified.append((_trigger_evidence(prediction, level, reason, evidence_id), prediction))
    classified.sort(
        key=lambda item: (
            branch_rank.get(item[0].branch, len(branch_rank)),
            item[0].view or "",
            item[0].evidence_id,
        ),
    )
    return tuple(classified)


def _nonfinite_prediction_faults(predictions: Sequence[BranchPrediction]) -> list[str]:
    """Return non-finite numeric fields that make strict fusion input invalid."""
    faults: list[str] = []
    for prediction in predictions:
        evidence_id = f"{prediction.view or 'unknown'}:{prediction.branch}"
        for field, value in (
            ("score", prediction.score),
            ("threshold", prediction.threshold),
            ("low_threshold", prediction.low_threshold),
            ("high_threshold", prediction.high_threshold),
        ):
            if value is not None and not isfinite(value):
                faults.append(f"non-finite {field}: {evidence_id}")
    return faults


STRICT_VERSION_FIELDS = ("model_version", "threshold_version", "roi_version", "template_version")


def _strict_identity(prediction: BranchPrediction) -> tuple[str, str, str, str, str, str]:
    """Return the complete production identity of one strict branch row."""
    return (
        prediction.product or "",
        prediction.profile or "",
        prediction.hand or "",
        prediction.side,
        prediction.view or "",
        prediction.branch,
    )


def _strict_contract_faults(
    predictions: Sequence[BranchPrediction],
    config: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return exact identity/capture faults and expected-version faults for strict ZS32."""
    invalid: list[str] = []
    review: list[str] = []
    identity_config = config.get("identity")
    if not isinstance(identity_config, Mapping):
        invalid.append("missing strict identity contract")
        identity_config = {}
    expected_product = _clean_text(identity_config.get("product"))
    expected_profile = _clean_text(identity_config.get("profile"))
    allowed_hands = _string_sequence(
        identity_config.get(
            "allowed_hands",
            identity_config.get("required_hands", identity_config.get("required_hand")),
        ),
    )
    required_side = _clean_text(identity_config.get("required_side"))
    for name, value in (
        ("product", expected_product),
        ("profile", expected_profile),
        ("allowed_hands", allowed_hands),
        ("required_side", required_side),
    ):
        if not value:
            invalid.append(f"missing strict identity field: {name}")

    expected_raw = config.get("expected_versions")
    if not isinstance(expected_raw, Sequence) or isinstance(expected_raw, (str, bytes)) or not expected_raw:
        review.append("missing expected_versions contract")
        expected_raw = ()
    expected_all: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for index, item in enumerate(expected_raw):
        if not isinstance(item, Mapping):
            review.append(f"invalid expected_versions record: {index}")
            continue
        identity = tuple(_clean_text(item.get(field)) or "" for field in ("hand", "side", "view", "branch"))
        if "" in identity:
            review.append(f"incomplete expected_versions identity: {index}")
            continue
        if identity in expected_all:
            review.append(f"duplicate expected_versions identity: {':'.join(identity)}")
            continue
        expected_all[identity] = item
        review.extend(
            f"missing {field} expectation for {':'.join(identity)}"
            for field in STRICT_VERSION_FIELDS
            if _clean_text(item.get(field)) is None
        )

    part_hands = sorted({prediction.hand or "" for prediction in predictions})
    if len(part_hands) != 1:
        invalid.append(f"mixed hand identity for part: {', '.join(part_hands)}")
    part_hand = part_hands[0] if len(part_hands) == 1 else ""
    if part_hand not in allowed_hands:
        invalid.append(f"unexpected hand for part: {part_hand or 'missing'}")
    part_ids = sorted({prediction.part_id for prediction in predictions})
    if len(part_ids) > 1:
        invalid.append(f"mixed part_id for part: {', '.join(part_ids)}")
    for field in ("capture_session", "group_id"):
        values = [_clean_text(getattr(prediction, field)) for prediction in predictions]
        if any(value is None for value in values):
            invalid.append(f"missing {field} for strict identity")
        distinct = sorted({value for value in values if value is not None})
        if len(distinct) > 1:
            invalid.append(f"mixed {field} for part: {', '.join(distinct)}")
    expected = {identity: value for identity, value in expected_all.items() if identity[0] == part_hand}
    review.extend(
        f"missing expected_versions hand contract: {hand}"
        for hand in allowed_hands
        if not any(identity[0] == hand for identity in expected_all)
    )

    observed: dict[tuple[str, str, str, str], list[BranchPrediction]] = defaultdict(list)
    for prediction in predictions:
        product, profile, hand, side, view, branch = _strict_identity(prediction)
        identity = (hand, side, view, branch)
        if product != (expected_product or "") or profile != (expected_profile or ""):
            invalid.append(f"product/profile mismatch for {':'.join(identity)}")
        if hand != part_hand or hand not in allowed_hands or side != (required_side or ""):
            invalid.append(f"unexpected strict identity: {':'.join(identity)}")
        if branch == "yolo":
            invalid.extend(_yolo_detection_faults(prediction))
        observed[identity].append(prediction)

    for identity, expectation in expected.items():
        rows = observed.get(identity, ())
        if not rows:
            invalid.append(f"missing required identity: {':'.join(identity)}")
            continue
        if len(rows) > 1:
            invalid.append(f"duplicate required identity: {':'.join(identity)}")
        for prediction in rows:
            for field in STRICT_VERSION_FIELDS:
                expected_value = _clean_text(expectation.get(field))
                actual_value = _clean_text(getattr(prediction, field))
                if expected_value is not None and actual_value != expected_value:
                    review.append(f"{field} mismatch for {':'.join(identity)}")
    if expected:
        invalid.extend(
            f"unexpected strict identity: {':'.join(identity)}" for identity in observed.keys() - expected.keys()
        )

    required_views = {identity[2] for identity in expected}
    for field in ("source_path", "source_hash", "manifest_identity"):
        views_by_value: dict[str, set[str]] = defaultdict(set)
        for prediction in predictions:
            if prediction.view not in required_views:
                continue
            value = _clean_text(getattr(prediction, field))
            if value is not None:
                views_by_value[value].add(str(prediction.view))
        invalid.extend(
            f"reused {field} across required views: {', '.join(sorted(views))}"
            for views in views_by_value.values()
            if len(views) > 1
        )

    return sorted(set(invalid)), sorted(set(review))


def _classification_faults(predictions: Sequence[BranchPrediction]) -> list[str]:
    """Return malformed evidence faults without hiding other valid STRONG evidence."""
    faults = []
    for prediction in predictions:
        if prediction.branch in GATE_BRANCHES:
            continue
        try:
            classify_evidence(prediction)
        except (ValueError, TypeError) as error:
            faults.append(str(error))
    return faults


def _nonpass_gate_triggers(
    predictions: Sequence[BranchPrediction],
    config: Mapping[str, Any],
    *,
    strict_zs32: bool,
) -> tuple[TriggerEvidence, ...]:
    """Retain every present quality or registration row that is not an explicit PASS."""
    triggers: list[TriggerEvidence] = []
    for prediction in predictions:
        if prediction.branch not in GATE_BRANCHES:
            continue
        status = _clean_text(prediction.status)
        requires_pass = _requires_pass(
            config,
            "quality_gate" if prediction.branch in {"quality", "quality_gate"} else "registration",
        )
        retain_status = strict_zs32 or requires_pass
        explicit_nonpass = status is not None and status.upper() != PASS_STATUS
        missing_strict_status = strict_zs32 and status is None
        if prediction.pred_label != 1 and not (retain_status and explicit_nonpass or missing_strict_status):
            continue
        observed = status or "missing status (explicit PASS required)"
        reason = prediction.reason or f"required {prediction.branch} PASS but got {observed}"
        triggers.append(_trigger_evidence(prediction, EvidenceLevel.GRAY, reason))
    return tuple(triggers)


def _system_trigger(reasons: Sequence[str]) -> TriggerEvidence:
    """Build one retained fail-closed trigger for incomplete strict inspection evidence."""
    reason = "; ".join(reasons)
    return TriggerEvidence(
        evidence_id="system:input_contract",
        branch="system",
        side="zs32",
        view=None,
        level=EvidenceLevel.GRAY.value,
        score=None,
        low_threshold=None,
        high_threshold=None,
        reason=reason,
    )


def _system_fault_decision(part_id: str, status: str, reasons: Sequence[str]) -> FusedDecision:
    """Create a fail-closed decision retaining strict input-validation reasons."""
    reason = "; ".join(reasons)
    trigger = _system_trigger(reasons)
    return FusedDecision(
        part_id=part_id,
        final_status=status,
        final_label=None,
        defect_side=None,
        defect_view=None,
        defect_slot=None,
        defect_type=None,
        triggered_branch="system",
        reason=reason,
        triggered_evidence=(trigger,),
    )


def fuse_part_predictions(
    part_id: str,
    predictions: Sequence[BranchPrediction],
    *,
    missing_required: Sequence[str] = (),
    config: Mapping[str, Any] | None = None,
) -> FusedDecision:
    """Fuse branch predictions into a fail-closed decision while retaining every trigger."""
    fusion_config = config or {}
    strict_zs32 = _clean_text(fusion_config.get("profile")) == "zs32_six_view_v1"
    missing = [item for item in missing_required if item]
    classified_triggers = _classified_triggers(predictions, fusion_config, strict_zs32=strict_zs32)
    gate_triggers = _nonpass_gate_triggers(predictions, fusion_config, strict_zs32=strict_zs32)
    triggers = (*tuple(item[0] for item in classified_triggers), *gate_triggers)
    invalid_faults: list[str] = []
    review_faults: list[str] = []
    if strict_zs32:
        invalid_faults.extend(_nonfinite_prediction_faults(predictions))
        contract_invalid, contract_review = _strict_contract_faults(predictions, fusion_config)
        invalid_faults.extend(contract_invalid)
        review_faults.extend(contract_review)
        review_faults.extend(_classification_faults(predictions))
        invalid_faults.extend(f"missing required input: {item}" for item in missing)
        invalid_faults.extend(
            f"missing required view: {item}" for item in missing_required_views(predictions, fusion_config)
        )
        invalid_faults.extend(
            f"missing required branch: {item}" for item in missing_required_branch_keys(predictions, fusion_config)
        )
        all_system_faults = [*sorted(set(invalid_faults)), *sorted(set(review_faults))]
        if all_system_faults:
            triggers = (*triggers, _system_trigger(all_system_faults))

        first_strong = next(
            (item for item in classified_triggers if item[0].level == EvidenceLevel.STRONG.value),
            None,
        )
        if first_strong is not None:
            _, prediction = first_strong
            status = _status_from_branch(fusion_config, prediction.branch)
            if status == "SUSPECT":
                status = "REVIEW"
            return _decision_from_prediction(
                part_id,
                status,
                1 if status != "REVIEW" else None,
                prediction,
                _branch_reason(prediction),
                triggers,
            )
        if invalid_faults:
            decision = _system_fault_decision(part_id, "INVALID_CAPTURE", sorted(set(invalid_faults)))
            return FusedDecision(**{**decision.__dict__, "triggered_evidence": triggers})
        if review_faults:
            decision = _system_fault_decision(part_id, "REVIEW", sorted(set(review_faults)))
            return FusedDecision(**{**decision.__dict__, "triggered_evidence": triggers})

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
            triggered_evidence=triggers,
        )

    for branch in ("quality", "quality_gate", "registration"):
        for prediction in predictions:
            status = _clean_text(prediction.status)
            nonpass_status = strict_zs32 and (status is None or status.upper() != PASS_STATUS)
            if prediction.branch == branch and (prediction.pred_label == 1 or nonpass_status):
                if nonpass_status:
                    return _gate_not_pass_decision(part_id, prediction, triggers)
                return _decision_from_prediction(
                    part_id,
                    "RETAKE",
                    None,
                    prediction,
                    _branch_reason(prediction),
                    triggers,
                )

    if _requires_pass(fusion_config, "quality_gate") and not any(
        prediction.branch in {"quality", "quality_gate"} for prediction in predictions
    ):
        return _gate_missing_decision(part_id, "quality_gate", triggers)
    if _requires_pass(fusion_config, "quality_gate"):
        nonpass_quality = _first_nonpass_gate(predictions, {"quality", "quality_gate"})
        if nonpass_quality is not None:
            return _gate_not_pass_decision(part_id, nonpass_quality, triggers)
        if not _gate_has_pass(predictions, {"quality", "quality_gate"}):
            return _gate_missing_decision(part_id, "quality_gate", triggers)
    if _requires_pass(fusion_config, "registration") and not any(
        prediction.branch == "registration" for prediction in predictions
    ):
        return _gate_missing_decision(part_id, "registration", triggers)
    if _requires_pass(fusion_config, "registration"):
        nonpass_registration = _first_nonpass_gate(predictions, {"registration"})
        if nonpass_registration is not None:
            return _gate_not_pass_decision(part_id, nonpass_registration, triggers)
        if not _gate_has_pass(predictions, {"registration"}):
            return _gate_missing_decision(part_id, "registration", triggers)

    first_strong = next(
        (item for item in classified_triggers if item[0].level == EvidenceLevel.STRONG.value),
        None,
    )
    if first_strong is not None:
        _, prediction = first_strong
        status = _status_from_branch(fusion_config, prediction.branch)
        return _decision_from_prediction(
            part_id,
            status,
            None if status == "SUSPECT" else 1,
            prediction,
            _branch_reason(prediction),
            triggers,
        )

    configured_missing_views = missing_required_views(predictions, fusion_config)
    missing_branches = missing_required_branch_keys(predictions, fusion_config)
    review_reasons = [*(f"missing required view: {item}" for item in configured_missing_views)]
    review_reasons.extend(f"missing required branch: {item}" for item in missing_branches)
    if triggers:
        review_reasons.extend(trigger.reason for trigger in triggers)
    if review_reasons:
        prediction = classified_triggers[0][1] if classified_triggers else None
        legacy_near_threshold = not strict_zs32 and any(
            "near threshold" in trigger.reason for trigger, _ in classified_triggers
        )
        return FusedDecision(
            part_id=part_id,
            final_status="SUSPECT" if legacy_near_threshold else "REVIEW",
            final_label=None,
            defect_side=None if prediction is None else prediction.side,
            defect_view=None if prediction is None else prediction.view,
            defect_slot=None if prediction is None else prediction.slot_id,
            defect_type=None if prediction is None else prediction.defect_type,
            triggered_branch=None if prediction is None else prediction.branch,
            reason="; ".join(review_reasons),
            triggered_evidence=triggers,
        )

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
        triggered_evidence=triggers,
    )


def _face_config(config: Mapping[str, Any], face: str) -> dict[str, Any]:
    """Return a shallow fusion config restricted to the three views of one face."""
    face_views = {face, f"{face}_left", f"{face}_right"}
    face_config = dict(config)
    ok_requires = config.get("ok_requires")
    if isinstance(ok_requires, Mapping):
        filtered_requires = dict(ok_requires)
        required_keys = _string_sequence(ok_requires.get("required_view_keys"))
        filtered_requires["required_view_keys"] = [
            key for key in required_keys if key.rsplit(":", maxsplit=1)[-1] in face_views
        ]
        face_config["ok_requires"] = filtered_requires
    required_branches = config.get("required_branches_by_view")
    if isinstance(required_branches, Mapping):
        face_config["required_branches_by_view"] = {
            view: branches for view, branches in required_branches.items() if str(view) in face_views
        }
    return face_config


def fuse_face_predictions(
    part_id: str,
    predictions: Sequence[BranchPrediction],
    *,
    face: str,
    config: Mapping[str, Any] | None = None,
) -> FaceDecision:
    """Fuse exactly one three-view face without releasing a final OK.

    Args:
        part_id (str): Stable part identity shared by every prediction.
        predictions (Sequence[BranchPrediction]): Rows for the selected face.
        face (str): Physical face, either ``front`` or ``back``.
        config (Mapping[str, Any] | None): Strict fusion configuration.

    Returns:
        FaceDecision: A staged CLEAR, REVIEW, or NG result with no final release.

    Raises:
        ValueError: If the face, part identity, or supplied views are invalid.
    """
    if face not in {"front", "back"}:
        msg = "face must be 'front' or 'back'"
        raise ValueError(msg)
    mismatched_parts = sorted({prediction.part_id for prediction in predictions if prediction.part_id != part_id})
    if mismatched_parts:
        msg = f"part_id mismatch: expected {part_id}, found {', '.join(mismatched_parts)}"
        raise ValueError(msg)
    required_views = {face, f"{face}_left", f"{face}_right"}
    wrong_views = sorted({str(prediction.view) for prediction in predictions if prediction.view not in required_views})
    if wrong_views:
        msg = f"predictions contain views outside {face} face: {', '.join(wrong_views)}"
        raise ValueError(msg)
    observed_views = {prediction.view for prediction in predictions}
    missing_views = sorted(required_views - observed_views)
    decision = fuse_part_predictions(
        part_id,
        predictions,
        missing_required=missing_views,
        config=_face_config(config or {}, face),
    )
    if decision.final_status == "OK":
        state = "CLEAR"
    elif decision.final_status == "REVIEW" or decision.final_status in {"RETAKE", "INVALID_CAPTURE"}:
        state = "REVIEW"
    else:
        state = "NG"
    return FaceDecision(
        part_id=part_id,
        face=face,
        stage_status=f"{face.upper()}_{state}",
        final_status=None,
        inspection_complete=False,
        triggered_evidence=decision.triggered_evidence,
        reason=decision.reason,
        defect_side=decision.defect_side,
        defect_view=decision.defect_view,
        defect_slot=decision.defect_slot,
        defect_type=decision.defect_type,
        triggered_branch=decision.triggered_branch,
    )


def combine_face_decisions(front: FaceDecision, back: FaceDecision) -> FusedDecision:
    """Combine matching staged faces into the only releasable part decision.

    Args:
        front (FaceDecision): Completed front-face stage.
        back (FaceDecision): Completed back-face stage.

    Returns:
        FusedDecision: ``OK`` only for CLEAR+CLEAR, otherwise REVIEW or NG.

    Raises:
        ValueError: If identities or face roles do not match.
    """
    if front.part_id != back.part_id:
        msg = f"part_id mismatch: front={front.part_id}, back={back.part_id}"
        raise ValueError(msg)
    if front.face != "front" or back.face != "back":
        msg = "combine_face_decisions requires front then back face decisions"
        raise ValueError(msg)
    valid_front_stages = {"FRONT_CLEAR", "FRONT_REVIEW", "FRONT_NG"}
    valid_back_stages = {"BACK_CLEAR", "BACK_REVIEW", "BACK_NG"}
    if front.stage_status not in valid_front_stages or back.stage_status not in valid_back_stages:
        msg = f"invalid stage_status pair: {front.stage_status}, {back.stage_status}"
        raise ValueError(msg)
    primary: FaceDecision | None
    if front.stage_status == "FRONT_NG":
        status, label, primary = "NG", 1, front
    elif back.stage_status == "BACK_NG":
        status, label, primary = "NG", 1, back
    elif front.stage_status == "FRONT_REVIEW":
        status, label, primary = "REVIEW", None, front
    elif back.stage_status == "BACK_REVIEW":
        status, label, primary = "REVIEW", None, back
    else:
        status, label, primary = "OK", 0, None
    triggers = (*front.triggered_evidence, *back.triggered_evidence)
    return FusedDecision(
        part_id=front.part_id,
        final_status=status,
        final_label=label,
        defect_side=None if primary is None else primary.defect_side,
        defect_view=None if primary is None else primary.defect_view,
        defect_slot=None if primary is None else primary.defect_slot,
        defect_type=None if primary is None else primary.defect_type,
        triggered_branch=None if primary is None else primary.triggered_branch,
        reason=f"{front.stage_status} + {back.stage_status}" if primary is None else primary.reason,
        triggered_evidence=triggers,
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
        hand=_first_value(row, ("hand", "part_hand")),
        product=_first_value(row, ("product", "product_id", "product_name")),
        profile=_first_value(row, ("profile", "profile_id", "fusion_profile")),
        source_hash=_first_value(row, ("source_hash", "source_sha256", "image_sha256")),
        evidence_hash=_first_value(row, ("evidence_hash", "evidence_sha256", "template_sha256")),
        manifest_identity=_first_value(row, ("manifest_identity", "manifest_image_id", "capture_id", "image_id")),
        capture_session=_first_value(row, ("capture_session", "session_id")),
        group_id=_first_value(row, ("group_id", "capture_group")),
        detections=_structured_detections(_first_value(row, ("detections", "yolo_detections"))),
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
            detections = row["detections"]
            row["detections"] = (
                "" if detections is None else json.dumps(detections, separators=(",", ":"), sort_keys=True)
            )
            writer.writerow({field: row.get(field) for field in BRANCH_FIELDNAMES})


def write_fused_decisions_csv(decisions: Sequence[FusedDecision], output_csv: Path) -> None:
    """Write fused inspection decisions to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FUSED_FIELDNAMES)
        writer.writeheader()
        for decision in decisions:
            row = asdict(decision)
            writer.writerow({field: row.get(field) for field in FUSED_FIELDNAMES})


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
        prediction.branch == "missing_prediction" or marker in (prediction.reason or "") for prediction in predictions
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
        counts[f"{defect_type}_recall"] = type_count["detected"] / type_count["total"] if type_count["total"] else 0.0
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
                "defect_type": gt_defect_type
                or decision.defect_type
                or (first_prediction.defect_type if first_prediction else ""),
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
