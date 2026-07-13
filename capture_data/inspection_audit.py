# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Immutable and atomic evidence records for ZS32 inspection decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from capture_data.fusion_engine import BranchPrediction, TriggerEvidence

CANONICAL_VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path_value: str | None) -> tuple[bool | None, str | None]:
    """Return artifact existence and its digest when present."""
    if path_value is None:
        return None, None
    path = Path(path_value)
    exists = path.is_file()
    return exists, sha256_file(path) if exists else None


def _normalized_risk(score: float | None, low: float | None, high: float | None) -> float | None:
    """Map branch-local dual thresholds onto a stable zero-to-one risk scale."""
    if score is None or low is None or high is None or not all(isfinite(value) for value in (score, low, high)):
        return None
    if low > high:
        return None
    if low == high:
        return float(score >= high)
    return max(0.0, min(1.0, 0.5 + 0.5 * (score - low) / (high - low)))


def _computed_evidence(prediction: BranchPrediction) -> tuple[str, str | None]:
    """Return the engine-computed evidence band and any validation error."""
    from capture_data.fusion_engine import classify_evidence

    try:
        return classify_evidence(prediction).value, None
    except (TypeError, ValueError) as error:
        return "INVALID", str(error)


def _unique_prediction_value(predictions: Sequence[BranchPrediction], *names: str) -> str | None:
    """Return one shared non-empty identity value, or an explicit mixed marker."""
    values = {
        str(value)
        for prediction in predictions
        for name in names
        if (value := getattr(prediction, name, None)) not in {None, ""}
    }
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return "MIXED:" + ",".join(sorted(values))


def _serialize_prediction(prediction: BranchPrediction) -> dict[str, Any]:
    """Serialize one normalized prediction without discarding absent artifacts."""
    source_exists, source_sha256 = _artifact(prediction.source_path)
    evidence_exists, evidence_sha256 = _artifact(prediction.evidence_path)
    serialized = asdict(prediction)
    invalid_numeric: dict[str, str] = {}
    for field in ("score", "threshold", "low_threshold", "high_threshold"):
        value = serialized[field]
        if value is not None and not isfinite(value):
            invalid_numeric[field] = str(value)
            serialized[field] = None
    score = serialized["score"]
    low = serialized["low_threshold"]
    high = serialized["high_threshold"]
    computed_evidence_level, evidence_error = _computed_evidence(prediction)
    return {
        **serialized,
        "invalid_numeric": invalid_numeric,
        "low_margin": None if score is None or low is None else score - low,
        "high_margin": None if score is None or high is None else score - high,
        "computed_evidence_level": computed_evidence_level,
        "evidence_error": evidence_error,
        "normalized_risk": _normalized_risk(score, low, high),
        "source_exists": source_exists,
        "source_sha256": source_sha256,
        "evidence_exists": evidence_exists,
        "evidence_sha256": evidence_sha256,
    }


def build_part_audit(
    part_id: str,
    predictions: Sequence[BranchPrediction],
    *,
    machine_status: str,
    triggered_evidence: Sequence[TriggerEvidence],
    inspection_complete: bool = False,
    session_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a complete six-view machine audit without mutating review state.

    Args:
        part_id (str): Stable identity for the inspected part.
        predictions (Sequence[BranchPrediction]): Normalized branch predictions.
        machine_status (str): Immutable machine decision.
        triggered_evidence (Sequence[TriggerEvidence]): Every GRAY or STRONG trigger.
        inspection_complete (bool): Whether all required inspection evidence is valid.
        session_id (str | None): Optional acquisition-session identity override.
        timestamp (str | None): Optional ISO-8601 inspection timestamp override.

    Returns:
        dict[str, Any]: JSON-serializable audit record initialized for review.

    Raises:
        ValueError: If a prediction has another part identity or unknown view.
    """
    views: dict[str, list[dict[str, Any]]] = {view: [] for view in CANONICAL_VIEWS}
    for prediction in predictions:
        if prediction.part_id != part_id:
            msg = f"part_id mismatch: expected {part_id}, found {prediction.part_id}"
            raise ValueError(msg)
        if prediction.view not in views:
            msg = f"unknown ZS32 view: {prediction.view}"
            raise ValueError(msg)
        views[prediction.view].append(_serialize_prediction(prediction))
    audit_timestamp = timestamp or _unique_prediction_value(predictions, "timestamp", "captured_at")
    if audit_timestamp is None:
        audit_timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "2.0",
        "part_id": part_id,
        "product": _unique_prediction_value(predictions, "product", "product_id"),
        "profile": _unique_prediction_value(predictions, "profile", "profile_id"),
        "hand": _unique_prediction_value(predictions, "hand"),
        "session_id": session_id or _unique_prediction_value(predictions, "session_id", "capture_session"),
        "timestamp": audit_timestamp,
        "inspection_complete": inspection_complete,
        "machine_status": machine_status,
        "machine": {
            "status": machine_status,
            "inspection_complete": inspection_complete,
            "trigger_ids": [trigger.evidence_id for trigger in triggered_evidence],
        },
        "views": views,
        "triggers": [asdict(trigger) for trigger in triggered_evidence],
        "review": {"status": "PENDING"},
        "review_status": None,
        "released_status": None,
    }


def write_part_audit(audit: Mapping[str, Any], output_path: Path) -> Path:
    """Atomically publish one audit JSON file.

    Args:
        audit (Mapping[str, Any]): JSON-serializable audit record.
        output_path (Path): Final JSON destination.

    Returns:
        Path: Published final path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return output_path
