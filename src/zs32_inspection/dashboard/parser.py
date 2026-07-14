"""Strict parser for one complete ZS32 eight-view dashboard result."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

import cv2

from .contracts import (
    MODELED_VIEWS,
    VIEW_ORDER,
    BranchEvidence,
    BranchState,
    InspectionIdentity,
    InspectionResult,
    ViewResult,
)

_BRANCH_NAMES = ("template", "patchcore", "yolo", "fusion")
_ERROR_STATUSES = {"error", "failed", "execution_error"}
_SKIPPED_STATUSES = {"skipped", "not_run", "not_executed"}


class DashboardResultError(ValueError):
    """Raised when a result cannot represent one complete ZS32 inspection."""


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DashboardResultError(f"could not load strict JSON manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DashboardResultError("runtime_manifest.json must contain a JSON object")
    return payload


def _required_text(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DashboardResultError(f"{context}.{field} must be a non-empty string")
    return value


def _parse_identity(manifest: Mapping[str, Any]) -> InspectionIdentity:
    product = _required_text(manifest, "product", context="manifest")
    if product != "ZS32":
        raise DashboardResultError(f"manifest.product must be 'ZS32', got {product!r}")
    manifest_identity = _required_text(manifest, "manifest_identity", context="manifest")
    if manifest_identity != "ZS32/right":
        raise DashboardResultError("manifest.manifest_identity must be 'ZS32/right'")
    values = {
        field: _required_text(manifest, field, context="manifest")
        for field in ("part_id", "capture_session", "group_id", "hand")
    }
    try:
        return InspectionIdentity(**values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise DashboardResultError(f"invalid manifest identity: {error}") from error


def _require_exact_views(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DashboardResultError("manifest.views must be an object with exactly eight views")
    expected = set(VIEW_ORDER)
    actual = set(value)
    if len(value) != len(VIEW_ORDER) or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DashboardResultError(f"manifest.views must contain exactly VIEW_ORDER; missing={missing}, extra={extra}")
    return value


def _resolve_result_path(result_dir: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DashboardResultError(f"{field} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = result_dir / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as error:
        raise DashboardResultError(f"{field} could not be resolved: {error}") from error
    if not resolved.is_relative_to(result_dir):
        raise DashboardResultError(f"{field} escapes result_dir: {value!r}")
    return resolved


def _require_file(result_dir: Path, value: Any, *, field: str) -> Path:
    path = _resolve_result_path(result_dir, value, field=field)
    if not path.is_file():
        raise DashboardResultError(f"{field} does not identify a file: {path}")
    return path


def _source_shape(value: Any, *, view: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise DashboardResultError(f"views.{view}.source_shape must be [height, width]")
    return value[0], value[1]


def _finite_score(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DashboardResultError(f"{field} must be a finite number or null")
    score = float(value)
    if not math.isfinite(score):
        raise DashboardResultError(f"{field} must be finite")
    return score


def _parse_roi(value: Any, *, field: str, source_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise DashboardResultError(f"{field} must be four integer xyxy coordinates")
    x1, y1, x2, y2 = value
    height, width = source_shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise DashboardResultError(f"{field} lies outside the source image")
    return x1, y1, x2, y2


def _parse_detections(value: Any, *, field: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise DashboardResultError(f"{field} must be a list of objects")
    return tuple(dict(item) for item in value)


def _branch_state(value: Mapping[str, Any], *, context: str) -> BranchState:
    raw_state = value.get("state")
    if raw_state is not None:
        if not isinstance(raw_state, str):
            raise DashboardResultError(f"{context}.state must be a string")
        try:
            return BranchState(raw_state.lower())
        except ValueError as error:
            raise DashboardResultError(f"{context}.state is invalid: {raw_state!r}") from error

    status = value.get("status")
    if not isinstance(status, str) or not status.strip():
        raise DashboardResultError(f"{context}.status is required when state is absent")
    executed = value.get("executed")
    if executed is not None and not isinstance(executed, bool):
        raise DashboardResultError(f"{context}.executed must be a boolean")
    normalized = status.strip().lower().replace(" ", "_")
    if normalized in _ERROR_STATUSES:
        return BranchState.ERROR
    if normalized == BranchState.UNSUPPORTED:
        return BranchState.UNSUPPORTED
    if executed is False or normalized in _SKIPPED_STATUSES:
        return BranchState.SKIPPED
    return BranchState.AVAILABLE


def _parse_branch(
    branch: str,
    value: Any,
    *,
    view: str,
    result_dir: Path,
    source_shape: tuple[int, int],
) -> BranchEvidence:
    context = f"views.{view}.branches.{branch}"
    if not isinstance(value, dict):
        return BranchEvidence(branch, BranchState.ERROR, "ERROR", None, f"{context} must be an object")

    score: float | None = None
    try:
        score = _finite_score(value.get("score"), field=f"{context}.score")
        state = _branch_state(value, context=context)
        status = value.get("status", state.value.upper())
        reason = value.get("reason", "")
        if not isinstance(status, str) or not isinstance(reason, str):
            raise DashboardResultError(f"{context}.status and reason must be strings")

        evidence_path = None
        if value.get("evidence_path") is not None:
            evidence_path = _require_file(result_dir, value["evidence_path"], field=f"{context}.evidence_path")
        if state is BranchState.AVAILABLE and evidence_path is None:
            raise DashboardResultError(f"{context}.evidence_path is required when available")
        for path_field, path_value in value.items():
            if (
                path_field.endswith("_path")
                and path_field not in {"evidence_path", "mask_path"}
                and path_value is not None
            ):
                _require_file(result_dir, path_value, field=f"{context}.{path_field}")

        roi_xyxy = None
        if value.get("roi_xyxy") is not None:
            roi_xyxy = _parse_roi(value["roi_xyxy"], field=f"{context}.roi_xyxy", source_shape=source_shape)
        mask_path = None
        mask_source = value.get("mask_source")
        if mask_source is not None and not isinstance(mask_source, str):
            raise DashboardResultError(f"{context}.mask_source must be a string or null")
        if value.get("mask_path") is not None:
            mask_path = _require_file(result_dir, value["mask_path"], field=f"{context}.mask_path")
            if roi_xyxy is None:
                raise DashboardResultError(f"{context}.roi_xyxy is required with mask_path")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise DashboardResultError(f"{context}.mask_path could not be decoded")
            x1, y1, x2, y2 = roi_xyxy
            if tuple(mask.shape) != (y2 - y1, x2 - x1):
                raise DashboardResultError(f"{context}.mask_path geometry does not match roi_xyxy")
            if not set(mask.reshape(-1).tolist()) <= {0, 255}:
                raise DashboardResultError(f"{context}.mask_path must contain a binary 0/255 mask")
        if branch == "patchcore" and state is BranchState.AVAILABLE and mask_path is None:
            raise DashboardResultError(f"{context}.mask_path is required when PatchCore is available")

        return BranchEvidence(
            branch=branch,
            state=state,
            status=status,
            score=score,
            reason=reason,
            evidence_path=evidence_path,
            mask_path=mask_path,
            mask_source=mask_source,
            roi_xyxy=roi_xyxy,
            detections=_parse_detections(value.get("detections"), field=f"{context}.detections"),
        )
    except DashboardResultError as error:
        original_reason = value.get("reason", "")
        reason = f"{original_reason}; {error}" if isinstance(original_reason, str) and original_reason else str(error)
        return BranchEvidence(
            branch=branch,
            state=BranchState.ERROR,
            status="ERROR",
            score=score,
            reason=reason,
        )


def _parse_view(
    value: Any,
    expected_view: str,
    identity: InspectionIdentity,
    result_dir: Path,
) -> ViewResult:
    context = f"views.{expected_view}"
    if not isinstance(value, dict):
        raise DashboardResultError(f"{context} must be an object")
    actual_view = _required_text(value, "view", context=context)
    if actual_view != expected_view:
        raise DashboardResultError(f"{context}.view must be {expected_view!r}, got {actual_view!r}")
    for field in ("part_id", "capture_session", "group_id", "hand"):
        actual = _required_text(value, field, context=context)
        expected = getattr(identity, field)
        if actual != expected:
            raise DashboardResultError(f"{context}.{field} conflicts with manifest identity")
    expected_manifest_identity = f"{identity.part_id}:{identity.hand}:{expected_view}"
    actual_manifest_identity = _required_text(value, "manifest_identity", context=context)
    if actual_manifest_identity != expected_manifest_identity:
        raise DashboardResultError(f"{context}.manifest_identity does not match {expected_manifest_identity!r}")

    expected_support = expected_view in MODELED_VIEWS
    model_supported = value.get("model_supported")
    if model_supported is not expected_support:
        raise DashboardResultError(f"{context}.model_supported must be {expected_support}")

    source_path = _require_file(result_dir, value.get("source_path"), field=f"{context}.source_path")
    source_sha256 = _required_text(value, "source_sha256", context=context)
    actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != actual_sha256:
        raise DashboardResultError(f"{context}.source_sha256 does not match source_path")
    source_shape = _source_shape(value.get("source_shape"), view=expected_view)
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise DashboardResultError(f"{context}.source_path could not be decoded by OpenCV")
    if tuple(image.shape[:2]) != source_shape:
        raise DashboardResultError(
            f"{context}.source_shape {source_shape!r} does not match decoded shape {tuple(image.shape[:2])!r}",
        )

    branch_values = value.get("branches")
    if branch_values is None:
        branch_values = {branch: value[branch] for branch in _BRANCH_NAMES if branch in value}
    if not isinstance(branch_values, dict) or set(branch_values) != set(_BRANCH_NAMES):
        raise DashboardResultError(f"{context}.branches must contain exactly {_BRANCH_NAMES!r}")
    branches = {
        branch: _parse_branch(
            branch,
            branch_value,
            view=expected_view,
            result_dir=result_dir,
            source_shape=source_shape,
        )
        for branch, branch_value in branch_values.items()
        if isinstance(branch, str) and branch
    }
    if len(branches) != len(branch_values):
        raise DashboardResultError(f"{context}.branches keys must be non-empty strings")
    capture = {
        field: value[field]
        for field in ("part_id", "capture_session", "group_id", "hand", "manifest_identity", "camera")
        if isinstance(value.get(field), str)
    }
    try:
        return ViewResult(
            view=expected_view,
            source_path=source_path,
            source_sha256=source_sha256,
            source_shape=source_shape,
            model_supported=model_supported,
            branches=branches,
            capture=capture,
        )
    except ValueError as error:
        raise DashboardResultError(f"invalid {context}: {error}") from error


def load_inspection_result(result_dir: Path) -> InspectionResult:
    """Load and validate one explicit eight-view ``runtime_manifest.json`` result."""
    try:
        resolved_result_dir = Path(result_dir).expanduser().resolve()
    except (OSError, RuntimeError) as error:
        raise DashboardResultError(f"result_dir could not be resolved: {error}") from error
    if not resolved_result_dir.is_dir():
        raise DashboardResultError(f"result_dir is not a directory: {resolved_result_dir}")
    manifest = _load_json(resolved_result_dir / "runtime_manifest.json")
    identity = _parse_identity(manifest)
    records = _require_exact_views(manifest.get("views"))
    views = tuple(_parse_view(records[view], view, identity, resolved_result_dir) for view in VIEW_ORDER)
    if len({view.source_path for view in views}) != len(VIEW_ORDER):
        raise DashboardResultError("the eight views must identify eight distinct source files")
    machine_status = _required_text(manifest, "machine_status", context="manifest")
    reason = manifest.get("reason", "")
    if not isinstance(reason, str):
        raise DashboardResultError("manifest.reason must be a string")
    mode = manifest.get("mode", "offline")
    if mode not in ("offline", "live"):
        raise DashboardResultError("manifest.mode must be 'offline' or 'live'")
    return InspectionResult(identity, views, machine_status, reason, mode)
