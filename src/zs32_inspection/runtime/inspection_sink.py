"""Atomic filesystem publication of complete ZS32 inspection evidence."""

from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from pathlib import Path
import stat
from typing import Any, Mapping

from zs32_inspection.capture.reader import load_capture_bundle
from zs32_inspection.capture.gates import CaptureGateResult
from zs32_inspection.domain.decisions import InspectionDecision, InspectionStatus
from zs32_inspection.domain.evidence import (
    EvidenceBranch,
    EvidenceLevel,
    ModelEvidence,
    TemplateEvidence,
    TemplateOutcome,
)
from zs32_inspection.domain.identity import Hand
from zs32_inspection.fusion.completeness import (
    EvidenceContext,
    require_complete_models,
    require_complete_templates,
)
from zs32_inspection.fusion.engine import StrictFusionEngine
from zs32_inspection.models.base import ModelInput, RawModelScore, sha256_file
from zs32_inspection.template.predictor import TemplateScore

from .audit import AuditTrigger, InspectionAudit, utc_now
from .orchestrator import (
    InspectionRequest,
    InspectionRun,
    LoadedRuntime,
    RuntimeTemplateSlotContract,
)
from .publisher import AtomicDirectoryPublisher, PublicationError, canonical_json_bytes


def _value(value: object) -> object:
    """Convert nested runtime values to deterministic JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    raise PublicationError(f"unsupported inspection JSON value: {type(value).__name__}")


def _template_row(row: TemplateEvidence) -> dict[str, object]:
    return {
        "inspection_id": row.inspection_id,
        "capture_set_id": row.capture_set_id,
        "part_instance_id": row.part_instance_id,
        "hand": row.hand.value,
        "view": row.view_id,
        "outcome": row.outcome.value,
        "score": row.score,
        "threshold": row.threshold,
        "template_sha256": row.template_sha256,
        "threshold_sha256": row.threshold_sha256,
        "source_sha256": row.source_sha256,
        "crop_sha256": row.crop_sha256,
        "roi_version": row.roi_config_id,
    }


def _model_row(row: ModelEvidence, raw: RawModelScore | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "inspection_id": row.inspection_id,
        "capture_set_id": row.capture_set_id,
        "part_instance_id": row.part_instance_id,
        "hand": row.hand.value,
        "view": row.view_id,
        "branch": row.branch.value,
        "level": row.level.value,
        "score": row.score,
        "model_family": row.model_family,
        "model_sha256": row.model_sha256,
        "threshold_sha256": row.threshold_sha256,
        "roi_version": row.roi_config_id,
        "source_sha256": row.source_sha256,
        "crop_sha256": row.crop_sha256,
    }
    if raw is not None:
        payload["detections"] = _value(raw.detections)
        payload["heatmap_sha256"] = raw.heatmap_sha256
        payload["overlay_sha256"] = raw.overlay_sha256
    return payload


def _raw_model_row(raw: RawModelScore, calibrated: ModelEvidence | None) -> dict[str, object]:
    """Retain raw evidence even when identity/completeness validation failed."""
    payload: dict[str, object] = {
        "inspection_id": raw.inspection_id,
        "capture_set_id": raw.capture_set_id,
        "part_instance_id": raw.part_instance_id,
        "hand": raw.hand,
        "view": raw.view,
        "branch": raw.branch,
        "score": raw.score,
        "model_family": raw.model_family,
        "model_sha256": raw.model_digest,
        "roi_version": raw.roi_config_id,
        "roi_sha256": raw.roi_digest,
        "source_sha256": raw.source_sha256,
        "crop_sha256": raw.crop_sha256,
        "detections": _value(raw.detections),
        "heatmap_sha256": raw.heatmap_sha256,
        "overlay_sha256": raw.overlay_sha256,
        "calibration_status": "not_calibrated" if calibrated is None else "calibrated",
        "level": None if calibrated is None else calibrated.level.value,
        "threshold_sha256": None if calibrated is None else calibrated.threshold_sha256,
    }
    return payload


def _branch_rows(run: InspectionRun, branch: EvidenceBranch) -> list[dict[str, object]]:
    calibrated = {
        row.view_id: row for row in run.model_evidence if row.branch is branch
    }
    raw_rows = [row for row in run.raw_model_scores if row.branch == branch.value]
    if raw_rows:
        return [_raw_model_row(row, calibrated.get(row.view)) for row in raw_rows]
    return [_model_row(row, None) for row in calibrated.values()]


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _decision(run: InspectionRun) -> dict[str, object]:
    decision = run.decision
    return {
        "inspection_id": decision.inspection_id,
        "evidence_status": None if decision.evidence_status is None else decision.evidence_status.value,
        "inspection_status": decision.inspection_status.value,
        "review_status": decision.review_status,
        "released_status": None if decision.released_status is None else decision.released_status.value,
        "reason_codes": list(decision.reason_codes),
    }


def _row_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _template_score_row(row: TemplateScore) -> dict[str, object]:
    return {
        "inspection_id": row.inspection_id,
        "capture_set_id": row.capture_set_id,
        "part_instance_id": row.part_instance_id,
        "hand": row.hand,
        "view": row.view,
        "risk_score": row.risk_score,
        "similarity": row.similarity,
        "model_sha256": row.model_digest,
        "roi_version": row.roi_config_id,
        "roi_sha256": row.roi_digest,
        "source_sha256": row.source_sha256,
        "crop_sha256": row.crop_sha256,
        "best_template_sha256": row.best_template_sha256,
        "offset_xy": list(row.offset_xy),
    }


def _reject(message: str) -> None:
    raise PublicationError(f"inspection evidence contract violation: {message}")


def _validate_template_score_references(runtime: LoadedRuntime, run: InspectionRun) -> None:
    """Bind every raw template result to one verified release reference file."""
    contracts = getattr(runtime, "template_slot_contracts", None)
    if not isinstance(contracts, Mapping):
        _reject("loaded runtime has no release-bound template reference contracts")
    for score in run.template_scores:
        if not isinstance(score, TemplateScore):
            _reject("template score evidence contains an invalid record type")
        try:
            key = (Hand.parse(score.hand), score.view)
            slot_contract = contracts[key]
        except (KeyError, TypeError, ValueError) as error:
            _reject(f"template score has no release reference contract: {score.hand}/{score.view}: {error}")
        if not isinstance(slot_contract, RuntimeTemplateSlotContract):
            _reject(f"template reference contract has an invalid type for {score.hand}/{score.view}")
        offset = score.offset_xy
        if (
            not isinstance(offset, tuple)
            or len(offset) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offset)
            or any(abs(value) > slot_contract.max_shift for value in offset)
        ):
            _reject(
                f"template score offset exceeds release max_shift for {score.hand}/{score.view}"
            )
        candidate = Path(score.best_template_path).expanduser()
        try:
            metadata = candidate.stat(follow_symlinks=False)
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            _reject(
                f"template score best reference is unavailable for {score.hand}/{score.view}: {error}"
            )
        if not candidate.is_absolute() or candidate != resolved:
            _reject(
                f"template score best reference path is not canonical for {score.hand}/{score.view}"
            )
        if (
            candidate.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            _reject(
                f"template score best reference must be a private regular file for {score.hand}/{score.view}"
            )
        reference_by_path = {item.path: item for item in slot_contract.references}
        reference = reference_by_path.get(resolved)
        if reference is None:
            _reject(
                f"template score best reference is outside the release allowlist for {score.hand}/{score.view}"
            )
        try:
            actual_sha256 = sha256_file(resolved)
        except OSError as error:
            _reject(
                f"template score best reference cannot be rehashed for {score.hand}/{score.view}: {error}"
            )
        if score.best_template_sha256 != reference.sha256 or actual_sha256 != reference.sha256:
            _reject(
                f"template score best reference hash differs from release for {score.hand}/{score.view}"
            )


def _revalidate_capture_publication(runtime: LoadedRuntime, run: InspectionRun) -> None:
    """Reload the complete capture tree and align it with the in-memory request.

    This check deliberately runs again in the staging validator so source files,
    manifests, gates, or the publication root cannot change unnoticed between
    request construction and the final inspection publication.
    """
    request = run.request
    try:
        manifest_path = Path(request.capture_manifest_path)
        gates_path = Path(request.capture_gates_path)
        set_root = manifest_path.parent
        expected_set_root = (
            Path(request.capture_root)
            / "images"
            / request.capture.capture_set_id
        )
        if (
            manifest_path.name != "capture_manifest.json"
            or gates_path != set_root / "capture_gates.json"
            or set_root.resolve(strict=True) != expected_set_root.resolve(strict=True)
        ):
            _reject("capture publication paths differ from the canonical session/set layout")
        bundle = load_capture_bundle(set_root, runtime.contract.topology)
        if bundle.domain_capture_set != request.capture:
            _reject("reloaded capture manifest differs from the in-memory CaptureSet")
        if tuple(bundle.gate_results) != tuple(request.capture_gate_results):
            _reject("reloaded capture gates differ from the in-memory gate evidence")
        if bundle.gate_provenance != request.capture_gate_provenance:
            _reject("reloaded capture gate provenance differs from the in-memory request")
        if sha256_file(manifest_path) != request.capture_manifest_sha256:
            _reject("capture manifest changed after InspectionRequest verification")
        if sha256_file(gates_path) != request.capture_gates_sha256:
            _reject("capture gate evidence changed after InspectionRequest verification")
        checksum_bytes = (set_root / "checksums.sha256").read_bytes()
        if hashlib.sha256(checksum_bytes).hexdigest() != request.capture_publication_root_sha256:
            _reject("capture publication root changed after InspectionRequest verification")
    except PublicationError:
        raise
    except Exception as error:
        _reject(f"capture publication revalidation failed: {error}")


def _require_no_stage_evidence(run: InspectionRun, *, status: InspectionStatus) -> None:
    fields = {
        "gate_results": run.gate_results,
        "crops": run.crops,
        "template_scores": run.template_scores,
        "template_evidence": run.template_evidence,
        "raw_model_scores": run.raw_model_scores,
        "model_evidence": run.model_evidence,
        "system_errors": run.system_errors,
    }
    populated = sorted(name for name, value in fields.items() if value)
    if populated:
        _reject(f"{status.value} must not carry downstream stage evidence: {populated}")


def _derive_invalid_capture_reason(runtime: LoadedRuntime, run: InspectionRun) -> str | None:
    """Independently derive the capture/contract failure used by INVALID_CAPTURE."""
    request = run.request
    capture = request.capture
    contract = runtime.contract
    if request.release_id != runtime.release_id:
        return None
    if capture.part.product != contract.product:
        return "capture_product_mismatch"
    if capture.part.hand not in contract.allowed_hands:
        return "capture_hand_not_enabled"
    if capture.topology_id != contract.topology.topology_id:
        return "capture_topology_id_mismatch"
    if capture.topology_sha256 != contract.topology.topology_sha256:
        return "capture_topology_digest_mismatch"
    try:
        capture.validate_required_views(contract.topology.required_views)
    except ValueError as error:
        return f"capture_view_identity_error:{error}"
    for view, image in capture.images.items():
        if (image.width, image.height) != (contract.roi.source_width, contract.roi.source_height):
            return f"capture_source_dimensions_mismatch:{view}"
        if (image.round_id, image.camera_slot_id, image.camera_serial) != contract.topology.binding_for_view(view):
            return f"capture_camera_identity_mismatch:{view}"
        path = request.capture_root / image.relative_path
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(request.capture_root)
        except (OSError, ValueError):
            return f"capture_source_path_invalid:{view}"
        try:
            if path.is_symlink() or not path.is_file() or sha256_file(path) != image.image_sha256:
                return f"capture_source_hash_mismatch:{view}"
        except OSError:
            return f"capture_source_hash_mismatch:{view}"
    try:
        expected_provenance = runtime.capture_gate_policy.provenance_for(
            capture.part.hand,
            policy_sha256=runtime.capture_gate_policy_file_sha256,
        )
    except Exception:
        return None
    if request.capture_gate_provenance != expected_provenance:
        return "capture_gate_policy_mismatch"
    return None


def _require_capture_matches_contract(runtime: LoadedRuntime, run: InspectionRun) -> None:
    request = run.request
    capture = request.capture
    contract = runtime.contract
    if request.release_id != runtime.release_id:
        _reject("non-terminal inspection requested a different release")
    if capture.part.product != contract.product:
        _reject("capture product differs from deployment contract")
    if capture.part.hand not in contract.allowed_hands:
        _reject("capture hand is not enabled by deployment contract")
    if (
        capture.topology_id != contract.topology.topology_id
        or capture.topology_sha256 != contract.topology.topology_sha256
    ):
        _reject("capture topology differs from deployment contract")
    try:
        capture.validate_required_views(contract.topology.required_views)
    except ValueError as error:
        _reject(f"capture required views are invalid: {error}")
    for view in contract.topology.required_views:
        image = capture.images[view]
        expected_binding = contract.topology.binding_for_view(view)
        if (image.round_id, image.camera_slot_id, image.camera_serial) != expected_binding:
            _reject(f"capture camera identity differs from contract for view {view!r}")
        if (image.width, image.height) != (contract.roi.source_width, contract.roi.source_height):
            _reject(f"capture dimensions differ from contract for view {view!r}")
    try:
        expected_provenance = runtime.capture_gate_policy.provenance_for(
            capture.part.hand,
            policy_sha256=runtime.capture_gate_policy_file_sha256,
        )
    except Exception as error:
        _reject(f"release capture-gate policy is unusable: {error}")
    if request.capture_gate_provenance != expected_provenance:
        _reject("capture gate provenance differs from verified release")


def _validated_gate_results(runtime: LoadedRuntime, run: InspectionRun) -> tuple[CaptureGateResult, ...]:
    expected_keys = {
        (gate, view)
        for gate in ("quality", "registration")
        for view in runtime.contract.topology.required_views
    }
    indexed: dict[tuple[str, str], CaptureGateResult] = {}
    for row in run.gate_results:
        if not isinstance(row, CaptureGateResult) or row.view_id is None:
            _reject("gate evidence contains an invalid or view-less row")
        key = (row.gate, row.view_id)
        if key in indexed:
            _reject(f"duplicate gate evidence row {key!r}")
        indexed[key] = row
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        _reject(f"gate evidence groups differ from contract; missing={missing}, extra={extra}")
    persisted: dict[tuple[str, str], CaptureGateResult] = {}
    for row in run.request.capture_gate_results:
        if not isinstance(row, CaptureGateResult) or row.view_id is None:
            _reject("persisted capture gate evidence contains an invalid row")
        key = (row.gate, row.view_id)
        if key in persisted:
            _reject(f"persisted capture gate evidence duplicates {key!r}")
        persisted[key] = row
    if persisted != indexed:
        _reject("run gate evidence differs from the verified capture publication")
    return tuple(
        indexed[(gate, view)]
        for gate in ("quality", "registration")
        for view in runtime.contract.topology.required_views
    )


def _validate_crops(
    runtime: LoadedRuntime,
    run: InspectionRun,
    *,
    require_complete: bool,
) -> dict[str, ModelInput]:
    if any(not isinstance(view, str) or not view for view in run.crops):
        _reject("crop evidence keys must be non-empty view strings")
    expected = set(runtime.contract.topology.required_views)
    actual = set(run.crops)
    if not actual <= expected:
        _reject(f"crop evidence contains unexpected views: {sorted(actual - expected)}")
    if require_complete and actual != expected:
        _reject(f"crop evidence is incomplete; missing={sorted(expected - actual)}")
    validated: dict[str, ModelInput] = {}
    for view, model_input in run.crops.items():
        if not isinstance(model_input, ModelInput):
            _reject(f"crop evidence for {view!r} is not a ModelInput")
        sample = model_input.sample
        source = run.request.capture.images[view]
        roi = runtime.contract.roi.hands[run.request.capture.part.hand].views[view]
        if (
            sample.part != run.request.capture.part
            or sample.capture_set_id != run.request.capture.capture_set_id
            or sample.view_id != view
            or sample.source_sha256 != source.image_sha256
            or sample.roi_config_id != runtime.contract.roi.roi_config_id
            or (sample.crop_width, sample.crop_height) != (roi.width, roi.height)
            or model_input.roi_digest != runtime.contract.roi.roi_sha256
        ):
            _reject(f"canonical crop identity differs from release/capture for view {view!r}")
        path = model_input.crop_path
        try:
            invalid_file = path.is_symlink() or not path.is_file() or sha256_file(path) != sample.crop_sha256
        except OSError:
            invalid_file = True
        if invalid_file:
            _reject(f"canonical crop bytes differ from evidence for view {view!r}")
        validated[view] = model_input
    return validated


def _validate_template_material(
    runtime: LoadedRuntime,
    run: InspectionRun,
    crops: Mapping[str, ModelInput],
    *,
    require_complete: bool,
) -> dict[str, TemplateEvidence]:
    required_views = runtime.contract.topology.required_views
    expected = set(required_views)
    score_by_view: dict[str, TemplateScore] = {}
    for score in run.template_scores:
        if not isinstance(score, TemplateScore):
            _reject("template score evidence contains an invalid record type")
        if score.view in score_by_view:
            _reject(f"duplicate template score for view {score.view!r}")
        if score.view not in expected or score.view not in crops:
            _reject(f"template score has no contract-bound crop for view {score.view!r}")
        crop = crops[score.view].sample
        binding = next(
            item
            for item in runtime.contract.template_bindings
            if item.hand is run.request.capture.part.hand and item.view_id == score.view
        )
        if (
            score.inspection_id != run.request.inspection_id
            or score.capture_set_id != run.request.capture.capture_set_id
            or score.part_instance_id != run.request.capture.part.part_instance_id
            or score.hand != run.request.capture.part.hand.value
            or score.model_digest != binding.asset.sha256
            or score.roi_config_id != runtime.contract.roi.roi_config_id
            or score.roi_digest != runtime.contract.roi.roi_sha256
            or score.source_sha256 != crop.source_sha256
            or score.crop_sha256 != crop.crop_sha256
        ):
            _reject(f"template score identity/hash differs from release/crop for view {score.view!r}")
        score_by_view[score.view] = score
    if require_complete and set(score_by_view) != expected:
        _reject(f"template scores are incomplete; missing={sorted(expected - set(score_by_view))}")

    context = EvidenceContext.from_contract(
        inspection_id=run.request.inspection_id,
        capture_set_id=run.request.capture.capture_set_id,
        part_instance_id=run.request.capture.part.part_instance_id,
        hand=run.request.capture.part.hand,
        contract=runtime.contract,
    )
    evidence_by_view: dict[str, TemplateEvidence] = {}
    for row in run.template_evidence:
        if not isinstance(row, TemplateEvidence):
            _reject("template evidence contains an invalid record type")
        if row.view_id in evidence_by_view:
            _reject(f"duplicate template evidence for view {row.view_id!r}")
        score = score_by_view.get(row.view_id)
        if score is None:
            _reject(f"template evidence has no retained raw score for view {row.view_id!r}")
        try:
            runtime.contract.validate_template_evidence(row)
        except ValueError as error:
            _reject(f"template evidence violates release contract: {error}")
        if (
            row.inspection_id != score.inspection_id
            or row.capture_set_id != score.capture_set_id
            or row.part_instance_id != score.part_instance_id
            or row.hand.value != score.hand
            or row.view_id != score.view
            or row.score != score.risk_score
            or row.template_sha256 != score.model_digest
            or row.roi_config_id != score.roi_config_id
            or row.source_sha256 != score.source_sha256
            or row.crop_sha256 != score.crop_sha256
        ):
            _reject(f"template raw/calibrated evidence disagree for view {row.view_id!r}")
        evidence_by_view[row.view_id] = row
    if require_complete:
        try:
            require_complete_templates(run.template_evidence, context)
        except ValueError as error:
            _reject(f"template evidence is incomplete or invalid: {error}")
    return evidence_by_view


def _validate_raw_visual(
    path: Path,
    digest: str,
    *,
    label: str,
    key: tuple[str, EvidenceBranch],
) -> None:
    candidate = path.expanduser()
    try:
        metadata = candidate.stat(follow_symlinks=False)
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _reject(f"anomaly {label} is unavailable for group {key!r}: {error}")
    if (
        not candidate.is_absolute()
        or candidate != resolved
        or candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        _reject(
            f"anomaly {label} must be a canonical private regular file for group {key!r}"
        )
    try:
        actual_sha256 = sha256_file(resolved)
    except OSError as error:
        _reject(f"anomaly {label} cannot be rehashed for group {key!r}: {error}")
    if actual_sha256 != digest:
        _reject(f"anomaly {label} hash differs for group {key!r}")


def _validate_model_material(
    runtime: LoadedRuntime,
    run: InspectionRun,
    crops: Mapping[str, ModelInput],
    *,
    require_complete: bool,
) -> dict[tuple[str, EvidenceBranch], ModelEvidence]:
    expected = {
        (view, branch)
        for view in runtime.contract.topology.required_views
        for branch in EvidenceBranch
    }
    raw_by_group: dict[tuple[str, EvidenceBranch], RawModelScore] = {}
    for raw in run.raw_model_scores:
        if not isinstance(raw, RawModelScore):
            _reject("raw model evidence contains an invalid record type")
        branch = EvidenceBranch(raw.branch)
        key = (raw.view, branch)
        if key in raw_by_group:
            _reject(f"duplicate raw model score for group {(raw.view, raw.branch)!r}")
        if key not in expected or raw.view not in crops:
            _reject(f"raw model score is outside required evidence groups: {(raw.view, raw.branch)!r}")
        crop = crops[raw.view].sample
        expected_digest = runtime.contract.yolo_model.sha256
        expected_family = "yolo"
        if branch is EvidenceBranch.ANOMALY:
            binding = next(
                item
                for item in runtime.contract.anomaly_bindings
                if item.hand is run.request.capture.part.hand and item.view_id == raw.view
            )
            expected_digest = binding.asset.sha256
            expected_family = runtime.contract.anomaly_family.value
        if (
            raw.inspection_id != run.request.inspection_id
            or raw.capture_set_id != run.request.capture.capture_set_id
            or raw.part_instance_id != run.request.capture.part.part_instance_id
            or raw.hand != run.request.capture.part.hand.value
            or raw.model_digest != expected_digest
            or raw.model_family != expected_family
            or raw.roi_config_id != runtime.contract.roi.roi_config_id
            or raw.roi_digest != runtime.contract.roi.roi_sha256
            or raw.source_sha256 != crop.source_sha256
            or raw.crop_sha256 != crop.crop_sha256
        ):
            _reject(f"raw model score identity/hash differs from release/crop for group {key!r}")
        if branch is EvidenceBranch.YOLO:
            if any(
                value is not None
                for value in (
                    raw.heatmap_path,
                    raw.heatmap_sha256,
                    raw.overlay_path,
                    raw.overlay_sha256,
                )
            ):
                _reject(f"YOLO raw score must not carry anomaly visuals for group {key!r}")
        else:
            if raw.heatmap_path is None or raw.heatmap_sha256 is None:
                _reject(f"anomaly raw score is missing its heatmap for group {key!r}")
            if raw.overlay_path is None or raw.overlay_sha256 is None:
                _reject(f"anomaly raw score is missing its overlay for group {key!r}")
            _validate_raw_visual(
                raw.heatmap_path,
                raw.heatmap_sha256,
                label="heatmap",
                key=key,
            )
            _validate_raw_visual(
                raw.overlay_path,
                raw.overlay_sha256,
                label="overlay",
                key=key,
            )
        raw_by_group[key] = raw

    evidence_by_group: dict[tuple[str, EvidenceBranch], ModelEvidence] = {}
    for row in run.model_evidence:
        if not isinstance(row, ModelEvidence):
            _reject("calibrated model evidence contains an invalid record type")
        key = (row.view_id, row.branch)
        if key in evidence_by_group:
            _reject(f"duplicate calibrated model evidence for group {key!r}")
        raw = raw_by_group.get(key)
        if raw is None:
            _reject(f"calibrated model evidence has no retained raw score for group {key!r}")
        try:
            runtime.contract.validate_model_evidence(row)
        except ValueError as error:
            _reject(f"model evidence violates release contract: {error}")
        if (
            row.inspection_id != raw.inspection_id
            or row.capture_set_id != raw.capture_set_id
            or row.part_instance_id != raw.part_instance_id
            or row.hand.value != raw.hand
            or row.view_id != raw.view
            or row.branch.value != raw.branch
            or row.score != raw.score
            or row.model_family != raw.model_family
            or row.model_sha256 != raw.model_digest
            or row.roi_config_id != raw.roi_config_id
            or row.source_sha256 != raw.source_sha256
            or row.crop_sha256 != raw.crop_sha256
        ):
            _reject(f"raw/calibrated model evidence disagree for group {key!r}")
        evidence_by_group[key] = row
    if require_complete:
        if set(raw_by_group) != expected:
            missing = sorted((view, branch.value) for view, branch in expected - set(raw_by_group))
            _reject(f"raw model evidence is incomplete; missing={missing}")
        context = EvidenceContext.from_contract(
            inspection_id=run.request.inspection_id,
            capture_set_id=run.request.capture.capture_set_id,
            part_instance_id=run.request.capture.part.part_instance_id,
            hand=run.request.capture.part.hand,
            contract=runtime.contract,
        )
        try:
            require_complete_models(run.model_evidence, context)
        except ValueError as error:
            _reject(f"calibrated model evidence is incomplete or invalid: {error}")
    return evidence_by_group


def _require_exact_decision(actual: InspectionDecision, expected: InspectionDecision) -> None:
    if actual != expected:
        _reject(
            "decision differs from strict evidence-derived decision; "
            f"expected={expected!r}, actual={actual!r}"
        )


def _validate_system_error_records(run: InspectionRun) -> None:
    if not run.system_errors or any(
        not isinstance(error, str) or not error.strip()
        for error in run.system_errors
    ):
        _reject("SYSTEM_ERROR requires non-empty retained system error evidence")
    joined_reasons = ";".join(run.decision.reason_codes)
    if any(error not in joined_reasons for error in run.system_errors):
        _reject("SYSTEM_ERROR reason codes do not account for retained system errors")


def validate_inspection_run_for_publication(runtime: LoadedRuntime, run: InspectionRun) -> None:
    """Independently prove release-bound evidence completeness before publication.

    This is deliberately separate from orchestration.  A caller that constructs
    an internally inconsistent ``InspectionRun`` cannot use the filesystem sink
    to publish a releasable or review decision.
    """
    if not isinstance(run, InspectionRun):
        _reject("sink input is not an InspectionRun")
    if not isinstance(run.request, InspectionRequest):
        _reject("inspection run request is not a verified InspectionRequest")
    if not isinstance(run.decision, InspectionDecision):
        _reject("inspection run decision is not an InspectionDecision")
    status = run.decision.inspection_status
    if not run.decision.reason_codes:
        _reject(f"{status.value} decision must retain at least one reason code")

    if status is InspectionStatus.INVALID_CAPTURE:
        _require_no_stage_evidence(run, status=status)
        expected_reason = _derive_invalid_capture_reason(runtime, run)
        if expected_reason is None:
            _reject("INVALID_CAPTURE is not supported by a capture/contract violation")
        if run.decision.reason_codes != (expected_reason,):
            _reject(
                "INVALID_CAPTURE reason does not match independently derived capture failure; "
                f"expected={expected_reason!r}"
            )
        return

    if status is InspectionStatus.SYSTEM_ERROR and not any(
        (
            run.gate_results,
            run.crops,
            run.template_scores,
            run.template_evidence,
            run.raw_model_scores,
            run.model_evidence,
        )
    ):
        _validate_system_error_records(run)
        if run.decision.evidence_status is not None:
            _reject("SYSTEM_ERROR evidence_status requires retained calibrated STRONG evidence")
        return

    _require_capture_matches_contract(runtime, run)
    ordered_gates = _validated_gate_results(runtime, run)

    if status is InspectionStatus.RETAKE:
        failures = tuple(row for row in ordered_gates if not row.passed)
        if not failures:
            _reject("RETAKE requires at least one failed capture gate")
        if any((run.crops, run.template_scores, run.template_evidence, run.raw_model_scores, run.model_evidence)):
            _reject("RETAKE must short-circuit before crop/template/model evidence")
        if run.system_errors:
            _reject("RETAKE cannot carry system error evidence")
        expected = InspectionDecision(
            inspection_id=run.request.inspection_id,
            evidence_status=None,
            inspection_status=InspectionStatus.RETAKE,
            review_status=None,
            released_status=None,
            reason_codes=tuple(
                f"retake:{row.gate}:{row.view_id or 'all'}:{row.reason}" for row in failures
            ),
        )
        _require_exact_decision(run.decision, expected)
        return

    if any(not row.passed for row in ordered_gates):
        _reject(f"{status.value} cannot cross a failed capture gate")

    is_system_error = status is InspectionStatus.SYSTEM_ERROR
    crops = _validate_crops(runtime, run, require_complete=not is_system_error)
    templates = _validate_template_material(
        runtime,
        run,
        crops,
        require_complete=not is_system_error,
    )
    if not is_system_error:
        if run.system_errors:
            _reject(f"{status.value} cannot carry system error evidence")
        context = EvidenceContext.from_contract(
            inspection_id=run.request.inspection_id,
            capture_set_id=run.request.capture.capture_set_id,
            part_instance_id=run.request.capture.part.part_instance_id,
            hand=run.request.capture.part.hand,
            contract=runtime.contract,
        )
        fusion = StrictFusionEngine(runtime.fusion_policy)
        template_decision = fusion.fuse_templates(context=context, evidence=run.template_evidence)
        if status is InspectionStatus.NG_TEMPLATE:
            if template_decision is None:
                _reject("NG_TEMPLATE has no template mismatch evidence")
            if run.raw_model_scores or run.model_evidence:
                _reject("NG_TEMPLATE must short-circuit both second-layer branches")
            _require_exact_decision(run.decision, template_decision)
            return
        if template_decision is not None:
            _reject(f"{status.value} contradicts template mismatch evidence")
        _validate_model_material(runtime, run, crops, require_complete=True)
        expected = fusion.fuse_models(context=context, evidence=run.model_evidence)
        _require_exact_decision(run.decision, expected)
        return

    _validate_system_error_records(run)
    expected_views = set(runtime.contract.topology.required_views)
    if set(templates) == expected_views and any(
        row.outcome is TemplateOutcome.NG_TEMPLATE for row in templates.values()
    ):
        _reject("SYSTEM_ERROR cannot suppress a complete NG_TEMPLATE decision")
    if run.raw_model_scores or run.model_evidence:
        if set(crops) != expected_views or set(templates) != expected_views:
            _reject("model-stage SYSTEM_ERROR requires complete PASS template evidence")
        if any(row.outcome is not TemplateOutcome.PASS for row in templates.values()):
            _reject("model-stage SYSTEM_ERROR cannot follow a template mismatch")
    model_rows = _validate_model_material(runtime, run, crops, require_complete=False)
    expected_evidence_status: InspectionStatus | None = None
    strong_branches = {
        row.branch for row in model_rows.values() if row.level is EvidenceLevel.STRONG
    }
    for branch in runtime.fusion_policy.strong_priority:
        if branch in strong_branches:
            expected_evidence_status = (
                InspectionStatus.NG_ANOMALY
                if branch is EvidenceBranch.ANOMALY
                else InspectionStatus.NG_YOLO
            )
            break
    if run.decision.evidence_status is not expected_evidence_status:
        _reject("SYSTEM_ERROR evidence_status is not supported by retained STRONG evidence")
    expected_groups = {
        (view, branch)
        for view in runtime.contract.topology.required_views
        for branch in EvidenceBranch
    }
    raw_groups = {(row.view, EvidenceBranch(row.branch)) for row in run.raw_model_scores}
    if raw_groups == expected_groups and set(model_rows) == expected_groups:
        _reject("SYSTEM_ERROR cannot suppress a complete valid second-layer decision")


class FilesystemInspectionSink:
    """Write all evidence through one no-replace atomic directory publisher."""

    def __init__(self, output_root: Path) -> None:
        self._output_root = Path(output_root)

    def publish(self, runtime: LoadedRuntime, run: InspectionRun) -> Path:
        """Publish request, exact capture manifest, crops, evidence, decision, and audit."""
        validate_inspection_run_for_publication(runtime, run)
        _validate_template_score_references(runtime, run)
        _revalidate_capture_publication(runtime, run)
        request = run.request
        request_payload = {
            "schema_version": 1,
            "inspection_id": request.inspection_id,
            "requested_release_id": request.release_id,
            "loaded_release_id": runtime.release_id,
            "release_root_sha256": runtime.release_root_sha256,
            "contract_sha256": runtime.contract.contract_sha256,
            "runtime_environment_receipt_sha256": (
                runtime.runtime_environment_receipt_sha256
            ),
            "runtime_environment_file_sha256": runtime.runtime_environment_file_sha256,
            "capture_set_id": request.capture.capture_set_id,
            "part_instance_id": request.capture.part.part_instance_id,
            "hand": request.capture.part.hand.value,
            "capture_manifest_sha256": request.capture_manifest_sha256,
            "capture_gates_sha256": request.capture_gates_sha256,
            "capture_publication_root_sha256": request.capture_publication_root_sha256,
        }
        request_sha256 = hashlib.sha256(canonical_json_bytes(request_payload)).hexdigest()
        anomaly_rows = _branch_rows(run, EvidenceBranch.ANOMALY)
        yolo_rows = _branch_rows(run, EvidenceBranch.YOLO)
        persisted_model_rows = anomaly_rows + yolo_rows
        persisted_by_group = {
            (row.get("view"), row.get("branch")): row
            for row in persisted_model_rows
            if row.get("level") is not None
        }
        template_rows = [_template_row(row) for row in run.template_evidence]
        gate_rows = [
            {
                "gate": row.gate,
                "view": row.view_id,
                "passed": row.passed,
                "reason": row.reason,
            }
            for row in run.gate_results
        ]
        template_score_rows = [_template_score_row(row) for row in run.template_scores]
        serialized_evidence = {
            "template_evidence.jsonl": _jsonl(template_rows),
            "template_scores.jsonl": _jsonl(template_score_rows),
            "anomaly_evidence.jsonl": _jsonl(anomaly_rows),
            "yolo_evidence.jsonl": _jsonl(yolo_rows),
            "gate_results.jsonl": _jsonl(gate_rows),
        }
        system_error_bytes = canonical_json_bytes({"errors": list(run.system_errors)})
        decision_payload = _decision(run)
        reason_triggers = [
            AuditTrigger(
                code=f"reason_{index:03d}",
                category=(
                    "system"
                    if run.decision.inspection_status.value == "SYSTEM_ERROR"
                    else "decision"
                ),
                message=reason,
            )
            for index, reason in enumerate(run.decision.reason_codes, start=1)
        ]
        evidence_triggers = [
            AuditTrigger(
                code=f"template_{row.view_id}_{row.outcome.value.lower()}",
                category="template",
                message=f"template {row.outcome.value} score={row.score:.17g}",
                view_id=row.view_id,
                branch="template",
                evidence_sha256=_row_digest(_template_row(row)),
            )
            for row in run.template_evidence
            if row.outcome.value != "PASS"
        ]
        evidence_triggers.extend(
            AuditTrigger(
                code=f"model_{row.branch.value}_{row.view_id}_{row.level.value.lower()}",
                category="model_evidence",
                message=f"{row.branch.value} {row.level.value} score={row.score:.17g}",
                view_id=row.view_id,
                branch=row.branch.value,
                evidence_sha256=_row_digest(
                    persisted_by_group[(row.view_id, row.branch.value)]
                ),
            )
            for row in run.model_evidence
            if row.level.value != "CLEAR"
        )
        gate_triggers = [
            AuditTrigger(
                code=f"gate_{row.gate}_{row.view_id}",
                category=row.gate,
                message=row.reason,
                view_id=row.view_id,
                branch=row.gate,
            )
            for row in run.gate_results
            if not row.passed
        ]
        triggers = tuple((*reason_triggers, *gate_triggers, *evidence_triggers))
        model_hashes: dict[str, str] = {"yolo/global": runtime.contract.yolo_model.sha256}
        model_hashes.update(
            {
                f"anomaly/{binding.hand.value}/{binding.view_id}": binding.asset.sha256
                for binding in runtime.contract.anomaly_bindings
            }
        )
        calibration_digests = {
            item.calibration_sha256
            for item in (*runtime.contract.template_thresholds, *runtime.contract.model_thresholds)
        }
        model_hashes.update(
            {
                f"template/{binding.hand.value}/{binding.view_id}": binding.asset.sha256
                for binding in runtime.contract.template_bindings
            }
        )
        config_hashes = {
            "deployment/contract": runtime.contract.contract_sha256,
            "deployment/topology": runtime.contract.topology.topology_sha256,
            "deployment/roi": runtime.contract.roi.roi_sha256,
            "fusion/policy": runtime.fusion_policy_file_sha256,
            "capture/gate_policy": runtime.capture_gate_policy_file_sha256,
            "capture/acquisition_config": (
                runtime.capture_gate_policy.acquisition_config.sha256
            ),
            "release/root": runtime.release_root_sha256,
            "runtime/environment_receipt": (
                runtime.runtime_environment_receipt_sha256
            ),
            "runtime/environment_file": runtime.runtime_environment_file_sha256,
        }
        if len(calibration_digests) == 1:
            config_hashes["calibration/artifact"] = next(iter(calibration_digests))
        audit = InspectionAudit(
            inspection_id=request.inspection_id,
            release_id=runtime.release_id,
            requested_release_id=request.release_id,
            contract_sha256=runtime.contract.contract_sha256,
            capture_set_id=request.capture.capture_set_id,
            part_instance_id=request.capture.part.part_instance_id,
            hand=request.capture.part.hand,
            topology_sha256=runtime.contract.topology.topology_sha256,
            roi_sha256=runtime.contract.roi.roi_sha256,
            runtime_environment_receipt_sha256=(
                runtime.runtime_environment_receipt_sha256
            ),
            runtime_environment_file_sha256=runtime.runtime_environment_file_sha256,
            request_sha256=request_sha256,
            capture_manifest_sha256=request.capture_manifest_sha256,
            capture_gates_sha256=request.capture_gates_sha256,
            capture_publication_root_sha256=request.capture_publication_root_sha256,
            decision=run.decision,
            triggers=triggers,
            source_sha256_by_view={
                view: image.image_sha256 for view, image in request.capture.images.items()
            },
            crop_sha256_by_view={
                view: model_input.sample.crop_sha256 for view, model_input in run.crops.items()
            },
            evidence_sha256_by_role={
                "capture/gates_persisted": request.capture_gates_sha256,
                "capture/gates_runtime": hashlib.sha256(
                    serialized_evidence["gate_results.jsonl"]
                ).hexdigest(),
                "template/raw": hashlib.sha256(
                    serialized_evidence["template_scores.jsonl"]
                ).hexdigest(),
                "template/calibrated": hashlib.sha256(
                    serialized_evidence["template_evidence.jsonl"]
                ).hexdigest(),
                "model/anomaly": hashlib.sha256(
                    serialized_evidence["anomaly_evidence.jsonl"]
                ).hexdigest(),
                "model/yolo": hashlib.sha256(
                    serialized_evidence["yolo_evidence.jsonl"]
                ).hexdigest(),
                "system/errors": hashlib.sha256(system_error_bytes).hexdigest(),
            },
            model_sha256_by_role=model_hashes,
            config_sha256_by_role=config_hashes,
            created_at=utc_now(),
        )

        required = {
            "request.json",
            "capture_manifest.json",
            "capture_gates.json",
            "template_evidence.jsonl",
            "template_scores.jsonl",
            "anomaly_evidence.jsonl",
            "yolo_evidence.jsonl",
            "gate_results.jsonl",
            "system_errors.json",
            "decision.json",
            "audit.json",
        }
        with AtomicDirectoryPublisher(self._output_root, request.inspection_id) as publisher:
            publisher.write_json("request.json", request_payload)
            publisher.copy_file(
                request.capture_manifest_path,
                "capture_manifest.json",
                expected_sha256=request.capture_manifest_sha256,
            )
            publisher.copy_file(
                request.capture_gates_path,
                "capture_gates.json",
                expected_sha256=request.capture_gates_sha256,
            )
            for view, model_input in sorted(run.crops.items()):
                relative = f"crops/{view}.png"
                publisher.copy_file(
                    model_input.crop_path,
                    relative,
                    expected_sha256=model_input.sample.crop_sha256,
                )
                required.add(relative)
            for relative, content in serialized_evidence.items():
                publisher.write_bytes(relative, content)
            publisher.write_bytes("system_errors.json", system_error_bytes)
            for raw in run.raw_model_scores:
                for kind, path, digest in (
                    ("heatmaps", raw.heatmap_path, raw.heatmap_sha256),
                    ("overlays", raw.overlay_path, raw.overlay_sha256),
                ):
                    if path is None:
                        continue
                    relative = f"{kind}/{raw.branch}/{raw.view}.png"
                    publisher.copy_file(
                        path,
                        relative,
                        expected_sha256=digest or "",
                    )
                    required.add(relative)
            publisher.write_json("decision.json", decision_payload)
            publisher.write_json("audit.json", audit.to_dict())
            return publisher.finalize(
                validator=lambda root: self._validate_staging(
                    root,
                    runtime,
                    run,
                    serialized_evidence,
                ),
                required_paths=frozenset(required),
            )

    @staticmethod
    def _validate_staging(
        root: Path,
        runtime: LoadedRuntime,
        run: InspectionRun,
        serialized_evidence: Mapping[str, bytes],
    ) -> None:
        """Recheck status channels and exact evidence row counts before rename."""
        validate_inspection_run_for_publication(runtime, run)
        _validate_template_score_references(runtime, run)
        _revalidate_capture_publication(runtime, run)
        decision = json.loads((root / "decision.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
        if decision["inspection_id"] != run.request.inspection_id:
            raise PublicationError("staged decision inspection identity mismatch")
        for field in ("evidence_status", "inspection_status", "review_status", "released_status"):
            if audit[field] != decision[field]:
                raise PublicationError(f"staged audit and decision disagree on {field}")
        for relative, expected in serialized_evidence.items():
            if (root / relative).read_bytes() != expected:
                raise PublicationError(f"staged evidence bytes differ from validated run: {relative}")
        system_errors = json.loads((root / "system_errors.json").read_text(encoding="utf-8"))
        if system_errors != {"errors": list(run.system_errors)}:
            raise PublicationError("staged system error evidence differs from validated run")
        expected_evidence_digests = {
            "capture/gates_persisted": run.request.capture_gates_sha256,
            "capture/gates_runtime": hashlib.sha256(
                serialized_evidence["gate_results.jsonl"]
            ).hexdigest(),
            "template/raw": hashlib.sha256(
                serialized_evidence["template_scores.jsonl"]
            ).hexdigest(),
            "template/calibrated": hashlib.sha256(
                serialized_evidence["template_evidence.jsonl"]
            ).hexdigest(),
            "model/anomaly": hashlib.sha256(
                serialized_evidence["anomaly_evidence.jsonl"]
            ).hexdigest(),
            "model/yolo": hashlib.sha256(
                serialized_evidence["yolo_evidence.jsonl"]
            ).hexdigest(),
            "system/errors": hashlib.sha256(
                (root / "system_errors.json").read_bytes()
            ).hexdigest(),
        }
        if audit.get("evidence_sha256_by_role") != expected_evidence_digests:
            raise PublicationError("staged audit evidence digests differ from evidence files")
        expected_counts = {
            "template_evidence.jsonl": len(run.template_evidence),
            "anomaly_evidence.jsonl": len(_branch_rows(run, EvidenceBranch.ANOMALY)),
            "yolo_evidence.jsonl": len(_branch_rows(run, EvidenceBranch.YOLO)),
        }
        for filename, expected in expected_counts.items():
            actual = len((root / filename).read_text(encoding="utf-8").splitlines())
            if actual != expected:
                raise PublicationError(f"staged {filename} row count mismatch: {actual} != {expected}")
