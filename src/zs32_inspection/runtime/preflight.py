"""Atomic audit for failures that occur before a trusted runtime can exist."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from zs32_inspection.domain.decisions import InspectionStatus

from .audit import utc_now
from .publisher import AtomicDirectoryPublisher, canonical_json_bytes


def publish_preflight_failure(
    *,
    output_root: Path,
    inspection_id: str,
    status: InspectionStatus,
    reason_code: str,
    release_path: Path,
    capture_set_root: Path,
    runtime_environment_audit: Mapping[str, object] | None = None,
) -> Path:
    """Publish a non-releasable result when release/capture trust cannot be built."""
    if status not in {InspectionStatus.SYSTEM_ERROR, InspectionStatus.INVALID_CAPTURE}:
        raise ValueError("preflight publication supports only SYSTEM_ERROR or INVALID_CAPTURE")
    if not isinstance(reason_code, str) or not reason_code.strip():
        raise ValueError("preflight failure requires a reason code")
    request = {
        "schema_version": 1,
        "inspection_id": inspection_id,
        "requested_release_path": str(Path(release_path).expanduser()),
        "requested_capture_set_root": str(Path(capture_set_root).expanduser()),
        "trust_state": "preflight_failed",
    }
    decision = {
        "inspection_id": inspection_id,
        "evidence_status": None,
        "inspection_status": status.value,
        "review_status": None,
        "released_status": None,
        "reason_codes": [reason_code],
    }
    system_errors = {
        "errors": [reason_code] if status is InspectionStatus.SYSTEM_ERROR else []
    }
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    audit = {
        "schema_version": 3,
        "audit_kind": "preflight_failure",
        "inspection_id": inspection_id,
        "evidence_status": None,
        "inspection_status": status.value,
        "review_status": None,
        "released_status": None,
        "reason_codes": [reason_code],
        "triggers": [
            {
                "code": "preflight_001",
                "category": "system" if status is InspectionStatus.SYSTEM_ERROR else "capture",
                "message": reason_code,
            }
        ],
        "requested_release_path": request["requested_release_path"],
        "requested_capture_set_root": request["requested_capture_set_root"],
        "source_sha256_by_view": {},
        "crop_sha256_by_view": {},
        "evidence_sha256_by_role": {
            "capture/gates_runtime": empty_sha256,
            "template/raw": empty_sha256,
            "template/calibrated": empty_sha256,
            "model/anomaly": empty_sha256,
            "model/yolo": empty_sha256,
            "system/errors": hashlib.sha256(
                canonical_json_bytes(system_errors)
            ).hexdigest(),
        },
        "model_sha256_by_role": {},
        "config_sha256_by_role": {},
        "created_at": utc_now(),
    }
    if runtime_environment_audit is not None:
        audit["runtime_environment"] = dict(runtime_environment_audit)
    required = frozenset(
        {
            "request.json",
            "decision.json",
            "audit.json",
            "system_errors.json",
            "template_evidence.jsonl",
            "template_scores.jsonl",
            "anomaly_evidence.jsonl",
            "yolo_evidence.jsonl",
            "gate_results.jsonl",
        }
    )
    with AtomicDirectoryPublisher(Path(output_root), inspection_id) as publisher:
        publisher.write_json("request.json", request)
        publisher.write_json("decision.json", decision)
        publisher.write_json("audit.json", audit)
        publisher.write_json("system_errors.json", system_errors)
        for relative in (
            "template_evidence.jsonl",
            "template_scores.jsonl",
            "anomaly_evidence.jsonl",
            "yolo_evidence.jsonl",
            "gate_results.jsonl",
        ):
            publisher.write_bytes(relative, b"")
        return publisher.finalize(
            validator=lambda staging: _validate_preflight(
                staging,
                request=request,
                decision=decision,
                audit=audit,
                system_errors=system_errors,
            ),
            required_paths=required,
        )


def _validate_preflight(
    root: Path,
    *,
    request: dict[str, object],
    decision: dict[str, object],
    audit: dict[str, object],
    system_errors: dict[str, object],
) -> None:
    import json

    persisted_request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    persisted = json.loads((root / "decision.json").read_text(encoding="utf-8"))
    persisted_audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    persisted_errors = json.loads((root / "system_errors.json").read_text(encoding="utf-8"))
    if persisted_request != request:
        raise RuntimeError("preflight request changed during publication")
    if persisted != decision or persisted.get("released_status") is not None:
        raise RuntimeError("preflight decision changed or became releasable")
    if persisted_audit != audit or persisted_errors != system_errors:
        raise RuntimeError("preflight audit or system-error evidence changed")
    evidence_digests = {
        "capture/gates_runtime": hashlib.sha256(
            (root / "gate_results.jsonl").read_bytes()
        ).hexdigest(),
        "template/raw": hashlib.sha256(
            (root / "template_scores.jsonl").read_bytes()
        ).hexdigest(),
        "template/calibrated": hashlib.sha256(
            (root / "template_evidence.jsonl").read_bytes()
        ).hexdigest(),
        "model/anomaly": hashlib.sha256(
            (root / "anomaly_evidence.jsonl").read_bytes()
        ).hexdigest(),
        "model/yolo": hashlib.sha256(
            (root / "yolo_evidence.jsonl").read_bytes()
        ).hexdigest(),
        "system/errors": hashlib.sha256(
            (root / "system_errors.json").read_bytes()
        ).hexdigest(),
    }
    if persisted_audit.get("evidence_sha256_by_role") != evidence_digests:
        raise RuntimeError("preflight audit evidence digests differ from staged files")
    for field in (
        "evidence_status",
        "inspection_status",
        "review_status",
        "released_status",
        "reason_codes",
    ):
        if persisted_audit.get(field) != persisted.get(field):
            raise RuntimeError(f"preflight audit and decision disagree on {field}")
    for relative in (
        "template_evidence.jsonl",
        "template_scores.jsonl",
        "anomaly_evidence.jsonl",
        "yolo_evidence.jsonl",
        "gate_results.jsonl",
    ):
        if (root / relative).read_bytes() != b"":
            raise RuntimeError(f"preflight evidence file must remain empty: {relative}")


__all__ = ["publish_preflight_failure"]
