"""Complete machine-decision audit records with separated status channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from zs32_inspection.domain.decisions import InspectionDecision, InspectionStatus
from zs32_inspection.domain.identity import Hand, require_non_empty, require_sha256


@dataclass(frozen=True, slots=True)
class AuditTrigger:
    """One retained gate, evidence, or system-fault trigger."""

    code: str
    category: str
    message: str
    view_id: str | None = None
    branch: str | None = None
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        for field in ("code", "category"):
            object.__setattr__(self, field, require_non_empty(getattr(self, field), field))
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("audit trigger message must be non-empty")
        if self.evidence_sha256 is not None:
            object.__setattr__(
                self,
                "evidence_sha256",
                require_sha256(self.evidence_sha256, "evidence_sha256"),
            )


@dataclass(frozen=True, slots=True)
class InspectionAudit:
    """Serializable audit envelope written beside every inspection result."""

    inspection_id: str
    release_id: str
    requested_release_id: str
    contract_sha256: str
    capture_set_id: str
    part_instance_id: str
    hand: Hand
    topology_sha256: str
    roi_sha256: str
    runtime_environment_receipt_sha256: str
    runtime_environment_file_sha256: str
    request_sha256: str
    capture_manifest_sha256: str
    capture_gates_sha256: str
    capture_publication_root_sha256: str
    decision: InspectionDecision
    triggers: tuple[AuditTrigger, ...]
    source_sha256_by_view: Mapping[str, str]
    crop_sha256_by_view: Mapping[str, str]
    evidence_sha256_by_role: Mapping[str, str]
    model_sha256_by_role: Mapping[str, str]
    config_sha256_by_role: Mapping[str, str]
    created_at: str

    def __post_init__(self) -> None:
        for field in (
            "inspection_id",
            "release_id",
            "requested_release_id",
            "capture_set_id",
            "part_instance_id",
            "created_at",
        ):
            object.__setattr__(self, field, require_non_empty(getattr(self, field), field))
        object.__setattr__(self, "hand", Hand.parse(self.hand))
        for field in (
            "contract_sha256",
            "topology_sha256",
            "roi_sha256",
            "runtime_environment_receipt_sha256",
            "runtime_environment_file_sha256",
            "request_sha256",
            "capture_manifest_sha256",
            "capture_gates_sha256",
            "capture_publication_root_sha256",
        ):
            object.__setattr__(self, field, require_sha256(getattr(self, field), field))
        if self.decision.inspection_id != self.inspection_id:
            raise ValueError("audit decision inspection_id mismatch")
        object.__setattr__(self, "triggers", tuple(self.triggers))
        for mapping_name in (
            "source_sha256_by_view",
            "crop_sha256_by_view",
            "evidence_sha256_by_role",
            "model_sha256_by_role",
            "config_sha256_by_role",
        ):
            mapping = dict(getattr(self, mapping_name))
            if any(not key for key in mapping):
                raise ValueError(f"{mapping_name} contains an empty key")
            for key, digest in mapping.items():
                require_sha256(digest, f"{mapping_name}[{key!r}]")
            object.__setattr__(self, mapping_name, MappingProxyType(mapping))

    def to_dict(self) -> dict[str, Any]:
        """Return the immutable machine audit; human review never rewrites it."""
        decision = {
            "inspection_id": self.decision.inspection_id,
            "evidence_status": (
                None if self.decision.evidence_status is None else self.decision.evidence_status.value
            ),
            "inspection_status": self.decision.inspection_status.value,
            "review_status": self.decision.review_status,
            "released_status": (
                None if self.decision.released_status is None else self.decision.released_status.value
            ),
            "reason_codes": list(self.decision.reason_codes),
        }
        return {
            "schema_version": 3,
            "inspection_id": self.inspection_id,
            "release_id": self.release_id,
            "requested_release_id": self.requested_release_id,
            "contract_sha256": self.contract_sha256,
            "capture_set_id": self.capture_set_id,
            "part_instance_id": self.part_instance_id,
            "hand": self.hand.value,
            "topology_sha256": self.topology_sha256,
            "roi_sha256": self.roi_sha256,
            "runtime_environment_receipt_sha256": self.runtime_environment_receipt_sha256,
            "runtime_environment_file_sha256": self.runtime_environment_file_sha256,
            "request_sha256": self.request_sha256,
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "capture_gates_sha256": self.capture_gates_sha256,
            "capture_publication_root_sha256": self.capture_publication_root_sha256,
            "evidence_status": decision["evidence_status"],
            "inspection_status": decision["inspection_status"],
            "review_status": decision["review_status"],
            "released_status": decision["released_status"],
            "decision": decision,
            "triggers": [asdict(trigger) for trigger in self.triggers],
            "source_sha256_by_view": dict(sorted(self.source_sha256_by_view.items())),
            "crop_sha256_by_view": dict(sorted(self.crop_sha256_by_view.items())),
            "evidence_sha256_by_role": dict(sorted(self.evidence_sha256_by_role.items())),
            "model_sha256_by_role": dict(sorted(self.model_sha256_by_role.items())),
            "config_sha256_by_role": dict(sorted(self.config_sha256_by_role.items())),
            "created_at": self.created_at,
        }


def utc_now() -> str:
    """Return a timezone-aware audit timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def system_error_decision(
    inspection_id: str,
    *,
    reason_code: str,
    evidence_status: InspectionStatus | None = None,
) -> InspectionDecision:
    """Preserve an observed NG while refusing to release an incomplete run."""
    return InspectionDecision(
        inspection_id=inspection_id,
        evidence_status=evidence_status,
        inspection_status=InspectionStatus.SYSTEM_ERROR,
        review_status=None,
        released_status=None,
        reason_codes=(reason_code,),
    )
