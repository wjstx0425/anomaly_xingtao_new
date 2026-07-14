"""Strict, canonical human approval receipt for one ZS32 release."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from zs32_inspection.domain.identity import PRODUCT
from zs32_inspection.models.base import SHA256_PATTERN

from .publisher import canonical_json_bytes


_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "product",
        "release_id",
        "promotion_id",
        "approver",
        "approved_at",
        "candidate_id",
        "candidate_digest",
        "validation_publication_id",
        "validation_publication_root_sha256",
        "validation_record_sha256",
        "contract_sha256",
        "calibration_artifact_sha256",
        "dataset_manifest_sha256",
        "heldout_acceptance_policy_sha256",
        "heldout_acceptance_decision_sha256",
        "evidence",
    }
)
_EVIDENCE_ROLES = frozenset({"golden_parity", "fat", "sat"})
_EVIDENCE_KEYS = frozenset({"status", "evidence_sha256", "reason"})
_EVIDENCE_STATUSES = frozenset({"PASS", "NOT_APPLICABLE"})


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"promotion receipt {label} must be a non-empty trimmed string")
    if any(character.isspace() for character in value) or any(
        token in value for token in ("/", "\\", "\x00")
    ):
        raise ValueError(f"promotion receipt {label} must be a safe identity")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"promotion receipt {label} must be a non-empty trimmed string")
    if any(token in value for token in ("\x00", "\n", "\r")):
        raise ValueError(f"promotion receipt {label} must be one line")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"promotion receipt {label} must be a lowercase SHA256")
    return value


def _timestamp(value: object) -> str:
    text = _text(value, "approved_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("promotion receipt approved_at must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("promotion receipt approved_at must include an explicit timezone")
    return text


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    """One explicit approval prerequisite and its immutable evidence identity."""

    status: str
    evidence_sha256: str | None
    reason: str | None

    @classmethod
    def from_mapping(cls, payload: object, *, role: str) -> PromotionEvidence:
        if not isinstance(payload, Mapping) or set(payload) != _EVIDENCE_KEYS:
            raise ValueError(f"promotion receipt evidence {role} has missing or unknown fields")
        status = payload.get("status")
        digest = payload.get("evidence_sha256")
        reason = payload.get("reason")
        if status not in _EVIDENCE_STATUSES:
            raise ValueError(
                f"promotion receipt evidence {role} status must be PASS or NOT_APPLICABLE"
            )
        if status == "PASS":
            if reason is not None:
                raise ValueError(f"PASS promotion evidence {role} must not include a reason")
            return cls(status=status, evidence_sha256=_sha256(digest, role), reason=None)
        if digest is not None:
            raise ValueError(
                f"NOT_APPLICABLE promotion evidence {role} must not include a digest"
            )
        return cls(
            status=status,
            evidence_sha256=None,
            reason=_text(reason, f"{role}.reason"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "evidence_sha256": self.evidence_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PromotionReceipt:
    """Human approval bound to exactly one candidate, validation, and release."""

    release_id: str
    promotion_id: str
    approver: str
    approved_at: str
    candidate_id: str
    candidate_digest: str
    validation_publication_id: str
    validation_publication_root_sha256: str
    validation_record_sha256: str
    contract_sha256: str
    calibration_artifact_sha256: str
    dataset_manifest_sha256: str
    heldout_acceptance_policy_sha256: str
    heldout_acceptance_decision_sha256: str
    evidence: Mapping[str, PromotionEvidence]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PromotionReceipt:
        if set(payload) != _RECEIPT_KEYS:
            raise ValueError("promotion receipt has missing or unknown fields")
        if (
            payload.get("schema") != "zs32.promotion_receipt"
            or payload.get("schema_version") != 2
            or payload.get("product") != PRODUCT
        ):
            raise ValueError("promotion receipt schema/product identity is invalid")
        evidence_payload = payload.get("evidence")
        if (
            not isinstance(evidence_payload, Mapping)
            or set(evidence_payload) != _EVIDENCE_ROLES
        ):
            raise ValueError(
                "promotion receipt evidence must contain exactly golden_parity/fat/sat"
            )
        evidence = {
            role: PromotionEvidence.from_mapping(evidence_payload[role], role=role)
            for role in sorted(_EVIDENCE_ROLES)
        }
        if evidence["golden_parity"].status != "PASS":
            raise ValueError("promotion receipt golden_parity evidence must be PASS")
        return cls(
            release_id=_identity(payload.get("release_id"), "release_id"),
            promotion_id=_identity(payload.get("promotion_id"), "promotion_id"),
            approver=_text(payload.get("approver"), "approver"),
            approved_at=_timestamp(payload.get("approved_at")),
            candidate_id=_identity(payload.get("candidate_id"), "candidate_id"),
            candidate_digest=_sha256(payload.get("candidate_digest"), "candidate_digest"),
            validation_publication_id=_identity(
                payload.get("validation_publication_id"),
                "validation_publication_id",
            ),
            validation_publication_root_sha256=_sha256(
                payload.get("validation_publication_root_sha256"),
                "validation_publication_root_sha256",
            ),
            validation_record_sha256=_sha256(
                payload.get("validation_record_sha256"),
                "validation_record_sha256",
            ),
            contract_sha256=_sha256(payload.get("contract_sha256"), "contract_sha256"),
            calibration_artifact_sha256=_sha256(
                payload.get("calibration_artifact_sha256"),
                "calibration_artifact_sha256",
            ),
            dataset_manifest_sha256=_sha256(
                payload.get("dataset_manifest_sha256"),
                "dataset_manifest_sha256",
            ),
            heldout_acceptance_policy_sha256=_sha256(
                payload.get("heldout_acceptance_policy_sha256"),
                "heldout_acceptance_policy_sha256",
            ),
            heldout_acceptance_decision_sha256=_sha256(
                payload.get("heldout_acceptance_decision_sha256"),
                "heldout_acceptance_decision_sha256",
            ),
            evidence=MappingProxyType(evidence),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "zs32.promotion_receipt",
            "schema_version": 2,
            "product": PRODUCT,
            "release_id": self.release_id,
            "promotion_id": self.promotion_id,
            "approver": self.approver,
            "approved_at": self.approved_at,
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "validation_publication_id": self.validation_publication_id,
            "validation_publication_root_sha256": self.validation_publication_root_sha256,
            "validation_record_sha256": self.validation_record_sha256,
            "contract_sha256": self.contract_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "heldout_acceptance_policy_sha256": self.heldout_acceptance_policy_sha256,
            "heldout_acceptance_decision_sha256": self.heldout_acceptance_decision_sha256,
            "evidence": {
                role: self.evidence[role].as_dict()
                for role in sorted(_EVIDENCE_ROLES)
            },
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def load_promotion_receipt(path: Path) -> PromotionReceipt:
    """Read a canonical, private regular receipt without accepting symlink aliases."""
    path = Path(path).expanduser()
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"promotion receipt cannot be inspected: {error}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o222
    ):
        raise ValueError(
            "promotion receipt must be a private read-only regular non-symlink file"
        )
    try:
        content = path.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"promotion receipt cannot be read: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("promotion receipt must be a string-keyed JSON object")
    receipt = PromotionReceipt.from_mapping(payload)
    if content != receipt.canonical_bytes():
        raise ValueError("promotion receipt must use exact canonical JSON bytes")
    return receipt
