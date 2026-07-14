"""Strict held-out acceptance policy and immutable PASS decision."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from zs32_inspection.domain.identity import PRODUCT
from zs32_inspection.models.base import SHA256_PATTERN
from zs32_inspection.runtime.publisher import canonical_json_bytes

from .reports import HeldOutPartMetrics


_POLICY_KEYS = frozenset(
    {"schema", "schema_version", "product", "policy_id", "policy_version", "limits"}
)
_LIMIT_KEYS = frozenset(
    {
        "max_defect_escape_rate",
        "max_normal_reject_rate",
        "max_review_rate",
        "min_normal_heldout_parts",
        "min_defect_heldout_parts",
    }
)
_DECISION_KEYS = frozenset(
    {"schema", "schema_version", "status", "policy", "metrics", "limits"}
)
_DECISION_POLICY_KEYS = frozenset(
    {"policy_id", "policy_version", "policy_sha256"}
)
_DECISION_METRIC_KEYS = frozenset(
    {
        "metrics_sha256",
        "normal_heldout_parts",
        "defect_heldout_parts",
        "defect_escape_rate",
        "normal_reject_rate",
        "review_rate",
    }
)


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"held-out acceptance {label} must be a non-empty trimmed string")
    if any(character.isspace() for character in value) or any(
        token in value for token in ("/", "\\", "\x00")
    ):
        raise ValueError(f"held-out acceptance {label} must be a safe identity")
    return value


def _rate(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"held-out acceptance {label} must be a finite rate in [0, 1]")
    return float(value)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"held-out acceptance {label} must be an integer >= 1")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"held-out acceptance {label} must be a lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class HeldOutAcceptanceLimits:
    """Production acceptance limits applied to one independent held-out split."""

    max_defect_escape_rate: float
    max_normal_reject_rate: float
    max_review_rate: float
    min_normal_heldout_parts: int
    min_defect_heldout_parts: int

    def __post_init__(self) -> None:
        for field in (
            "max_defect_escape_rate",
            "max_normal_reject_rate",
            "max_review_rate",
        ):
            object.__setattr__(self, field, _rate(getattr(self, field), field))
        for field in ("min_normal_heldout_parts", "min_defect_heldout_parts"):
            object.__setattr__(
                self,
                field,
                _positive_integer(getattr(self, field), field),
            )

    @classmethod
    def from_mapping(cls, payload: object) -> HeldOutAcceptanceLimits:
        if not isinstance(payload, Mapping) or set(payload) != _LIMIT_KEYS:
            raise ValueError("held-out acceptance limits have missing or unknown fields")
        return cls(
            max_defect_escape_rate=_rate(
                payload.get("max_defect_escape_rate"), "max_defect_escape_rate"
            ),
            max_normal_reject_rate=_rate(
                payload.get("max_normal_reject_rate"), "max_normal_reject_rate"
            ),
            max_review_rate=_rate(payload.get("max_review_rate"), "max_review_rate"),
            min_normal_heldout_parts=_positive_integer(
                payload.get("min_normal_heldout_parts"), "min_normal_heldout_parts"
            ),
            min_defect_heldout_parts=_positive_integer(
                payload.get("min_defect_heldout_parts"), "min_defect_heldout_parts"
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "max_defect_escape_rate": self.max_defect_escape_rate,
            "max_normal_reject_rate": self.max_normal_reject_rate,
            "max_review_rate": self.max_review_rate,
            "min_normal_heldout_parts": self.min_normal_heldout_parts,
            "min_defect_heldout_parts": self.min_defect_heldout_parts,
        }


@dataclass(frozen=True, slots=True)
class HeldOutAcceptancePolicy:
    """Versioned policy whose exact canonical bytes are its content identity."""

    policy_id: str
    policy_version: str
    limits: HeldOutAcceptanceLimits

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identity(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "policy_version",
            _identity(self.policy_version, "policy_version"),
        )
        if not isinstance(self.limits, HeldOutAcceptanceLimits):
            raise TypeError("held-out acceptance policy requires validated limits")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> HeldOutAcceptancePolicy:
        if not isinstance(payload, Mapping) or set(payload) != _POLICY_KEYS:
            raise ValueError("held-out acceptance policy has missing or unknown fields")
        if (
            payload.get("schema") != "zs32.heldout_acceptance_policy"
            or payload.get("schema_version") != 1
            or payload.get("product") != PRODUCT
        ):
            raise ValueError("held-out acceptance policy schema/product identity is invalid")
        return cls(
            policy_id=_identity(payload.get("policy_id"), "policy_id"),
            policy_version=_identity(payload.get("policy_version"), "policy_version"),
            limits=HeldOutAcceptanceLimits.from_mapping(payload.get("limits")),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "zs32.heldout_acceptance_policy",
            "schema_version": 1,
            "product": PRODUCT,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "limits": self.limits.as_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class HeldOutAcceptanceDecision:
    """One deterministic PASS decision bound to exact policy and metrics bytes."""

    policy_id: str
    policy_version: str
    policy_sha256: str
    metrics_sha256: str
    normal_heldout_parts: int
    defect_heldout_parts: int
    defect_escape_rate: float
    normal_reject_rate: float
    review_rate: float
    limits: HeldOutAcceptanceLimits

    def __post_init__(self) -> None:
        for field in ("policy_id", "policy_version"):
            object.__setattr__(
                self,
                field,
                _identity(getattr(self, field), f"decision.{field}"),
            )
        for field in ("policy_sha256", "metrics_sha256"):
            object.__setattr__(
                self,
                field,
                _sha256(getattr(self, field), f"decision.{field}"),
            )
        for field in ("normal_heldout_parts", "defect_heldout_parts"):
            object.__setattr__(
                self,
                field,
                _positive_integer(getattr(self, field), f"decision.{field}"),
            )
        for field in ("defect_escape_rate", "normal_reject_rate", "review_rate"):
            object.__setattr__(
                self,
                field,
                _rate(getattr(self, field), f"decision.{field}"),
            )
        if not isinstance(self.limits, HeldOutAcceptanceLimits):
            raise TypeError("held-out acceptance decision requires validated limits")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> HeldOutAcceptanceDecision:
        if not isinstance(payload, Mapping) or set(payload) != _DECISION_KEYS:
            raise ValueError("held-out acceptance decision has missing or unknown fields")
        if (
            payload.get("schema") != "zs32.heldout_acceptance_decision"
            or payload.get("schema_version") != 1
            or payload.get("status") != "PASS"
        ):
            raise ValueError("held-out acceptance decision schema/status is invalid")
        policy = payload.get("policy")
        metrics = payload.get("metrics")
        if not isinstance(policy, Mapping) or set(policy) != _DECISION_POLICY_KEYS:
            raise ValueError("held-out acceptance decision policy binding is malformed")
        if not isinstance(metrics, Mapping) or set(metrics) != _DECISION_METRIC_KEYS:
            raise ValueError("held-out acceptance decision metrics binding is malformed")
        return cls(
            policy_id=_identity(policy.get("policy_id"), "decision.policy_id"),
            policy_version=_identity(
                policy.get("policy_version"), "decision.policy_version"
            ),
            policy_sha256=_sha256(
                policy.get("policy_sha256"), "decision.policy_sha256"
            ),
            metrics_sha256=_sha256(
                metrics.get("metrics_sha256"), "decision.metrics_sha256"
            ),
            normal_heldout_parts=_positive_integer(
                metrics.get("normal_heldout_parts"), "decision.normal_heldout_parts"
            ),
            defect_heldout_parts=_positive_integer(
                metrics.get("defect_heldout_parts"), "decision.defect_heldout_parts"
            ),
            defect_escape_rate=_rate(
                metrics.get("defect_escape_rate"), "decision.defect_escape_rate"
            ),
            normal_reject_rate=_rate(
                metrics.get("normal_reject_rate"), "decision.normal_reject_rate"
            ),
            review_rate=_rate(metrics.get("review_rate"), "decision.review_rate"),
            limits=HeldOutAcceptanceLimits.from_mapping(payload.get("limits")),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "zs32.heldout_acceptance_decision",
            "schema_version": 1,
            "status": "PASS",
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "policy_sha256": self.policy_sha256,
            },
            "metrics": {
                "metrics_sha256": self.metrics_sha256,
                "normal_heldout_parts": self.normal_heldout_parts,
                "defect_heldout_parts": self.defect_heldout_parts,
                "defect_escape_rate": self.defect_escape_rate,
                "normal_reject_rate": self.normal_reject_rate,
                "review_rate": self.review_rate,
            },
            "limits": self.limits.as_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def load_heldout_acceptance_policy_bytes(content: bytes) -> HeldOutAcceptancePolicy:
    """Load only exact canonical policy JSON; the bytes are the policy identity."""
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"held-out acceptance policy cannot be parsed: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("held-out acceptance policy must be a string-keyed JSON object")
    policy = HeldOutAcceptancePolicy.from_mapping(payload)
    if content != policy.canonical_bytes():
        raise ValueError("held-out acceptance policy must use exact canonical JSON bytes")
    return policy


def load_heldout_acceptance_decision_bytes(content: bytes) -> HeldOutAcceptanceDecision:
    """Load a canonical PASS decision for independent downstream verification."""
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"held-out acceptance decision cannot be parsed: {error}") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("held-out acceptance decision must be a string-keyed JSON object")
    decision = HeldOutAcceptanceDecision.from_mapping(payload)
    if content != decision.canonical_bytes():
        raise ValueError("held-out acceptance decision must use exact canonical JSON bytes")
    return decision


def evaluate_heldout_acceptance(
    policy: HeldOutAcceptancePolicy,
    metrics: HeldOutPartMetrics,
    *,
    metrics_sha256: str,
) -> HeldOutAcceptanceDecision:
    """Fail closed unless every sample-size and rate limit passes."""
    if not isinstance(policy, HeldOutAcceptancePolicy):
        raise TypeError("held-out acceptance requires a validated policy")
    if not isinstance(metrics, HeldOutPartMetrics):
        raise TypeError("held-out acceptance requires validated part metrics")
    metrics_digest = _sha256(metrics_sha256, "metrics_sha256")
    limits = policy.limits
    violations: list[str] = []
    if metrics.evaluation_complete is not True:
        violations.append("held-out evaluation is incomplete")
    if metrics.normal_part_count < limits.min_normal_heldout_parts:
        violations.append(
            f"normal held-out parts {metrics.normal_part_count} < {limits.min_normal_heldout_parts}"
        )
    if metrics.defect_part_count < limits.min_defect_heldout_parts:
        violations.append(
            f"defect held-out parts {metrics.defect_part_count} < {limits.min_defect_heldout_parts}"
        )
    observed_rates = (
        ("defect escape", metrics.defect_escape_rate, limits.max_defect_escape_rate),
        ("normal reject", metrics.normal_reject_rate, limits.max_normal_reject_rate),
        ("review", metrics.review_rate, limits.max_review_rate),
    )
    for label, observed, maximum in observed_rates:
        if observed is None:
            violations.append(f"{label} rate is unavailable")
        elif observed > maximum:
            violations.append(f"{label} rate {observed} > {maximum}")
    if violations:
        raise ValueError("held-out acceptance failed: " + "; ".join(violations))
    assert metrics.defect_escape_rate is not None
    assert metrics.normal_reject_rate is not None
    assert metrics.review_rate is not None
    return HeldOutAcceptanceDecision(
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        policy_sha256=policy.sha256,
        metrics_sha256=metrics_digest,
        normal_heldout_parts=metrics.normal_part_count,
        defect_heldout_parts=metrics.defect_part_count,
        defect_escape_rate=float(metrics.defect_escape_rate),
        normal_reject_rate=float(metrics.normal_reject_rate),
        review_rate=float(metrics.review_rate),
        limits=limits,
    )
