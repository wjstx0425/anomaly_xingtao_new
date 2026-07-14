"""Versioned strict-fusion policy with deterministic content identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from zs32_inspection.domain.evidence import EvidenceBranch
from zs32_inspection.domain.errors import DeploymentContractError


@dataclass(frozen=True, slots=True)
class FusionPolicy:
    """Resolve simultaneous strong branches without weakening either result.

    The policy never votes across views.  ``strong_priority`` only selects the
    primary status when both branches are already STRONG; all triggers remain
    in the decision audit.
    """

    policy_id: str
    schema_version: int = 1
    strong_priority: tuple[EvidenceBranch, ...] = (
        EvidenceBranch.ANOMALY,
        EvidenceBranch.YOLO,
    )

    def __post_init__(self) -> None:
        policy_id = self.policy_id.strip() if isinstance(self.policy_id, str) else ""
        if not policy_id or any(character.isspace() for character in policy_id):
            raise DeploymentContractError("fusion policy_id must be a non-empty token")
        object.__setattr__(self, "policy_id", policy_id)
        if self.schema_version != 1:
            raise DeploymentContractError("unsupported fusion policy schema_version")
        try:
            priority = tuple(EvidenceBranch(item) for item in self.strong_priority)
        except ValueError as error:
            raise DeploymentContractError(f"invalid strong branch priority: {error}") from error
        if set(priority) != set(EvidenceBranch) or len(priority) != len(EvidenceBranch):
            raise DeploymentContractError("strong_priority must contain anomaly and yolo exactly once")
        object.__setattr__(self, "strong_priority", priority)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical policy payload."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "no_majority_vote": True,
            "all_required_clear_for_ok": True,
            "strong_priority": [branch.value for branch in self.strong_priority],
        }

    @property
    def sha256(self) -> str:
        """Return the deterministic policy digest embedded in a release."""
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


DEFAULT_POLICY = FusionPolicy(policy_id="zs32-strict-fusion-v1")


def parse_fusion_policy(payload: Mapping[str, object]) -> FusionPolicy:
    """Parse the strict policy schema without accepting hidden defaults."""
    expected = {
        "schema_version",
        "policy_id",
        "no_majority_vote",
        "all_required_clear_for_ok",
        "strong_priority",
    }
    if set(payload) != expected:
        raise DeploymentContractError(
            f"fusion policy keys invalid; missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )
    if payload["no_majority_vote"] is not True or payload["all_required_clear_for_ok"] is not True:
        raise DeploymentContractError("ZS32 fusion policy cannot weaken strict clear/strong semantics")
    schema_version = payload["schema_version"]
    policy_id = payload["policy_id"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise DeploymentContractError("fusion schema_version must be an integer")
    if not isinstance(policy_id, str):
        raise DeploymentContractError("fusion policy_id must be a string")
    priority = payload["strong_priority"]
    if isinstance(priority, (str, bytes)) or not isinstance(priority, Sequence):
        raise DeploymentContractError("fusion strong_priority must be an array")
    if any(not isinstance(item, str) for item in priority):
        raise DeploymentContractError("fusion strong_priority values must be strings")
    try:
        return FusionPolicy(
            schema_version=schema_version,
            policy_id=policy_id,
            strong_priority=tuple(EvidenceBranch(item) for item in priority),
        )
    except (TypeError, ValueError) as error:
        raise DeploymentContractError(f"invalid fusion policy: {error}") from error
