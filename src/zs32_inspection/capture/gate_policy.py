"""Immutable capture-gate policy and capture evidence provenance.

The policy is the sole bridge between deployment release, acquisition, and
inspection.  Human-readable gate reasons are deliberately excluded from this
identity; only content digests can authorize a quality/registration policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from zs32_inspection.domain.contracts import ArtifactRef
from zs32_inspection.domain.identity import Hand, PRODUCT, require_non_empty, require_sha256
from zs32_inspection.domain.topology import CaptureTopology


CAPTURE_GATE_EVIDENCE_SCHEMA_VERSION = 2


def _artifact(payload: object, label: str) -> ArtifactRef:
    if not isinstance(payload, Mapping) or set(payload) != {
        "artifact_id", "version", "sha256", "relative_path"
    }:
        raise ValueError(f"{label} must be a strict ArtifactRef object")
    if any(not isinstance(payload[key], str) for key in payload):
        raise TypeError(f"{label} ArtifactRef fields must be strings")
    return ArtifactRef(
        artifact_id=payload["artifact_id"],
        version=payload["version"],
        sha256=payload["sha256"],
        relative_path=payload["relative_path"],
    )


@dataclass(frozen=True, slots=True)
class HandCaptureGatePolicy:
    """Independent quality/profile/reference assets for one physical hand."""

    hand: Hand
    quality_profile: ArtifactRef
    registration_profile: ArtifactRef
    registration_references: Mapping[str, ArtifactRef]

    def __post_init__(self) -> None:
        object.__setattr__(self, "hand", Hand.parse(self.hand))
        if not isinstance(self.quality_profile, ArtifactRef) or not isinstance(
            self.registration_profile, ArtifactRef
        ):
            raise TypeError("gate profiles must be ArtifactRef values")
        prefix = f"capture/gates/{self.hand.value}"
        if self.quality_profile.relative_path != f"{prefix}/quality.json":
            raise ValueError("quality profile path must be canonical and hand-specific")
        if self.registration_profile.relative_path != f"{prefix}/registration.json":
            raise ValueError("registration profile path must be canonical and hand-specific")
        if not isinstance(self.registration_references, Mapping) or not self.registration_references:
            raise ValueError("registration_references must be a non-empty per-view mapping")
        references: dict[str, ArtifactRef] = {}
        paths: set[str] = set()
        for raw_view, reference in self.registration_references.items():
            view = require_non_empty(raw_view, "registration reference view")
            if not isinstance(reference, ArtifactRef):
                raise TypeError(f"registration reference for {view!r} must be an ArtifactRef")
            if reference.relative_path in paths:
                raise ValueError("registration reference paths must be unique per view")
            if reference.relative_path != f"{prefix}/references/{view}.png":
                raise ValueError(
                    "registration reference path must be canonical for its hand/view"
                )
            paths.add(reference.relative_path)
            references[view] = reference
        object.__setattr__(self, "registration_references", MappingProxyType(references))

    def as_dict(self) -> dict[str, object]:
        return {
            "quality_profile": self.quality_profile.as_dict(),
            "registration_profile": self.registration_profile.as_dict(),
            "registration_references": {
                view: artifact.as_dict()
                for view, artifact in sorted(self.registration_references.items())
            },
        }


@dataclass(frozen=True, slots=True)
class CaptureGatePolicy:
    """Canonical topology- and hand-bound gate asset manifest."""

    schema_version: int
    policy_id: str
    product: str
    topology_id: str
    topology_sha256: str
    acquisition_config: ArtifactRef
    hands: Mapping[Hand, HandCaptureGatePolicy]

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.product != PRODUCT:
            raise ValueError("capture gate policy must be schema_version=2 and product='ZS32'")
        object.__setattr__(self, "policy_id", require_non_empty(self.policy_id, "policy_id"))
        object.__setattr__(self, "topology_id", require_non_empty(self.topology_id, "topology_id"))
        object.__setattr__(
            self,
            "topology_sha256",
            require_sha256(self.topology_sha256, "topology_sha256"),
        )
        if not isinstance(self.acquisition_config, ArtifactRef):
            raise TypeError("capture acquisition_config must be an ArtifactRef")
        if self.acquisition_config.relative_path != "capture/acquisition/hikvision.json":
            raise ValueError(
                "capture acquisition_config path must be capture/acquisition/hikvision.json"
            )
        if not isinstance(self.hands, Mapping) or not self.hands:
            raise ValueError("capture gate policy must contain at least one explicit hand")
        hands: dict[Hand, HandCaptureGatePolicy] = {}
        for raw_hand, policy in self.hands.items():
            hand = Hand.parse(raw_hand)
            if not isinstance(policy, HandCaptureGatePolicy) or policy.hand is not hand:
                raise TypeError(f"capture gate policy entry conflicts for hand {hand.value!r}")
            hands[hand] = policy
        object.__setattr__(self, "hands", MappingProxyType(hands))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "product": self.product,
            "topology_id": self.topology_id,
            "topology_sha256": self.topology_sha256,
            "acquisition_config": self.acquisition_config.as_dict(),
            "hands": {
                hand.value: policy.as_dict()
                for hand, policy in sorted(self.hands.items(), key=lambda item: item[0].value)
            },
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CaptureGatePolicy":
        expected = {
            "schema_version", "policy_id", "product", "topology_id", "topology_sha256",
            "acquisition_config", "hands",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("capture gate policy fields differ from the strict schema")
        raw_hands = payload["hands"]
        if not isinstance(raw_hands, Mapping) or any(not isinstance(key, str) for key in raw_hands):
            raise TypeError("capture gate policy hands must be an object")
        hands: dict[Hand, HandCaptureGatePolicy] = {}
        for raw_hand, raw_policy in raw_hands.items():
            hand = Hand.parse(raw_hand)
            if not isinstance(raw_policy, Mapping) or set(raw_policy) != {
                "quality_profile", "registration_profile", "registration_references"
            }:
                raise ValueError(f"capture gate policy for {hand.value!r} is malformed")
            raw_references = raw_policy["registration_references"]
            if not isinstance(raw_references, Mapping) or any(
                not isinstance(view, str) for view in raw_references
            ):
                raise TypeError("registration_references must be an object keyed by view")
            hands[hand] = HandCaptureGatePolicy(
                hand=hand,
                quality_profile=_artifact(raw_policy["quality_profile"], "quality_profile"),
                registration_profile=_artifact(
                    raw_policy["registration_profile"], "registration_profile"
                ),
                registration_references={
                    view: _artifact(item, f"registration_references.{view}")
                    for view, item in raw_references.items()
                },
            )
        return cls(
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            policy_id=payload["policy_id"],  # type: ignore[arg-type]
            product=payload["product"],  # type: ignore[arg-type]
            topology_id=payload["topology_id"],  # type: ignore[arg-type]
            topology_sha256=payload["topology_sha256"],  # type: ignore[arg-type]
            acquisition_config=_artifact(
                payload["acquisition_config"],
                "acquisition_config",
            ),
            hands=hands,
        )

    def validate_topology(
        self,
        topology: CaptureTopology,
        *,
        allowed_hands: tuple[Hand, ...],
    ) -> None:
        if (
            self.product != topology.product
            or self.topology_id != topology.topology_id
            or self.topology_sha256 != topology.topology_sha256
        ):
            raise ValueError("capture gate policy topology identity differs from deployment")
        if set(self.hands) != set(allowed_hands):
            raise ValueError("capture gate policy hand set differs from enabled deployment hands")
        expected_views = set(topology.required_views)
        for hand, policy in self.hands.items():
            if set(policy.registration_references) != expected_views:
                raise ValueError(
                    f"registration reference view set differs for {hand.value}; "
                    f"missing={sorted(expected_views - set(policy.registration_references))}, "
                    f"extra={sorted(set(policy.registration_references) - expected_views)}"
                )

    def verify_assets(self, root: Path) -> None:
        """Rehash every profile/reference directly below one trusted asset root."""
        artifacts = [self.acquisition_config]
        for hand_policy in self.hands.values():
            artifacts.extend((
                hand_policy.quality_profile,
                hand_policy.registration_profile,
                *hand_policy.registration_references.values(),
            ))
        for artifact in artifacts:
            path = root.joinpath(*Path(artifact.relative_path).parts)
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"capture gate asset is missing: {artifact.relative_path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
                raise ValueError(f"capture gate asset hash mismatch: {artifact.relative_path}")

    def provenance_for(self, hand: Hand, *, policy_sha256: str) -> "CaptureGateProvenance":
        canonical_hand = Hand.parse(hand)
        try:
            hand_policy = self.hands[canonical_hand]
        except KeyError as error:
            raise ValueError(f"capture gate policy does not enable hand {canonical_hand.value!r}") from error
        return CaptureGateProvenance(
            policy_id=self.policy_id,
            policy_sha256=policy_sha256,
            hand=canonical_hand,
            topology_sha256=self.topology_sha256,
            acquisition_config_sha256=self.acquisition_config.sha256,
            quality_profile_sha256=hand_policy.quality_profile.sha256,
            registration_profile_sha256=hand_policy.registration_profile.sha256,
            registration_reference_sha256_by_view={
                view: artifact.sha256
                for view, artifact in hand_policy.registration_references.items()
            },
        )


@dataclass(frozen=True, slots=True)
class CaptureGateProvenance:
    """Structured immutable gate identity persisted beside every capture."""

    policy_id: str
    policy_sha256: str
    hand: Hand
    topology_sha256: str
    acquisition_config_sha256: str
    quality_profile_sha256: str
    registration_profile_sha256: str
    registration_reference_sha256_by_view: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", require_non_empty(self.policy_id, "policy_id"))
        object.__setattr__(self, "hand", Hand.parse(self.hand))
        for field in (
            "policy_sha256", "topology_sha256", "acquisition_config_sha256",
            "quality_profile_sha256",
            "registration_profile_sha256",
        ):
            object.__setattr__(self, field, require_sha256(getattr(self, field), field))
        references: dict[str, str] = {}
        if not isinstance(self.registration_reference_sha256_by_view, Mapping):
            raise TypeError("registration reference provenance must be a mapping")
        for raw_view, raw_digest in self.registration_reference_sha256_by_view.items():
            view = require_non_empty(raw_view, "registration reference view")
            references[view] = require_sha256(raw_digest, f"registration reference {view}")
        if not references:
            raise ValueError("registration reference provenance must not be empty")
        object.__setattr__(
            self,
            "registration_reference_sha256_by_view",
            MappingProxyType(references),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "hand": self.hand.value,
            "topology_sha256": self.topology_sha256,
            "acquisition_config_sha256": self.acquisition_config_sha256,
            "quality_profile_sha256": self.quality_profile_sha256,
            "registration_profile_sha256": self.registration_profile_sha256,
            "registration_reference_sha256_by_view": dict(
                sorted(self.registration_reference_sha256_by_view.items())
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CaptureGateProvenance":
        expected = {
            "policy_id", "policy_sha256", "hand", "topology_sha256",
            "acquisition_config_sha256", "quality_profile_sha256",
            "registration_profile_sha256",
            "registration_reference_sha256_by_view",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("capture gate provenance fields differ from the strict schema")
        references = payload["registration_reference_sha256_by_view"]
        if not isinstance(references, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in references.items()
        ):
            raise TypeError("registration reference provenance must be a string mapping")
        return cls(
            policy_id=payload["policy_id"],  # type: ignore[arg-type]
            policy_sha256=payload["policy_sha256"],  # type: ignore[arg-type]
            hand=Hand.parse(payload["hand"]),  # type: ignore[arg-type]
            topology_sha256=payload["topology_sha256"],  # type: ignore[arg-type]
            acquisition_config_sha256=payload["acquisition_config_sha256"],  # type: ignore[arg-type]
            quality_profile_sha256=payload["quality_profile_sha256"],  # type: ignore[arg-type]
            registration_profile_sha256=payload["registration_profile_sha256"],  # type: ignore[arg-type]
            registration_reference_sha256_by_view=references,  # type: ignore[arg-type]
        )


def load_capture_gate_policy_bytes(content: bytes) -> CaptureGatePolicy:
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"capture gate policy cannot be parsed: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("capture gate policy root must be an object")
    return CaptureGatePolicy.from_mapping(payload)
