"""Load one exact, immutable bootstrap capture-gate publication."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from zs32_inspection.config.schemas import parse_topology
from zs32_inspection.domain.topology import CaptureTopology
from zs32_inspection.runtime.publisher import (
    VerifiedAtomicPublication,
    canonical_json_bytes,
    verify_atomic_publication,
)

from .contracts import CapturePlan
from .gate_policy import CaptureGatePolicy, load_capture_gate_policy_bytes
from .hikvision import HikvisionCaptureConfig
from .opencv_quality import QualityGateProfile
from .opencv_registration import RegistrationGateProfile


AssetReader = Callable[[str], bytes]


def _json_object(read_bytes: AssetReader, relative_path: str, label: str) -> dict[str, object]:
    try:
        payload = json.loads(read_bytes(relative_path))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be parsed: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object")
    return payload


def _png_dimensions(payload: bytes, label: str) -> tuple[int, int]:
    """Read the mandatory PNG signature and IHDR dimensions without image libraries."""
    if (
        len(payload) < 24
        or payload[:8] != b"\x89PNG\r\n\x1a\n"
        or payload[8:12] != b"\x00\x00\x00\r"
        or payload[12:16] != b"IHDR"
    ):
        raise ValueError(f"{label} is not a PNG with a canonical IHDR header")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has invalid PNG dimensions")
    return width, height


def verify_capture_gate_asset_semantics(
    topology: CaptureTopology,
    policy: CaptureGatePolicy,
    read_bytes: AssetReader,
) -> None:
    """Parse and cross-bind every gate asset before it can become trusted input."""
    plan = CapturePlan.from_topology(topology)
    HikvisionCaptureConfig.from_mapping(
        _json_object(
            read_bytes,
            policy.acquisition_config.relative_path,
            "capture acquisition config",
        )
    )
    for hand, hand_policy in policy.hands.items():
        quality = QualityGateProfile.from_mapping(
            _json_object(
                read_bytes,
                hand_policy.quality_profile.relative_path,
                f"{hand.value} quality profile",
            )
        )
        registration = RegistrationGateProfile.from_mapping(
            _json_object(
                read_bytes,
                hand_policy.registration_profile.relative_path,
                f"{hand.value} registration profile",
            )
        )
        quality.validate_plan(plan)
        registration.validate_plan(plan)
        if quality.hand != hand.value or registration.hand != hand.value:
            raise ValueError(f"gate profile hand differs from policy hand {hand.value!r}")
        for view_id in plan.required_views:
            reference = registration.views[view_id].reference
            policy_reference = hand_policy.registration_references[view_id]
            if (
                reference.asset_id,
                reference.relative_path,
                reference.sha256,
            ) != (
                policy_reference.artifact_id,
                policy_reference.relative_path,
                policy_reference.sha256,
            ):
                raise ValueError(
                    f"registration profile reference differs from policy for {hand.value}/{view_id}"
                )
            dimensions = _png_dimensions(
                read_bytes(reference.relative_path),
                f"registration reference {hand.value}/{view_id}",
            )
            if dimensions != (reference.width, reference.height):
                raise ValueError(
                    f"registration reference dimensions differ for {hand.value}/{view_id}: "
                    f"profile={(reference.width, reference.height)}, png={dimensions}"
                )


@dataclass(frozen=True, slots=True)
class VerifiedCaptureGatePublication:
    """Semantically verified policy, topology, assets, and publication root."""

    publication: VerifiedAtomicPublication
    topology: CaptureTopology
    policy: CaptureGatePolicy
    policy_sha256: str


def load_verified_capture_gate_publication(
    root: Path,
) -> VerifiedCaptureGatePublication:
    """Verify the exact bootstrap gate publication before trusting provenance."""
    policy_relative = "capture/gates/policy.json"
    publication = verify_atomic_publication(
        root,
        required_paths=frozenset({"topology.json", policy_relative}),
    )
    try:
        topology_payload = json.loads(publication.read_bytes("topology.json"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"gate publication topology.json cannot be parsed: {error}") from error
    if not isinstance(topology_payload, dict):
        raise ValueError("gate publication topology.json must contain an object")
    topology = parse_topology(topology_payload)
    if publication.read_bytes("topology.json") != canonical_json_bytes(
        topology.as_dict(include_sha256=False)
    ):
        raise ValueError("gate publication topology.json is not canonical")
    policy_bytes = publication.read_bytes(policy_relative)
    policy = load_capture_gate_policy_bytes(policy_bytes)
    if policy_bytes != canonical_json_bytes(policy.as_dict()):
        raise ValueError("gate publication capture policy is not canonical")
    policy.validate_topology(topology, allowed_hands=tuple(policy.hands))
    asset_paths = {policy.acquisition_config.relative_path}
    asset_paths.update(
        artifact.relative_path
        for hand_policy in policy.hands.values()
        for artifact in (
            hand_policy.quality_profile,
            hand_policy.registration_profile,
            *hand_policy.registration_references.values(),
        )
    )
    allowed = frozenset({"topology.json", policy_relative, *asset_paths})
    publication = verify_atomic_publication(
        publication.root,
        required_paths=allowed,
        allowed_paths=allowed,
        expected_publication_id=publication.publication_id,
    )
    artifacts = [policy.acquisition_config]
    for hand_policy in policy.hands.values():
        artifacts.extend((
            hand_policy.quality_profile,
            hand_policy.registration_profile,
            *hand_policy.registration_references.values(),
        ))
    for artifact in artifacts:
        if publication.checksums.get(artifact.relative_path) != artifact.sha256:
            raise ValueError(
                "gate publication asset differs from policy: "
                f"{artifact.relative_path}"
            )
        publication.read_bytes(artifact.relative_path)
    verify_capture_gate_asset_semantics(topology, policy, publication.read_bytes)
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    if publication.checksums.get(policy_relative) != policy_sha256:
        raise ValueError("gate publication policy digest changed during verification")
    return VerifiedCaptureGatePublication(
        publication=publication,
        topology=topology,
        policy=policy,
        policy_sha256=policy_sha256,
    )


__all__ = [
    "VerifiedCaptureGatePublication",
    "load_verified_capture_gate_publication",
    "verify_capture_gate_asset_semantics",
]
