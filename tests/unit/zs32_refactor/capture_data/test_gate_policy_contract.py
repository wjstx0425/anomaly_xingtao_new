"""Linux-only immutable capture-gate policy contract tests."""

from __future__ import annotations

import sys

import pytest

from tests.unit.zs32_refactor.domain._fixtures import topology_mapping
from zs32_inspection.capture.gate_policy import CaptureGatePolicy, HandCaptureGatePolicy
from zs32_inspection.config.schemas import parse_topology
from zs32_inspection.domain.contracts import ArtifactRef
from zs32_inspection.domain.identity import Hand


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 verification is Linux-only")


def _artifact(identity: str, digest: str, path: str) -> ArtifactRef:
    return ArtifactRef(identity, "v1", digest, path)


@pytest.mark.parametrize("camera_count", [3, 4, 5])
def test_policy_requires_one_registration_reference_per_dynamic_view(camera_count: int) -> None:
    topology = parse_topology(topology_mapping(camera_count))
    hand_policy = HandCaptureGatePolicy(
        Hand.RIGHT,
        _artifact("quality", "1" * 64, "capture/gates/right/quality.json"),
        _artifact("registration", "2" * 64, "capture/gates/right/registration.json"),
        {
            view: _artifact(
                f"reference-{view}",
                "3" * 64,
                f"capture/gates/right/references/{view}.png",
            )
            for view in topology.required_views
        },
    )
    policy = CaptureGatePolicy(
        2,
        "policy-right-v1",
        "ZS32",
        topology.topology_id,
        topology.topology_sha256,
        _artifact(
            "hikvision-acquisition",
            "5" * 64,
            "capture/acquisition/hikvision.json",
        ),
        {Hand.RIGHT: hand_policy},
    )

    policy.validate_topology(topology, allowed_hands=(Hand.RIGHT,))
    provenance = policy.provenance_for(Hand.RIGHT, policy_sha256="4" * 64)
    assert set(provenance.registration_reference_sha256_by_view) == set(
        topology.required_views
    )
    assert provenance.acquisition_config_sha256 == "5" * 64
    assert "reason" not in provenance.as_dict()


def test_left_cannot_be_inferred_from_right_policy() -> None:
    topology = parse_topology(topology_mapping(3))
    hand_policy = HandCaptureGatePolicy(
        Hand.RIGHT,
        _artifact("quality", "1" * 64, "capture/gates/right/quality.json"),
        _artifact("registration", "2" * 64, "capture/gates/right/registration.json"),
        {
            view: _artifact(
                f"reference-{view}", "3" * 64,
                f"capture/gates/right/references/{view}.png",
            )
            for view in topology.required_views
        },
    )
    policy = CaptureGatePolicy(
        2, "policy-right-v1", "ZS32", topology.topology_id,
        topology.topology_sha256,
        _artifact(
            "hikvision-acquisition",
            "5" * 64,
            "capture/acquisition/hikvision.json",
        ),
        {Hand.RIGHT: hand_policy},
    )

    with pytest.raises(ValueError, match="does not enable hand 'left'"):
        policy.provenance_for(Hand.LEFT, policy_sha256="4" * 64)
