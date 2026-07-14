"""Topology schema and dynamic view-count contract tests."""

from __future__ import annotations

import copy

import pytest

from zs32_inspection.config.schemas import parse_topology
from zs32_inspection.domain.errors import TopologyValidationError

from ._fixtures import topology_mapping


@pytest.mark.parametrize(("camera_count", "view_count"), [(3, 6), (4, 8), (5, 10)])
def test_topology_compiles_camera_count_to_required_views(camera_count: int, view_count: int) -> None:
    """Two rounds dynamically produce 6/8/10 unique required views."""
    topology = parse_topology(topology_mapping(camera_count))

    assert len(topology.camera_slots) == camera_count
    assert topology.expected_view_count == view_count
    assert len(set(topology.required_views)) == view_count


def test_duplicate_camera_serial_is_rejected() -> None:
    """Stable camera identity cannot depend on two slots sharing one serial."""
    payload = topology_mapping(3)
    payload["camera_slots"][1]["serial"] = payload["camera_slots"][0]["serial"]

    with pytest.raises(TopologyValidationError, match="serial values must be unique"):
        parse_topology(payload)


def test_slot_missing_one_round_is_rejected() -> None:
    """Every camera must provide exactly one image in every required round."""
    payload = topology_mapping(3)
    del payload["camera_slots"][1]["views"]["back"]

    with pytest.raises(TopologyValidationError, match="round map mismatch"):
        parse_topology(payload)


def test_two_camera_rounds_cannot_claim_the_same_view() -> None:
    """One image identity can come from only one physical slot/round pair."""
    payload = topology_mapping(3)
    payload["camera_slots"][1]["views"]["front"] = payload["camera_slots"][0]["views"]["front"]

    with pytest.raises(TopologyValidationError, match="unique view"):
        parse_topology(payload)


def test_required_views_cannot_disagree_with_slot_mapping() -> None:
    """required_views is verified, not trusted as a second mutable truth."""
    payload = topology_mapping(3)
    payload["required_views"][-1] = "invented_view"

    with pytest.raises(TopologyValidationError, match="required_views must equal"):
        parse_topology(payload)


def test_topology_digest_is_stable_and_content_sensitive() -> None:
    """Equivalent config hashes match while a serial change changes identity."""
    payload = topology_mapping(3)
    first = parse_topology(payload)
    second = parse_topology(copy.deepcopy(payload))
    changed = copy.deepcopy(payload)
    changed["camera_slots"][0]["serial"] = "A_DIFFERENT_SERIAL"

    assert first.topology_sha256 == second.topology_sha256
    assert first.topology_sha256 != parse_topology(changed).topology_sha256
