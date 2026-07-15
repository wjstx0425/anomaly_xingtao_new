"""Topology schema and dynamic view-count contract tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from zs32_inspection.config.loaders import load_topology
from zs32_inspection.config.schemas import parse_topology
from zs32_inspection.domain.errors import TopologyValidationError

from ._fixtures import topology_mapping


FOUR_CAMERA_TOPOLOGY = (
    Path(__file__).parents[4] / "configs/zs32/topology/zs32_4cam_double_side_v1.json"
)


@pytest.mark.parametrize(("camera_count", "view_count"), [(3, 6), (4, 8), (5, 10)])
def test_topology_compiles_camera_count_to_required_views(camera_count: int, view_count: int) -> None:
    """Two rounds dynamically produce 6/8/10 unique required views."""
    topology = parse_topology(topology_mapping(camera_count))

    assert len(topology.camera_slots) == camera_count
    assert topology.expected_view_count == view_count
    assert len(set(topology.required_views)) == view_count


def test_checked_in_four_camera_topology_binds_secondary_front_camera() -> None:
    """The production authoring config binds DB0968108 to two canonical views."""
    topology = load_topology(FOUR_CAMERA_TOPOLOGY)

    assert topology.topology_id == "zs32-4cam-double-side-v1"
    assert topology.expected_view_count == 8
    assert topology.required_views == (
        "front",
        "front_left",
        "front_right",
        "front_secondary",
        "back",
        "back_left",
        "back_right",
        "back_secondary",
    )
    assert topology.binding_for_view("front_secondary") == (
        "front",
        "front_secondary",
        "DB0968108",
    )
    assert topology.binding_for_view("back_secondary") == (
        "back",
        "front_secondary",
        "DB0968108",
    )


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
