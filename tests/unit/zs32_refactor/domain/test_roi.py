"""Authoritative hand-aware ROI contract tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from zs32_inspection.config.loaders import load_roi_config
from zs32_inspection.config.schemas import parse_roi_config, parse_topology
from zs32_inspection.domain.contracts import RoiReadiness
from zs32_inspection.domain.errors import RoiValidationError, SchemaValidationError
from zs32_inspection.domain.identity import Hand

from ._fixtures import roi_mapping, topology_mapping


def test_authoritative_repository_roi_uses_the_only_v2_schema() -> None:
    """The checked-in right ROI and explicit pending left ROI parse without translation."""
    repository_root = Path(__file__).resolve().parents[4]
    roi = load_roi_config(repository_root / "configs/zs32/roi/zs32_roi_v2.json")

    assert roi.roi_config_id == "zs32-roi-v2"
    assert roi.hands[Hand.RIGHT].views["front"].as_list() == [150, 1020, 3910, 2520]
    assert roi.hands[Hand.LEFT].status is RoiReadiness.PENDING


def test_v2_roi_keeps_right_ready_and_left_explicitly_pending() -> None:
    """Pending left ROI remains a first-class blocking state with no coordinates."""
    topology = parse_topology(topology_mapping(3))
    roi = parse_roi_config(roi_mapping(topology))

    assert roi.hands[Hand.RIGHT].status is RoiReadiness.READY
    assert set(roi.hands[Hand.RIGHT].views) == set(topology.required_views)
    assert roi.hands[Hand.LEFT].status is RoiReadiness.PENDING
    assert not roi.hands[Hand.LEFT].views


def test_roi_outside_source_image_is_rejected() -> None:
    """Half-open x2/y2 cannot exceed declared 4K source dimensions."""
    topology = parse_topology(topology_mapping(3))
    payload = roi_mapping(topology)
    payload["hands"]["right"]["views"][topology.required_views[0]]["xyxy"][2] = 1001

    with pytest.raises(RoiValidationError, match="exceeds image"):
        parse_roi_config(payload)


def test_pending_roi_cannot_hide_unapproved_coordinates() -> None:
    """A pending hand cannot smuggle mirrored or copied coordinates into config."""
    topology = parse_topology(topology_mapping(3))
    payload = roi_mapping(topology)
    payload["hands"]["left"]["views"] = copy.deepcopy(payload["hands"]["right"]["views"])

    with pytest.raises(RoiValidationError, match="must not contain unapproved"):
        parse_roi_config(payload)


def test_old_or_parallel_roi_field_names_are_rejected() -> None:
    """Only roi_version/source_image_size define the external v2 schema."""
    topology = parse_topology(topology_mapping(3))
    payload = roi_mapping(topology)
    payload["roi_config_id"] = payload.pop("roi_version")

    with pytest.raises(SchemaValidationError, match="missing=.*roi_version"):
        parse_roi_config(payload)
