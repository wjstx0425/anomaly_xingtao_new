# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure ZS32 multi-camera capture API."""

from __future__ import annotations

import pytest

from capture_data import collect_multicamera_dataset as multicam


def test_default_device_order_maps_to_six_views() -> None:
    """The three camera slots should keep their physical meaning in both rounds."""
    assert multicam.validate_devices([0, 1, 2]) == (0, 1, 2)
    assert [multicam.view_for("front", slot) for slot in range(3)] == [
        "front",
        "front_left",
        "front_right",
    ]
    assert [multicam.view_for("back", slot) for slot in range(3)] == [
        "back",
        "back_left",
        "back_right",
    ]


@pytest.mark.parametrize("devices", [[], [0, 1], [0, 1, 2, 3], [0, 0, 2]])
def test_validate_devices_rejects_non_unique_triples(devices: list[int]) -> None:
    """Device selection must contain exactly three unique indices."""
    with pytest.raises(ValueError, match="exactly three unique"):
        multicam.validate_devices(devices)


def test_parser_rejects_non_left_hand() -> None:
    """The first ZS32 release should accept only left-hand samples."""
    with pytest.raises(SystemExit):
        multicam.build_parser().parse_args(
            ["--devices", "0", "1", "2", "--hand", "right", "--label", "normal", "--hdr"]
        )


def test_parser_exposes_capture_schema_and_hdr_defaults() -> None:
    """The pure parser should expose the agreed multi-camera HDR CLI schema."""
    args = multicam.build_parser().parse_args(["--label", "normal", "--hdr"])

    assert args.devices == [0, 1, 2]
    assert args.hand == "left"
    assert args.defect_type == ""
    assert args.part_id == "part001"
    assert args.group_count == 1
    assert args.images_per_group == 1
    assert args.manual_load is False
    assert args.short_exposure == 7000.0
    assert args.long_exposure == 40000.0
    assert args.gain is None
    assert args.fps is None
    assert args.hdr_settle_frames == 5
    assert args.timeout_ms == 3000
    assert args.short_dark_threshold == 70.0
    assert args.long_clip_threshold == 245.0
    assert args.blend_width == 18.0
    assert args.blur_size == 31
    assert args.align_hdr is False
    assert args.save_hdr_sources is False
    assert args.hdr_max_retries == 2
    assert args.hdr_max_clip_pct == 12.0
    assert args.root == "./dataset"
    assert args.list_devices is False


def test_parser_requires_hdr_for_capture() -> None:
    """Single-exposure capture is deliberately outside the first release."""
    with pytest.raises(SystemExit):
        multicam.build_parser().parse_args(["--label", "normal"])


def test_parser_accepts_list_devices_without_capture_arguments() -> None:
    """Device discovery should not require capture-only label or HDR flags."""
    args = multicam.build_parser().parse_args(["--list-devices"])

    assert args.list_devices is True
