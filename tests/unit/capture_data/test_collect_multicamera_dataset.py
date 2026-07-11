# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure ZS32 multi-camera capture API."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from capture_data import collect_multicamera_dataset as multicam


class FakeAdapter:
    """Record camera lifecycle and acquisition calls without camera hardware."""

    def __init__(self, *, fail_open: int | None = None, fail_start: int | None = None) -> None:
        self.events: list[str] = []
        self.fail_open = fail_open
        self.fail_start = fail_start

    def open(self, device: multicam.DeviceDescription, gain: float, fps: float) -> multicam.CameraHandle:
        self.events.append(f"open:{device.index}")
        if device.index == self.fail_open:
            raise RuntimeError("open failed")
        return multicam.CameraHandle(device, object(), SimpleNamespace(), bytearray(8), started=False)

    def start(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"start:{handle.device.index}")
        if handle.device.index == self.fail_start:
            raise RuntimeError("start failed")
        handle.started = True

    def trigger(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"trigger:{handle.device.index}")

    def read(self, handle: multicam.CameraHandle, timeout_ms: int) -> np.ndarray:
        self.events.append(f"read:{handle.device.index}")
        return np.full((1, 1, 3), handle.device.index, dtype=np.uint8)

    def stop(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"stop:{handle.device.index}")

    def close(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"close:{handle.device.index}")

    def destroy(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"destroy:{handle.device.index}")


def make_devices() -> list[multicam.DeviceDescription]:
    """Create three deterministic fake device descriptions."""
    return [multicam.DeviceDescription(index, f"model-{index}", f"serial-{index}") for index in range(3)]


def test_trigger_and_read_groups_all_triggers_before_reads() -> None:
    """All cameras should receive a trigger before any camera is read."""
    adapter = FakeAdapter()
    handles = [adapter.open(device, gain=0.0, fps=10.0) for device in make_devices()]
    adapter.events.clear()

    frames = multicam.trigger_and_read(handles, adapter, timeout_ms=3000)

    assert adapter.events == [
        "trigger:0",
        "trigger:1",
        "trigger:2",
        "read:0",
        "read:1",
        "read:2",
    ]
    assert len({id(handle.data_buf) for handle in handles}) == 3
    assert len({id(handle.frame_info) for handle in handles}) == 3
    assert len(frames) == 3


@pytest.mark.parametrize((failure, failed_index), [("open", 2), ("start", 2)])
def test_open_cameras_cleans_up_partial_setup_in_reverse_order(failure: str, failed_index: int) -> None:
    """Successfully created handles should be cleaned exactly once after setup failure."""
    adapter = FakeAdapter(
        fail_open=failed_index if failure == "open" else None,
        fail_start=failed_index if failure == "start" else None,
    )

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        with multicam.open_cameras(make_devices(), adapter, gain=0.0, fps=10.0):
            pytest.fail("setup failure should prevent entering the context")

    cleanup_events = [
        event for event in adapter.events if event.split(":", maxsplit=1)[0] in {"stop", "close", "destroy"}
    ]
    if failure == "open":
        assert cleanup_events == [
            "stop:1",
            "close:1",
            "destroy:1",
            "stop:0",
            "close:0",
            "destroy:0",
        ]
    else:
        assert cleanup_events == [
            "close:2",
            "destroy:2",
            "stop:1",
            "close:1",
            "destroy:1",
            "stop:0",
            "close:0",
            "destroy:0",
        ]


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
