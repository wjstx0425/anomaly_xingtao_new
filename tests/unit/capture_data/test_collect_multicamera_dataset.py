# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure ZS32 multi-camera capture API."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from capture_data import collect_multicamera_dataset as multicam


class FakeAdapter:
    """Record camera lifecycle and acquisition calls without camera hardware."""

    def __init__(
        self,
        *,
        fail_open: int | None = None,
        fail_start: int | None = None,
        fail_read: int | None = None,
    ) -> None:
        self.events: list[str] = []
        self.fail_open = fail_open
        self.fail_start = fail_start
        self.fail_read = fail_read
        self.exposures: dict[int, float] = {}

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
        if handle.device.index == self.fail_read:
            raise RuntimeError("read timed out")
        exposure = int(self.exposures.get(handle.device.index, 0))
        return np.full((1, 1, 3), exposure // 1000 + handle.device.index, dtype=np.uint8)

    def set_exposure(self, handle: multicam.CameraHandle, exposure: float) -> None:
        self.events.append(f"exposure:{handle.device.index}:{int(exposure)}")
        self.exposures[handle.device.index] = exposure

    def stop(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"stop:{handle.device.index}")

    def close(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"close:{handle.device.index}")

    def destroy(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"destroy:{handle.device.index}")


def make_devices() -> list[multicam.DeviceDescription]:
    """Create three deterministic fake device descriptions."""
    return [multicam.DeviceDescription(index, f"model-{index}", f"serial-{index}") for index in range(3)]


def make_hdr_config(**overrides: object) -> SimpleNamespace:
    """Create the minimal configuration consumed by grouped HDR capture."""
    values: dict[str, object] = {
        "short_exposure": 4000.0,
        "long_exposure": 35000.0,
        "hdr_settle_frames": 0,
        "timeout_ms": 3000,
        "align_hdr": False,
        "short_dark_threshold": 70.0,
        "long_clip_threshold": 245.0,
        "blend_width": 18.0,
        "blur_size": 31,
        "hdr_max_retries": 0,
        "hdr_max_clip_pct": 12.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_storage_args(root: Path, **overrides: object) -> SimpleNamespace:
    """Create capture and storage arguments for an isolated session."""
    values = vars(make_hdr_config()).copy()
    values.update(
        {
            "root": str(root),
            "hand": "left",
            "label": "normal",
            "defect_type": "",
            "part_id": "part001",
            "save_hdr_sources": False,
        }
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_round_results(round_offset: int = 0) -> list[multicam.HdrViewResult]:
    """Create three deterministic in-memory HDR results."""
    return [
        multicam.HdrViewResult(
            camera_slot=slot,
            short_image=np.full((2, 2, 3), round_offset + slot, dtype=np.uint8),
            long_image=np.full((2, 2, 3), round_offset + slot + 10, dtype=np.uint8),
            fused_image=np.full((2, 2, 3), round_offset + slot + 20, dtype=np.uint8),
            fused_clip_pct=float(slot),
            attempt=1,
        )
        for slot in range(3)
    ]


def make_handles(adapter: FakeAdapter) -> list[multicam.CameraHandle]:
    """Open three fake handles and clear setup events."""
    handles = [adapter.open(device, gain=0.0, fps=10.0) for device in make_devices()]
    adapter.events.clear()
    return handles


def test_capture_hdr_round_groups_exposures_and_fuses_matching_camera_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each exposure is grouped across cameras and fusion keeps slot pairs together."""
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    fusion_pairs: list[tuple[int, int]] = []
    fusion_options: list[dict[str, object]] = []

    def fake_fuse(images: list[np.ndarray], **kwargs: object) -> np.ndarray:
        fusion_pairs.append((int(images[0][0, 0, 0]), int(images[1][0, 0, 0])))
        fusion_options.append(kwargs)
        return images[0]

    monkeypatch.setattr(multicam, "fuse_exposures", fake_fuse)

    results = multicam.capture_hdr_round(handles, adapter, make_hdr_config())

    assert adapter.events == [
        "exposure:0:4000",
        "exposure:1:4000",
        "exposure:2:4000",
        "trigger:0",
        "trigger:1",
        "trigger:2",
        "read:0",
        "read:1",
        "read:2",
        "exposure:0:35000",
        "exposure:1:35000",
        "exposure:2:35000",
        "trigger:0",
        "trigger:1",
        "trigger:2",
        "read:0",
        "read:1",
        "read:2",
    ]
    assert fusion_pairs == [(4, 35), (5, 36), (6, 37)]
    assert fusion_options == [
        {
            "method": "selective",
            "align": False,
            "short_dark_threshold": 70.0,
            "long_clip_threshold": 245.0,
            "blend_width": 18.0,
            "blur_size": 31,
        }
    ] * 3
    assert [result.camera_slot for result in results] == [0, 1, 2]
    assert [result.attempt for result in results] == [1, 1, 1]


def test_capture_hdr_round_retries_the_complete_pair_when_any_view_is_clipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clipped view should repeat both exposure passes for every camera."""
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    fusion_count = 0

    def fake_fuse(images: list[np.ndarray], **kwargs: object) -> np.ndarray:
        nonlocal fusion_count
        del images, kwargs
        fusion_count += 1
        value = 255 if fusion_count == 2 else 0
        return np.full((1, 1, 3), value, dtype=np.uint8)

    monkeypatch.setattr(multicam, "fuse_exposures", fake_fuse)

    results = multicam.capture_hdr_round(
        handles,
        adapter,
        make_hdr_config(hdr_max_retries=1, hdr_max_clip_pct=12.0),
    )

    assert adapter.events.count("exposure:0:4000") == 2
    assert adapter.events.count("exposure:0:35000") == 2
    assert [result.attempt for result in results] == [2, 2, 2]


def test_capture_hdr_round_does_not_return_partial_results_after_timeout() -> None:
    """Any camera timeout should fail the round instead of returning successful views."""
    adapter = FakeAdapter(fail_read=1)
    handles = make_handles(adapter)

    with pytest.raises(RuntimeError, match="read timed out"):
        multicam.capture_hdr_round(handles, adapter, make_hdr_config())


def test_capture_sample_writes_one_complete_six_view_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two successful rounds should create one explicitly complete six-view sample."""
    args = make_storage_args(tmp_path, save_hdr_sources=True)
    paths = multicam.create_session(args, datetime(2026, 7, 11, 12, 34, 56, 123456))
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    rounds = iter([make_round_results(), make_round_results(30)])
    monkeypatch.setattr(multicam, "capture_hdr_round", lambda *_args: next(rounds))

    assert multicam.capture_sample(handles, adapter, args, paths, "group001", 1) is True

    assert set(paths.view_dirs) == {
        "front",
        "front_left",
        "front_right",
        "back",
        "back_left",
        "back_right",
    }
    assert all(len(list(directory.glob("*_fused.png"))) == 1 for directory in paths.view_dirs.values())
    assert all(len(list(directory.glob("*_short.png"))) == 1 for directory in paths.view_dirs.values())
    assert all(len(list(directory.glob("*_long.png"))) == 1 for directory in paths.view_dirs.values())
    with paths.manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    image_rows = [row for row in rows if row["record_type"] == "image"]
    summary = [row for row in rows if row["record_type"] == "sample"]
    assert len(image_rows) == 6
    assert len({row["session_id"] for row in image_rows}) == 1
    assert len({row["sample_id"] for row in image_rows}) == 1
    assert {row["group_id"] for row in image_rows} == {"group001"}
    assert {row["sample_status"] for row in image_rows} == {"complete"}
    assert all(row["source_short"] and row["source_long"] for row in image_rows)
    assert len(summary) == 1 and summary[0]["sample_status"] == "complete"


def test_capture_sample_records_imwrite_failure_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A false imwrite result must make the sample incomplete with precise context."""
    args = make_storage_args(tmp_path)
    paths = multicam.create_session(args, datetime(2026, 7, 11, 12, 34, 56, 1))
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    monkeypatch.setattr(multicam, "capture_hdr_round", lambda *_args: make_round_results())
    writes = 0

    def fail_second_write(_path: str, _image: np.ndarray) -> bool:
        nonlocal writes
        writes += 1
        return writes != 2

    monkeypatch.setattr(multicam.cv2, "imwrite", fail_second_write)

    assert multicam.capture_sample(handles, adapter, args, paths, "group001", 2) is False

    with paths.manifest_path.open(newline="", encoding="utf-8") as file:
        summary = [row for row in csv.DictReader(file) if row["record_type"] == "sample"][0]
    assert summary["sample_status"] == "incomplete"
    assert summary["failed_round"] == "front"
    assert summary["failed_view"] == "front_left"
    assert summary["failed_device_index"] == "1"
    assert "cv2.imwrite returned false" in summary["error"]


def test_capture_sample_records_back_capture_failure_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A back-round camera failure must not promote the saved front round to complete."""
    args = make_storage_args(tmp_path)
    paths = multicam.create_session(args, datetime(2026, 7, 11, 12, 34, 56, 2))
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    calls = 0

    def capture(*_args: object) -> list[multicam.HdrViewResult]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("GetOneFrameTimeout failed for device 2 serial serial-2")
        return make_round_results()

    monkeypatch.setattr(multicam, "capture_hdr_round", capture)

    assert multicam.capture_sample(handles, adapter, args, paths, "group009", 3) is False

    with paths.manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    front_rows = [row for row in rows if row["record_type"] == "image"]
    summary = [row for row in rows if row["record_type"] == "sample"][0]
    assert len(front_rows) == 3
    assert {row["sample_status"] for row in front_rows} == {"incomplete"}
    assert summary["failed_round"] == "back"
    assert summary["failed_view"] == "back_right"
    assert summary["failed_device_index"] == "2"
    assert "GetOneFrameTimeout failed" in summary["error"]


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
