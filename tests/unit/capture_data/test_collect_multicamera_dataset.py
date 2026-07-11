# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure ZS32 multi-camera capture API."""

from __future__ import annotations

import csv
import ctypes
import math
from contextlib import contextmanager
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
        fail_stop: int | None = None,
        fail_restore: int | None = None,
    ) -> None:
        self.events: list[str] = []
        self.fail_open = fail_open
        self.fail_start = fail_start
        self.fail_read = fail_read
        self.fail_stop = fail_stop
        self.fail_restore = fail_restore
        self.exposures: dict[int, float] = {}

    def list_devices(self) -> list[multicam.DeviceDescription]:
        """Return deterministic fake devices."""
        self.events.append("list")
        return make_devices()

    def open(self, device: multicam.DeviceDescription, gain: float) -> multicam.CameraHandle:
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
        if handle.device.index == self.fail_stop:
            raise RuntimeError("stop failed")

    def restore_continuous(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"restore:{handle.device.index}")
        if handle.device.index == self.fail_restore:
            raise RuntimeError("restore failed")

    def close(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"close:{handle.device.index}")

    def destroy(self, handle: multicam.CameraHandle) -> None:
        self.events.append(f"destroy:{handle.device.index}")


class _SdkTransport(ctypes.Structure):
    _fields_ = [("chModelName", ctypes.c_char * 64), ("chSerialNumber", ctypes.c_char * 64)]


class _SdkSpecialInfo(ctypes.Union):
    _fields_ = [("stGigEInfo", _SdkTransport), ("stUsb3VInfo", _SdkTransport)]


class _SdkDeviceInfo(ctypes.Structure):
    _fields_ = [("nTLayerType", ctypes.c_uint), ("SpecialInfo", _SdkSpecialInfo)]


class _SdkDeviceInfoList:
    def __init__(self) -> None:
        self.nDeviceNum = 0
        self.pDeviceInfo: list[ctypes.POINTER[_SdkDeviceInfo]] = []


class _SdkFrameInfo(ctypes.Structure):
    _fields_: list[tuple[str, object]] = []


class _SdkFloatValue(ctypes.Structure):
    _fields_ = [("fCurValue", ctypes.c_float), ("fMax", ctypes.c_float), ("fMin", ctypes.c_float)]


class FakeSdk:
    """Small ctypes-compatible SDK double for enumeration snapshot tests."""

    MV_GIGE_DEVICE = 1
    MV_USB_DEVICE = 2
    MV_ACCESS_Exclusive = 1
    MV_CC_DEVICE_INFO = _SdkDeviceInfo
    MV_CC_DEVICE_INFO_LIST = _SdkDeviceInfoList
    MV_FRAME_OUT_INFO_EX = _SdkFrameInfo
    MVCC_FLOATVALUE = _SdkFloatValue

    def __init__(
        self,
        devices: list[multicam.DeviceDescription],
        *,
        fail_event: str | None = None,
        interrupt_event: str | None = None,
    ) -> None:
        self.enum_calls = 0
        self.created_serials: list[str] = []
        self.float_names: list[str] = []
        self.events: list[str] = []
        self.fail_event = fail_event
        self.interrupt_event = interrupt_event
        self._infos: list[_SdkDeviceInfo] = []
        for device in devices:
            info = _SdkDeviceInfo()
            info.nTLayerType = self.MV_USB_DEVICE
            info.SpecialInfo.stUsb3VInfo.chModelName = device.model.encode()
            info.SpecialInfo.stUsb3VInfo.chSerialNumber = device.serial.encode()
            self._infos.append(info)
        sdk = self

        def record(event: str) -> int:
            sdk.events.append(event)
            if event == sdk.interrupt_event:
                raise KeyboardInterrupt
            return 1 if event == sdk.fail_event else 0

        class Camera:
            def __init__(self) -> None:
                self.serial = "unbound"

            @staticmethod
            def MV_CC_EnumDevices(_device_types: int, device_list: _SdkDeviceInfoList) -> int:
                sdk.enum_calls += 1
                device_list.nDeviceNum = len(sdk._infos)
                device_list.pDeviceInfo = [ctypes.pointer(info) for info in sdk._infos]
                return 0

            def MV_CC_CreateHandle(self, info: _SdkDeviceInfo) -> int:
                self.serial = multicam._decode_sdk_text(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
                sdk.created_serials.append(self.serial)
                return record(f"create:{self.serial}")

            def MV_CC_OpenDevice(self, _access: int, _key: int) -> int:
                return record(f"open:{self.serial}")

            def MV_CC_SetEnumValue(self, name: str, value: int) -> int:
                return record(f"enum:{self.serial}:{name}:{value}")

            def MV_CC_SetFloatValue(self, name: str, _value: float) -> int:
                sdk.float_names.append(name)
                return record(f"float:{self.serial}:{name}:{_value:g}")

            def MV_CC_GetFloatValue(self, name: str, value: _SdkFloatValue) -> int:
                value.fMin = 0.0
                value.fMax = 100000.0 if name == "ExposureTime" else 24.0
                return record(f"get_float:{self.serial}:{name}")

            def MV_CC_SetBoolValue(self, _name: str, _value: bool) -> int:
                return 0

            def MV_CC_StartGrabbing(self) -> int:
                return record(f"start:{self.serial}")

            def MV_CC_StopGrabbing(self) -> int:
                return record(f"stop:{self.serial}")

            def MV_CC_CloseDevice(self) -> int:
                return record(f"close:{self.serial}")

            def MV_CC_DestroyHandle(self) -> int:
                return record(f"destroy:{self.serial}")

        self.MvCamera = Camera


def make_devices() -> list[multicam.DeviceDescription]:
    """Create three deterministic fake device descriptions."""
    serials = ("DA9805574", "DA9625347", "DB0998274")
    return [multicam.DeviceDescription(index, f"model-{index}", serial) for index, serial in enumerate(serials)]


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
        "capture_interval": 0.1,
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
            "hdr": True,
            "exposure": 4000.0,
            "gain": None,
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


def test_single_round_sets_all_exposures_before_grouped_trigger_and_read() -> None:
    """Single exposure keeps the three cameras in one grouped acquisition pass."""
    adapter = FakeAdapter()
    handles = [
        multicam.CameraHandle(device, object(), object(), bytearray(8), started=True)
        for device in make_devices()
    ]

    results = multicam.capture_single_round(
        handles,
        adapter,
        exposure=4000.0,
        timeout_ms=3000,
        pacer=multicam.GroupedTriggerPacer(0.0),
    )

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
    ]
    assert [result.camera_slot for result in results] == [0, 1, 2]
    assert all(result.source_short is None and result.source_long is None for result in results)


def test_parser_uses_application_capture_interval_without_hardware_fps() -> None:
    args = multicam.build_parser().parse_args(["--label", "normal"])

    assert args.capture_interval == 0.2
    assert not hasattr(args, "fps")


def make_handles(adapter: FakeAdapter) -> list[multicam.CameraHandle]:
    """Open three fake handles and clear setup events."""
    handles = [adapter.open(device, gain=0.0) for device in make_devices()]
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


def test_capture_hdr_round_paces_settle_short_and_long_trigger_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjacent grouped trigger passes should respect the configured frame interval."""
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    now = 0.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(multicam, "fuse_exposures", lambda images, **kwargs: images[0])
    pacer = multicam.GroupedTriggerPacer(0.1, monotonic=monotonic, sleep=sleep)

    multicam.capture_hdr_round(handles, adapter, make_hdr_config(hdr_settle_frames=1), pacer=pacer)

    assert sleeps == pytest.approx([0.1, 0.1, 0.1])
    acquisition_events = [event for event in adapter.events if event.startswith(("trigger", "read"))]
    assert acquisition_events == [
        event
        for _ in range(4)
        for event in (
            "trigger:0",
            "trigger:1",
            "trigger:2",
            "read:0",
            "read:1",
            "read:2",
        )
    ]


def test_capture_group_paces_across_adjacent_hdr_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every trigger pass in one group should share one continuous pacing timeline."""
    args = make_storage_args(tmp_path, images_per_group=2)
    paths = multicam.create_session(args, datetime(2026, 7, 11, 12, 34, 56, 7))
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    now = 0.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(multicam, "fuse_exposures", lambda images, **kwargs: images[0])
    monkeypatch.setattr(multicam, "save_round", lambda *_args: [])
    monkeypatch.setattr(multicam, "_write_manifest", lambda *_args: None)
    pacer = multicam.GroupedTriggerPacer(0.1, monotonic=monotonic, sleep=sleep)

    multicam.capture_group(handles, adapter, args, paths, "group001", pacer=pacer)

    # Four HDR rounds each issue short and long grouped trigger passes. Only
    # the first pass in the group is immediate, including across round edges.
    assert sleeps == pytest.approx([0.1] * 7)
    assert adapter.events.count("trigger:0") == 8


def test_capture_hdr_round_retry_keeps_using_the_supplied_pacer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry should not reset pacing between its complete HDR-pair attempts."""
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    now = 0.0
    sleeps: list[float] = []
    fusion_count = 0

    def monotonic() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    def fake_fuse(images: list[np.ndarray], **kwargs: object) -> np.ndarray:
        nonlocal fusion_count
        del images, kwargs
        fusion_count += 1
        return np.full((1, 1, 3), 255 if fusion_count == 1 else 0, dtype=np.uint8)

    monkeypatch.setattr(multicam, "fuse_exposures", fake_fuse)
    pacer = multicam.GroupedTriggerPacer(0.1, monotonic=monotonic, sleep=sleep)

    multicam.capture_hdr_round(
        handles,
        adapter,
        make_hdr_config(hdr_max_retries=1, hdr_max_clip_pct=12.0),
        pacer=pacer,
    )

    assert sleeps == pytest.approx([0.1, 0.1, 0.1])


@pytest.mark.parametrize("interval", [-1.0, float("nan")])
def test_grouped_trigger_pacer_rejects_invalid_interval(interval: float) -> None:
    """Negative or non-finite application pacing intervals are invalid."""
    with pytest.raises(ValueError, match="capture interval"):
        multicam.GroupedTriggerPacer(interval)


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
    monkeypatch.setattr(multicam, "capture_round", lambda *_args, **_kwargs: next(rounds))

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
    monkeypatch.setattr(multicam, "capture_round", lambda *_args, **_kwargs: make_round_results())
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

    def capture(*_args: object, **_kwargs: object) -> list[multicam.HdrViewResult]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("GetOneFrameTimeout failed for device 2 serial serial-2")
        return make_round_results()

    monkeypatch.setattr(multicam, "capture_round", capture)

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


def test_save_round_rejects_duplicate_camera_slots(tmp_path: Path) -> None:
    """A three-result round must still contain each physical camera slot exactly once."""
    args = make_storage_args(tmp_path)
    paths = multicam.create_session(args, datetime(2026, 7, 11, 12, 34, 56, 3))
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    results = make_round_results()
    results[1] = multicam.HdrViewResult(
        camera_slot=0,
        short_image=results[1].short_image,
        long_image=results[1].long_image,
        fused_image=results[1].fused_image,
        fused_clip_pct=results[1].fused_clip_pct,
        attempt=results[1].attempt,
    )

    with pytest.raises(RuntimeError, match=r"camera slots.*\{0, 1, 2\}"):
        multicam.save_round(results, "front", "sample", "group001", 4, handles, args, paths)

    assert not list(tmp_path.rglob("*.png"))


def test_capture_sample_checks_six_distinct_views_before_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed stored rows must not promote a sample to complete."""
    args = make_storage_args(tmp_path)
    paths = multicam.create_session(args, datetime(2026, 7, 11, 12, 34, 56, 4))
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    monkeypatch.setattr(multicam, "capture_round", lambda *_args, **_kwargs: make_round_results())

    def malformed_save_round(
        results: object,
        round_name: str,
        sample_id: str,
        group_id: str,
        image_index: int,
        round_handles: object,
        round_args: object,
        round_paths: multicam.SessionPaths,
    ) -> list[dict[str, str]]:
        del results, round_handles, round_args
        return [
            {
                **dict.fromkeys(multicam.MANIFEST_COLUMNS, ""),
                "record_type": "image",
                "session_id": round_paths.session_id,
                "sample_id": sample_id,
                "group_id": group_id,
                "image_index": str(image_index),
                "round": round_name,
                "view": multicam.ROUND_VIEWS[round_name][0],
                "sample_status": "incomplete",
            }
        ] * 3

    monkeypatch.setattr(multicam, "save_round", malformed_save_round)

    assert multicam.capture_sample(handles, adapter, args, paths, "group001", 5) is False

    with paths.manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    summary = [row for row in rows if row["record_type"] == "sample"][0]
    assert summary["sample_status"] == "incomplete"
    assert "six distinct canonical views" in summary["error"]


def test_capture_sample_rolls_back_published_files_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-publish failure must leave no round files and retain failure identity."""
    args = make_storage_args(tmp_path)
    paths = multicam.create_session(args, datetime(2026, 7, 11, 12, 34, 56, 5))
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    monkeypatch.setattr(multicam, "capture_round", lambda *_args, **_kwargs: make_round_results())
    original_replace = Path.replace
    replacements = 0

    def fail_second_replace(source: Path, destination: Path) -> Path:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("simulated publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_second_replace)

    assert multicam.capture_sample(handles, adapter, args, paths, "group001", 6) is False

    assert not list(tmp_path.rglob("*.png"))
    with paths.manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    summary = [row for row in rows if row["record_type"] == "sample"][0]
    assert summary["sample_status"] == "incomplete"
    assert summary["failed_round"] == "front"
    assert summary["failed_view"] == "front_left"
    assert summary["failed_device_index"] == "1"
    assert "simulated publish failure" in summary["error"]


def test_trigger_and_read_groups_all_triggers_before_reads() -> None:
    """All cameras should receive a trigger before any camera is read."""
    adapter = FakeAdapter()
    handles = [adapter.open(device, gain=0.0) for device in make_devices()]
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


@pytest.mark.parametrize(("failure", "failed_index"), [("open", 2), ("start", 2)])
def test_open_cameras_cleans_up_partial_setup_in_reverse_order(failure: str, failed_index: int) -> None:
    """Successfully created handles should be cleaned exactly once after setup failure."""
    adapter = FakeAdapter(
        fail_open=failed_index if failure == "open" else None,
        fail_start=failed_index if failure == "start" else None,
    )

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        with multicam.open_cameras(make_devices(), adapter, gain=0.0):
            pytest.fail("setup failure should prevent entering the context")

    cleanup_events = [
        event
        for event in adapter.events
        if event.split(":", maxsplit=1)[0] in {"stop", "restore", "close", "destroy"}
    ]
    if failure == "open":
        assert cleanup_events == [
            "stop:1",
            "restore:1",
            "close:1",
            "destroy:1",
            "stop:0",
            "restore:0",
            "close:0",
            "destroy:0",
        ]
    else:
        assert cleanup_events == [
            "restore:2",
            "close:2",
            "destroy:2",
            "stop:1",
            "restore:1",
            "close:1",
            "destroy:1",
            "stop:0",
            "restore:0",
            "close:0",
            "destroy:0",
        ]


def test_open_cameras_restores_every_camera_after_normal_exit() -> None:
    """A successful session should stop, restore, close, and destroy every camera."""
    adapter = FakeAdapter()

    with multicam.open_cameras(make_devices(), adapter, gain=0.0):
        adapter.events.append("capture")

    assert adapter.events[-13:] == [
        "capture",
        "stop:2",
        "restore:2",
        "close:2",
        "destroy:2",
        "stop:1",
        "restore:1",
        "close:1",
        "destroy:1",
        "stop:0",
        "restore:0",
        "close:0",
        "destroy:0",
    ]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_open_cameras_preserves_capture_base_exception_and_restores_all(error_type: type[BaseException]) -> None:
    """Capture errors and interrupts should retain their identity after complete cleanup."""
    adapter = FakeAdapter()

    with pytest.raises(error_type, match="capture failed"):
        with multicam.open_cameras(make_devices(), adapter, gain=0.0):
            raise error_type("capture failed")

    assert adapter.events[-12:] == [
        "stop:2",
        "restore:2",
        "close:2",
        "destroy:2",
        "stop:1",
        "restore:1",
        "close:1",
        "destroy:1",
        "stop:0",
        "restore:0",
        "close:0",
        "destroy:0",
    ]


def test_close_cameras_reports_stop_failure_after_finishing_all_cleanup() -> None:
    """A stop failure must not block restore or cleanup of later cameras."""
    adapter = FakeAdapter(fail_stop=2)
    handles = make_handles(adapter)
    for handle in handles:
        handle.started = True

    with pytest.raises(RuntimeError, match="stop failed"):
        multicam.close_cameras(handles, adapter)

    assert adapter.events == [
        "stop:2",
        "restore:2",
        "close:2",
        "destroy:2",
        "stop:1",
        "restore:1",
        "close:1",
        "destroy:1",
        "stop:0",
        "restore:0",
        "close:0",
        "destroy:0",
    ]


def test_close_cameras_reports_restore_failure_after_close_destroy_and_other_cameras() -> None:
    """A restore failure must not block close, destroy, or cleanup of another camera."""
    adapter = FakeAdapter(fail_restore=1)
    handles = make_handles(adapter)

    with pytest.raises(RuntimeError, match="restore failed"):
        multicam.close_cameras(handles, adapter)

    assert adapter.events == [
        "restore:2",
        "close:2",
        "destroy:2",
        "restore:1",
        "close:1",
        "destroy:1",
        "restore:0",
        "close:0",
        "destroy:0",
    ]


def test_cleanup_error_is_reported_without_overriding_capture_error() -> None:
    """Cleanup diagnostics should attach to, rather than replace, the capture exception."""
    adapter = FakeAdapter(fail_restore=2)

    with pytest.raises(ValueError, match="primary capture failure") as raised:
        with multicam.open_cameras(make_devices(), adapter, gain=0.0):
            raise ValueError("primary capture failure")

    assert any("restore failed" in note for note in raised.value.__notes__)
    assert adapter.events[-3:] == ["restore:0", "close:0", "destroy:0"]


def test_select_devices_by_serial_ignores_enumeration_order() -> None:
    """Physical camera slots should be selected independently of SDK order."""
    available = [
        multicam.DeviceDescription(0, "right-model", "DB0998274"),
        multicam.DeviceDescription(1, "front-model", "DA9805574"),
        multicam.DeviceDescription(2, "left-model", "DA9625347"),
    ]

    selected = multicam.select_devices_by_serial(multicam.DEFAULT_SERIALS, available)

    assert [device.serial for device in selected] == ["DA9805574", "DA9625347", "DB0998274"]


def test_serial_selected_camera_slots_map_to_six_views() -> None:
    """The serial-selected camera slots should keep their meaning in both rounds."""
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


def test_select_devices_by_serial_rejects_duplicate_requested_serials() -> None:
    """One physical camera cannot occupy multiple requested slots."""
    requested = multicam.CameraSerials("DA9805574", "DA9805574", "DB0998274")

    with pytest.raises(ValueError, match="requested camera serials must be unique"):
        multicam.select_devices_by_serial(requested, make_devices())


def test_select_devices_by_serial_rejects_missing_serials() -> None:
    """Every requested physical camera must exist in the enumeration snapshot."""
    available = make_devices()[:2]

    with pytest.raises(ValueError, match="camera serials are unavailable.*DB0998274"):
        multicam.select_devices_by_serial(multicam.DEFAULT_SERIALS, available)


def test_select_devices_by_serial_rejects_duplicate_enumerated_serials() -> None:
    """An ambiguous SDK snapshot must not silently select one duplicate serial."""
    available = [*make_devices(), multicam.DeviceDescription(9, "duplicate", "DA9805574")]

    with pytest.raises(ValueError, match="enumeration contains duplicate camera serials.*DA9805574"):
        multicam.select_devices_by_serial(multicam.DEFAULT_SERIALS, available)


def test_parser_rejects_non_left_hand() -> None:
    """The first ZS32 release should accept only left-hand samples."""
    with pytest.raises(SystemExit):
        multicam.build_parser().parse_args(
            ["--hand", "right", "--label", "normal", "--hdr"]
        )


def test_parser_exposes_capture_schema_and_hdr_defaults() -> None:
    """The pure parser should expose the agreed multi-camera HDR CLI schema."""
    args = multicam.build_parser().parse_args(["--label", "normal", "--hdr"])

    assert args.front_serial == "DA9805574"
    assert args.left_serial == "DA9625347"
    assert args.right_serial == "DB0998274"
    assert args.hand == "left"
    assert args.defect_type == ""
    assert args.part_id == "part001"
    assert args.group_count == 1
    assert args.images_per_group == 1
    assert args.manual_load is False
    assert args.exposure == 4000.0
    assert args.short_exposure == 7000.0
    assert args.long_exposure == 40000.0
    assert args.gain is None
    assert args.capture_interval == 0.2
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


def test_single_exposure_is_default_capture_mode() -> None:
    """Capture should default to one 4000 microsecond exposure without HDR."""
    args = multicam.build_parser().parse_args(["--label", "normal"])

    assert args.hdr is False
    assert args.exposure == 4000.0


@pytest.mark.parametrize("hdr_only_flag", ["--save-hdr-sources", "--align-hdr"])
def test_parser_rejects_hdr_only_flags_in_single_exposure_mode(hdr_only_flag: str) -> None:
    """Artifact and fusion flags must not be silently ignored outside HDR mode."""
    with pytest.raises(SystemExit):
        multicam.build_parser().parse_args(["--label", "normal", hdr_only_flag])


@pytest.mark.parametrize(
    "invalid_option",
    [
        ["--group-count", "0"],
        ["--images-per-group", "-1"],
        ["--timeout-ms", "0"],
        ["--exposure", "0"],
        ["--short-exposure", "-1"],
        ["--long-exposure", "0"],
        ["--exposure", "nan"],
        ["--short-exposure", "nan"],
        ["--long-exposure", "nan"],
        ["--capture-interval", "nan"],
    ],
)
def test_parser_rejects_invalid_capture_ranges(invalid_option: list[str]) -> None:
    """Invalid counts, timeouts, and exposures should fail before SDK loading."""
    with pytest.raises(SystemExit):
        multicam.build_parser().parse_args(["--label", "normal", *invalid_option])


def test_parser_rejects_negative_future_capture_interval_default() -> None:
    """Task 2 validation should protect the interval option introduced by Task 3."""
    parser = multicam.build_parser()
    parser.set_defaults(capture_interval=-0.1)

    with pytest.raises(SystemExit):
        parser.parse_args(["--label", "normal"])


@pytest.mark.parametrize(
    "option", ["--exposure", "--short-exposure", "--long-exposure", "--capture-interval"]
)
def test_main_rejects_non_finite_positive_float_before_sdk_load(
    option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-finite positive-float options must fail before the Hikvision SDK is loaded."""

    def fail_sdk_load() -> FakeAdapter:
        raise AssertionError("HikvisionAdapter.load must not run for invalid CLI input")

    monkeypatch.setattr(multicam.HikvisionAdapter, "load", fail_sdk_load)

    with pytest.raises(SystemExit):
        multicam.main(["--label", "normal", option, "nan"])


def test_parser_rejects_removed_device_indices_option() -> None:
    """Formal capture should bind cameras only by stable serial identity."""
    with pytest.raises(SystemExit):
        multicam.build_parser().parse_args(["--label", "normal", "--devices", "0", "1", "2"])


def test_parser_accepts_list_devices_without_capture_arguments() -> None:
    """Device discovery should not require capture-only label or HDR flags."""
    args = multicam.build_parser().parse_args(["--list-devices"])

    assert args.list_devices is True


def test_main_lists_devices_without_opening_cameras(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Discovery should print stable identities and avoid camera setup."""
    adapter = FakeAdapter()
    monkeypatch.setattr(multicam.HikvisionAdapter, "load", lambda: adapter)

    assert multicam.main(["--list-devices"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "0\tmodel-0\tDA9805574",
        "1\tmodel-1\tDA9625347",
        "2\tmodel-2\tDB0998274",
    ]
    assert adapter.events == ["list"]


def test_capture_startup_enumerates_sdk_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """All three camera handles must come from one cached SDK enumeration snapshot."""
    enumerated = [make_devices()[2], make_devices()[0], make_devices()[1]]
    sdk = FakeSdk(enumerated)
    adapter = multicam.HikvisionAdapter(sdk)
    monkeypatch.setattr(multicam.HikvisionAdapter, "load", lambda: adapter)
    monkeypatch.setattr(multicam, "create_session", lambda *_args: object())
    monkeypatch.setattr(multicam, "capture_group", lambda *_args, **_kwargs: [True])

    assert multicam.main(["--label", "normal"]) == 0

    assert sdk.enum_calls == 1
    assert sdk.created_serials == ["DA9805574", "DA9625347", "DB0998274"]
    assert "AcquisitionFrameRate" not in sdk.float_names


def test_hikvision_adapter_open_rejects_cached_serial_mismatch() -> None:
    """A stale or mismatched description must never open a different cached camera."""
    sdk = FakeSdk(make_devices())
    adapter = multicam.HikvisionAdapter(sdk)
    adapter.list_devices()
    mismatched = multicam.DeviceDescription(0, "model-0", "wrong-serial")

    with pytest.raises(RuntimeError, match="serial mismatch.*wrong-serial.*DA9805574"):
        adapter.open(mismatched, gain=0.0)

    assert sdk.enum_calls == 1
    assert sdk.created_serials == []


def test_hikvision_adapter_open_uses_range_validation_without_hardware_fps() -> None:
    """Camera setup should validate gain and never touch hardware frame-rate nodes."""
    sdk = FakeSdk(make_devices())
    adapter = multicam.HikvisionAdapter(sdk)
    device = adapter.list_devices()[0]

    handle = adapter.open(device, gain=3.5)

    assert sdk.events == [
        "create:DA9805574",
        "open:DA9805574",
        "enum:DA9805574:ExposureAuto:0",
        "enum:DA9805574:GainAuto:0",
        "get_float:DA9805574:Gain",
        "float:DA9805574:Gain:3.5",
        "enum:DA9805574:TriggerMode:1",
        "enum:DA9805574:TriggerSource:7",
    ]
    assert "AcquisitionFrameRate" not in sdk.float_names
    assert all("AcquisitionFrameRateEnable" not in event for event in sdk.events)

    adapter.restore_continuous(handle)
    adapter.close(handle)
    adapter.destroy(handle)


def test_hikvision_adapter_restores_raw_handle_when_open_device_fails() -> None:
    """A created raw handle should be restored, closed, and destroyed after open failure."""
    sdk = FakeSdk(make_devices(), fail_event="open:DA9805574")
    adapter = multicam.HikvisionAdapter(sdk)
    device = adapter.list_devices()[0]

    with pytest.raises(RuntimeError, match="OpenDevice failed"):
        adapter.open(device, gain=0.0)

    assert sdk.events == [
        "create:DA9805574",
        "open:DA9805574",
        "enum:DA9805574:TriggerMode:0",
        "close:DA9805574",
        "destroy:DA9805574",
    ]


@pytest.mark.parametrize("interrupt", [False, True])
def test_hikvision_adapter_restores_after_failure_following_trigger_mode(interrupt: bool) -> None:
    """Any BaseException after software trigger setup should roll the raw handle back."""
    event = "enum:DA9805574:TriggerSource:7"
    sdk = FakeSdk(
        make_devices(),
        fail_event=None if interrupt else event,
        interrupt_event=event if interrupt else None,
    )
    adapter = multicam.HikvisionAdapter(sdk)
    device = adapter.list_devices()[0]
    expected_error: type[BaseException] = KeyboardInterrupt if interrupt else RuntimeError

    with pytest.raises(expected_error):
        adapter.open(device, gain=0.0)

    assert sdk.events == [
        "create:DA9805574",
        "open:DA9805574",
        "enum:DA9805574:ExposureAuto:0",
        "enum:DA9805574:GainAuto:0",
        "get_float:DA9805574:Gain",
        "float:DA9805574:Gain:0",
        "enum:DA9805574:TriggerMode:1",
        "enum:DA9805574:TriggerSource:7",
        "enum:DA9805574:TriggerMode:0",
        "close:DA9805574",
        "destroy:DA9805574",
    ]


@pytest.mark.parametrize(
    ("name", "value", "minimum", "maximum"),
    [
        ("Gain", -0.5, 0.0, 24.0),
        ("Gain", 25.0, 0.0, 24.0),
        ("ExposureTime", 100001.0, 0.0, 100000.0),
    ],
)
def test_validate_float_range_reports_parameter_value_and_bounds(
    name: str, value: float, minimum: float, maximum: float
) -> None:
    """SDK range errors should expose the requested parameter and legal bounds."""
    sdk = FakeSdk(make_devices())
    adapter = multicam.HikvisionAdapter(sdk)
    device = adapter.list_devices()[0]
    handle = adapter.open(device, gain=0.0)

    with pytest.raises(
        ValueError,
        match=rf"{name}.*{value}.*{minimum}.*{maximum}",
    ):
        adapter.validate_float_range(handle.cam, name, value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_validate_float_range_rejects_non_finite_values(value: float) -> None:
    """NaN and infinities must be rejected before a camera node is written."""
    sdk = FakeSdk(make_devices())
    adapter = multicam.HikvisionAdapter(sdk)
    device = adapter.list_devices()[0]
    handle = adapter.open(device, gain=0.0)

    with pytest.raises(ValueError, match="Gain.*finite"):
        adapter.validate_float_range(handle.cam, "Gain", value)


def test_set_exposure_validates_every_write() -> None:
    """Single and HDR callers share one range check before each exposure write."""
    sdk = FakeSdk(make_devices())
    adapter = multicam.HikvisionAdapter(sdk)
    device = adapter.list_devices()[0]
    handle = adapter.open(device, gain=0.0)
    sdk.events.clear()

    adapter.set_exposure(handle, 7000.0)
    adapter.set_exposure(handle, 40000.0)

    assert sdk.events == [
        "get_float:DA9805574:ExposureTime",
        "float:DA9805574:ExposureTime:7000",
        "get_float:DA9805574:ExposureTime",
        "float:DA9805574:ExposureTime:40000",
    ]


def test_main_normalizes_default_capture_interval_for_capture_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default frame rate should configure application-level HDR pacing."""
    adapter = FakeAdapter()
    captured_intervals: list[float] = []
    opened_serials: list[str] = []

    @contextmanager
    def fake_open_cameras(
        devices: list[multicam.DeviceDescription],
        camera_adapter: FakeAdapter,
        gain: float,
    ) -> object:
        del camera_adapter, gain
        opened_serials.extend(device.serial for device in devices)
        yield []

    def fake_capture_group(
        handles: object,
        camera_adapter: object,
        config: SimpleNamespace,
        paths: object,
        group_id: object,
        prompt: object,
        *,
        pacer: multicam.GroupedTriggerPacer,
    ) -> None:
        del handles, camera_adapter, paths, group_id, prompt, pacer
        captured_intervals.append(config.capture_interval)

    monkeypatch.setattr(multicam.HikvisionAdapter, "load", lambda: adapter)
    monkeypatch.setattr(multicam, "create_session", lambda *args: SimpleNamespace())
    monkeypatch.setattr(multicam, "open_cameras", fake_open_cameras)
    monkeypatch.setattr(multicam, "capture_group", fake_capture_group)

    assert multicam.main(["--label", "normal", "--hdr"]) == 0
    assert captured_intervals == [0.2]
    assert opened_serials == ["DA9805574", "DA9625347", "DB0998274"]


def test_parser_requires_defect_type_for_defect_capture() -> None:
    """Defect samples must name their defect class."""
    with pytest.raises(SystemExit):
        multicam.build_parser().parse_args(["--label", "defect"])


def test_capture_sample_prompts_before_front_and_back_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One paired sample should request placement immediately before each side."""
    args = make_storage_args(tmp_path)
    paths = multicam.create_session(args, datetime(2026, 7, 11, 12, 34, 56, 5))
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    prompts: list[str] = []
    monkeypatch.setattr(multicam, "capture_round", lambda *_args, **_kwargs: make_round_results())
    monkeypatch.setattr(multicam, "save_round", lambda *_args: [])
    monkeypatch.setattr(multicam, "_write_manifest", lambda *_args: None)

    multicam.capture_sample(handles, adapter, args, paths, "group001", 1, prompt=prompts.append)

    assert prompts == [
        "放好 ZS32 左手件正面后按 Enter 或 s...",
        "将同一个 ZS32 左手件翻到背面后按 Enter 或 s...",
    ]


def test_capture_group_prompts_once_and_captures_all_fronts_before_backs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group should be placed once per side while preserving paired sample IDs."""
    args = make_storage_args(tmp_path, images_per_group=3)
    paths = multicam.create_session(args, datetime(2026, 7, 11, 12, 34, 56, 6))
    adapter = FakeAdapter()
    handles = make_handles(adapter)
    prompts: list[str] = []
    saved_rounds: list[tuple[str, str, int]] = []
    monkeypatch.setattr(multicam, "capture_round", lambda *_args, **_kwargs: make_round_results())

    def record_round(
        _results: object,
        round_name: str,
        sample_id: str,
        _group_id: str,
        image_index: int,
        *_args: object,
    ) -> list[dict[str, str]]:
        saved_rounds.append((round_name, sample_id, image_index))
        return [
            {
                **dict.fromkeys(multicam.MANIFEST_COLUMNS, ""),
                "record_type": "image",
                "sample_id": sample_id,
                "round": round_name,
                "view": view,
            }
            for view in multicam.ROUND_VIEWS[round_name]
        ]

    monkeypatch.setattr(multicam, "save_round", record_round)
    monkeypatch.setattr(multicam, "_write_manifest", lambda *_args: None)

    assert multicam.capture_group(
        handles, adapter, args, paths, "group001", prompt=prompts.append
    ) == [True, True, True]

    assert prompts == [
        "放好 ZS32 左手件正面后按 Enter 或 s...",
        "将同一个 ZS32 左手件翻到背面后按 Enter 或 s...",
    ]
    assert [round_name for round_name, _, _ in saved_rounds] == ["front"] * 3 + ["back"] * 3
    assert saved_rounds == [
        ("front", "part001_group001_000001", 1),
        ("front", "part001_group001_000002", 2),
        ("front", "part001_group001_000003", 3),
        ("back", "part001_group001_000001", 1),
        ("back", "part001_group001_000002", 2),
        ("back", "part001_group001_000003", 3),
    ]


@pytest.mark.parametrize(("manual_load", "expected_prompt_count"), [(False, 0), (True, 4)])
def test_main_prompts_twice_per_group_only_for_manual_load(
    manual_load: bool,
    expected_prompt_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic capture must not block, while manual loading prompts per side."""
    adapter = FakeAdapter()
    received_prompts: list[object] = []
    received_pacers: list[object] = []
    prompt_calls: list[str] = []
    monkeypatch.setattr(multicam.HikvisionAdapter, "load", lambda: adapter)
    monkeypatch.setattr(multicam, "create_session", lambda *_args: object())
    monkeypatch.setattr("builtins.input", prompt_calls.append)

    def record_group(
        *_args: object,
        prompt: object = None,
        pacer: object = None,
        **_kwargs: object,
    ) -> list[bool]:
        received_prompts.append(prompt)
        received_pacers.append(pacer)
        if callable(prompt):
            prompt("front")
            prompt("back")
        return [True]

    monkeypatch.setattr(multicam, "capture_group", record_group)
    argv = ["--label", "normal", "--hdr", "--group-count", "2"]
    if manual_load:
        argv.append("--manual-load")

    assert multicam.main(argv) == 0
    assert len(received_prompts) == 2
    assert isinstance(received_pacers[0], multicam.GroupedTriggerPacer)
    assert received_pacers[1] is received_pacers[0]
    assert all(callable(prompt) if manual_load else prompt is None for prompt in received_prompts)
    assert len(prompt_calls) == expected_prompt_count


def test_main_cleans_up_cameras_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator interrupt should return cleanly through the camera context."""
    adapter = FakeAdapter()
    monkeypatch.setattr(multicam.HikvisionAdapter, "load", lambda: adapter)
    monkeypatch.setattr(multicam, "create_session", lambda *_args: object())

    def interrupt(*_args: object, **_kwargs: object) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(multicam, "capture_group", interrupt)
    assert multicam.main(["--label", "normal", "--hdr"]) == 130
    assert adapter.events.count("open:0") == 1
    assert adapter.events.count("open:1") == 1
    assert adapter.events.count("open:2") == 1
    assert adapter.events[-12:] == [
        "stop:2",
        "restore:2",
        "close:2",
        "destroy:2",
        "stop:1",
        "restore:1",
        "close:1",
        "destroy:1",
        "stop:0",
        "restore:0",
        "close:0",
        "destroy:0",
    ]
