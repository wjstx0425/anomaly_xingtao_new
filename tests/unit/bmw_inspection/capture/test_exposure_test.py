"""Exposure brackets, replay integrity and camera cleanup without hardware."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bmw_inspection.capture.exposure_fusion import fuse_exposures
from bmw_inspection.capture.hardware import CameraHandle, DeviceDescription, HikvisionAdapter
from bmw_inspection.cli import exposure_test as experiment


class FakeAdapter:
    def __init__(self, fail_read: bool = False) -> None:
        self.events = []
        self.exposure = 0.0
        self.fail_read = fail_read
        self.read_count = 0

    def list_devices(self):
        return [DeviceDescription(0, "fake", "DA9805574")]

    def open(self, device, gain):
        self.events.append(("open", gain))
        return CameraHandle(device, None, None, None, False)

    def start(self, handle):
        handle.started = True

    def set_exposure(self, handle, exposure):
        self.exposure = exposure
        self.events.append(("exposure", exposure))

    def get_exposure(self, handle):
        return self.exposure + 0.25

    def trigger(self, handle):
        self.events.append(("trigger", self.exposure))

    def read(self, handle, timeout_ms):
        self.read_count += 1
        if self.fail_read:
            raise RuntimeError("simulated read failure")
        # Make discarded and retained frames visibly different.
        image = np.full((64, 96, 3), int(self.exposure / 50) + self.read_count, dtype=np.uint8)
        image[:, 30:40] = 200
        return image

    def stop(self, handle):
        self.events.append(("stop",))

    def restore_continuous(self, handle):
        self.events.append(("restore",))

    def close(self, handle):
        self.events.append(("close",))

    def destroy(self, handle):
        self.events.append(("destroy",))


def _args(output: Path, *extra):
    return experiment.build_parser().parse_args([
        "--output", str(output), "--serial", "DA9805574", "--no-prompt", *extra,
    ])


def _install(monkeypatch, adapter):
    monkeypatch.setattr(experiment.HikvisionAdapter, "load", lambda: adapter)
    monkeypatch.setattr(experiment.GroupedTriggerPacer, "wait", lambda self: None)


def test_capture_settles_saves_readback_and_matches_existing_fusion(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    _install(monkeypatch, adapter)
    output = experiment.run(_args(tmp_path / "live", "--repeat", "2"))
    session = json.loads((output / "session.json").read_text())
    assert session["status"] == "complete"
    assert len(session["records"]) == 4
    assert adapter.read_count == 8  # Each requested frame follows one discarded frame.
    assert [event[1] for event in adapter.events if event[0] == "exposure"] == [1500, 6000, 1500, 6000]
    assert adapter.events[-4:] == [("stop",), ("restore",), ("close",), ("destroy",)]
    first, second = session["records"][:2]
    assert first["exposure_readback_us"] == 1500.25
    assert experiment._read(output, first)[0, 0, 0] == 32
    assert experiment._read(output, second)[0, 0, 0] == 124
    report = json.loads((output / "report.json").read_text())
    assert len(report["records"]) == 8
    selective_record = next(row for row in report["records"] if row["kind"] == "selective")
    hdr = session["metadata"]["hdr"]
    expected = fuse_exposures(
        [experiment._read(output, first), experiment._read(output, second)], align=False,
        **{key: hdr[key] for key in ("short_dark_threshold", "long_clip_threshold", "blend_width", "blur_size")},
    )
    assert np.array_equal(experiment._read(output, selective_record), expected)
    assert (output / "index.html").is_file()


def test_read_failure_releases_camera_and_marks_session_failed(tmp_path, monkeypatch):
    adapter = FakeAdapter(fail_read=True)
    _install(monkeypatch, adapter)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated read failure"):
        experiment.run(_args(output))
    assert adapter.events[-4:] == [("stop",), ("restore",), ("close",), ("destroy",)]
    assert json.loads((output / "session.json").read_text())["status"] == "failed"


def test_replay_needs_no_sdk_and_preserves_source_pixels(tmp_path, monkeypatch):
    _install(monkeypatch, FakeAdapter())
    source = experiment.run(_args(tmp_path / "source"))
    before = (source / "session.json").read_bytes()
    monkeypatch.setattr(experiment.HikvisionAdapter, "load", lambda: pytest.fail("replay opened SDK"))
    args = experiment.build_parser().parse_args([
        "--replay", str(source / "session.json"), "--output", str(tmp_path / "replayed"),
        "--config", str(tmp_path / "nonexistent.json"),
    ])
    output = experiment.run(args)
    replay = json.loads((output / "session.json").read_text())
    original = json.loads(before)
    assert replay["status"] == "complete"
    for left, right in zip(original["records"], replay["records"], strict=True):
        assert np.array_equal(experiment._read(source, left), experiment._read(output, right))
    assert (source / "session.json").read_bytes() == before


def test_replay_rejects_tampered_source(tmp_path, monkeypatch):
    _install(monkeypatch, FakeAdapter())
    source = experiment.run(_args(tmp_path / "source"))
    session = json.loads((source / "session.json").read_text())
    (source / session["records"][0]["path"]).write_bytes(b"tampered")
    args = experiment.build_parser().parse_args([
        "--replay", str(source / "session.json"), "--output", str(tmp_path / "replayed"),
    ])
    with pytest.raises(ValueError, match="checksum"):
        experiment.run(args)
    assert json.loads((tmp_path / "replayed/session.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("values", [[0, 1], [1, 1], [2, 1], [1], [1, float("nan")], [1, float("inf")]])
def test_invalid_exposures(values):
    with pytest.raises(ValueError):
        experiment._exposures(values)


@pytest.mark.parametrize("pairs", [["1:3"], ["2:1"], ["1:2:3"], ["1:2", "1:2"]])
def test_invalid_pairs(pairs):
    with pytest.raises(ValueError):
        experiment._pairs(pairs, [1, 2])


def test_existing_output_rejected_before_sdk_load(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment.HikvisionAdapter, "load", lambda: pytest.fail("opened SDK"))
    with pytest.raises(FileExistsError):
        experiment.run(_args(tmp_path))


def test_missing_serial_rejected_without_opening(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    _install(monkeypatch, adapter)
    with pytest.raises(RuntimeError, match="missing cameras"):
        experiment.run(_args(tmp_path / "out", "--serial", "DB0998274"))
    assert adapter.events == []


def test_exposure_readback_checks_sdk_error():
    class Camera:
        def MV_CC_GetFloatValue(self, key, value):
            assert key == "ExposureTime"
            value.fCurValue = 1500.125
            return 0

    adapter = HikvisionAdapter(SimpleNamespace(MVCC_FLOATVALUE=SimpleNamespace))
    handle = CameraHandle(DeviceDescription(0, "fake", "serial"), Camera(), None, None, False)
    assert adapter.get_exposure(handle) == 1500.125
    handle.cam.MV_CC_GetFloatValue = lambda *_: 1
    with pytest.raises(RuntimeError, match="GetFloatValue"):
        adapter.get_exposure(handle)


def test_unconfigured_serial_can_capture_with_generic_view(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    adapter.list_devices = lambda: [DeviceDescription(0, "new-camera", "DB1624062")]
    _install(monkeypatch, adapter)
    output = experiment.run(_args(tmp_path / "new", "--serial", "DB1624062", "--exposures-us", "600", "750"))
    session = json.loads((output / "session.json").read_text())
    assert session["status"] == "complete"
    assert len(session["records"]) == 2
    assert all(row["camera_serial"] == "DB1624062" for row in session["records"])
    assert all(row["view"] == "camera_01_front" for row in session["records"])
    assert session["metadata"]["selected_slots"][0]["serial"] == "DB1624062"


def test_duplicate_serial_rejected_before_sdk_load(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment.HikvisionAdapter, "load", lambda: pytest.fail("opened SDK"))
    with pytest.raises(ValueError, match="distinct"):
        experiment.run(_args(tmp_path / "duplicate", "--serial", "DB1624062", "DB1624062"))
