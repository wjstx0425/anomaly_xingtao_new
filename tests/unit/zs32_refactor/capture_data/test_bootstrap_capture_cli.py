# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the isolated four-camera bootstrap capture command."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from zs32_inspection.capture import (
    CaptureFrame,
    IncompleteCaptureError,
    PartialRoundCaptureError,
    RoundConfirmation,
)
from zs32_inspection.config.loaders import load_topology

from zs32_inspection.cli import bootstrap_capture


REPOSITORY_ROOT = Path(__file__).parents[4]
TOPOLOGY = REPOSITORY_ROOT / "configs/zs32/topology/zs32_4cam_double_side_v1.json"


class _Coordinator:
    instances: list[_Coordinator] = []

    def __init__(self, operator_id: str, timeout_seconds: float, *, manual_load: bool) -> None:
        self.operator_id = operator_id
        self.timeout_seconds = timeout_seconds
        self.manual_load = manual_load
        self.rounds: list[tuple[str, int, int]] = []
        self.__class__.instances.append(self)

    def confirm_round(self, request, round_plan, *, round_index, round_count):
        self.rounds.append((round_plan.round_id, round_index, round_count))
        return RoundConfirmation(
            round_plan.round_id,
            round_plan.prompt,
            self.operator_id,
            "2026-07-14T00:00:00Z",
            "2026-07-14T00:00:01Z",
        )


class _Adapter:
    instances: list[_Adapter] = []
    failed_round: str | None = None
    interrupt_round: str | None = None

    def __init__(self, config) -> None:
        self.config = config
        self.rounds: list[str] = []
        self.exited = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.exited = True
        return None

    def capture_round(self, request, round_plan, cameras):
        self.rounds.append(round_plan.round_id)
        frames = tuple(
            CaptureFrame(
                round_id=round_plan.round_id,
                view_id=camera.views[round_plan.round_id],
                camera_slot_id=camera.slot_id,
                camera_serial=camera.serial,
                device_index=index,
                image_bytes=(
                    b"\x89PNG\r\n\x1a\n"
                    + f"{round_plan.round_id}:{camera.slot_id}".encode()
                ),
                width=4024,
                height=3036,
                capture_mode="hdr_fused",
                exposure=None,
                gain=0.0,
                captured_at="2026-07-14T00:00:02Z",
                capture_parameters={"short_exposure": 1500.0, "long_exposure": 5500.0},
            )
            for index, camera in enumerate(cameras)
        )
        if round_plan.round_id == self.failed_round:
            raise PartialRoundCaptureError("camera timeout", frames[:1])
        if round_plan.round_id == self.interrupt_round:
            raise KeyboardInterrupt
        return frames


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    _Coordinator.instances.clear()
    _Adapter.instances.clear()
    _Adapter.failed_round = None
    _Adapter.interrupt_round = None
    monkeypatch.setattr(bootstrap_capture, "BootstrapRoundCoordinator", _Coordinator)
    monkeypatch.setattr(bootstrap_capture, "HikvisionCameraAdapter", _Adapter)


def _argv(output_root: Path) -> list[str]:
    return [
        "--topology",
        str(TOPOLOGY),
        "--root",
        str(output_root),
        "--capture-session",
        "session-test",
        "--operator-id",
        "operator-test",
        "--hand",
        "left",
        "--label",
        "normal",
        "--part-id",
        "zs32_left_normal_part2",
        "--group-count",
        "1",
        "--images-per-group",
        "1",
        "--manual-load",
        "--hdr",
        "--short-exposure",
        "1500",
        "--long-exposure",
        "5500",
        "--gain",
        "0",
        "--capture-interval",
        "0.2",
        "--hdr-settle-frames",
        "1",
        "--timeout-ms",
        "2000",
    ]


def test_four_camera_bootstrap_writes_eight_bound_pngs_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        bootstrap_capture,
        "command_result",
        lambda command, payload: results.append((command, dict(payload))),
    )

    assert bootstrap_capture._run(_argv(tmp_path)) == 0

    assert _Coordinator.instances[0].rounds == [("front", 1, 2), ("back", 2, 2)]
    assert _Adapter.instances[0].rounds == ["front", "back"]
    assert len(results) == 1
    assert results[0][1]["output_layout"] == "bootstrap"
    assert results[0][1]["manifest_path"] is None
    published = Path(str(results[0][1]["published_paths"][0]))
    assert published.is_relative_to(tmp_path / "_bootstrap")
    assert not (tmp_path / "session-test" / "images").exists()

    image_paths = sorted((published / "images").glob("*.png"))
    assert len(image_paths) == 8
    manifest = json.loads((published / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    topology = load_topology(TOPOLOGY)
    assert manifest["bootstrap_only"] is True
    assert manifest["eligible_for_dataset"] is False
    assert manifest["production_release_allowed"] is False
    assert manifest["topology_sha256"] == topology.topology_sha256
    assert manifest["acquisition_config"]["hdr"] is True
    assert manifest["acquisition_config"]["short_exposure"] == 1500.0
    assert manifest["acquisition_config"]["long_exposure"] == 5500.0

    rows = {row["view_id"]: row for row in manifest["images"]}
    assert set(rows) == set(topology.required_views)
    for view_id, row in rows.items():
        round_id, slot_id, serial = topology.binding_for_view(view_id)
        image_path = published / row["relative_path"]
        assert row["round_id"] == round_id
        assert row["camera_slot_id"] == slot_id
        assert row["camera_serial"] == serial
        assert row["image_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()


def test_four_camera_legacy_layout_writes_historical_tree_and_one_manifest_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _argv(tmp_path)
    argv[argv.index("--capture-session") + 1] = "20260714_120000_000001"
    argv.append("--legacy-layout")
    results: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        bootstrap_capture,
        "command_result",
        lambda command, payload: results.append((command, dict(payload))),
    )

    assert bootstrap_capture._run(argv) == 0

    manifest = tmp_path / "manifests" / "20260714_120000_000001.csv"
    image_paths = sorted(tmp_path.glob("left/*/normal/*/images/*.png"))
    expected = (
        tmp_path
        / "left"
        / "front_secondary"
        / "normal"
        / "20260714_120000_000001"
        / "images"
        / "left_front_secondary_normal_zs32_left_normal_part2_"
        "group001_000001_fused.png"
    )
    assert manifest.is_file()
    assert len(image_paths) == 8
    assert expected.is_file()
    assert not (tmp_path / "_bootstrap").exists()
    assert results == [
        (
            "zs32-bootstrap-capture",
            {
                "bootstrap_only": True,
                "eligible_for_dataset": False,
                "capture_session": "20260714_120000_000001",
                "capture_set_count": 1,
                "topology_id": load_topology(TOPOLOGY).topology_id,
                "topology_sha256": load_topology(TOPOLOGY).topology_sha256,
                "acquisition_config_sha256": results[0][1][
                    "acquisition_config_sha256"
                ],
                "published_paths": [str(manifest)],
                "image_sha256_by_set": results[0][1]["image_sha256_by_set"],
                "output_layout": "legacy",
                "manifest_path": str(manifest),
            },
        )
    ]


def test_legacy_layout_preflight_rejects_conflict_before_hardware_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _argv(tmp_path)
    session = "20260714_120000_000001"
    argv[argv.index("--capture-session") + 1] = session
    argv.append("--legacy-layout")
    manifest = tmp_path / "manifests" / f"{session}.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("conflict", encoding="utf-8")

    class _ForbiddenAdapter:
        def __init__(self, config) -> None:
            raise AssertionError("legacy preflight must run before hardware initialization")

    monkeypatch.setattr(bootstrap_capture, "HikvisionCameraAdapter", _ForbiddenAdapter)

    with pytest.raises(FileExistsError, match="already exists"):
        bootstrap_capture._run(argv)


def test_legacy_payload_keeps_one_manifest_path_for_multiple_capture_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _argv(tmp_path)
    session = "20260714_120000_000001"
    argv[argv.index("--capture-session") + 1] = session
    argv[argv.index("--group-count") + 1] = "2"
    argv.append("--legacy-layout")
    results: list[dict[str, object]] = []
    monkeypatch.setattr(
        bootstrap_capture,
        "command_result",
        lambda command, payload: results.append(dict(payload)),
    )

    assert bootstrap_capture._run(argv) == 0

    manifest = tmp_path / "manifests" / f"{session}.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert results[0]["capture_set_count"] == 2
    assert results[0]["published_paths"] == [str(manifest)]
    assert results[0]["manifest_path"] == str(manifest)
    assert len(list(tmp_path.glob("left/*/normal/*/images/*.png"))) == 16
    assert len(rows) == 18
    assert sum(row["record_type"] == "image" for row in rows) == 16
    assert sum(
        row["record_type"] == "sample" and row["sample_status"] == "complete"
        for row in rows
    ) == 2


def test_default_session_formats_preserve_bootstrap_and_add_historical_legacy_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = datetime(2026, 7, 14, 12, 0, 0, 123456, tzinfo=timezone.utc)

    class _FixedDatetime:
        @staticmethod
        def now() -> datetime:
            return fixed

    monkeypatch.setattr(bootstrap_capture, "datetime", _FixedDatetime)

    assert bootstrap_capture._default_session(legacy_layout=False) == (
        "bootstrap-20260714-120000-123456"
    )
    assert bootstrap_capture._default_session(legacy_layout=True) == (
        "20260714_120000_123456"
    )
    assert re.fullmatch(
        r"\d{8}_\d{6}_\d{6}",
        bootstrap_capture._default_session(legacy_layout=True),
    )


def test_batch_opens_camera_adapter_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _argv(tmp_path)
    argv[argv.index("--group-count") + 1] = "2"
    results: list[dict[str, object]] = []
    monkeypatch.setattr(
        bootstrap_capture,
        "command_result",
        lambda command, payload: results.append(dict(payload)),
    )

    assert bootstrap_capture._run(argv) == 0

    assert len(_Adapter.instances) == 1
    assert _Adapter.instances[0].rounds == ["front", "back", "front", "back"]
    assert results[0]["capture_set_count"] == 2


def test_images_per_group_greater_than_one_is_rejected_before_hardware(
    tmp_path: Path,
) -> None:
    argv = _argv(tmp_path)
    argv[argv.index("--images-per-group") + 1] = "2"

    with pytest.raises(SystemExit) as raised:
        bootstrap_capture._run(argv)

    assert raised.value.code == 2
    assert _Adapter.instances == []


@pytest.mark.parametrize("failed_round", ["front", "back"])
def test_round_failure_never_publishes_bootstrap_complete(
    failed_round: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Adapter.failed_round = failed_round
    emitted = False

    def _unexpected_result(command, payload) -> None:
        nonlocal emitted
        emitted = True

    monkeypatch.setattr(bootstrap_capture, "command_result", _unexpected_result)

    with pytest.raises(IncompleteCaptureError) as raised:
        bootstrap_capture._run(_argv(tmp_path))

    assert emitted is False
    assert _Adapter.instances[0].exited is True
    assert not list((tmp_path / "_bootstrap").rglob("bootstrap_manifest.json"))
    assert not (tmp_path / "session-test" / "images").exists()
    failure_path = Path(raised.value.diagnostic_path)
    failure = json.loads((failure_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "incomplete"
    assert failure["bootstrap_only"] is True
    assert failure["failed_round"] == failed_round


def test_keyboard_interrupt_retains_partial_data_and_closes_adapter(
    tmp_path: Path,
) -> None:
    _Adapter.interrupt_round = "back"

    with pytest.raises(IncompleteCaptureError) as raised:
        bootstrap_capture._run(_argv(tmp_path))

    assert _Adapter.instances[0].exited is True
    failure_path = Path(raised.value.diagnostic_path)
    failure = json.loads((failure_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["failed_round"] == "back"
    assert len(failure["partial_images"]) == 4


def test_help_does_not_touch_camera(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ForbiddenAdapter:
        def __init__(self, config) -> None:
            raise AssertionError("--help must not initialize hardware")

    monkeypatch.setattr(bootstrap_capture, "HikvisionCameraAdapter", _ForbiddenAdapter)
    with pytest.raises(SystemExit) as raised:
        bootstrap_capture.main(["--help"])

    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--topology" in output
    assert "--group-count" in output
    assert "--short-exposure" in output
