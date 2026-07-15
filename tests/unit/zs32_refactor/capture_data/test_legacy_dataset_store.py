# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the four-camera historical dataset writer."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import zs32_inspection.capture.legacy_dataset as legacy_dataset
from zs32_inspection.capture.bootstrap import BootstrapCaptureMetadata
from zs32_inspection.capture.contracts import (
    CameraBinding,
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    CaptureRoundPlan,
    RoundConfirmation,
)
from zs32_inspection.capture.legacy_dataset import (
    LEGACY_MANIFEST_COLUMNS,
    LegacyCaptureContext,
    LegacyDatasetCaptureStore,
    legacy_image_path,
)
from zs32_inspection.domain.identity import Hand, PartIdentity


SESSION_ID = "20260714_120000_000001"
PART_ID = "part"
PNG = b"\x89PNG\r\n\x1a\n"


def _plan() -> CapturePlan:
    cameras = (
        CameraBinding("center", "SERIAL0", {"front": "front", "back": "back"}),
        CameraBinding(
            "left_oblique",
            "SERIAL1",
            {"front": "front_left", "back": "back_left"},
        ),
        CameraBinding(
            "right_oblique",
            "SERIAL2",
            {"front": "front_right", "back": "back_right"},
        ),
        CameraBinding(
            "front_secondary",
            "SERIAL3",
            {"front": "front_secondary", "back": "back_secondary"},
        ),
    )
    return CapturePlan(
        product="ZS32",
        topology_id="zs32-4cam-test",
        topology_sha256="a" * 64,
        rounds=(
            CaptureRoundPlan("front", "capture front"),
            CaptureRoundPlan("back", "capture back"),
        ),
        cameras=cameras,
        required_views=(
            "front",
            "front_left",
            "front_right",
            "front_secondary",
            "back",
            "back_left",
            "back_right",
            "back_secondary",
        ),
    )


def _request(
    *,
    group_id: str = "group001",
    image_index: int = 1,
    part_id: str = PART_ID,
) -> CaptureRequest:
    part_instance_id = f"{part_id}_{group_id}"
    return CaptureRequest(
        SESSION_ID,
        f"{part_instance_id}_{image_index:06d}",
        PartIdentity(part_instance_id, Hand.LEFT),
    )


def _frames(plan: CapturePlan) -> tuple[CaptureFrame, ...]:
    frames: list[CaptureFrame] = []
    for round_plan in plan.rounds:
        for device_index, camera in enumerate(plan.cameras):
            view = camera.views[round_plan.round_id]
            frames.append(
                CaptureFrame(
                    round_id=round_plan.round_id,
                    view_id=view,
                    camera_slot_id=camera.slot_id,
                    camera_serial=camera.serial,
                    device_index=device_index,
                    image_bytes=PNG + f"{round_plan.round_id}:{camera.slot_id}".encode(),
                    width=4024,
                    height=3036,
                    capture_mode="hdr_fused",
                    exposure=None,
                    gain=0.0,
                    captured_at=f"2026-07-14T12:00:0{len(frames)}Z",
                    capture_parameters={
                        "short_exposure": 1500.0,
                        "long_exposure": 5500.0,
                        "hdr_attempt": 2,
                        "fused_clip_pct": float(device_index),
                    },
                )
            )
    return tuple(frames)


def _confirmations(plan: CapturePlan) -> tuple[RoundConfirmation, ...]:
    return tuple(
        RoundConfirmation(
            item.round_id,
            item.prompt,
            "operator-test",
            "2026-07-14T11:59:58Z",
            "2026-07-14T11:59:59Z",
        )
        for item in plan.rounds
    )


def _metadata(*, label: str = "normal", defect_type: str | None = None) -> BootstrapCaptureMetadata:
    return BootstrapCaptureMetadata(
        label=label,
        defect_type=defect_type,
        acquisition_config={"hdr": True},
        acquisition_config_sha256="b" * 64,
    )


def _read_manifest(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    return tuple(reader.fieldnames or ()), rows


def test_publish_complete_writes_normal_paths_and_exact_nine_row_manifest(tmp_path: Path) -> None:
    plan = _plan()
    request = _request()
    frames = _frames(plan)
    frame_by_view = {frame.view_id: frame for frame in frames}
    store = LegacyDatasetCaptureStore(tmp_path, LegacyCaptureContext("normal", None, PART_ID))

    manifest = store.publish_complete(
        request,
        plan,
        frames,
        _confirmations(plan),
        _metadata(),
        started_at="2026-07-14T11:59:57Z",
    )

    expected = (
        tmp_path
        / "left"
        / "front_secondary"
        / "normal"
        / SESSION_ID
        / "images"
        / "left_front_secondary_normal_part_group001_000001_fused.png"
    )
    assert expected.read_bytes() == frame_by_view["front_secondary"].image_bytes
    fieldnames, rows = _read_manifest(manifest)
    assert manifest == tmp_path / "manifests" / f"{SESSION_ID}.csv"
    assert fieldnames == LEGACY_MANIFEST_COLUMNS
    assert len(rows) == 9
    assert [row["record_type"] for row in rows].count("image") == 8
    assert rows[-1]["record_type"] == "sample"
    assert rows[-1]["sample_status"] == "complete"
    assert {row["sample_status"] for row in rows[:-1]} == {"complete"}
    image_row = next(row for row in rows if row["view"] == "front_secondary")
    assert image_row["group_id"] == "group001"
    assert image_row["image_index"] == "1"
    assert image_row["short_exposure"] == "1500.0"
    assert image_row["long_exposure"] == "5500.0"
    assert image_row["hdr_attempt"] == "2"
    assert image_row["fused_clip_pct"] == "3.0"
    assert image_row["file"] == str(expected)


def test_publish_incomplete_routes_defect_frames_and_never_writes_complete_sample(
    tmp_path: Path,
) -> None:
    plan = _plan()
    request = _request(group_id="group009", image_index=3)
    frames = _frames(plan)[:5]
    store = LegacyDatasetCaptureStore(
        tmp_path,
        LegacyCaptureContext("defect", "less", PART_ID),
    )

    manifest = store.publish_incomplete(
        request,
        plan,
        frames,
        _confirmations(plan)[:2],
        _metadata(label="defect", defect_type="less"),
        started_at="2026-07-14T11:59:57Z",
        failed_round="back",
        failure_kind="system_error",
        failure_reason="capture failed in round back for device 1 serial SERIAL1",
    )

    expected = (
        tmp_path
        / "left"
        / "front_secondary"
        / "defect"
        / "less"
        / SESSION_ID
        / "images"
        / "left_front_secondary_defect_less_part_group009_000003_fused.png"
    )
    assert expected.read_bytes() == frames[3].image_bytes
    fieldnames, rows = _read_manifest(manifest)
    assert fieldnames == LEGACY_MANIFEST_COLUMNS
    assert len(rows) == len(frames) + 1
    assert {row["sample_status"] for row in rows} == {"incomplete"}
    assert not any(
        row["record_type"] == "sample" and row["sample_status"] == "complete"
        for row in rows
    )
    assert rows[-1]["failed_round"] == "back"
    assert rows[-1]["failed_view"] == "back_left"
    assert rows[-1]["failed_device_index"] == "1"
    assert rows[-1]["error"] == "capture failed in round back for device 1 serial SERIAL1"


@pytest.mark.parametrize(
    ("label", "defect_type"),
    [("unknown", None), ("normal", "less"), ("defect", None), ("defect", "../less")],
)
def test_context_rejects_invalid_label_and_defect_type(
    label: str,
    defect_type: str | None,
) -> None:
    with pytest.raises(ValueError):
        LegacyCaptureContext(label, defect_type, PART_ID)


@pytest.mark.parametrize(
    "capture_request",
    [
        _request(part_id="other"),
        CaptureRequest(
            SESSION_ID,
            "part_group001_000001",
            PartIdentity("part_group002", Hand.LEFT),
        ),
        CaptureRequest(
            SESSION_ID,
            "part_group001_image001",
            PartIdentity("part_group001", Hand.LEFT),
        ),
    ],
)
def test_structured_request_ids_must_match_context_prefix_and_each_other(
    tmp_path: Path,
    capture_request: CaptureRequest,
) -> None:
    with pytest.raises(ValueError, match="capture request"):
        legacy_image_path(
            tmp_path,
            capture_request,
            _frames(_plan())[0],
            LegacyCaptureContext("normal", None, PART_ID),
        )


def test_preflight_rejects_duplicate_destinations(tmp_path: Path) -> None:
    store = LegacyDatasetCaptureStore(tmp_path, LegacyCaptureContext("normal", None, PART_ID))

    with pytest.raises(FileExistsError, match="duplicate legacy destination"):
        store.preflight((_request(), _request()), _plan())


@pytest.mark.parametrize("conflict", ["final", "temporary", "manifest"])
def test_preflight_rejects_existing_publication_paths(tmp_path: Path, conflict: str) -> None:
    plan = _plan()
    request = _request()
    context = LegacyCaptureContext("normal", None, PART_ID)
    store = LegacyDatasetCaptureStore(tmp_path, context)
    destination = legacy_image_path(tmp_path, request, _frames(plan)[0], context)
    if conflict == "manifest":
        path = tmp_path / "manifests" / f"{SESSION_ID}.csv"
    elif conflict == "temporary":
        path = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    else:
        path = destination
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"conflict")

    with pytest.raises(FileExistsError, match="already exists"):
        store.preflight((request,), plan)


def test_preflight_rejects_symlinked_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    store = LegacyDatasetCaptureStore(linked_root, LegacyCaptureContext("normal", None, PART_ID))

    with pytest.raises(ValueError, match="symlink"):
        store.preflight((_request(),), _plan())


def test_publish_does_not_remove_a_foreign_manifest_temporary_file(tmp_path: Path) -> None:
    manifest_temporary = tmp_path / "manifests" / f".{SESSION_ID}.tmp.csv"
    manifest_temporary.parent.mkdir(parents=True)
    manifest_temporary.write_bytes(b"foreign temporary")
    store = LegacyDatasetCaptureStore(tmp_path, LegacyCaptureContext("normal", None, PART_ID))

    with pytest.raises(FileExistsError):
        store.publish_complete(
            _request(),
            _plan(),
            _frames(_plan()),
            _confirmations(_plan()),
            _metadata(),
            started_at="2026-07-14T11:59:57Z",
        )

    assert manifest_temporary.read_bytes() == b"foreign temporary"
    assert not list(tmp_path.rglob("*_fused.png"))


def test_publish_cleans_its_partial_temporary_file_when_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LegacyDatasetCaptureStore(tmp_path, LegacyCaptureContext("normal", None, PART_ID))

    def _fail_fsync(_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(legacy_dataset.os, "fsync", _fail_fsync)

    with pytest.raises(OSError, match="fsync failed"):
        store.publish_complete(
            _request(),
            _plan(),
            _frames(_plan()),
            _confirmations(_plan()),
            _metadata(),
            started_at="2026-07-14T11:59:57Z",
        )

    assert not list(tmp_path.rglob("*.png"))
    assert not list(tmp_path.rglob("*.csv"))


def test_first_manifest_publish_never_overwrites_a_racing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LegacyDatasetCaptureStore(tmp_path, LegacyCaptureContext("normal", None, PART_ID))
    manifest = tmp_path / "manifests" / f"{SESSION_ID}.csv"
    manifest_temporary = manifest.with_name(f".{SESSION_ID}.tmp.csv")
    original_write = legacy_dataset._write_exclusive

    def _write_with_manifest_race(path: Path, content: bytes) -> None:
        original_write(path, content)
        if path == manifest_temporary:
            manifest.write_bytes(b"foreign manifest")

    monkeypatch.setattr(legacy_dataset, "_write_exclusive", _write_with_manifest_race)

    with pytest.raises(FileExistsError):
        store.publish_complete(
            _request(),
            _plan(),
            _frames(_plan()),
            _confirmations(_plan()),
            _metadata(),
            started_at="2026-07-14T11:59:57Z",
        )

    assert manifest.read_bytes() == b"foreign manifest"
    assert not list(tmp_path.rglob("*_fused.png"))
