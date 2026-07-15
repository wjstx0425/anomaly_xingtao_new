# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Historical per-view dataset publication for four-camera ZS32 captures."""

from __future__ import annotations

import csv
import ctypes
import io
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .bootstrap import BootstrapCaptureMetadata
from .contracts import (
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    RoundConfirmation,
    require_identifier,
)


LEGACY_MANIFEST_COLUMNS = (
    "record_type",
    "session_id",
    "sample_id",
    "group_id",
    "image_index",
    "round",
    "view",
    "device_index",
    "camera_serial",
    "capture_mode",
    "exposure",
    "gain",
    "file",
    "source_short",
    "source_long",
    "short_exposure",
    "long_exposure",
    "hdr_attempt",
    "fused_clip_pct",
    "captured_at",
    "sample_status",
    "failed_round",
    "failed_view",
    "failed_device_index",
    "error",
)

_GROUP_ID = re.compile(r"group[0-9]{3,}")
_IMAGE_INDEX = re.compile(r"[0-9]{6,}")


@dataclass(frozen=True, slots=True)
class LegacyCaptureContext:
    """Immutable dataset label and filename identity for one capture batch."""

    label: str
    defect_type: str | None
    part_id: str

    def __post_init__(self) -> None:
        if self.label not in {"normal", "defect"}:
            raise ValueError("legacy label must be 'normal' or 'defect'")
        object.__setattr__(self, "part_id", require_identifier(self.part_id, "part_id"))
        if self.label == "normal":
            if self.defect_type is not None:
                raise ValueError("normal legacy captures must not define defect_type")
            return
        if self.defect_type is None:
            raise ValueError("defect legacy captures require defect_type")
        object.__setattr__(
            self,
            "defect_type",
            require_identifier(self.defect_type, "defect_type"),
        )


def _request_ids(request: CaptureRequest, context: LegacyCaptureContext) -> tuple[str, int]:
    part_prefix = f"{context.part_id}_"
    if not request.part_instance_id.startswith(part_prefix):
        raise ValueError(
            "capture request part_instance_id must use the configured part_id prefix: "
            f"{request.part_instance_id!r}"
        )
    group_id = request.part_instance_id[len(part_prefix) :]
    if _GROUP_ID.fullmatch(group_id) is None:
        raise ValueError(
            "capture request part_instance_id must end in a structured group ID: "
            f"{request.part_instance_id!r}"
        )
    capture_prefix = f"{request.part_instance_id}_"
    if not request.capture_set_id.startswith(capture_prefix):
        raise ValueError(
            "capture request capture_set_id must extend part_instance_id: "
            f"{request.capture_set_id!r}"
        )
    image_token = request.capture_set_id[len(capture_prefix) :]
    if _IMAGE_INDEX.fullmatch(image_token) is None or int(image_token) < 1:
        raise ValueError(
            "capture request capture_set_id must end in a positive zero-padded image index: "
            f"{request.capture_set_id!r}"
        )
    return group_id, int(image_token)


def _label_parts(context: LegacyCaptureContext) -> tuple[str, ...]:
    if context.label == "normal":
        return ("normal",)
    assert context.defect_type is not None
    return ("defect", context.defect_type)


def _filename_label(context: LegacyCaptureContext) -> str:
    if context.label == "normal":
        return "normal"
    assert context.defect_type is not None
    return f"defect_{context.defect_type}"


def _image_destination(
    root: Path,
    request: CaptureRequest,
    view_id: str,
    context: LegacyCaptureContext,
    *,
    kind: str,
) -> Path:
    group_id, image_index = _request_ids(request, context)
    directory = root / request.hand / view_id
    for part in _label_parts(context):
        directory /= part
    directory = directory / request.capture_session / "images"
    filename = (
        f"{request.hand}_{view_id}_{_filename_label(context)}_{context.part_id}_"
        f"{group_id}_{image_index:06d}_{kind}.png"
    )
    return directory / filename


def legacy_image_path(
    root: Path,
    request: CaptureRequest,
    frame: CaptureFrame,
    context: LegacyCaptureContext,
) -> Path:
    """Return the historical destination derived from structured capture identities."""
    kind = "fused" if frame.capture_mode == "hdr_fused" else "single"
    return _image_destination(root, request, frame.view_id, context, kind=kind)


def _string(value: object | None) -> str:
    return "" if value is None else str(value)


def legacy_image_row(
    request: CaptureRequest,
    frame: CaptureFrame,
    context: LegacyCaptureContext,
    destination: Path,
    *,
    sample_status: str,
) -> dict[str, str]:
    """Map one encoded frame to the exact historical 25-column CSV contract."""
    if sample_status not in {"complete", "incomplete"}:
        raise ValueError("sample_status must be 'complete' or 'incomplete'")
    group_id, image_index = _request_ids(request, context)
    parameters = frame.capture_parameters
    return {
        "record_type": "image",
        "session_id": request.capture_session,
        "sample_id": request.capture_set_id,
        "group_id": group_id,
        "image_index": str(image_index),
        "round": frame.round_id,
        "view": frame.view_id,
        "device_index": _string(frame.device_index),
        "camera_serial": frame.camera_serial,
        "capture_mode": frame.capture_mode,
        "exposure": _string(frame.exposure),
        "gain": _string(frame.gain),
        "file": str(destination),
        "source_short": "",
        "source_long": "",
        "short_exposure": _string(parameters.get("short_exposure", "")),
        "long_exposure": _string(parameters.get("long_exposure", "")),
        "hdr_attempt": _string(parameters.get("hdr_attempt", "")),
        "fused_clip_pct": _string(parameters.get("fused_clip_pct", "")),
        "captured_at": frame.captured_at,
        "sample_status": sample_status,
        "failed_round": "",
        "failed_view": "",
        "failed_device_index": "",
        "error": "",
    }


def _temporary_image_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.tmp{destination.suffix}")


def _manifest_path(root: Path, session_id: str) -> Path:
    return root / "manifests" / f"{session_id}.csv"


def _temporary_manifest_path(manifest: Path) -> Path:
    return manifest.with_name(f".{manifest.stem}.tmp{manifest.suffix}")


def _csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=LEGACY_MANIFEST_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        os.close(descriptor)
        descriptor = -1
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if sys.platform != "linux":
        raise RuntimeError("legacy atomic no-replace publication requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Linux libc does not expose renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error_number = ctypes.get_errno()
        raise FileExistsError(
            error_number,
            f"atomic legacy image publication failed: {os.strerror(error_number)}",
            destination,
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_plan(plan: CapturePlan) -> None:
    if plan.expected_view_count != 8:
        raise ValueError(
            "legacy four-camera publication requires exactly eight topology views; "
            f"got {plan.expected_view_count}"
        )


def _validate_frames(
    plan: CapturePlan,
    frames: Sequence[CaptureFrame],
    *,
    complete: bool,
) -> None:
    _validate_plan(plan)
    observed: set[str] = set()
    camera_by_slot = {camera.slot_id: camera for camera in plan.cameras}
    for frame in frames:
        if frame.view_id in observed:
            raise ValueError(f"duplicate legacy frame view: {frame.view_id!r}")
        observed.add(frame.view_id)
        try:
            camera = camera_by_slot[frame.camera_slot_id]
            expected_view = camera.views[frame.round_id]
        except KeyError as error:
            raise ValueError(
                f"legacy frame is not bound by the capture plan: {frame.round_id!r}/"
                f"{frame.camera_slot_id!r}"
            ) from error
        if frame.camera_serial != camera.serial or frame.view_id != expected_view:
            raise ValueError(
                "legacy frame serial/view identity differs from the capture plan: "
                f"{frame.camera_serial!r}/{frame.view_id!r}"
            )
    unexpected = observed - set(plan.required_views)
    if unexpected:
        raise ValueError(f"legacy frames contain unexpected views: {sorted(unexpected)}")
    if complete and (len(frames) != 8 or observed != set(plan.required_views)):
        raise ValueError(
            "complete legacy sample requires exactly eight topology-bound views; "
            f"got {sorted(observed)}"
        )


def _summary_row(
    request: CaptureRequest,
    context: LegacyCaptureContext,
    frames: Sequence[CaptureFrame],
    metadata: BootstrapCaptureMetadata,
    *,
    sample_status: str,
    failed_round: str = "",
    failed_view: str = "",
    failed_device_index: str = "",
    error: str = "",
) -> dict[str, str]:
    group_id, image_index = _request_ids(request, context)
    if frames:
        modes = {frame.capture_mode for frame in frames}
        capture_mode = next(iter(modes)) if len(modes) == 1 else "mixed"
    else:
        capture_mode = "hdr_fused" if metadata.acquisition_config.get("hdr") is True else "single"
    return {
        **dict.fromkeys(LEGACY_MANIFEST_COLUMNS, ""),
        "record_type": "sample",
        "session_id": request.capture_session,
        "sample_id": request.capture_set_id,
        "group_id": group_id,
        "image_index": str(image_index),
        "capture_mode": capture_mode,
        "sample_status": sample_status,
        "failed_round": failed_round,
        "failed_view": failed_view,
        "failed_device_index": failed_device_index,
        "error": error,
    }


def _failure_identity(
    plan: CapturePlan,
    frames: Sequence[CaptureFrame],
    *,
    failed_round: str,
    failure_reason: str,
) -> tuple[str, str]:
    view = ""
    device_index = ""
    for candidate in sorted(plan.required_views, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_])", failure_reason):
            view = candidate
            break
    device_match = re.search(
        r"(?:device(?:_index)?|camera)\s*[=:]?\s*(\d+)",
        failure_reason,
        flags=re.IGNORECASE,
    )
    if device_match is not None:
        device_index = device_match.group(1)

    serial_by_device = {
        str(frame.device_index): frame.camera_serial
        for frame in frames
        if frame.device_index is not None
    }
    serial = serial_by_device.get(device_index)
    if serial is None:
        for camera in plan.cameras:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(camera.serial)}(?![A-Za-z0-9])", failure_reason):
                serial = camera.serial
                break
    camera_by_serial = {camera.serial: camera for camera in plan.cameras}
    if serial is not None and failed_round in {item.round_id for item in plan.rounds}:
        camera = camera_by_serial.get(serial)
        if camera is not None:
            view = camera.views[failed_round]
            if not device_index:
                device_index = next(
                    (
                        str(frame.device_index)
                        for frame in frames
                        if frame.camera_serial == serial and frame.device_index is not None
                    ),
                    "",
                )

    if not view and failed_round in {item.round_id for item in plan.rounds}:
        observed_serials = {
            frame.camera_serial for frame in frames if frame.round_id == failed_round
        }
        missing = [camera for camera in plan.cameras if camera.serial not in observed_serials]
        if len(missing) == 1:
            camera = missing[0]
            view = camera.views[failed_round]
            device_index = next(
                (
                    str(frame.device_index)
                    for frame in frames
                    if frame.camera_serial == camera.serial and frame.device_index is not None
                ),
                device_index,
            )
    return view, device_index


class LegacyDatasetCaptureStore:
    """Publish historical per-view PNGs and one append-only session manifest."""

    def __init__(self, root: Path, context: LegacyCaptureContext) -> None:
        if not isinstance(context, LegacyCaptureContext):
            raise TypeError("context must be LegacyCaptureContext")
        self.root = root.expanduser().absolute()
        self.context = context
        self._manifest_path: Path | None = None
        self._owned_manifest_sessions: set[str] = set()

    @property
    def manifest_path(self) -> Path:
        """Return the single session manifest selected by preflight/publication."""
        if self._manifest_path is None:
            raise RuntimeError("legacy manifest path is unknown before preflight/publication")
        return self._manifest_path

    def preflight(
        self,
        requests: Sequence[CaptureRequest],
        plan: CapturePlan,
    ) -> None:
        """Reject all known batch conflicts without creating filesystem state."""
        _validate_plan(plan)
        if self.root.is_symlink():
            raise ValueError(f"legacy dataset root must not be a symlink: {self.root}")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(f"legacy dataset root must be a directory or absent: {self.root}")
        if not requests:
            raise ValueError("legacy preflight requires at least one capture request")
        sessions = {request.capture_session for request in requests}
        if len(sessions) != 1:
            raise ValueError("legacy capture batch must use exactly one capture session")

        destinations: set[tuple[Path, str]] = set()
        for request in requests:
            _request_ids(request, self.context)
            for view_id in plan.required_views:
                identity = (
                    _image_destination(
                        self.root,
                        request,
                        view_id,
                        self.context,
                        kind="single",
                    ).parent,
                    _image_destination(
                        self.root,
                        request,
                        view_id,
                        self.context,
                        kind="single",
                    ).stem.removesuffix("_single"),
                )
                if identity in destinations:
                    raise FileExistsError(f"duplicate legacy destination: {identity[0] / identity[1]}")
                destinations.add(identity)
                for kind in ("single", "fused"):
                    destination = _image_destination(
                        self.root,
                        request,
                        view_id,
                        self.context,
                        kind=kind,
                    )
                    for candidate in (destination, _temporary_image_path(destination)):
                        if candidate.exists() or candidate.is_symlink():
                            raise FileExistsError(f"legacy publication path already exists: {candidate}")

        session_id = next(iter(sessions))
        manifest = _manifest_path(self.root, session_id)
        for candidate in (manifest, _temporary_manifest_path(manifest)):
            if candidate.exists() or candidate.is_symlink():
                raise FileExistsError(f"legacy publication path already exists: {candidate}")
        self._manifest_path = manifest

    def publish_complete(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
        confirmations: Sequence[RoundConfirmation],
        metadata: BootstrapCaptureMetadata,
        *,
        started_at: str,
    ) -> Path:
        """Publish exactly eight images plus one complete sample row."""
        del confirmations, started_at
        _validate_frames(plan, frames, complete=True)
        rows = [
            legacy_image_row(
                request,
                frame,
                self.context,
                legacy_image_path(self.root, request, frame, self.context),
                sample_status="complete",
            )
            for frame in frames
        ]
        rows.append(
            _summary_row(
                request,
                self.context,
                frames,
                metadata,
                sample_status="complete",
            )
        )
        return self._publish(request, frames, rows)

    def publish_incomplete(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
        confirmations: Sequence[RoundConfirmation],
        metadata: BootstrapCaptureMetadata,
        *,
        started_at: str,
        failed_round: str,
        failure_kind: str,
        failure_reason: str,
    ) -> Path:
        """Publish only available frames and one explicitly incomplete sample row."""
        del confirmations, started_at
        _validate_frames(plan, frames, complete=False)
        failed_view, failed_device_index = _failure_identity(
            plan,
            frames,
            failed_round=failed_round,
            failure_reason=failure_reason,
        )
        rows = [
            legacy_image_row(
                request,
                frame,
                self.context,
                legacy_image_path(self.root, request, frame, self.context),
                sample_status="incomplete",
            )
            for frame in frames
        ]
        rows.append(
            _summary_row(
                request,
                self.context,
                frames,
                metadata,
                sample_status="incomplete",
                failed_round=failed_round,
                failed_view=failed_view,
                failed_device_index=failed_device_index,
                error=failure_reason or failure_kind,
            )
        )
        return self._publish(request, frames, rows)

    def _publish(
        self,
        request: CaptureRequest,
        frames: Sequence[CaptureFrame],
        rows: Sequence[dict[str, str]],
    ) -> Path:
        manifest = _manifest_path(self.root, request.capture_session)
        self._manifest_path = manifest
        manifest_existed = manifest.exists() or manifest.is_symlink()
        if manifest_existed and request.capture_session not in self._owned_manifest_sessions:
            raise FileExistsError(f"legacy manifest already exists: {manifest}")
        previous_manifest = manifest.read_bytes() if manifest_existed else None
        previous_rows = self._read_rows(manifest) if manifest_existed else []
        destinations = [
            legacy_image_path(self.root, request, frame, self.context) for frame in frames
        ]
        if len(destinations) != len(set(destinations)):
            raise ValueError("legacy frames map to duplicate destinations")

        staged: list[Path] = []
        published: list[Path] = []
        manifest_temporary = _temporary_manifest_path(manifest)
        manifest_temporary_created = False
        manifest_replaced = False
        try:
            for destination, frame in zip(destinations, frames, strict=True):
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = _temporary_image_path(destination)
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(f"legacy image already exists: {destination}")
                _write_exclusive(temporary, frame.image_bytes)
                staged.append(temporary)
            for temporary, destination in zip(tuple(staged), destinations, strict=True):
                _rename_noreplace(temporary, destination)
                published.append(destination)
                _fsync_directory(destination.parent)

            manifest.parent.mkdir(parents=True, exist_ok=True)
            _write_exclusive(manifest_temporary, _csv_bytes((*previous_rows, *rows)))
            manifest_temporary_created = True
            if previous_manifest is None:
                _rename_noreplace(manifest_temporary, manifest)
            else:
                os.replace(manifest_temporary, manifest)
            manifest_temporary_created = False
            manifest_replaced = True
            _fsync_directory(manifest.parent)
        except BaseException:
            if manifest_temporary_created:
                manifest_temporary.unlink(missing_ok=True)
            for temporary in staged:
                temporary.unlink(missing_ok=True)
            for destination in published:
                destination.unlink(missing_ok=True)
            if manifest_replaced:
                if previous_manifest is None:
                    manifest.unlink(missing_ok=True)
                else:
                    restore = _temporary_manifest_path(manifest)
                    restore.unlink(missing_ok=True)
                    _write_exclusive(restore, previous_manifest)
                    os.replace(restore, manifest)
            raise

        self._owned_manifest_sessions.add(request.capture_session)
        return manifest

    @staticmethod
    def _read_rows(manifest: Path) -> list[dict[str, str]]:
        if manifest.is_symlink() or not manifest.is_file():
            raise FileExistsError(f"legacy manifest must be a regular file: {manifest}")
        with manifest.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if tuple(reader.fieldnames or ()) != LEGACY_MANIFEST_COLUMNS:
            raise ValueError(f"legacy manifest columns differ from contract: {manifest}")
        return rows


__all__ = [
    "LEGACY_MANIFEST_COLUMNS",
    "LegacyCaptureContext",
    "LegacyDatasetCaptureStore",
    "legacy_image_path",
    "legacy_image_row",
]
