# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Isolated topology-driven capture used to bootstrap four-camera gate assets.

Bootstrap captures deliberately bypass quality and registration gates because
those assets may not exist yet.  Their output lives outside the canonical raw
capture layout and is explicitly marked ineligible for dataset/release use.
"""

from __future__ import annotations

import select
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from zs32_inspection.runtime.publisher import AtomicDirectoryPublisher
from zs32_inspection.dashboard.contracts import ConfirmationCommand, ProgressRecord
from zs32_inspection.dashboard.control import consume_confirmation, write_progress

from .contracts import (
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    CaptureResult,
    CaptureRoundPlan,
    RoundConfirmation,
    ensure_unique_frames,
    utc_now,
)
from .errors import CaptureFailure, CaptureRetakeRequired, CaptureSystemError, InvalidCaptureError
from .service import (
    CaptureSource,
    IncompleteCaptureError,
    PartialRoundCaptureError,
    RoundCoordinator,
)


@dataclass(frozen=True, slots=True)
class BootstrapCaptureMetadata:
    """Auditable non-production context attached to every bootstrap set."""

    label: str
    defect_type: str | None
    acquisition_config: Mapping[str, object]
    acquisition_config_sha256: str


class BootstrapCaptureStore(Protocol):
    """Publication contract shared by isolated and legacy capture stores."""

    def publish_complete(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
        confirmations: Sequence[RoundConfirmation],
        metadata: BootstrapCaptureMetadata,
        *,
        started_at: str,
    ) -> Path: ...

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
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class BootstrapRoundCoordinator:
    """Fast Enter-based operator confirmation for front/back bootstrap rounds."""

    operator_id: str
    timeout_seconds: float = 120.0
    manual_load: bool = False

    def __post_init__(self) -> None:
        if not self.operator_id or any(character.isspace() for character in self.operator_id):
            raise ValueError("operator_id must be a non-empty identifier without whitespace")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("round confirmation timeout_seconds must be positive")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

    def confirm_round(
        self,
        request: CaptureRequest,
        round_plan: CaptureRoundPlan,
        *,
        round_index: int,
        round_count: int,
    ) -> RoundConfirmation:
        """Wait for Enter immediately before one physical capture round."""
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise CaptureSystemError("bootstrap round confirmation requires a foreground TTY")
        action = (
            "上料并摆好正面"
            if round_index == 1 and self.manual_load
            else round_plan.prompt
        )
        prompted_at = utc_now()
        print(
            f"\n[{round_index}/{round_count}] part={request.part_instance_id} "
            f"hand={request.hand}: {action}\n"
            "确认位置后按 Enter 开始采集；输入 q 取消"
            f"（{self.timeout_seconds:g} 秒超时）：",
            end="",
            flush=True,
        )
        readable, _, _ = select.select([sys.stdin], [], [], self.timeout_seconds)
        if not readable:
            raise CaptureRetakeRequired(
                f"operator confirmation timed out for round {round_plan.round_id!r}"
            )
        response = sys.stdin.readline()
        if response == "":
            raise CaptureRetakeRequired(
                f"operator confirmation input closed for round {round_plan.round_id!r}"
            )
        if response.strip().lower() in {"q", "quit", "cancel"}:
            raise CaptureRetakeRequired(
                f"operator cancelled round {round_plan.round_id!r}"
            )
        if response.strip():
            raise CaptureRetakeRequired(
                f"press Enter only to confirm round {round_plan.round_id!r}, or q to cancel"
            )
        return RoundConfirmation(
            round_id=round_plan.round_id,
            prompt=round_plan.prompt,
            confirmed_by=self.operator_id,
            prompted_at=prompted_at,
            confirmed_at=utc_now(),
        )


class DashboardRoundCoordinator:
    """File-backed round confirmation for a dashboard-owned headless capture."""

    def __init__(
        self,
        operator_id: str,
        progress_path: Path,
        control_path: Path,
        *,
        timeout_seconds: float = 120.0,
        poll_interval: float = 0.05,
    ) -> None:
        self.operator_id = operator_id
        self.progress_path = progress_path
        self.control_path = control_path
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self._expected: ConfirmationCommand | None = None

    def publish_waiting(self, request: CaptureRequest, round_plan: CaptureRoundPlan) -> None:
        confirmation_id = uuid.uuid4().hex
        self._expected = ConfirmationCommand(
            "confirm_round", round_plan.round_id, confirmation_id, request.part_instance_id
        )
        write_progress(
            self.progress_path,
            ProgressRecord(
                request.part_instance_id,
                request.capture_session,
                f"waiting_{round_plan.round_id}",
                round_plan.prompt,
                utc_now(),
                confirmation_id,
            ),
        )

    def consume_current_confirmation(
        self, request: CaptureRequest, round_plan: CaptureRoundPlan
    ) -> bool:
        expected = self._expected
        return bool(
            expected
            and expected.part_id == request.part_instance_id
            and expected.round == round_plan.round_id
            and consume_confirmation(self.control_path, expected)
        )

    def confirm_round(
        self,
        request: CaptureRequest,
        round_plan: CaptureRoundPlan,
        *,
        round_index: int,
        round_count: int,
    ) -> RoundConfirmation:
        del round_index, round_count
        if self._expected is None:
            self.publish_waiting(request, round_plan)
        prompted_at = utc_now()
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            if self.consume_current_confirmation(request, round_plan):
                self._expected = None
                write_progress(
                    self.progress_path,
                    ProgressRecord(
                        request.part_instance_id,
                        request.capture_session,
                        f"capturing_{round_plan.round_id}",
                        round_plan.prompt,
                        utc_now(),
                    ),
                )
                return RoundConfirmation(
                    round_plan.round_id,
                    round_plan.prompt,
                    self.operator_id,
                    prompted_at,
                    utc_now(),
                )
            time.sleep(self.poll_interval)
        raise CaptureRetakeRequired(
            f"operator confirmation timed out for round {round_plan.round_id!r}"
        )


class AtomicBootstrapCaptureStore:
    """Publish bootstrap sets atomically into isolated complete/incomplete roots."""

    def __init__(self, output_root: Path) -> None:
        expanded = output_root.expanduser()
        if expanded.exists() and (expanded.is_symlink() or not expanded.is_dir()):
            raise ValueError(f"bootstrap output root must be a directory or absent: {expanded}")
        self.output_root = expanded.resolve()

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
        """Atomically publish one complete, explicitly non-production capture set."""
        destination_root = self.output_root / "_bootstrap" / request.capture_session
        image_rows = [_image_row(frame) for frame in frames]
        manifest = _base_manifest(request, plan, metadata)
        manifest.update(
            {
                "status": "complete",
                "started_at": started_at,
                "completed_at": utc_now(),
                "round_confirmations": [_confirmation_row(item) for item in confirmations],
                "images": image_rows,
            }
        )
        required = frozenset(
            {"bootstrap_manifest.json"}
            | {str(row["relative_path"]) for row in image_rows}
        )
        with AtomicDirectoryPublisher(destination_root, request.capture_set_id) as publisher:
            for frame, row in zip(frames, image_rows, strict=True):
                publisher.write_bytes(str(row["relative_path"]), frame.image_bytes)
            publisher.write_json("bootstrap_manifest.json", manifest)
            return publisher.finalize(
                validator=lambda root: _validate_complete_staging(root, plan),
                required_paths=required,
            )

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
        """Retain partial evidence separately without creating a complete manifest."""
        destination_root = self.output_root / "_bootstrap_incomplete" / request.capture_session
        partial_rows: list[dict[str, object]] = []
        with AtomicDirectoryPublisher(destination_root, request.capture_set_id) as publisher:
            for index, frame in enumerate(frames, start=1):
                relative_path = f"partial/{index:02d}_{frame.round_id}_{frame.view_id}.png"
                row = _image_row(frame, relative_path=relative_path)
                partial_rows.append(row)
                publisher.write_bytes(relative_path, frame.image_bytes)
            failure = _base_manifest(request, plan, metadata)
            failure.update(
                {
                    "status": "incomplete",
                    "started_at": started_at,
                    "failed_at": utc_now(),
                    "failed_round": failed_round,
                    "failure_kind": failure_kind,
                    "failure_reason": failure_reason,
                    "round_confirmations": [_confirmation_row(item) for item in confirmations],
                    "partial_images": partial_rows,
                }
            )
            publisher.write_json("failure.json", failure)
            required = frozenset(
                {"failure.json"} | {str(row["relative_path"]) for row in partial_rows}
            )
            return publisher.finalize(validator=lambda root: None, required_paths=required)


class BootstrapCaptureService:
    """Capture one topology-defined part without claiming formal gate approval."""

    def __init__(
        self,
        source: CaptureSource,
        store: BootstrapCaptureStore,
        *,
        round_coordinator: RoundCoordinator,
    ) -> None:
        self._source = source
        self._store = store
        self._round_coordinator = round_coordinator

    def capture(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        metadata: BootstrapCaptureMetadata,
    ) -> CaptureResult:
        """Capture all rounds or retain partial evidence in the incomplete root."""
        started_at = utc_now()
        frames: list[CaptureFrame] = []
        confirmations: list[RoundConfirmation] = []
        failed_round = "before_first_round"
        try:
            for round_index, round_plan in enumerate(plan.rounds, start=1):
                failed_round = round_plan.round_id
                try:
                    confirmation = self._round_coordinator.confirm_round(
                        request,
                        round_plan,
                        round_index=round_index,
                        round_count=len(plan.rounds),
                    )
                except CaptureFailure:
                    raise
                except Exception as error:
                    raise CaptureSystemError(
                        f"round confirmation failed for {round_plan.round_id!r}: {error}"
                    ) from error
                if not isinstance(confirmation, RoundConfirmation) or (
                    confirmation.round_id,
                    confirmation.prompt,
                ) != (round_plan.round_id, round_plan.prompt):
                    raise CaptureSystemError(
                        f"round confirmation identity differs for {round_plan.round_id!r}"
                    )
                confirmations.append(confirmation)
                try:
                    round_frames = tuple(
                        self._source.capture_round(request, round_plan, plan.cameras)
                    )
                except PartialRoundCaptureError as error:
                    frames.extend(error.partial_frames)
                    raise
                except Exception as error:
                    raise CaptureSystemError(
                        f"capture source runtime failed in round {round_plan.round_id!r}: {error}"
                    ) from error
                _validate_round(plan, round_plan, round_frames)
                frames.extend(round_frames)
            try:
                ensure_unique_frames(frames)
            except (TypeError, ValueError) as error:
                raise InvalidCaptureError(str(error)) from error
            actual_views = {frame.view_id for frame in frames}
            if actual_views != set(plan.required_views):
                raise InvalidCaptureError(
                    "capture set views do not match topology: "
                    f"missing={sorted(set(plan.required_views) - actual_views)}, "
                    f"unexpected={sorted(actual_views - set(plan.required_views))}"
                )
        except (Exception, KeyboardInterrupt) as error:
            classified = (
                error
                if isinstance(error, CaptureFailure)
                else CaptureSystemError(f"bootstrap capture orchestration failed: {error}")
            )
            try:
                diagnostic = self._store.publish_incomplete(
                    request,
                    plan,
                    frames,
                    confirmations,
                    metadata,
                    started_at=started_at,
                    failed_round=failed_round,
                    failure_kind=classified.failure_kind,
                    failure_reason=str(classified),
                )
            except Exception as publication_error:
                raise CaptureSystemError(
                    "bootstrap capture failed and diagnostic publication also failed: "
                    f"original={classified.failure_kind}:{classified}; "
                    f"publication={publication_error}"
                ) from publication_error
            raise IncompleteCaptureError(
                request.capture_set_id,
                classified.failure_kind,
                str(classified),
                str(diagnostic),
            ) from classified

        published = self._store.publish_complete(
            request,
            plan,
            frames,
            confirmations,
            metadata,
            started_at=started_at,
        )
        return CaptureResult(
            request.capture_session,
            request.capture_set_id,
            request.part_instance_id,
            str(published),
            {frame.view_id: frame.image_sha256 for frame in frames},
        )


def _validate_round(
    plan: CapturePlan,
    round_plan: CaptureRoundPlan,
    frames: Sequence[CaptureFrame],
) -> None:
    expected = {
        camera.slot_id: (camera.serial, camera.views[round_plan.round_id])
        for camera in plan.cameras
    }
    if len(frames) != len(expected):
        raise InvalidCaptureError(
            f"round {round_plan.round_id!r} returned {len(frames)} frames; expected {len(expected)}"
        )
    observed: set[str] = set()
    for frame in frames:
        if frame.round_id != round_plan.round_id:
            raise InvalidCaptureError(
                f"frame round identity mismatch: expected {round_plan.round_id!r}, "
                f"got {frame.round_id!r}"
            )
        if frame.camera_slot_id in observed:
            raise InvalidCaptureError(
                f"round {round_plan.round_id!r} has duplicate camera slot "
                f"{frame.camera_slot_id!r}"
            )
        observed.add(frame.camera_slot_id)
        try:
            expected_serial, expected_view = expected[frame.camera_slot_id]
        except KeyError as error:
            raise InvalidCaptureError(
                f"unexpected camera slot: {frame.camera_slot_id!r}"
            ) from error
        if frame.camera_serial != expected_serial or frame.view_id != expected_view:
            raise InvalidCaptureError(
                f"round/slot binding mismatch for {round_plan.round_id!r}/"
                f"{frame.camera_slot_id!r}: expected serial/view "
                f"{expected_serial!r}/{expected_view!r}, got "
                f"{frame.camera_serial!r}/{frame.view_id!r}"
            )
    if observed != set(expected):
        raise InvalidCaptureError(
            f"round {round_plan.round_id!r} camera slots differ from topology"
        )


def _base_manifest(
    request: CaptureRequest,
    plan: CapturePlan,
    metadata: BootstrapCaptureMetadata,
) -> dict[str, object]:
    return {
        "schema": "zs32.bootstrap_capture",
        "schema_version": 1,
        "purpose": "gate_bootstrap",
        "bootstrap_only": True,
        "eligible_for_dataset": False,
        "production_release_allowed": False,
        "capture_session": request.capture_session,
        "capture_set_id": request.capture_set_id,
        "part_instance_id": request.part_instance_id,
        "hand": request.hand,
        "label": metadata.label,
        "defect_type": metadata.defect_type,
        "topology_id": plan.topology_id,
        "topology_sha256": plan.topology_sha256,
        "required_views": list(plan.required_views),
        "acquisition_config": dict(metadata.acquisition_config),
        "acquisition_config_sha256": metadata.acquisition_config_sha256,
    }


def _image_row(
    frame: CaptureFrame,
    *,
    relative_path: str | None = None,
) -> dict[str, object]:
    return {
        "round_id": frame.round_id,
        "view_id": frame.view_id,
        "camera_slot_id": frame.camera_slot_id,
        "camera_serial": frame.camera_serial,
        "device_index": frame.device_index,
        "relative_path": relative_path or f"images/{frame.view_id}.png",
        "image_sha256": frame.image_sha256,
        "width": frame.width,
        "height": frame.height,
        "media_type": frame.media_type,
        "capture_mode": frame.capture_mode,
        "exposure": frame.exposure,
        "gain": frame.gain,
        "captured_at": frame.captured_at,
        "capture_parameters": dict(frame.capture_parameters),
    }


def _confirmation_row(item: RoundConfirmation) -> dict[str, str]:
    return {
        "round_id": item.round_id,
        "prompt": item.prompt,
        "confirmed_by": item.confirmed_by,
        "prompted_at": item.prompted_at,
        "confirmed_at": item.confirmed_at,
    }


def _validate_complete_staging(root: Path, plan: CapturePlan) -> None:
    import json

    payload = json.loads((root / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    if payload.get("bootstrap_only") is not True or payload.get("status") != "complete":
        raise ValueError("bootstrap manifest is missing complete/non-production markers")
    rows = payload.get("images")
    if not isinstance(rows, list) or {row.get("view_id") for row in rows} != set(
        plan.required_views
    ):
        raise ValueError("bootstrap manifest images differ from topology required views")


__all__ = [
    "AtomicBootstrapCaptureStore",
    "BootstrapCaptureMetadata",
    "BootstrapCaptureService",
    "BootstrapRoundCoordinator",
]
