"""Manifest rows and completeness rules for immutable raw capture data."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping, Sequence

from zs32_inspection.domain.identity import CaptureSet, Hand, ViewImage, require_sha256

from .contracts import (
    CaptureFrame,
    CapturePlan,
    CaptureRequest,
    RoundConfirmation,
    ensure_unique_frames,
    require_identifier,
    utc_now,
)
from .gates import CaptureGateResult
from .gate_policy import CaptureGateProvenance


CAPTURE_SET_COLUMNS = (
    "capture_set_id",
    "part_instance_id",
    "product",
    "hand",
    "topology_id",
    "topology_sha256",
    "expected_view_count",
    "captured_view_count",
    "status",
    "started_at",
    "completed_at",
    "failure_reason",
)

IMAGE_COLUMNS = (
    "capture_set_id",
    "part_instance_id",
    "round_id",
    "view_id",
    "camera_slot_id",
    "camera_serial",
    "device_index",
    "relative_path",
    "image_sha256",
    "width",
    "height",
    "capture_mode",
    "exposure",
    "gain",
    "capture_parameters_json",
    "captured_at",
    "status",
    "error",
)


@dataclass(frozen=True, slots=True)
class CaptureSetRow:
    """One row in ``capture_sets.csv``."""

    capture_set_id: str
    part_instance_id: str
    product: str
    hand: str
    topology_id: str
    topology_sha256: str
    expected_view_count: int
    captured_view_count: int
    status: str
    started_at: str
    completed_at: str
    failure_reason: str

    def __post_init__(self) -> None:
        if self.product != "ZS32":
            raise ValueError("capture manifest product must be ZS32")
        Hand.parse(self.hand)
        require_sha256(self.topology_sha256, "topology_sha256")
        if not self.capture_set_id or not self.part_instance_id or not self.topology_id:
            raise ValueError("capture set identity fields must not be empty")
        if self.expected_view_count <= 0 or self.captured_view_count < 0:
            raise ValueError("capture view counts must be positive/non-negative")
        if self.status not in {"complete", "incomplete"}:
            raise ValueError(f"invalid capture set status: {self.status!r}")
        if not self.started_at or not self.completed_at:
            raise ValueError("capture set timestamps must not be empty")
        if self.status == "complete" and (
            self.captured_view_count != self.expected_view_count or self.failure_reason
        ):
            raise ValueError("complete capture set must have exact count and no failure_reason")
        if self.status == "incomplete" and not self.failure_reason:
            raise ValueError("incomplete capture set requires failure_reason")

    def to_csv_row(self) -> dict[str, str]:
        """Serialize values without leaking Python-specific types."""
        return {name: str(value) for name, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class CaptureImageRow:
    """One row in ``images.csv``."""

    capture_set_id: str
    part_instance_id: str
    round_id: str
    view_id: str
    camera_slot_id: str
    camera_serial: str
    device_index: int | None
    relative_path: str
    image_sha256: str
    width: int
    height: int
    capture_mode: str
    exposure: float | None
    gain: float | None
    capture_parameters_json: str
    captured_at: str
    status: str
    error: str

    def __post_init__(self) -> None:
        if not all(
            value
            for value in (
                self.capture_set_id,
                self.part_instance_id,
                self.round_id,
                self.view_id,
                self.camera_slot_id,
                self.camera_serial,
                self.capture_mode,
                self.captured_at,
            )
        ):
            raise ValueError("capture image identity fields must not be empty")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"capture image relative_path is unsafe: {self.relative_path!r}")
        require_sha256(self.image_sha256, "image_sha256")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("capture image dimensions must be positive")
        if self.device_index is not None and self.device_index < 0:
            raise ValueError("capture image device_index must be non-negative")
        for field in ("exposure", "gain"):
            value = getattr(self, field)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"capture image {field} must be finite")
        try:
            parameters = json.loads(self.capture_parameters_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("capture_parameters_json must be valid JSON") from error
        if not isinstance(parameters, dict):
            raise ValueError("capture_parameters_json must contain a JSON object")
        if any(not isinstance(key, str) for key in parameters) or any(
            not isinstance(value, (str, int, float, bool, type(None)))
            for value in parameters.values()
        ):
            raise ValueError("capture_parameters_json values must be JSON scalars")
        for key in parameters:
            require_identifier(key, "capture parameter key")
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in parameters.values()
        ):
            raise ValueError("capture_parameters_json numeric values must be finite")
        canonical_parameters = json.dumps(
            parameters,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical_parameters != self.capture_parameters_json:
            raise ValueError("capture_parameters_json must use canonical JSON serialization")
        if self.status not in {"complete", "incomplete"}:
            raise ValueError(f"invalid capture image status: {self.status!r}")
        if self.status == "complete" and self.error:
            raise ValueError("complete capture image cannot have an error")
        if self.status == "incomplete" and not self.error:
            raise ValueError("incomplete capture image requires an error")

    def to_csv_row(self) -> dict[str, str]:
        """Serialize optional values as empty CSV cells."""
        values = asdict(self)
        return {name: "" if value is None else str(value) for name, value in values.items()}


@dataclass(frozen=True, slots=True)
class CaptureBundle:
    """A complete set ready for atomic publication."""

    capture_session: str
    domain_capture_set: CaptureSet
    capture_set: CaptureSetRow
    images: tuple[CaptureImageRow, ...]
    gate_results: tuple[CaptureGateResult, ...]
    gate_provenance: CaptureGateProvenance
    round_confirmations: tuple[RoundConfirmation, ...]
    payload_by_view: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if not isinstance(self.domain_capture_set, CaptureSet):
            raise TypeError("domain_capture_set must be a domain CaptureSet")
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "gate_results", tuple(self.gate_results))
        object.__setattr__(self, "round_confirmations", tuple(self.round_confirmations))
        if not isinstance(self.gate_provenance, CaptureGateProvenance):
            raise TypeError("capture bundle requires structured gate_provenance")
        object.__setattr__(
            self,
            "payload_by_view",
            MappingProxyType(dict(self.payload_by_view)),
        )

    def validate(self, plan: CapturePlan) -> None:
        """Prove that every required view is present exactly once with matching hashes."""
        if self.capture_set.status != "complete":
            raise ValueError("only a complete capture set may enter the raw complete store")
        if self.capture_set.product != plan.product:
            raise ValueError("capture bundle product does not match capture plan")
        self.domain_capture_set.validate_required_views(plan.required_views)
        if self.domain_capture_set.capture_set_id != self.capture_set.capture_set_id:
            raise ValueError("domain capture_set_id conflicts with CSV manifest")
        if self.domain_capture_set.capture_session_id != self.capture_session:
            raise ValueError("domain capture_session_id conflicts with capture bundle")
        if (
            self.domain_capture_set.part.part_instance_id,
            self.domain_capture_set.part.hand.value,
            self.domain_capture_set.part.product,
        ) != (
            self.capture_set.part_instance_id,
            self.capture_set.hand,
            self.capture_set.product,
        ):
            raise ValueError("domain PartIdentity conflicts with CSV capture manifest")
        if self.capture_set.topology_id != plan.topology_id:
            raise ValueError("capture bundle topology_id does not match capture plan")
        if self.capture_set.topology_sha256 != plan.topology_sha256:
            raise ValueError("capture bundle topology_sha256 does not match capture plan")
        if (
            self.gate_provenance.hand.value != self.capture_set.hand
            or self.gate_provenance.topology_sha256 != plan.topology_sha256
            or set(self.gate_provenance.registration_reference_sha256_by_view)
            != set(plan.required_views)
        ):
            raise ValueError("capture gate provenance differs from hand/topology/required views")
        if len(self.round_confirmations) != len(plan.rounds):
            raise ValueError("capture requires exactly one operator confirmation per round")
        if tuple(item.round_id for item in self.round_confirmations) != tuple(
            item.round_id for item in plan.rounds
        ):
            raise ValueError("round confirmations differ from topology order")
        for confirmation, round_plan in zip(
            self.round_confirmations,
            plan.rounds,
            strict=True,
        ):
            if confirmation.prompt != round_plan.prompt:
                raise ValueError(
                    f"round confirmation prompt differs for {round_plan.round_id!r}"
                )
        if (
            self.domain_capture_set.topology_id,
            self.domain_capture_set.topology_sha256,
        ) != (plan.topology_id, plan.topology_sha256):
            raise ValueError("domain CaptureSet topology conflicts with capture plan")
        if self.capture_set.expected_view_count != plan.expected_view_count:
            raise ValueError("capture bundle expected_view_count does not match topology")
        if self.capture_set.captured_view_count != plan.expected_view_count:
            raise ValueError("complete capture bundle has an incomplete captured_view_count")
        gate_result_by_identity: dict[tuple[str, str], CaptureGateResult] = {}
        for result in self.gate_results:
            if result.view_id is None:
                raise ValueError("complete capture gate evidence must identify one required view")
            identity = (result.gate, result.view_id)
            if identity in gate_result_by_identity:
                raise ValueError(f"duplicate capture gate evidence: {identity}")
            if not result.passed:
                raise ValueError(f"complete capture bundle contains failed gate evidence: {identity}")
            gate_result_by_identity[identity] = result
        expected_gate_identities = {
            (gate, view_id)
            for gate in ("quality", "registration")
            for view_id in plan.required_views
        }
        if set(gate_result_by_identity) != expected_gate_identities:
            raise ValueError(
                "complete capture gate evidence is not exactly quality+registration per view"
            )
        rows_by_view: dict[str, CaptureImageRow] = {}
        for row in self.images:
            if row.status != "complete":
                raise ValueError(f"complete bundle contains non-complete image row: {row.view_id}")
            if row.capture_set_id != self.capture_set.capture_set_id:
                raise ValueError("image row capture_set_id does not match capture set")
            if row.part_instance_id != self.capture_set.part_instance_id:
                raise ValueError("image row part_instance_id does not match capture set")
            if row.view_id in rows_by_view:
                raise ValueError(f"duplicate image row for view: {row.view_id}")
            expected_round = ""
            expected_slot = ""
            expected_serial = ""
            for camera in plan.cameras:
                for round_id, view_id in camera.views.items():
                    if view_id == row.view_id:
                        expected_round = round_id
                        expected_slot = camera.slot_id
                        expected_serial = camera.serial
                        break
            if (
                row.round_id,
                row.camera_slot_id,
                row.camera_serial,
            ) != (expected_round, expected_slot, expected_serial):
                raise ValueError(
                    f"image topology identity mismatch for {row.view_id}: "
                    f"expected {(expected_round, expected_slot, expected_serial)}, got "
                    f"{(row.round_id, row.camera_slot_id, row.camera_serial)}"
                )
            domain_image = self.domain_capture_set.images.get(row.view_id)
            if domain_image is None or (
                domain_image.round_id,
                domain_image.camera_slot_id,
                domain_image.camera_serial,
                domain_image.relative_path,
                domain_image.image_sha256,
                domain_image.width,
                domain_image.height,
            ) != (
                row.round_id,
                row.camera_slot_id,
                row.camera_serial,
                row.relative_path,
                row.image_sha256,
                row.width,
                row.height,
            ):
                raise ValueError(f"domain ViewImage conflicts with CSV row for {row.view_id}")
            rows_by_view[row.view_id] = row
        if set(rows_by_view) != set(plan.required_views):
            raise ValueError(
                "complete capture views differ from topology; "
                f"captured={sorted(rows_by_view)}, required={sorted(plan.required_views)}"
            )
        if set(self.payload_by_view) != set(plan.required_views):
            raise ValueError("capture image payloads differ from required views")
        import hashlib

        for view_id, row in rows_by_view.items():
            payload = self.payload_by_view[view_id]
            if hashlib.sha256(payload).hexdigest() != row.image_sha256:
                raise ValueError(f"image payload hash mismatch for view: {view_id}")
            expected_path = f"images/{self.capture_set.capture_set_id}/{view_id}.png"
            if row.relative_path != expected_path:
                raise ValueError(
                    f"image relative_path must be canonical for {view_id}: {row.relative_path!r}"
                )


def build_capture_rows(
    request: CaptureRequest,
    plan: CapturePlan,
    frames: Sequence[CaptureFrame],
    gate_results: Sequence[CaptureGateResult],
    gate_provenance: CaptureGateProvenance,
    round_confirmations: Sequence[RoundConfirmation],
    *,
    started_at: str,
    completed_at: str | None = None,
) -> CaptureBundle:
    """Build and validate the complete two-table capture manifest."""
    ensure_unique_frames(frames)
    completed_at = completed_at or utc_now()
    by_view = {frame.view_id: frame for frame in frames}
    if set(by_view) != set(plan.required_views):
        raise ValueError(
            f"cannot build complete capture rows: missing={sorted(set(plan.required_views) - set(by_view))}, "
            f"unexpected={sorted(set(by_view) - set(plan.required_views))}"
        )
    for frame in frames:
        expected_view = plan.view_for(frame.round_id, frame.camera_slot_id)
        expected_serial = next(
            camera.serial for camera in plan.cameras if camera.slot_id == frame.camera_slot_id
        )
        if frame.view_id != expected_view or frame.camera_serial != expected_serial:
            raise ValueError(
                f"frame identity conflicts with topology for {frame.round_id}/"
                f"{frame.camera_slot_id}: expected view/serial "
                f"{expected_view}/{expected_serial}, got {frame.view_id}/{frame.camera_serial}"
            )
    image_rows = tuple(
        CaptureImageRow(
            capture_set_id=request.capture_set_id,
            part_instance_id=request.part_instance_id,
            round_id=frame.round_id,
            view_id=frame.view_id,
            camera_slot_id=frame.camera_slot_id,
            camera_serial=frame.camera_serial,
            device_index=frame.device_index,
            relative_path=f"images/{request.capture_set_id}/{frame.view_id}.png",
            image_sha256=frame.image_sha256,
            width=frame.width,
            height=frame.height,
            capture_mode=frame.capture_mode,
            exposure=frame.exposure,
            gain=frame.gain,
            capture_parameters_json=frame.capture_parameters_json,
            captured_at=frame.captured_at,
            status="complete",
            error="",
        )
        for frame in sorted(frames, key=lambda item: plan.required_views.index(item.view_id))
    )
    row = CaptureSetRow(
        capture_set_id=request.capture_set_id,
        part_instance_id=request.part_instance_id,
        product=plan.product,
        hand=request.hand,
        topology_id=plan.topology_id,
        topology_sha256=plan.topology_sha256,
        expected_view_count=plan.expected_view_count,
        captured_view_count=len(frames),
        status="complete",
        started_at=started_at,
        completed_at=completed_at,
        failure_reason="",
    )
    domain_images = {
        row.view_id: ViewImage(
            view_id=row.view_id,
            round_id=row.round_id,
            camera_slot_id=row.camera_slot_id,
            camera_serial=row.camera_serial,
            relative_path=row.relative_path,
            image_sha256=row.image_sha256,
            width=row.width,
            height=row.height,
        )
        for row in image_rows
    }
    domain_capture_set = CaptureSet(
        capture_set_id=request.capture_set_id,
        capture_session_id=request.capture_session,
        part=request.part,
        topology_id=plan.topology_id,
        topology_sha256=plan.topology_sha256,
        images=domain_images,
    )
    domain_capture_set.validate_required_views(plan.required_views)
    gate_order = {"quality": 0, "registration": 1}
    view_order = {view_id: index for index, view_id in enumerate(plan.required_views)}
    ordered_gate_results = tuple(
        sorted(
            gate_results,
            key=lambda result: (
                gate_order.get(result.gate, len(gate_order)),
                view_order.get(result.view_id or "", len(view_order)),
            ),
        )
    )
    bundle = CaptureBundle(
        capture_session=request.capture_session,
        domain_capture_set=domain_capture_set,
        capture_set=row,
        images=image_rows,
        gate_results=ordered_gate_results,
        gate_provenance=gate_provenance,
        round_confirmations=tuple(round_confirmations),
        payload_by_view={item.view_id: item.image_bytes for item in frames},
    )
    bundle.validate(plan)
    return bundle


def build_incomplete_rows(
    request: CaptureRequest,
    plan: CapturePlan,
    frames: Sequence[CaptureFrame],
    *,
    started_at: str,
    failure_reason: str,
) -> tuple[CaptureSetRow, tuple[CaptureImageRow, ...]]:
    """Build diagnostic rows that can never be mistaken for canonical input."""
    reason = failure_reason.strip() or "unspecified capture failure"
    rows = tuple(
        CaptureImageRow(
            capture_set_id=request.capture_set_id,
            part_instance_id=request.part_instance_id,
            round_id=frame.round_id,
            view_id=frame.view_id,
            camera_slot_id=frame.camera_slot_id,
            camera_serial=frame.camera_serial,
            device_index=frame.device_index,
            relative_path=(
                f"images/{request.capture_set_id}/{index:04d}_{frame.view_id}.png"
            ),
            image_sha256=frame.image_sha256,
            width=frame.width,
            height=frame.height,
            capture_mode=frame.capture_mode,
            exposure=frame.exposure,
            gain=frame.gain,
            capture_parameters_json=frame.capture_parameters_json,
            captured_at=frame.captured_at,
            status="incomplete",
            error=reason,
        )
        for index, frame in enumerate(frames, start=1)
    )
    return (
        CaptureSetRow(
            capture_set_id=request.capture_set_id,
            part_instance_id=request.part_instance_id,
            product=plan.product,
            hand=request.hand,
            topology_id=plan.topology_id,
            topology_sha256=plan.topology_sha256,
            expected_view_count=plan.expected_view_count,
            captured_view_count=len(frames),
            status="incomplete",
            started_at=started_at,
            completed_at=utc_now(),
            failure_reason=reason,
        ),
        rows,
    )
