"""Stable, hardware-free contracts for one multi-round capture.

The camera SDK adapter is deliberately outside this module.  This module only
describes the already-compiled acquisition plan and the encoded frames returned
by an adapter.  Consequently the same service works for three, four, or five
cameras without a camera-count constant.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping, Sequence

from zs32_inspection.domain.identity import PartIdentity
from zs32_inspection.domain.topology import CaptureTopology


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def require_identifier(value: str, field: str) -> str:
    """Return a stripped identifier or raise for unsafe/path-like values."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string identifier")
    normalized = value.strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{field} must be a non-empty safe identifier: {value!r}")
    return normalized


def require_sha256(value: str, field: str) -> str:
    """Validate a lowercase SHA256 digest."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a SHA256 string")
    normalized = value.strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return normalized


def utc_now() -> str:
    """Return an audit-friendly UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class CaptureRoundPlan:
    """One ordered physical acquisition round."""

    round_id: str
    prompt: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_id", require_identifier(self.round_id, "round_id"))
        if not self.prompt.strip():
            raise ValueError("capture round prompt must not be empty")


@dataclass(frozen=True, slots=True)
class RoundConfirmation:
    """Auditable operator authorization immediately preceding one physical round."""

    round_id: str
    prompt: str
    confirmed_by: str
    prompted_at: str
    confirmed_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_id", require_identifier(self.round_id, "round_id"))
        object.__setattr__(
            self,
            "confirmed_by",
            require_identifier(self.confirmed_by, "confirmed_by"),
        )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("round confirmation prompt must not be empty")
        parsed: list[datetime] = []
        for field_name in ("prompted_at", "confirmed_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.endswith("Z"):
                raise ValueError(f"{field_name} must be a UTC timestamp ending in Z")
            try:
                parsed.append(datetime.fromisoformat(value[:-1] + "+00:00"))
            except ValueError as error:
                raise ValueError(f"{field_name} is not a valid UTC timestamp") from error
        if parsed[1] < parsed[0]:
            raise ValueError("round confirmation cannot precede its prompt")


@dataclass(frozen=True, slots=True)
class CameraBinding:
    """Stable physical slot/serial with one view assignment per round."""

    slot_id: str
    serial: str
    views: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", require_identifier(self.slot_id, "camera slot_id"))
        object.__setattr__(self, "serial", require_identifier(self.serial, "camera serial"))
        if not self.views:
            raise ValueError(f"camera slot {self.slot_id!r} has no round/view bindings")
        normalized: dict[str, str] = {}
        for round_id, view_id in self.views.items():
            checked_round = require_identifier(round_id, "camera binding round_id")
            checked_view = require_identifier(view_id, "camera binding view_id")
            if checked_round in normalized:
                raise ValueError(f"duplicate round binding for camera slot {self.slot_id}: {checked_round}")
            normalized[checked_round] = checked_view
        object.__setattr__(self, "views", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """Capture-only projection of a validated topology.

    A config/domain compiler should create this object only after validating the
    full topology schema.  It nevertheless validates all capture invariants so
    an invalid or hand-built plan fails closed at the hardware boundary.
    """

    product: str
    topology_id: str
    topology_sha256: str
    rounds: tuple[CaptureRoundPlan, ...]
    cameras: tuple[CameraBinding, ...]
    required_views: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.product != "ZS32":
            raise ValueError(f"capture plan product must be ZS32, got {self.product!r}")
        object.__setattr__(self, "topology_id", require_identifier(self.topology_id, "topology_id"))
        object.__setattr__(
            self,
            "topology_sha256",
            require_sha256(self.topology_sha256, "topology_sha256"),
        )
        rounds = tuple(self.rounds)
        cameras = tuple(self.cameras)
        required_views = tuple(
            require_identifier(view, "required_view") for view in self.required_views
        )
        if not rounds or not cameras or not required_views:
            raise ValueError("capture plan requires rounds, cameras, and required views")
        round_ids = tuple(item.round_id for item in rounds)
        slot_ids = tuple(item.slot_id for item in cameras)
        serials = tuple(item.serial for item in cameras)
        for field, values in (
            ("round_id", round_ids),
            ("camera slot_id", slot_ids),
            ("camera serial", serials),
            ("required view", required_views),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"capture plan contains duplicate {field}: {values}")
        emitted: list[str] = []
        expected_rounds = set(round_ids)
        for camera in cameras:
            if set(camera.views) != expected_rounds:
                raise ValueError(
                    f"camera slot {camera.slot_id!r} must bind every round exactly once; "
                    f"expected {sorted(expected_rounds)}, got {sorted(camera.views)}"
                )
            emitted.extend(camera.views[round_id] for round_id in round_ids)
        if len(emitted) != len(set(emitted)):
            raise ValueError(f"topology maps multiple camera/round pairs to one view: {emitted}")
        if set(emitted) != set(required_views):
            raise ValueError(
                "topology camera/round mappings must equal required_views; "
                f"mapped={sorted(emitted)}, required={sorted(required_views)}"
            )
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "required_views", required_views)

    @property
    def expected_view_count(self) -> int:
        """Number of images required for one complete capture set."""
        return len(self.required_views)

    def view_for(self, round_id: str, slot_id: str) -> str:
        """Resolve the unique view identity for a round and physical slot."""
        for camera in self.cameras:
            if camera.slot_id == slot_id:
                try:
                    return camera.views[round_id]
                except KeyError as error:
                    raise ValueError(
                        f"round {round_id!r} is not mapped for camera slot {slot_id!r}"
                    ) from error
        raise ValueError(f"unknown camera slot: {slot_id!r}")

    @classmethod
    def from_topology(cls, topology: CaptureTopology) -> CapturePlan:
        """Create the capture adapter projection from the canonical domain topology."""
        if not isinstance(topology, CaptureTopology):
            raise TypeError("topology must be a domain CaptureTopology")
        return cls(
            product=topology.product,
            topology_id=topology.topology_id,
            topology_sha256=topology.topology_sha256,
            rounds=tuple(CaptureRoundPlan(item.round_id, item.prompt) for item in topology.rounds),
            cameras=tuple(
                CameraBinding(item.slot_id, item.serial, item.views)
                for item in topology.camera_slots
            ),
            required_views=topology.required_views,
        )


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    """Identity supplied by the application for one physical part."""

    capture_session: str
    capture_set_id: str
    part: PartIdentity

    def __post_init__(self) -> None:
        for field in ("capture_session", "capture_set_id"):
            object.__setattr__(self, field, require_identifier(getattr(self, field), field))
        if not isinstance(self.part, PartIdentity):
            raise TypeError("capture request part must be a domain PartIdentity")

    @property
    def part_instance_id(self) -> str:
        """Return the canonical physical-part identifier."""
        return self.part.part_instance_id

    @property
    def hand(self) -> str:
        """Return the canonical hand string used by CSV manifests."""
        return self.part.hand.value


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    """One encoded source image returned by the hardware adapter."""

    round_id: str
    view_id: str
    camera_slot_id: str
    camera_serial: str
    device_index: int | None
    image_bytes: bytes
    width: int
    height: int
    capture_mode: str
    exposure: float | None
    gain: float | None
    captured_at: str
    media_type: str = "image/png"
    capture_parameters: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for field in ("round_id", "view_id", "camera_slot_id", "camera_serial"):
            object.__setattr__(self, field, require_identifier(getattr(self, field), field))
        if self.device_index is not None and (
            isinstance(self.device_index, bool)
            or not isinstance(self.device_index, int)
            or self.device_index < 0
        ):
            raise ValueError("device_index must be a non-negative integer when present")
        if not isinstance(self.image_bytes, bytes) or not self.image_bytes:
            raise ValueError("capture frame image_bytes must be non-empty bytes")
        if not self.image_bytes.startswith(_PNG_SIGNATURE):
            raise ValueError("capture frame image_bytes must have a PNG signature")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.width, self.height)
        ):
            raise ValueError("capture frame width and height must be positive")
        if self.media_type != "image/png":
            raise ValueError("canonical raw capture currently requires lossless image/png")
        if not isinstance(self.capture_mode, str) or not self.capture_mode.strip():
            raise ValueError("capture_mode must not be empty")
        for field in ("exposure", "gain"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"capture frame {field} must be finite")
        if not isinstance(self.captured_at, str) or not self.captured_at.strip():
            raise ValueError("captured_at must not be empty")
        parameters: dict[str, str | int | float | bool | None] = {}
        for key, value in self.capture_parameters.items():
            checked_key = require_identifier(key, "capture parameter key")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"capture parameter {checked_key!r} must be a JSON scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"capture parameter {checked_key!r} must be finite")
            parameters[checked_key] = value
        object.__setattr__(self, "capture_parameters", MappingProxyType(parameters))

    @property
    def image_sha256(self) -> str:
        """Content digest of the encoded source image."""
        return hashlib.sha256(self.image_bytes).hexdigest()

    @property
    def capture_parameters_json(self) -> str:
        """Return deterministic adapter-specific acquisition provenance."""
        return json.dumps(
            dict(self.capture_parameters),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Published complete capture identity returned to the application."""

    capture_session: str
    capture_set_id: str
    part_instance_id: str
    published_path: str
    image_sha256_by_view: Mapping[str, str]

    def __post_init__(self) -> None:
        for field in ("capture_session", "capture_set_id"):
            require_identifier(getattr(self, field), field)
        if not isinstance(self.part_instance_id, str) or not self.part_instance_id.strip():
            raise ValueError("capture result part_instance_id must not be empty")
        if not self.published_path:
            raise ValueError("capture result published_path must not be empty")
        digests = {
            require_identifier(view, "capture result view_id"): require_sha256(
                digest,
                f"capture result image hash for {view}",
            )
            for view, digest in self.image_sha256_by_view.items()
        }
        if not digests:
            raise ValueError("capture result image hash mapping must not be empty")
        object.__setattr__(
            self,
            "image_sha256_by_view",
            MappingProxyType(digests),
        )


def ensure_unique_frames(frames: Sequence[CaptureFrame]) -> None:
    """Reject duplicate view, round/slot, or conflicting camera identities."""
    views: set[str] = set()
    round_slots: set[tuple[str, str]] = set()
    serial_by_slot: dict[str, str] = {}
    slot_by_serial: dict[str, str] = {}
    for frame in frames:
        if frame.view_id in views:
            raise ValueError(f"duplicate captured view: {frame.view_id}")
        views.add(frame.view_id)
        key = (frame.round_id, frame.camera_slot_id)
        if key in round_slots:
            raise ValueError(f"duplicate frame for round/slot: {key}")
        round_slots.add(key)
        previous_serial = serial_by_slot.setdefault(frame.camera_slot_id, frame.camera_serial)
        if previous_serial != frame.camera_serial:
            raise ValueError(
                f"camera slot {frame.camera_slot_id!r} changed serial from "
                f"{previous_serial!r} to {frame.camera_serial!r}"
            )
        previous_slot = slot_by_serial.setdefault(frame.camera_serial, frame.camera_slot_id)
        if previous_slot != frame.camera_slot_id:
            raise ValueError(
                f"camera serial {frame.camera_serial!r} appeared in slots "
                f"{previous_slot!r} and {frame.camera_slot_id!r}"
            )
