"""Configuration-driven 3/4/5-camera topology domain contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .errors import TopologyValidationError
from .identity import PRODUCT, require_non_empty, require_sha256

SUPPORTED_CAMERA_COUNTS = frozenset({3, 4, 5})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _safe_identifier(value: str, field: str) -> str:
    try:
        normalized = require_non_empty(value, field)
    except ValueError as error:
        raise TopologyValidationError(str(error)) from error
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise TopologyValidationError(f"{field} must be a path-safe identifier: {value!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class CaptureRound:
    """One synchronous capture round."""

    round_id: str
    prompt: str

    def __post_init__(self) -> None:
        """Validate stable round identity and an explicit operator prompt."""
        try:
            object.__setattr__(self, "round_id", _safe_identifier(self.round_id, "round_id"))
            object.__setattr__(self, "prompt", require_non_empty(self.prompt, "prompt"))
        except ValueError as error:
            raise TopologyValidationError(str(error)) from error


@dataclass(frozen=True, slots=True)
class CameraSlot:
    """A physical camera slot bound to one serial and one view per round."""

    slot_id: str
    serial: str
    views: Mapping[str, str]

    def __post_init__(self) -> None:
        """Freeze the round-to-view map and reject empty identities."""
        try:
            object.__setattr__(self, "slot_id", _safe_identifier(self.slot_id, "slot_id"))
            object.__setattr__(self, "serial", _safe_identifier(self.serial, "serial"))
        except ValueError as error:
            raise TopologyValidationError(str(error)) from error
        if not isinstance(self.views, Mapping) or not self.views:
            msg = f"camera slot {self.slot_id!r} must define a non-empty views mapping"
            raise TopologyValidationError(msg)
        frozen: dict[str, str] = {}
        for round_id, view_id in self.views.items():
            try:
                canonical_round = _safe_identifier(round_id, "views round_id")
                canonical_view = _safe_identifier(view_id, "views view_id")
            except ValueError as error:
                raise TopologyValidationError(str(error)) from error
            if canonical_round in frozen:
                msg = f"camera slot {self.slot_id!r} repeats round {canonical_round!r}"
                raise TopologyValidationError(msg)
            frozen[canonical_round] = canonical_view
        object.__setattr__(self, "views", MappingProxyType(frozen))


@dataclass(frozen=True, slots=True)
class CaptureTopology:
    """Complete, unambiguous camera/round/view identity contract."""

    schema_version: int
    topology_id: str
    product: str
    rounds: tuple[CaptureRound, ...]
    camera_slots: tuple[CameraSlot, ...]
    required_views: tuple[str, ...]
    topology_sha256: str

    def __post_init__(self) -> None:
        """Validate 3/4/5-camera topology completeness and uniqueness."""
        if self.schema_version != 1:
            msg = f"unsupported topology schema_version {self.schema_version!r}; expected 1"
            raise TopologyValidationError(msg)
        if self.product != PRODUCT:
            msg = f"topology product must be {PRODUCT!r}, got {self.product!r}"
            raise TopologyValidationError(msg)
        try:
            object.__setattr__(self, "topology_id", _safe_identifier(self.topology_id, "topology_id"))
            object.__setattr__(
                self,
                "topology_sha256",
                require_sha256(self.topology_sha256, "topology_sha256"),
            )
        except ValueError as error:
            raise TopologyValidationError(str(error)) from error
        object.__setattr__(self, "rounds", tuple(self.rounds))
        object.__setattr__(self, "camera_slots", tuple(self.camera_slots))
        object.__setattr__(
            self,
            "required_views",
            tuple(_safe_identifier(view, "required_view") for view in self.required_views),
        )
        if len(self.rounds) != 2:
            msg = f"ZS32 double-side topology requires exactly 2 rounds, got {len(self.rounds)}"
            raise TopologyValidationError(msg)
        if len(self.camera_slots) not in SUPPORTED_CAMERA_COUNTS:
            msg = f"camera count must be one of {sorted(SUPPORTED_CAMERA_COUNTS)}, got {len(self.camera_slots)}"
            raise TopologyValidationError(msg)
        if not all(isinstance(item, CaptureRound) for item in self.rounds):
            msg = "rounds must contain only CaptureRound objects"
            raise TopologyValidationError(msg)
        if not all(isinstance(item, CameraSlot) for item in self.camera_slots):
            msg = "camera_slots must contain only CameraSlot objects"
            raise TopologyValidationError(msg)

        round_ids = tuple(item.round_id for item in self.rounds)
        if len(set(round_ids)) != len(round_ids):
            msg = f"round_id values must be unique: {round_ids}"
            raise TopologyValidationError(msg)
        slot_ids = tuple(item.slot_id for item in self.camera_slots)
        serials = tuple(item.serial for item in self.camera_slots)
        if len(set(slot_ids)) != len(slot_ids):
            msg = f"slot_id values must be unique: {slot_ids}"
            raise TopologyValidationError(msg)
        if len(set(serials)) != len(serials):
            msg = f"camera serial values must be unique: {serials}"
            raise TopologyValidationError(msg)

        generated_views: list[str] = []
        expected_rounds = set(round_ids)
        for slot in self.camera_slots:
            actual_rounds = set(slot.views)
            if actual_rounds != expected_rounds:
                missing = sorted(expected_rounds - actual_rounds)
                extra = sorted(actual_rounds - expected_rounds)
                msg = f"slot {slot.slot_id!r} round map mismatch; missing={missing}, extra={extra}"
                raise TopologyValidationError(msg)
            generated_views.extend(slot.views[round_id] for round_id in round_ids)
        if len(set(generated_views)) != len(generated_views):
            msg = f"every slot/round pair must produce a unique view: {generated_views}"
            raise TopologyValidationError(msg)
        if len(set(self.required_views)) != len(self.required_views):
            msg = f"required_views contains duplicates: {self.required_views}"
            raise TopologyValidationError(msg)
        if set(self.required_views) != set(generated_views):
            missing = sorted(set(generated_views) - set(self.required_views))
            extra = sorted(set(self.required_views) - set(generated_views))
            msg = f"required_views must equal generated slot/round views; missing={missing}, extra={extra}"
            raise TopologyValidationError(msg)
        encoded = json.dumps(
            self.as_dict(include_sha256=False),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != self.topology_sha256:
            msg = "topology_sha256 does not match the canonical topology payload"
            raise TopologyValidationError(msg)

    @property
    def expected_view_count(self) -> int:
        """Return 6, 8, or 10 from the validated topology."""
        return len(self.required_views)

    def binding_for_view(self, view_id: str) -> tuple[str, str, str]:
        """Return ``(round_id, slot_id, serial)`` for one required view."""
        for capture_round in self.rounds:
            for slot in self.camera_slots:
                if slot.views[capture_round.round_id] == view_id:
                    return capture_round.round_id, slot.slot_id, slot.serial
        msg = f"view {view_id!r} is not required by topology {self.topology_id!r}"
        raise TopologyValidationError(msg)

    def as_dict(self, *, include_sha256: bool = False) -> dict[str, object]:
        """Return the canonical external topology schema."""
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "topology_id": self.topology_id,
            "product": self.product,
            "rounds": [
                {"round_id": capture_round.round_id, "prompt": capture_round.prompt}
                for capture_round in self.rounds
            ],
            "camera_slots": [
                {"slot_id": slot.slot_id, "serial": slot.serial, "views": dict(sorted(slot.views.items()))}
                for slot in self.camera_slots
            ],
            "required_views": list(self.required_views),
        }
        if include_sha256:
            payload["topology_sha256"] = self.topology_sha256
        return payload
