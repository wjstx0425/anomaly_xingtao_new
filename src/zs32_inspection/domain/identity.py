"""Stable product, part, capture, and image identity contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping

from .errors import IdentityValidationError

PRODUCT = "ZS32"
_HEX_DIGITS = frozenset("0123456789abcdef")


def require_non_empty(value: str, field_name: str) -> str:
    """Return a stripped identifier or reject it."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise IdentityValidationError(msg)
    normalized = value.strip()
    if any(character.isspace() for character in normalized):
        msg = f"{field_name} must not contain whitespace: {value!r}"
        raise IdentityValidationError(msg)
    return normalized


def require_sha256(value: str, field_name: str) -> str:
    """Return a canonical lowercase SHA256 digest or reject it."""
    if not isinstance(value, str):
        msg = f"{field_name} must be a SHA256 string"
        raise IdentityValidationError(msg)
    digest = value.lower()
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        msg = f"{field_name} must contain exactly 64 hexadecimal characters"
        raise IdentityValidationError(msg)
    return digest


class Hand(str, Enum):
    """Physical ZS32 hand variant."""

    LEFT = "left"
    RIGHT = "right"

    @classmethod
    def parse(cls, value: str | Hand) -> Hand:
        """Parse a hand without guessing or mirroring."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as error:
            msg = f"hand must be one of {[item.value for item in cls]}, got {value!r}"
            raise IdentityValidationError(msg) from error


@dataclass(frozen=True, slots=True)
class PartIdentity:
    """Identity shared by every image of one physical part."""

    part_instance_id: str
    hand: Hand
    product: str = PRODUCT

    def __post_init__(self) -> None:
        """Validate a ZS32-only part identity."""
        object.__setattr__(self, "part_instance_id", require_non_empty(self.part_instance_id, "part_instance_id"))
        object.__setattr__(self, "hand", Hand.parse(self.hand))
        if self.product != PRODUCT:
            msg = f"only product {PRODUCT!r} is supported, got {self.product!r}"
            raise IdentityValidationError(msg)


@dataclass(frozen=True, slots=True)
class ViewImage:
    """One immutable raw image with camera and topology identity."""

    view_id: str
    round_id: str
    camera_slot_id: str
    camera_serial: str
    relative_path: str
    image_sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject ambiguous identity, unsafe paths, and invalid image metadata."""
        for field_name in ("view_id", "round_id", "camera_slot_id", "camera_serial"):
            object.__setattr__(self, field_name, require_non_empty(getattr(self, field_name), field_name))
        relative_path = require_non_empty(self.relative_path, "relative_path")
        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts:
            msg = f"relative_path must stay inside its capture bundle: {relative_path!r}"
            raise IdentityValidationError(msg)
        object.__setattr__(self, "relative_path", path.as_posix())
        object.__setattr__(self, "image_sha256", require_sha256(self.image_sha256, "image_sha256"))
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            msg = f"width must be a positive integer, got {self.width!r}"
            raise IdentityValidationError(msg)
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            msg = f"height must be a positive integer, got {self.height!r}"
            raise IdentityValidationError(msg)


@dataclass(frozen=True, slots=True)
class CaptureSet:
    """A complete two-round capture of one physical part."""

    capture_set_id: str
    capture_session_id: str
    part: PartIdentity
    topology_id: str
    topology_sha256: str
    images: Mapping[str, ViewImage]

    def __post_init__(self) -> None:
        """Freeze the view map and enforce key/image identity agreement."""
        object.__setattr__(self, "capture_set_id", require_non_empty(self.capture_set_id, "capture_set_id"))
        object.__setattr__(
            self,
            "capture_session_id",
            require_non_empty(self.capture_session_id, "capture_session_id"),
        )
        object.__setattr__(self, "topology_id", require_non_empty(self.topology_id, "topology_id"))
        object.__setattr__(self, "topology_sha256", require_sha256(self.topology_sha256, "topology_sha256"))
        if not isinstance(self.part, PartIdentity):
            msg = "part must be a PartIdentity"
            raise IdentityValidationError(msg)
        if not isinstance(self.images, Mapping) or not self.images:
            msg = "images must be a non-empty mapping keyed by view_id"
            raise IdentityValidationError(msg)
        frozen: dict[str, ViewImage] = {}
        for view_id, image in self.images.items():
            canonical_view = require_non_empty(view_id, "images key")
            if not isinstance(image, ViewImage):
                msg = f"images[{canonical_view!r}] must be a ViewImage"
                raise IdentityValidationError(msg)
            if canonical_view != image.view_id:
                msg = f"images key {canonical_view!r} conflicts with ViewImage.view_id {image.view_id!r}"
                raise IdentityValidationError(msg)
            if canonical_view in frozen:
                msg = f"duplicate view image: {canonical_view!r}"
                raise IdentityValidationError(msg)
            frozen[canonical_view] = image
        object.__setattr__(self, "images", MappingProxyType(frozen))

    def validate_required_views(self, required_views: tuple[str, ...]) -> None:
        """Reject a capture whose view set is not exactly the topology contract."""
        expected = set(required_views)
        actual = set(self.images)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            msg = f"capture view identity mismatch; missing={missing}, extra={extra}"
            raise IdentityValidationError(msg)


@dataclass(frozen=True, slots=True)
class RoiSample:
    """One canonical crop derived exactly once from an immutable raw image."""

    part: PartIdentity
    capture_set_id: str
    view_id: str
    source_sha256: str
    crop_sha256: str
    roi_config_id: str
    crop_width: int
    crop_height: int

    def __post_init__(self) -> None:
        """Validate the identity and content-addressed crop metadata."""
        if not isinstance(self.part, PartIdentity):
            msg = "part must be a PartIdentity"
            raise IdentityValidationError(msg)
        for field_name in ("capture_set_id", "view_id", "roi_config_id"):
            object.__setattr__(self, field_name, require_non_empty(getattr(self, field_name), field_name))
        for field_name in ("source_sha256", "crop_sha256"):
            object.__setattr__(self, field_name, require_sha256(getattr(self, field_name), field_name))
        if (
            isinstance(self.crop_width, bool)
            or not isinstance(self.crop_width, int)
            or isinstance(self.crop_height, bool)
            or not isinstance(self.crop_height, int)
            or self.crop_width <= 0
            or self.crop_height <= 0
        ):
            msg = "crop dimensions must be positive"
            raise IdentityValidationError(msg)
