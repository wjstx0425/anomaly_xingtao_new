"""Stable canonical crop encoder provenance without importing OpenCV."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class CanonicalPngCodec:
    """Frozen encoder identity that explains canonical crop byte hashes."""

    encoder: str
    implementation_version: str
    format: str
    media_type: str
    compression: int
    spatial_operation: str

    def __post_init__(self) -> None:
        if self.encoder != "opencv.imencode":
            raise ValueError("canonical PNG encoder must be opencv.imencode")
        if (
            not isinstance(self.implementation_version, str)
            or not self.implementation_version.strip()
            or self.implementation_version != self.implementation_version.strip()
            or any(
                character in self.implementation_version
                for character in ("\x00", "\n", "\r")
            )
        ):
            raise ValueError("canonical PNG implementation_version must be non-empty")
        if self.format != "png" or self.media_type != "image/png":
            raise ValueError("canonical crop codec must be lossless PNG")
        if self.spatial_operation != "single_roi_xyxy_half_open_crop_no_resize":
            raise ValueError("canonical PNG spatial operation differs from the crop contract")
        if (
            isinstance(self.compression, bool)
            or not isinstance(self.compression, int)
            or not 0 <= self.compression <= 9
        ):
            raise ValueError("PNG compression must be an integer from 0 through 9")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CanonicalPngCodec":
        expected = {
            "encoder",
            "implementation_version",
            "format",
            "media_type",
            "compression",
            "spatial_operation",
        }
        if set(payload) != expected:
            raise ValueError(
                "canonical PNG codec fields differ from strict schema; "
                f"missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)}"
            )
        return cls(
            encoder=payload["encoder"],  # type: ignore[arg-type]
            implementation_version=payload["implementation_version"],  # type: ignore[arg-type]
            format=payload["format"],  # type: ignore[arg-type]
            media_type=payload["media_type"],  # type: ignore[arg-type]
            compression=payload["compression"],  # type: ignore[arg-type]
            spatial_operation=payload["spatial_operation"],  # type: ignore[arg-type]
        )

    @classmethod
    def opencv(
        cls,
        compression: int,
        *,
        implementation_version: str,
    ) -> "CanonicalPngCodec":
        return cls(
            encoder="opencv.imencode",
            implementation_version=implementation_version,
            format="png",
            media_type="image/png",
            compression=compression,
            spatial_operation="single_roi_xyxy_half_open_crop_no_resize",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "encoder": self.encoder,
            "implementation_version": self.implementation_version,
            "format": self.format,
            "media_type": self.media_type,
            "compression": self.compression,
            "spatial_operation": self.spatial_operation,
        }

    @property
    def sha256(self) -> str:
        payload = (
            json.dumps(
                self.as_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
