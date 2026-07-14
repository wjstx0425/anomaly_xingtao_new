"""OpenCV adapter that performs the one permitted spatial crop."""

from __future__ import annotations

import cv2
import numpy as np

from zs32_inspection.domain.contracts import RoiBox

from .codec_contract import CanonicalPngCodec
from .roi import CroppedImage


class OpenCvPngCropper:
    """Decode lossless raw PNG, apply exactly one half-open ROI, and re-encode PNG.

    This adapter never resizes, pads, mirrors, rotates, or performs an additional
    crop.  Model-specific resize/normalization happens only after the canonical
    crop release has been built.
    """

    def __init__(self, *, compression: int = 1) -> None:
        self._codec = CanonicalPngCodec.opencv(
            compression,
            implementation_version=cv2.__version__,
        )

    @property
    def canonical_png_codec(self) -> CanonicalPngCodec:
        """Return the immutable encoder settings bound into dataset provenance."""
        return self._codec

    def crop_png(
        self,
        source_png: bytes,
        *,
        source_width: int,
        source_height: int,
        roi: RoiBox,
    ) -> CroppedImage:
        """Apply the authoritative pixel ``xyxy_half_open`` rectangle exactly once."""
        if not isinstance(source_png, bytes) or not source_png:
            raise ValueError("source PNG must be non-empty bytes")
        roi.validate_image_bounds(source_width, source_height)
        encoded = np.frombuffer(source_png, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("source bytes are not a decodable image")
        actual_height, actual_width = image.shape[:2]
        if (actual_width, actual_height) != (source_width, source_height):
            raise ValueError(
                "decoded source dimensions conflict with capture manifest: "
                f"decoded={actual_width}x{actual_height}, "
                f"manifest={source_width}x{source_height}"
            )
        crop = np.ascontiguousarray(image[roi.y1 : roi.y2, roi.x1 : roi.x2])
        if crop.shape[:2] != (roi.height, roi.width):
            raise RuntimeError(
                f"unexpected canonical crop shape: {crop.shape[:2]} != {(roi.height, roi.width)}"
            )
        written, payload = cv2.imencode(
            ".png",
            crop,
            [cv2.IMWRITE_PNG_COMPRESSION, self._codec.compression],
        )
        if not written:
            raise RuntimeError("OpenCV failed to encode canonical ROI crop")
        return CroppedImage(payload.tobytes(), roi.width, roi.height)
