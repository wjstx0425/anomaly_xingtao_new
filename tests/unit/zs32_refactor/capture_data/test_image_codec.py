"""Linux-only proof that the canonical codec crops once without resize."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from zs32_inspection.data.image_codec import OpenCvPngCropper
from zs32_inspection.domain.contracts import RoiBox


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="authoritative runtime is Linux only")


def test_opencv_cropper_applies_exact_half_open_pixel_slice() -> None:
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    written, source = cv2.imencode(".png", image)
    assert written
    cropper = OpenCvPngCropper(compression=1)
    crop = cropper.crop_png(
        source.tobytes(),
        source_width=8,
        source_height=6,
        roi=RoiBox(2, 1, 7, 5),
    )
    decoded = cv2.imdecode(np.frombuffer(crop.image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert crop.width == 5
    assert crop.height == 4
    assert np.array_equal(decoded, image[1:5, 2:7])
    assert cropper.canonical_png_codec.as_dict() == {
        "encoder": "opencv.imencode",
        "implementation_version": cv2.__version__,
        "format": "png",
        "media_type": "image/png",
        "compression": 1,
        "spatial_operation": "single_roi_xyxy_half_open_crop_no_resize",
    }
