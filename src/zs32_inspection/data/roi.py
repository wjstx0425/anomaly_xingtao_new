"""Single-crop ROI and auditable 4K-to-canonical YOLO label migration."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from zs32_inspection.domain.contracts import RoiBox


AnnotationState = Literal["annotated", "confirmed_empty", "missing"]
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class LabelDocument:
    """Explicitly distinguish a label file, confirmed emptiness, and missing work."""

    state: AnnotationState
    text: str | None
    source_path: str

    def __post_init__(self) -> None:
        if self.state not in {"annotated", "confirmed_empty", "missing"}:
            raise ValueError(f"unsupported annotation state: {self.state!r}")
        if self.state == "missing" and self.text is not None:
            raise ValueError("a missing annotation must not carry label text")
        if self.state != "missing" and self.text is None:
            raise ValueError(f"annotation state {self.state!r} requires label text")
        if self.state == "confirmed_empty" and self.text and self.text.strip():
            raise ValueError("confirmed_empty label text must contain no boxes")
        if self.state == "annotated" and (not self.text or not self.text.strip()):
            raise ValueError("annotated label document must contain at least one box")

    @property
    def sha256(self) -> str:
        """Hash exact source label bytes; missing documents have no digest."""
        if self.text is None:
            raise ValueError("missing annotations do not have a source label SHA256")
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NormalizedYoloBox:
    """One validated normalized YOLO ``class cx cy width height`` box."""

    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or self.class_id != 0:
            raise ValueError(f"ZS32 YOLO class_id must be 0, got {self.class_id!r}")
        values = (self.cx, self.cy, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("YOLO coordinates must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("YOLO box width and height must be positive")
        x1 = self.cx - self.width / 2
        y1 = self.cy - self.height / 2
        x2 = self.cx + self.width / 2
        y2 = self.cy + self.height / 2
        if min(x1, y1) < 0 or max(x2, y2) > 1:
            raise ValueError(f"YOLO box must lie inside normalized source image: {values!r}")

    def pixel_xyxy(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        return (
            (self.cx - self.width / 2) * image_width,
            (self.cy - self.height / 2) * image_height,
            (self.cx + self.width / 2) * image_width,
            (self.cy + self.height / 2) * image_height,
        )


@dataclass(frozen=True, slots=True)
class BboxTransformAudit:
    """Before/after evidence for one source bounding box."""

    line_number: int
    original_line: str
    original_normalized: tuple[float, float, float, float]
    original_pixel_xyxy: tuple[float, float, float, float]
    roi_xyxy: tuple[int, int, int, int]
    transformed_normalized: tuple[float, float, float, float] | None
    transformed_pixel_xyxy: tuple[float, float, float, float] | None
    clipped: bool
    outside_roi: bool
    dropped_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LabelMigrationAudit:
    """Hashes and every bbox outcome for one source label document."""

    source_path: str
    annotation_state: str
    source_label_sha256: str
    crop_label_sha256: str
    source_box_count: int
    crop_box_count: int
    clipped_box_count: int
    outside_roi_box_count: int
    dropped_box_count: int
    boxes: tuple[BboxTransformAudit, ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["boxes"] = [item.as_dict() for item in self.boxes]
        return payload


@dataclass(frozen=True, slots=True)
class MigratedLabels:
    """Canonical crop-space YOLO labels with their mandatory audit."""

    text: str
    audit: LabelMigrationAudit


@dataclass(frozen=True, slots=True)
class CroppedImage:
    """Lossless crop bytes returned by a codec adapter."""

    image_bytes: bytes
    width: int
    height: int
    media_type: str = "image/png"

    def __post_init__(self) -> None:
        if not isinstance(self.image_bytes, bytes) or not self.image_bytes:
            raise ValueError("cropped image bytes must be non-empty bytes")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.width, self.height)
        ):
            raise ValueError("cropped image bytes and dimensions must be non-empty/positive")
        if not self.image_bytes.startswith(_PNG_SIGNATURE):
            raise ValueError("canonical crop bytes must have a PNG signature")
        if self.media_type != "image/png":
            raise ValueError("canonical crops must use lossless image/png")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.image_bytes).hexdigest()


class ImageCropper(Protocol):
    """Image codec port; it may crop only the supplied authoritative RoiBox."""

    def crop_png(
        self,
        source_png: bytes,
        *,
        source_width: int,
        source_height: int,
        roi: RoiBox,
    ) -> CroppedImage: ...


def parse_yolo_labels(text: str, *, source: str) -> tuple[tuple[int, str, NormalizedYoloBox], ...]:
    """Strictly parse non-empty YOLO lines while preserving line identity."""
    output: list[tuple[int, str, NormalizedYoloBox]] = []
    for line_number, original in enumerate(text.splitlines(), start=1):
        line = original.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label at {source}:{line_number}: {original!r}")
        try:
            class_id = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(
                f"invalid YOLO label at {source}:{line_number}: {original!r}"
            ) from error
        output.append(
            (line_number, original, NormalizedYoloBox(class_id, cx, cy, width, height))
        )
    return tuple(output)


def _transform_box(
    box: NormalizedYoloBox,
    *,
    line_number: int,
    original_line: str,
    source_width: int,
    source_height: int,
    roi: RoiBox,
) -> tuple[str | None, BboxTransformAudit]:
    original = box.pixel_xyxy(source_width, source_height)
    x1, y1, x2, y2 = original
    ix1 = max(x1, roi.x1)
    iy1 = max(y1, roi.y1)
    ix2 = min(x2, roi.x2)
    iy2 = min(y2, roi.y2)
    outside = ix2 <= ix1 or iy2 <= iy1
    roi_tuple = (roi.x1, roi.y1, roi.x2, roi.y2)
    original_normalized = (box.cx, box.cy, box.width, box.height)
    if outside:
        return None, BboxTransformAudit(
            line_number,
            original_line,
            original_normalized,
            original,
            roi_tuple,
            None,
            None,
            False,
            True,
            "outside_roi",
        )
    clipped = (ix1, iy1, ix2, iy2) != (x1, y1, x2, y2)
    crop_x1, crop_y1 = ix1 - roi.x1, iy1 - roi.y1
    crop_x2, crop_y2 = ix2 - roi.x1, iy2 - roi.y1
    width = (crop_x2 - crop_x1) / roi.width
    height = (crop_y2 - crop_y1) / roi.height
    cx = (crop_x1 + crop_x2) / (2 * roi.width)
    cy = (crop_y1 + crop_y2) / (2 * roi.height)
    transformed = (cx, cy, width, height)
    line = f"{box.class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
    return line, BboxTransformAudit(
        line_number,
        original_line,
        original_normalized,
        original,
        roi_tuple,
        transformed,
        (crop_x1, crop_y1, crop_x2, crop_y2),
        clipped,
        False,
        None,
    )


def migrate_yolo_labels(
    document: LabelDocument,
    *,
    source_width: int,
    source_height: int,
    roi: RoiBox,
) -> MigratedLabels:
    """Move 4K labels into crop space and report every clipped/dropped box."""
    if document.state == "missing":
        raise ValueError(
            f"annotation is missing for {document.source_path}; missing is not confirmed empty"
        )
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source image dimensions must be positive")
    roi.validate_image_bounds(source_width, source_height)
    source_text = document.text or ""
    parsed = parse_yolo_labels(source_text, source=document.source_path)
    if document.state == "confirmed_empty" and parsed:
        raise ValueError("confirmed_empty annotation unexpectedly contains boxes")
    if document.state == "annotated" and not parsed:
        raise ValueError("annotated document produced no YOLO boxes")
    output_lines: list[str] = []
    audits: list[BboxTransformAudit] = []
    for line_number, original, box in parsed:
        transformed, audit = _transform_box(
            box,
            line_number=line_number,
            original_line=original,
            source_width=source_width,
            source_height=source_height,
            roi=roi,
        )
        audits.append(audit)
        if transformed is not None:
            output_lines.append(transformed)
    output_text = "\n".join(output_lines) + ("\n" if output_lines else "")
    audit = LabelMigrationAudit(
        source_path=document.source_path,
        annotation_state=document.state,
        source_label_sha256=document.sha256,
        crop_label_sha256=hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        source_box_count=len(parsed),
        crop_box_count=len(output_lines),
        clipped_box_count=sum(item.clipped for item in audits),
        outside_roi_box_count=sum(item.outside_roi for item in audits),
        dropped_box_count=sum(item.dropped_reason is not None for item in audits),
        boxes=tuple(audits),
    )
    return MigratedLabels(output_text, audit)


def validate_crop_result(crop: CroppedImage, roi: RoiBox) -> None:
    """Reject adapters that resize or otherwise alter canonical crop geometry."""
    if (crop.width, crop.height) != (roi.width, roi.height):
        raise ValueError(
            "crop adapter changed authoritative ROI geometry: "
            f"expected {roi.width}x{roi.height}, got {crop.width}x{crop.height}"
        )
