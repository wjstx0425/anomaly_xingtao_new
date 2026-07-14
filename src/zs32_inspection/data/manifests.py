"""Canonical dataset manifest rows and release metadata."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from zs32_inspection.capture.gate_policy import CaptureGateProvenance
from zs32_inspection.domain.identity import (
    Hand,
    PartIdentity,
    RoiSample,
    require_non_empty,
    require_sha256,
)


CANONICAL_MANIFEST_COLUMNS = (
    "dataset_release_id",
    "part_instance_id",
    "capture_set_id",
    "hand",
    "view",
    "source_path",
    "source_sha256",
    "crop_path",
    "crop_sha256",
    "roi_version",
    "roi_sha256",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "crop_width",
    "crop_height",
    "label",
    "defect_type",
    "split",
    "is_synthetic",
    "annotation_state",
    "source_label_sha256",
    "crop_label_sha256",
    "source_box_count",
    "crop_box_count",
    "clipped_box_count",
    "outside_roi_box_count",
    "dropped_box_count",
)

ADAPTER_BASE_COLUMNS = (
    "sample_id",
    "part_instance_id",
    "capture_set_id",
    "hand",
    "view",
    "split",
    "label",
    "defect_type",
    "canonical_crop_path",
    "canonical_crop_sha256",
)

YOLO_ADAPTER_COLUMNS = (
    *ADAPTER_BASE_COLUMNS,
    "crop_label_path",
    "crop_label_sha256",
)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"payload is not canonical JSON: {error}") from error


@dataclass(frozen=True, slots=True)
class SemanticsApprovalEnvelope:
    """Mandatory human approval bound to one canonical semantics document."""

    reviewed_by: str
    reviewed_at: str
    approved: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reviewed_by, str)
            or not self.reviewed_by.strip()
            or self.reviewed_by != self.reviewed_by.strip()
            or any(character in self.reviewed_by for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError("semantics reviewed_by must be a canonical non-empty identity")
        if (
            not isinstance(self.reviewed_at, str)
            or not self.reviewed_at.strip()
            or self.reviewed_at != self.reviewed_at.strip()
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
                self.reviewed_at,
            )
            is None
        ):
            raise ValueError("semantics reviewed_at must be a canonical ISO-8601 timestamp")
        try:
            parsed = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("semantics reviewed_at must be an ISO-8601 timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("semantics reviewed_at must include an explicit timezone")
        if self.approved is not True:
            raise ValueError("semantics approval must be exactly true")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "SemanticsApprovalEnvelope":
        expected = {"reviewed_by", "reviewed_at", "approved"}
        if set(payload) != expected:
            raise ValueError(
                "semantics approval fields differ from strict schema; "
                f"missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)}"
            )
        return cls(
            reviewed_by=payload["reviewed_by"],  # type: ignore[arg-type]
            reviewed_at=payload["reviewed_at"],  # type: ignore[arg-type]
            approved=payload["approved"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "approved": self.approved,
        }


@dataclass(frozen=True, slots=True)
class CanonicalSemanticsSnapshot:
    """Canonical bytes and approval identity of the source governance document."""

    canonical_bytes: bytes
    approval: SemanticsApprovalEnvelope

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_bytes, bytes) or not self.canonical_bytes:
            raise ValueError("canonical semantics snapshot must contain bytes")
        try:
            payload = json.loads(self.canonical_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("canonical semantics snapshot is not valid UTF-8 JSON") from error
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "approval", "captures"
        }:
            raise ValueError("canonical semantics snapshot fields differ from strict schema")
        if payload["schema_version"] != 2:
            raise ValueError("canonical semantics snapshot schema_version must be 2")
        if not isinstance(payload["captures"], list) or not payload["captures"]:
            raise ValueError("canonical semantics snapshot requires non-empty captures")
        if not isinstance(payload["approval"], dict):
            raise ValueError("canonical semantics approval must be an object")
        parsed_approval = SemanticsApprovalEnvelope.from_mapping(payload["approval"])
        if parsed_approval != self.approval:
            raise ValueError("canonical semantics approval differs from snapshot bytes")
        if self.canonical_bytes != _canonical_json_bytes(payload):
            raise ValueError("semantics snapshot bytes are not canonical JSON")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CanonicalSemanticsSnapshot":
        if set(payload) != {"schema_version", "approval", "captures"}:
            raise ValueError("capture semantics fields differ from strict schema")
        approval_payload = payload.get("approval")
        if not isinstance(approval_payload, Mapping):
            raise ValueError("capture semantics approval must be an object")
        approval = SemanticsApprovalEnvelope.from_mapping(approval_payload)
        return cls(_canonical_json_bytes(dict(payload)), approval)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalSampleRow:
    """One canonical crop and its complete source/label audit identity."""

    dataset_release_id: str
    part_instance_id: str
    capture_set_id: str
    hand: str
    view: str
    source_path: str
    source_sha256: str
    crop_path: str
    crop_sha256: str
    roi_version: str
    roi_sha256: str
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int
    crop_width: int
    crop_height: int
    label: str
    defect_type: str
    split: str
    is_synthetic: bool
    annotation_state: str
    source_label_sha256: str
    crop_label_sha256: str
    source_box_count: int
    crop_box_count: int
    clipped_box_count: int
    outside_roi_box_count: int
    dropped_box_count: int

    def __post_init__(self) -> None:
        if not all(
            value
            for value in (
                self.dataset_release_id,
                self.part_instance_id,
                self.capture_set_id,
                self.view,
                self.roi_version,
            )
        ):
            raise ValueError("canonical sample identity fields must not be empty")
        for field in ("source_path", "crop_path"):
            path = PurePosixPath(getattr(self, field))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"canonical {field} must be a safe relative path")
        if self.hand not in {item.value for item in Hand}:
            raise ValueError(f"invalid canonical sample hand: {self.hand!r}")
        if self.label not in {"normal", "defect"}:
            raise ValueError(f"label must be normal or defect, got {self.label!r}")
        if self.label == "normal" and self.defect_type:
            raise ValueError("normal canonical sample cannot have defect_type")
        if self.label == "defect" and not self.defect_type:
            raise ValueError("defect canonical sample requires defect_type")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError(f"invalid canonical split: {self.split!r}")
        if self.annotation_state not in {"annotated", "confirmed_empty"}:
            raise ValueError(f"invalid annotation_state: {self.annotation_state!r}")
        if self.is_synthetic:
            raise ValueError("canonical source releases do not accept synthetic images")
        for field in (
            "source_sha256",
            "crop_sha256",
            "roi_sha256",
            "source_label_sha256",
            "crop_label_sha256",
        ):
            require_sha256(getattr(self, field), field)
        roi_values = (
            self.roi_x1,
            self.roi_y1,
            self.roi_x2,
            self.roi_y2,
            self.crop_width,
            self.crop_height,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in roi_values):
            raise ValueError("canonical ROI and crop dimensions must be integers")
        if self.roi_x1 < 0 or self.roi_y1 < 0:
            raise ValueError("canonical ROI origin must be non-negative")
        if self.roi_x2 <= self.roi_x1 or self.roi_y2 <= self.roi_y1:
            raise ValueError("canonical ROI must have positive half-open area")
        if (self.crop_width, self.crop_height) != (
            self.roi_x2 - self.roi_x1,
            self.roi_y2 - self.roi_y1,
        ):
            raise ValueError("canonical crop dimensions must equal ROI half-open extent")
        counts = (
            self.source_box_count,
            self.crop_box_count,
            self.clipped_box_count,
            self.outside_roi_box_count,
            self.dropped_box_count,
        )
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counts
        ):
            raise ValueError("canonical annotation counts must be non-negative integers")
        if self.crop_box_count + self.dropped_box_count != self.source_box_count:
            raise ValueError("crop_box_count + dropped_box_count must equal source_box_count")
        if self.clipped_box_count > self.crop_box_count:
            raise ValueError("clipped_box_count cannot exceed retained crop boxes")
        if self.outside_roi_box_count > self.dropped_box_count:
            raise ValueError("outside_roi_box_count cannot exceed dropped boxes")
        if self.annotation_state == "confirmed_empty" and self.source_box_count:
            raise ValueError("confirmed_empty sample cannot contain source boxes")
        if self.annotation_state == "annotated" and self.source_box_count == 0:
            raise ValueError("annotated sample must contain source boxes")

    def to_csv_row(self) -> dict[str, str]:
        values = asdict(self)
        return {
            key: (str(value).lower() if isinstance(value, bool) else str(value))
            for key, value in values.items()
        }

    def to_roi_sample(self) -> RoiSample:
        """Project one manifest row into the canonical domain RoiSample."""
        return RoiSample(
            part=PartIdentity(self.part_instance_id, Hand.parse(self.hand)),
            capture_set_id=self.capture_set_id,
            view_id=self.view,
            source_sha256=self.source_sha256,
            crop_sha256=self.crop_sha256,
            roi_config_id=self.roi_version,
            crop_width=self.crop_width,
            crop_height=self.crop_height,
        )


@dataclass(frozen=True, slots=True)
class CaptureProvenanceRow:
    """One capture-level gate identity retained by the canonical release."""

    capture_set_id: str
    part_instance_id: str
    hand: str
    policy_id: str
    policy_sha256: str
    topology_sha256: str
    acquisition_config_sha256: str
    quality_profile_sha256: str
    registration_profile_sha256: str
    registration_reference_sha256_by_view: Mapping[str, str]

    def __post_init__(self) -> None:
        for field in ("capture_set_id", "part_instance_id"):
            value = require_non_empty(getattr(self, field), field)
            if any(token in value for token in ("/", "\\", "\x00")):
                raise ValueError(f"capture provenance {field} is unsafe")
            object.__setattr__(self, field, value)
        provenance = CaptureGateProvenance(
            policy_id=self.policy_id,
            policy_sha256=self.policy_sha256,
            hand=Hand.parse(self.hand),
            topology_sha256=self.topology_sha256,
            acquisition_config_sha256=self.acquisition_config_sha256,
            quality_profile_sha256=self.quality_profile_sha256,
            registration_profile_sha256=self.registration_profile_sha256,
            registration_reference_sha256_by_view=(
                self.registration_reference_sha256_by_view
            ),
        )
        object.__setattr__(self, "hand", provenance.hand.value)
        object.__setattr__(
            self,
            "registration_reference_sha256_by_view",
            provenance.registration_reference_sha256_by_view,
        )

    @classmethod
    def from_capture(
        cls,
        *,
        capture_set_id: str,
        part_instance_id: str,
        provenance: CaptureGateProvenance,
    ) -> "CaptureProvenanceRow":
        return cls(
            capture_set_id=capture_set_id,
            part_instance_id=part_instance_id,
            hand=provenance.hand.value,
            policy_id=provenance.policy_id,
            policy_sha256=provenance.policy_sha256,
            topology_sha256=provenance.topology_sha256,
            acquisition_config_sha256=provenance.acquisition_config_sha256,
            quality_profile_sha256=provenance.quality_profile_sha256,
            registration_profile_sha256=provenance.registration_profile_sha256,
            registration_reference_sha256_by_view=(
                provenance.registration_reference_sha256_by_view
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CaptureProvenanceRow":
        expected = {
            "capture_set_id",
            "part_instance_id",
            "hand",
            "policy_id",
            "policy_sha256",
            "topology_sha256",
            "acquisition_config_sha256",
            "quality_profile_sha256",
            "registration_profile_sha256",
            "registration_reference_sha256_by_view",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("capture provenance row fields differ from the strict schema")
        references = payload["registration_reference_sha256_by_view"]
        if not isinstance(references, Mapping) or any(
            not isinstance(view, str) or not isinstance(digest, str)
            for view, digest in references.items()
        ):
            raise TypeError("capture provenance references must be a string mapping")
        scalar_fields = expected - {"registration_reference_sha256_by_view"}
        if any(not isinstance(payload[field], str) for field in scalar_fields):
            raise TypeError("capture provenance scalar fields must be strings")
        return cls(
            capture_set_id=payload["capture_set_id"],  # type: ignore[arg-type]
            part_instance_id=payload["part_instance_id"],  # type: ignore[arg-type]
            hand=payload["hand"],  # type: ignore[arg-type]
            policy_id=payload["policy_id"],  # type: ignore[arg-type]
            policy_sha256=payload["policy_sha256"],  # type: ignore[arg-type]
            topology_sha256=payload["topology_sha256"],  # type: ignore[arg-type]
            acquisition_config_sha256=payload["acquisition_config_sha256"],  # type: ignore[arg-type]
            quality_profile_sha256=payload["quality_profile_sha256"],  # type: ignore[arg-type]
            registration_profile_sha256=payload["registration_profile_sha256"],  # type: ignore[arg-type]
            registration_reference_sha256_by_view=references,  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "capture_set_id": self.capture_set_id,
            "part_instance_id": self.part_instance_id,
            "hand": self.hand,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "topology_sha256": self.topology_sha256,
            "acquisition_config_sha256": self.acquisition_config_sha256,
            "quality_profile_sha256": self.quality_profile_sha256,
            "registration_profile_sha256": self.registration_profile_sha256,
            "registration_reference_sha256_by_view": dict(
                sorted(self.registration_reference_sha256_by_view.items())
            ),
        }


@dataclass(frozen=True, slots=True)
class DatasetReleaseManifest:
    """Content-addressed identity of one immutable canonical dataset release."""

    schema_version: int
    dataset_release_id: str
    product: str
    topology_id: str
    topology_sha256: str
    roi_version: str
    roi_sha256: str
    capture_gate_policy_sha256: str
    bbox_migration_policy: str
    required_views: tuple[str, ...]
    hands: tuple[str, ...]
    sample_count: int
    part_count: int
    capture_set_count: int
    canonical_manifest_sha256: str
    capture_provenance_sha256: str
    bbox_audit_sha256: str
    canonical_semantics_sha256: str
    calibration_targets_sha256: str
    calibration_target_count: int
    dataset_provenance_sha256: str
    split_assignments_sha256: str
    split_policy: Mapping[str, object]
    split_policy_sha256: str
    canonical_png_codec: Mapping[str, object]
    canonical_png_codec_sha256: str
    adapter_manifest_sha256: Mapping[str, str]
    created_at: str

    def __post_init__(self) -> None:
        if self.schema_version != 4 or self.product != "ZS32":
            raise ValueError("dataset release must use schema_version 4 and product ZS32")
        if not all((self.dataset_release_id, self.topology_id, self.roi_version)):
            raise ValueError("dataset release identities must not be empty")
        if self.bbox_migration_policy != "clip_partial_drop_outside_audit_all":
            raise ValueError("unsupported bbox migration policy")
        for field in (
            "topology_sha256",
            "roi_sha256",
            "capture_gate_policy_sha256",
            "canonical_manifest_sha256",
            "capture_provenance_sha256",
            "bbox_audit_sha256",
            "canonical_semantics_sha256",
            "calibration_targets_sha256",
            "dataset_provenance_sha256",
            "split_assignments_sha256",
            "split_policy_sha256",
            "canonical_png_codec_sha256",
        ):
            require_sha256(getattr(self, field), field)
        if set(self.split_policy) != {
            "algorithm", "seed", "calibration_ratio", "test_ratio"
        }:
            raise ValueError("dataset release split_policy fields differ from strict schema")
        if set(self.canonical_png_codec) != {
            "encoder", "implementation_version", "format", "media_type",
            "compression", "spatial_operation"
        }:
            raise ValueError("dataset release canonical_png_codec fields differ from strict schema")
        if hashlib.sha256(
            _canonical_json_bytes(dict(self.split_policy))
        ).hexdigest() != self.split_policy_sha256:
            raise ValueError("dataset release split_policy SHA256 mismatch")
        if hashlib.sha256(
            _canonical_json_bytes(dict(self.canonical_png_codec))
        ).hexdigest() != self.canonical_png_codec_sha256:
            raise ValueError("dataset release canonical_png_codec SHA256 mismatch")
        for adapter, digest in self.adapter_manifest_sha256.items():
            if adapter not in {"yolo", "anomalib", "template"}:
                raise ValueError(f"unknown dataset adapter manifest: {adapter!r}")
            require_sha256(digest, f"adapter_manifest_sha256.{adapter}")
        if set(self.adapter_manifest_sha256) != {"yolo", "anomalib", "template"}:
            raise ValueError("dataset release requires yolo, anomalib, and template adapter manifests")
        count_fields = (
            "sample_count",
            "part_count",
            "capture_set_count",
            "calibration_target_count",
        )
        if any(
            isinstance(getattr(self, field), bool)
            or not isinstance(getattr(self, field), int)
            for field in count_fields
        ):
            raise ValueError("dataset release count fields must be strict integers")
        if not self.required_views or not self.hands or self.sample_count <= 0:
            raise ValueError("dataset release manifest cannot be empty")
        if len(self.required_views) != len(set(self.required_views)):
            raise ValueError("dataset release required_views must be unique")
        if not set(self.hands).issubset({item.value for item in Hand}):
            raise ValueError(f"dataset release contains invalid hands: {self.hands}")
        if len(self.hands) != len(set(self.hands)):
            raise ValueError("dataset release hands must be unique")
        if self.part_count <= 0 or self.capture_set_count <= 0:
            raise ValueError("dataset release part/capture counts must be positive")
        if self.calibration_target_count != 3 * self.sample_count:
            raise ValueError("dataset release requires exactly three calibration targets per sample")
        if (
            not isinstance(self.created_at, str)
            or self.created_at != self.created_at.strip()
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
                self.created_at,
            )
            is None
        ):
            raise ValueError("dataset release created_at must be canonical RFC3339")
        try:
            created_at = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("dataset release created_at is invalid") from error
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("dataset release created_at must include a timezone")
        object.__setattr__(self, "required_views", tuple(self.required_views))
        object.__setattr__(self, "hands", tuple(self.hands))
        object.__setattr__(
            self,
            "adapter_manifest_sha256",
            MappingProxyType(dict(self.adapter_manifest_sha256)),
        )
        object.__setattr__(
            self,
            "split_policy",
            MappingProxyType(dict(self.split_policy)),
        )
        object.__setattr__(
            self,
            "canonical_png_codec",
            MappingProxyType(dict(self.canonical_png_codec)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_release_id": self.dataset_release_id,
            "product": self.product,
            "topology_id": self.topology_id,
            "topology_sha256": self.topology_sha256,
            "roi_version": self.roi_version,
            "roi_sha256": self.roi_sha256,
            "capture_gate_policy_sha256": self.capture_gate_policy_sha256,
            "bbox_migration_policy": self.bbox_migration_policy,
            "required_views": list(self.required_views),
            "hands": list(self.hands),
            "sample_count": self.sample_count,
            "part_count": self.part_count,
            "capture_set_count": self.capture_set_count,
            "canonical_manifest_sha256": self.canonical_manifest_sha256,
            "capture_provenance_sha256": self.capture_provenance_sha256,
            "bbox_audit_sha256": self.bbox_audit_sha256,
            "canonical_semantics_sha256": self.canonical_semantics_sha256,
            "calibration_targets_sha256": self.calibration_targets_sha256,
            "calibration_target_count": self.calibration_target_count,
            "dataset_provenance_sha256": self.dataset_provenance_sha256,
            "split_assignments_sha256": self.split_assignments_sha256,
            "split_policy": dict(self.split_policy),
            "split_policy_sha256": self.split_policy_sha256,
            "canonical_png_codec": dict(self.canonical_png_codec),
            "canonical_png_codec_sha256": self.canonical_png_codec_sha256,
            "adapter_manifest_sha256": dict(sorted(self.adapter_manifest_sha256.items())),
            "created_at": self.created_at,
        }
