"""Build and verify immutable canonical ROI dataset releases."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence, cast

from zs32_inspection.capture.contracts import CapturePlan
from zs32_inspection.capture.gate_policy import CaptureGateProvenance
from zs32_inspection.capture.gate_publication import (
    VerifiedCaptureGatePublication,
    load_verified_capture_gate_publication,
)
from zs32_inspection.config.guards import require_bound_capture_gate_policy
from zs32_inspection.domain.contracts import InspectionRecipe, RoiBox, RoiConfig
from zs32_inspection.domain.identity import CaptureSet, Hand, RoiSample
from zs32_inspection.domain.topology import CaptureTopology
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    canonical_json_bytes,
    sha256_file,
    tree_checksums,
)

from .codec_contract import CanonicalPngCodec
from .calibration_targets import (
    CalibrationTargetsSnapshot,
    validate_calibration_targets,
)
from .manifests import (
    ADAPTER_BASE_COLUMNS,
    CANONICAL_MANIFEST_COLUMNS,
    YOLO_ADAPTER_COLUMNS,
    CanonicalSemanticsSnapshot,
    CanonicalSampleRow,
    CaptureProvenanceRow,
    DatasetReleaseManifest,
)
from .roi import (
    ImageCropper,
    LabelDocument,
    LabelMigrationAudit,
    migrate_yolo_labels,
    validate_crop_result,
)
from .splitter import (
    SplitAssignment,
    SplitPolicy,
    assign_grouped_splits,
    validate_part_grouped_splits,
)


def _safe_relative_path(value: str, field: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a safe relative POSIX path: {value!r}")
    return path.as_posix()


def _semantics_capture_bindings(
    snapshot: CanonicalSemanticsSnapshot,
    required_views: Sequence[str],
) -> dict[str, Mapping[str, object]]:
    payload = json.loads(snapshot.canonical_bytes.decode("utf-8"))
    bindings: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(payload["captures"]):
        if not isinstance(raw, Mapping) or set(raw) != {
            "capture_set_path", "label", "defect_type", "annotations"
        }:
            raise ValueError(
                f"canonical semantics captures[{index}] fields differ from strict schema"
            )
        capture_path = _safe_relative_path(
            raw["capture_set_path"],  # type: ignore[arg-type]
            f"semantics captures[{index}].capture_set_path",
        )
        if capture_path in bindings:
            raise ValueError(f"duplicate semantics capture_set_path: {capture_path!r}")
        label = raw["label"]
        defect_type = raw["defect_type"]
        if label not in {"normal", "defect"} or not isinstance(defect_type, str):
            raise ValueError(f"invalid semantics label fields for {capture_path!r}")
        if (label == "normal" and defect_type) or (label == "defect" and not defect_type.strip()):
            raise ValueError(f"invalid semantics defect_type for {capture_path!r}")
        annotations = raw["annotations"]
        if not isinstance(annotations, Mapping) or set(annotations) != set(required_views):
            raise ValueError(
                f"semantics annotation views differ from topology for {capture_path!r}"
            )
        for view, annotation in annotations.items():
            if not isinstance(annotation, Mapping):
                raise ValueError(f"semantics annotation {capture_path}/{view} must be an object")
            required = {"state", "source_path"}
            allowed = required | {"text", "label_path"}
            if not required.issubset(annotation) or not set(annotation).issubset(allowed):
                raise ValueError(
                    f"semantics annotation fields differ from strict schema: {capture_path}/{view}"
                )
            if ("text" in annotation) == ("label_path" in annotation):
                raise ValueError(
                    f"semantics annotation must contain exactly one label source: "
                    f"{capture_path}/{view}"
                )
            if annotation["state"] not in {"annotated", "confirmed_empty"}:
                raise ValueError(
                    f"semantics annotation is not approved complete: {capture_path}/{view}"
                )
            _safe_relative_path(
                annotation["source_path"],  # type: ignore[arg-type]
                f"semantics annotation source_path {capture_path}/{view}",
            )
            if "text" in annotation and not isinstance(annotation["text"], str):
                raise ValueError(
                    f"semantics annotation text must be a string: {capture_path}/{view}"
                )
            if "label_path" in annotation:
                _safe_relative_path(
                    annotation["label_path"],  # type: ignore[arg-type]
                    f"semantics annotation label_path {capture_path}/{view}",
                )
        bindings[capture_path] = raw
    return bindings


def _capture_path_from_source(
    source_path: str,
    *,
    capture_set_id: str,
    view_id: str,
) -> str:
    path = PurePosixPath(_safe_relative_path(source_path, "source_path"))
    if (
        len(path.parts) < 5
        or path.parts[0] != "raw"
        or path.parts[-3] != "images"
        or path.parts[-2] != capture_set_id
        or path.name != f"{view_id}.png"
    ):
        raise ValueError(
            "source_path must use raw/<session>/images/<capture_set_id>/<view>.png layout"
        )
    return PurePosixPath(*path.parts[1:-1]).as_posix()


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class SourceSample:
    """Raw payload plus explicit semantic labels for one domain ViewImage."""

    capture_set: CaptureSet
    view_id: str
    source_png: bytes
    source_path: str
    label: str
    defect_type: str
    annotation: LabelDocument
    gate_provenance: CaptureGateProvenance
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.capture_set, CaptureSet):
            raise TypeError("source sample capture_set must be a domain CaptureSet")
        if self.view_id not in self.capture_set.images:
            raise ValueError(f"view {self.view_id!r} is absent from source capture set")
        if not self.source_png:
            raise ValueError("source sample PNG payload must not be empty")
        if not self.source_png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("source sample payload must have a PNG signature")
        _safe_relative_path(self.source_path, "source_path")
        if not isinstance(self.annotation, LabelDocument):
            raise TypeError("source sample annotation must be a LabelDocument")
        if not isinstance(self.gate_provenance, CaptureGateProvenance):
            raise TypeError("source sample gate_provenance must be structured")
        if self.gate_provenance.hand is not self.capture_set.part.hand:
            raise ValueError("source sample gate provenance hand differs from capture")
        if self.gate_provenance.topology_sha256 != self.capture_set.topology_sha256:
            raise ValueError("source sample gate provenance topology differs from capture")
        if self.label not in {"normal", "defect"}:
            raise ValueError(f"source label must be normal or defect, got {self.label!r}")
        if self.label == "normal" and self.defect_type:
            raise ValueError("normal source sample cannot have a defect_type")
        if self.label == "normal" and self.annotation.state != "confirmed_empty":
            raise ValueError("normal source sample requires an explicit confirmed_empty annotation")
        if self.label == "defect" and not self.defect_type.strip():
            raise ValueError("defect source sample requires defect_type")
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("source sample is_synthetic must be a boolean")
        if self.is_synthetic:
            raise ValueError("synthetic images cannot enter the canonical source release")
        view = self.capture_set.images[self.view_id]
        source_parts = PurePosixPath(self.source_path).parts
        view_parts = PurePosixPath(view.relative_path).parts
        if source_parts[-len(view_parts) :] != view_parts:
            raise ValueError(
                f"source_path does not end with domain ViewImage.relative_path: "
                f"{self.source_path!r} vs {view.relative_path!r}"
            )
        actual = hashlib.sha256(self.source_png).hexdigest()
        if actual != view.image_sha256:
            raise ValueError(
                f"source payload hash mismatch for {self.capture_set.capture_set_id}/{self.view_id}: "
                f"{actual} != {view.image_sha256}"
            )


@dataclass(frozen=True, slots=True)
class DatasetBuildSpec:
    """All frozen identities needed to build one canonical dataset release."""

    dataset_release_id: str
    output_root: Path
    topology: CaptureTopology
    roi_config: RoiConfig
    recipe: InspectionRecipe
    enabled_hands: tuple[Hand, ...]
    split_assignments: tuple[SplitAssignment, ...]
    split_policy: SplitPolicy
    semantics_snapshot: CanonicalSemanticsSnapshot
    canonical_png_codec: CanonicalPngCodec
    created_at: str
    gate_publication: VerifiedCaptureGatePublication
    calibration_targets: CalibrationTargetsSnapshot

    def __post_init__(self) -> None:
        if not self.dataset_release_id.strip() or any(
            token in self.dataset_release_id for token in ("/", "\\", "\x00", "\n", "\r")
        ):
            raise ValueError(f"unsafe dataset_release_id: {self.dataset_release_id!r}")
        if not isinstance(self.topology, CaptureTopology):
            raise TypeError("dataset topology must be a domain CaptureTopology")
        if not isinstance(self.roi_config, RoiConfig):
            raise TypeError("dataset ROI must be a domain RoiConfig")
        if not isinstance(self.recipe, InspectionRecipe):
            raise TypeError("dataset recipe must be a domain InspectionRecipe")
        if not isinstance(self.gate_publication, VerifiedCaptureGatePublication):
            raise TypeError(
                "dataset gate_publication must be a VerifiedCaptureGatePublication"
            )
        verified_gate_publication = load_verified_capture_gate_publication(
            self.gate_publication.publication.root
        )
        object.__setattr__(self, "gate_publication", verified_gate_publication)
        CapturePlan.from_topology(self.topology)
        if self.roi_config.topology_id != self.topology.topology_id:
            raise ValueError("ROI topology_id does not match dataset topology")
        hands = tuple(sorted((Hand.parse(item) for item in self.enabled_hands), key=lambda item: item.value))
        if not hands or len(hands) != len(set(hands)):
            raise ValueError("enabled_hands must be non-empty and unique")
        for hand in hands:
            self.roi_config.require_ready(hand, self.topology.required_views)
        if self.recipe.product != self.topology.product:
            raise ValueError("recipe product differs from dataset topology")
        if self.recipe.topology_id != self.topology.topology_id:
            raise ValueError("recipe topology differs from dataset topology")
        if self.recipe.roi_config_id != self.roi_config.roi_config_id:
            raise ValueError("recipe ROI differs from canonical dataset ROI")
        if set(self.recipe.allowed_hands) != set(hands):
            raise ValueError("recipe allowed_hands differ from dataset enabled_hands")
        require_bound_capture_gate_policy(self.recipe)
        if (
            self.recipe.capture_gate_policy.sha256
            != self.gate_publication.policy_sha256
        ):
            raise ValueError("verified capture gate policy differs from dataset recipe")
        self.gate_publication.policy.validate_topology(
            self.topology,
            allowed_hands=hands,
        )
        object.__setattr__(self, "enabled_hands", hands)
        if not isinstance(self.split_policy, SplitPolicy):
            raise TypeError("dataset split_policy must be structured")
        if not isinstance(self.semantics_snapshot, CanonicalSemanticsSnapshot):
            raise TypeError("dataset semantics_snapshot must be canonical and approved")
        if not isinstance(self.canonical_png_codec, CanonicalPngCodec):
            raise TypeError("dataset canonical_png_codec must be structured")
        if not isinstance(self.calibration_targets, CalibrationTargetsSnapshot):
            raise TypeError("dataset calibration_targets must be structured and approved")
        if (
            self.calibration_targets.dataset_release_id != self.dataset_release_id
            or self.calibration_targets.topology_id != self.topology.topology_id
            or self.calibration_targets.roi_version != self.roi_config.roi_config_id
        ):
            raise ValueError("calibration target identities differ from dataset build spec")
        object.__setattr__(self, "split_assignments", tuple(self.split_assignments))
        assignment_ids = tuple(item.part_instance_id for item in self.split_assignments)
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("split_assignments contains duplicate part_instance_id values")
        if not self.created_at.strip():
            raise ValueError("dataset release created_at must not be empty")


@dataclass(frozen=True, slots=True)
class DatasetRelease:
    """Published release path, content manifest, and canonical RoiSamples."""

    path: Path
    manifest: DatasetReleaseManifest
    samples: tuple[RoiSample, ...]


@dataclass(frozen=True, slots=True)
class _PreparedSample:
    row: CanonicalSampleRow
    roi_sample: RoiSample
    crop_bytes: bytes
    crop_width: int
    crop_height: int
    source_label_text: str
    label_text: str
    label_audit: LabelMigrationAudit
    sample_id: str


class CanonicalDatasetBuilder:
    """Crop each raw image once and atomically publish all adapter views."""

    def __init__(self, cropper: ImageCropper) -> None:
        self._cropper = cropper

    def build(
        self,
        spec: DatasetBuildSpec,
        sources: Sequence[SourceSample],
    ) -> DatasetRelease:
        """Build an immutable release or leave no consumable destination."""
        if not sources:
            raise ValueError("canonical dataset build requires source samples")
        self._validate_source_sets(spec, sources)
        semantics_by_path = _semantics_capture_bindings(
            spec.semantics_snapshot,
            spec.topology.required_views,
        )
        sources_by_path: dict[str, list[SourceSample]] = {}
        for source in sources:
            sources_by_path.setdefault(
                _capture_path_from_source(
                    source.source_path,
                    capture_set_id=source.capture_set.capture_set_id,
                    view_id=source.view_id,
                ),
                [],
            ).append(source)
        if set(semantics_by_path) != set(sources_by_path):
            raise ValueError(
                "canonical semantics capture identities differ from dataset sources; "
                f"missing={sorted(set(sources_by_path) - set(semantics_by_path))}, "
                f"extra={sorted(set(semantics_by_path) - set(sources_by_path))}"
            )
        for capture_path, capture_sources in sources_by_path.items():
            semantics = semantics_by_path[capture_path]
            annotations = cast(
                Mapping[str, Mapping[str, object]],
                semantics["annotations"],
            )
            for source in capture_sources:
                annotation = annotations[source.view_id]
                if (
                    source.label != semantics["label"]
                    or source.defect_type != semantics["defect_type"]
                    or source.annotation.state != annotation["state"]
                    or source.annotation.source_path != annotation["source_path"]
                    or (
                        "text" in annotation
                        and source.annotation.text != annotation["text"]
                    )
                ):
                    raise ValueError(
                        "canonical semantics differs from source labels for "
                        f"{capture_path!r}"
                    )
        actual_codec = getattr(self._cropper, "canonical_png_codec", None)
        if actual_codec != spec.canonical_png_codec:
            raise ValueError(
                "cropper canonical PNG codec differs from dataset build provenance"
            )
        split_by_part = validate_part_grouped_splits(
            ((item.part_instance_id, item.split) for item in spec.split_assignments),
            required_part_ids=sorted({item.capture_set.part.part_instance_id for item in sources}),
        )
        strata_by_part: dict[str, str] = {}
        for source in sources:
            part_id = source.capture_set.part.part_instance_id
            stratum = (
                f"{source.capture_set.part.hand.value}:"
                f"{source.label}:{source.defect_type or 'none'}"
            )
            previous = strata_by_part.setdefault(part_id, stratum)
            if previous != stratum:
                raise ValueError(f"part {part_id!r} has conflicting split strata")
        expected_assignments = assign_grouped_splits(
            strata_by_part,
            policy=spec.split_policy,
        )
        if spec.split_assignments != expected_assignments:
            raise ValueError(
                "split assignments are not the deterministic result of the bound SplitPolicy"
            )
        prepared = tuple(
            self._prepare_sample(spec, source, split_by_part[source.capture_set.part.part_instance_id])
            for source in sorted(
                sources,
                key=lambda item: (
                    item.capture_set.part.part_instance_id,
                    item.capture_set.capture_set_id,
                    spec.topology.required_views.index(item.view_id),
                ),
            )
        )
        validate_calibration_targets(
            spec.calibration_targets,
            tuple(item.row for item in prepared),
            dataset_release_id=spec.dataset_release_id,
            topology_id=spec.topology.topology_id,
            roi_version=spec.roi_config.roi_config_id,
        )
        canonical_bytes = _csv_bytes(
            CANONICAL_MANIFEST_COLUMNS,
            [item.row.to_csv_row() for item in prepared],
        )
        bbox_bytes = _jsonl_bytes(
            [
                {
                    "capture_set_id": item.row.capture_set_id,
                    "part_instance_id": item.row.part_instance_id,
                    "view": item.row.view,
                    "audit": item.label_audit.as_dict(),
                }
                for item in prepared
            ]
        )
        provenance_rows = self._capture_provenance_rows(spec, sources)
        provenance_bytes = _jsonl_bytes([row.as_dict() for row in provenance_rows])
        split_bytes = canonical_json_bytes(
            {
                "schema_version": 2,
                "policy": spec.split_policy.as_dict(),
                "policy_sha256": spec.split_policy.sha256,
                "assignments": [
                    {
                        "part_instance_id": item.part_instance_id,
                        "split": item.split,
                        "stratum": item.stratum,
                    }
                    for item in sorted(spec.split_assignments, key=lambda value: value.part_instance_id)
                ],
            }
        )
        dataset_provenance_bytes = canonical_json_bytes(
            {
                "schema_version": 2,
                "canonical_semantics": {
                    "relative_path": "canonical_semantics.json",
                    "sha256": spec.semantics_snapshot.sha256,
                    "approval": spec.semantics_snapshot.approval.as_dict(),
                },
                "calibration_targets": {
                    "relative_path": "calibration_targets.json",
                    "sha256": spec.calibration_targets.sha256,
                    "target_count": len(spec.calibration_targets.targets),
                    "approval": spec.calibration_targets.approval.as_dict(),
                },
                "split": {
                    "relative_path": "split_assignments.json",
                    "artifact_sha256": hashlib.sha256(split_bytes).hexdigest(),
                    "policy": spec.split_policy.as_dict(),
                    "policy_sha256": spec.split_policy.sha256,
                },
                "canonical_png": {
                    "codec": spec.canonical_png_codec.as_dict(),
                    "codec_sha256": spec.canonical_png_codec.sha256,
                },
            }
        )
        adapter_payloads = self._adapter_manifests(prepared)
        manifest = DatasetReleaseManifest(
            schema_version=4,
            dataset_release_id=spec.dataset_release_id,
            product="ZS32",
            topology_id=spec.topology.topology_id,
            topology_sha256=spec.topology.topology_sha256,
            roi_version=spec.roi_config.roi_config_id,
            roi_sha256=spec.roi_config.roi_sha256,
            capture_gate_policy_sha256=spec.recipe.capture_gate_policy.sha256,
            bbox_migration_policy="clip_partial_drop_outside_audit_all",
            required_views=spec.topology.required_views,
            hands=tuple(item.value for item in spec.enabled_hands),
            sample_count=len(prepared),
            part_count=len({item.row.part_instance_id for item in prepared}),
            capture_set_count=len({item.row.capture_set_id for item in prepared}),
            canonical_manifest_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
            capture_provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
            bbox_audit_sha256=hashlib.sha256(bbox_bytes).hexdigest(),
            canonical_semantics_sha256=spec.semantics_snapshot.sha256,
            calibration_targets_sha256=spec.calibration_targets.sha256,
            calibration_target_count=len(spec.calibration_targets.targets),
            dataset_provenance_sha256=hashlib.sha256(
                dataset_provenance_bytes
            ).hexdigest(),
            split_assignments_sha256=hashlib.sha256(split_bytes).hexdigest(),
            split_policy=spec.split_policy.as_dict(),
            split_policy_sha256=spec.split_policy.sha256,
            canonical_png_codec=spec.canonical_png_codec.as_dict(),
            canonical_png_codec_sha256=spec.canonical_png_codec.sha256,
            adapter_manifest_sha256={
                adapter: hashlib.sha256(payload).hexdigest()
                for adapter, payload in adapter_payloads.items()
            },
            created_at=spec.created_at,
        )
        required_paths = {
            "dataset_release.json",
            "canonical_manifest.csv",
            "capture_provenance.jsonl",
            "bbox_migration_audit.jsonl",
            "split_assignments.json",
            "canonical_semantics.json",
            "calibration_targets.json",
            "dataset_provenance.json",
            "roi_contract.json",
            "topology_contract.json",
            *(f"adapters/{adapter}/manifest.csv" for adapter in adapter_payloads),
        }
        for item in prepared:
            required_paths.add(item.row.crop_path)
            required_paths.add(f"labels/source/{item.sample_id}.txt")
            required_paths.add(f"adapters/yolo/labels/{item.row.split}/{item.sample_id}.txt")
        with AtomicDirectoryPublisher(spec.output_root, spec.dataset_release_id) as publisher:
            for item in prepared:
                publisher.write_bytes(item.row.crop_path, item.crop_bytes)
                publisher.write_bytes(
                    f"labels/source/{item.sample_id}.txt",
                    item.source_label_text.encode("utf-8"),
                )
                publisher.write_bytes(
                    f"adapters/yolo/labels/{item.row.split}/{item.sample_id}.txt",
                    item.label_text.encode("utf-8"),
                )
            publisher.write_bytes("canonical_manifest.csv", canonical_bytes)
            publisher.write_bytes("capture_provenance.jsonl", provenance_bytes)
            publisher.write_bytes("bbox_migration_audit.jsonl", bbox_bytes)
            publisher.write_bytes("split_assignments.json", split_bytes)
            publisher.write_bytes(
                "canonical_semantics.json",
                spec.semantics_snapshot.canonical_bytes,
            )
            publisher.write_bytes(
                "calibration_targets.json",
                spec.calibration_targets.canonical_bytes,
            )
            publisher.write_bytes("dataset_provenance.json", dataset_provenance_bytes)
            publisher.write_json("roi_contract.json", self._roi_contract(spec.roi_config))
            publisher.write_json(
                "topology_contract.json",
                self._topology_contract(spec.topology),
            )
            for adapter, payload in adapter_payloads.items():
                publisher.write_bytes(f"adapters/{adapter}/manifest.csv", payload)
            publisher.write_json("dataset_release.json", manifest.as_dict())
            destination = publisher.finalize(
                validator=lambda staging: _verify_release_content(staging, manifest),
                required_paths=frozenset(required_paths),
            )
        return DatasetRelease(destination, manifest, tuple(item.roi_sample for item in prepared))

    @staticmethod
    def _validate_source_sets(spec: DatasetBuildSpec, sources: Sequence[SourceSample]) -> None:
        by_capture: dict[str, list[SourceSample]] = {}
        seen_source_identity: set[tuple[str, str]] = set()
        part_identity_by_id = {}
        semantics_by_part: dict[str, tuple[str, str]] = {}
        provenance_by_hand: dict[Hand, CaptureGateProvenance] = {}
        for source in sources:
            identity = (source.capture_set.capture_set_id, source.view_id)
            if identity in seen_source_identity:
                raise ValueError(f"duplicate canonical source identity: {identity}")
            seen_source_identity.add(identity)
            capture = source.capture_set
            capture.validate_required_views(spec.topology.required_views)
            if capture.topology_id != spec.topology.topology_id:
                raise ValueError(f"capture {capture.capture_set_id} topology_id mismatch")
            if capture.topology_sha256 != spec.topology.topology_sha256:
                raise ValueError(f"capture {capture.capture_set_id} topology_sha256 mismatch")
            if capture.part.hand not in spec.enabled_hands:
                raise ValueError(f"capture {capture.capture_set_id} hand is not enabled")
            provenance = source.gate_provenance
            expected_provenance = spec.gate_publication.policy.provenance_for(
                capture.part.hand,
                policy_sha256=spec.gate_publication.policy_sha256,
            )
            if provenance != expected_provenance:
                raise ValueError(
                    f"capture {capture.capture_set_id} gate provenance differs from "
                    "the verified policy publication"
                )
            if set(provenance.registration_reference_sha256_by_view) != set(
                spec.topology.required_views
            ):
                raise ValueError(
                    f"capture {capture.capture_set_id} gate reference views differ from topology"
                )
            previous_provenance = provenance_by_hand.setdefault(
                capture.part.hand,
                provenance,
            )
            if previous_provenance != provenance:
                raise ValueError(
                    f"capture gate provenance is mixed within hand {capture.part.hand.value!r}"
                )
            previous_part = part_identity_by_id.setdefault(
                capture.part.part_instance_id,
                capture.part,
            )
            if previous_part != capture.part:
                raise ValueError(
                    f"part_instance_id {capture.part.part_instance_id!r} has conflicting hand/product identity"
                )
            semantics = (source.label, source.defect_type)
            previous_semantics = semantics_by_part.setdefault(
                capture.part.part_instance_id,
                semantics,
            )
            if previous_semantics != semantics:
                raise ValueError(
                    f"part_instance_id {capture.part.part_instance_id!r} has conflicting "
                    f"label/defect_type semantics"
                )
            by_capture.setdefault(capture.capture_set_id, []).append(source)
        for capture_set_id, items in by_capture.items():
            observed = {item.view_id for item in items}
            if observed != set(spec.topology.required_views):
                raise ValueError(
                    f"capture {capture_set_id} dataset source views are incomplete; "
                    f"missing={sorted(set(spec.topology.required_views) - observed)}, "
                    f"extra={sorted(observed - set(spec.topology.required_views))}"
                )
            first = items[0].capture_set
            if any(item.capture_set != first for item in items):
                raise ValueError(f"capture {capture_set_id} has conflicting capture/domain identity")
            semantics = {(item.label, item.defect_type) for item in items}
            if len(semantics) != 1:
                raise ValueError(
                    f"capture {capture_set_id} has conflicting label/defect_type semantics: "
                    f"{sorted(semantics)}"
                )
        observed_hands = {item.capture_set.part.hand for item in sources}
        if observed_hands != set(spec.enabled_hands):
            raise ValueError(
                "dataset source hands differ from enabled_hands; "
                f"observed={sorted(item.value for item in observed_hands)}, "
                f"enabled={sorted(item.value for item in spec.enabled_hands)}"
            )

    @staticmethod
    def _capture_provenance_rows(
        spec: DatasetBuildSpec,
        sources: Sequence[SourceSample],
    ) -> tuple[CaptureProvenanceRow, ...]:
        by_capture: dict[str, SourceSample] = {}
        for source in sources:
            by_capture.setdefault(source.capture_set.capture_set_id, source)
        return tuple(
            CaptureProvenanceRow.from_capture(
                capture_set_id=source.capture_set.capture_set_id,
                part_instance_id=source.capture_set.part.part_instance_id,
                provenance=source.gate_provenance,
            )
            for source in sorted(
                by_capture.values(),
                key=lambda item: item.capture_set.capture_set_id,
            )
        )

    def _prepare_sample(
        self,
        spec: DatasetBuildSpec,
        source: SourceSample,
        split: str,
    ) -> _PreparedSample:
        capture = source.capture_set
        raw = capture.images[source.view_id]
        hand_config = spec.roi_config.require_ready(capture.part.hand, spec.topology.required_views)
        roi = hand_config.views[source.view_id]
        if (raw.width, raw.height) != (
            spec.roi_config.source_width,
            spec.roi_config.source_height,
        ):
            raise ValueError(
                f"source dimensions do not match ROI config for {capture.capture_set_id}/"
                f"{source.view_id}: {raw.width}x{raw.height}"
            )
        crop = self._cropper.crop_png(
            source.source_png,
            source_width=raw.width,
            source_height=raw.height,
            roi=roi,
        )
        validate_crop_result(crop, roi)
        labels = migrate_yolo_labels(
            source.annotation,
            source_width=raw.width,
            source_height=raw.height,
            roi=roi,
        )
        sample_id = hashlib.sha256(
            f"{capture.capture_set_id}\x00{source.view_id}".encode("utf-8")
        ).hexdigest()[:24]
        crop_path = (
            f"crops/{capture.part.hand.value}/{source.view_id}/{sample_id}.png"
        )
        row = CanonicalSampleRow(
            dataset_release_id=spec.dataset_release_id,
            part_instance_id=capture.part.part_instance_id,
            capture_set_id=capture.capture_set_id,
            hand=capture.part.hand.value,
            view=source.view_id,
            source_path=_safe_relative_path(source.source_path, "source_path"),
            source_sha256=raw.image_sha256,
            crop_path=crop_path,
            crop_sha256=crop.sha256,
            roi_version=spec.roi_config.roi_config_id,
            roi_sha256=spec.roi_config.roi_sha256,
            roi_x1=roi.x1,
            roi_y1=roi.y1,
            roi_x2=roi.x2,
            roi_y2=roi.y2,
            crop_width=crop.width,
            crop_height=crop.height,
            label=source.label,
            defect_type=source.defect_type,
            split=split,
            is_synthetic=False,
            annotation_state=labels.audit.annotation_state,
            source_label_sha256=labels.audit.source_label_sha256,
            crop_label_sha256=labels.audit.crop_label_sha256,
            source_box_count=labels.audit.source_box_count,
            crop_box_count=labels.audit.crop_box_count,
            clipped_box_count=labels.audit.clipped_box_count,
            outside_roi_box_count=labels.audit.outside_roi_box_count,
            dropped_box_count=labels.audit.dropped_box_count,
        )
        roi_sample = row.to_roi_sample()
        return _PreparedSample(
            row=row,
            roi_sample=roi_sample,
            crop_bytes=crop.image_bytes,
            crop_width=crop.width,
            crop_height=crop.height,
            source_label_text=source.annotation.text or "",
            label_text=labels.text,
            label_audit=labels.audit,
            sample_id=sample_id,
        )

    @staticmethod
    def _adapter_manifests(prepared: Sequence[_PreparedSample]) -> dict[str, bytes]:
        base_columns = ADAPTER_BASE_COLUMNS
        base_rows = [
            {
                "sample_id": item.sample_id,
                "part_instance_id": item.row.part_instance_id,
                "capture_set_id": item.row.capture_set_id,
                "hand": item.row.hand,
                "view": item.row.view,
                "split": item.row.split,
                "label": item.row.label,
                "defect_type": item.row.defect_type,
                "canonical_crop_path": item.row.crop_path,
                "canonical_crop_sha256": item.row.crop_sha256,
            }
            for item in prepared
        ]
        yolo_columns = YOLO_ADAPTER_COLUMNS
        yolo_rows = [
            {
                **row,
                "crop_label_path": f"adapters/yolo/labels/{item.row.split}/{item.sample_id}.txt",
                "crop_label_sha256": item.row.crop_label_sha256,
            }
            for row, item in zip(base_rows, prepared, strict=True)
        ]
        return {
            "yolo": _csv_bytes(yolo_columns, yolo_rows),
            "anomalib": _csv_bytes(base_columns, base_rows),
            "template": _csv_bytes(base_columns, base_rows),
        }

    @staticmethod
    def _roi_contract(roi_config: RoiConfig) -> dict[str, object]:
        """Snapshot every explicit hand status/box while retaining the source digest."""
        return {
            "schema_version": roi_config.schema_version,
            "roi_version": roi_config.roi_config_id,
            "roi_source_sha256": roi_config.roi_sha256,
            "product": roi_config.product,
            "topology_id": roi_config.topology_id,
            "coordinate_system": roi_config.coordinate_system,
            "source_image_size": {
                "width": roi_config.source_width,
                "height": roi_config.source_height,
            },
            "hands": {
                hand.value: {
                    "status": config.status.value,
                    "reason": config.reason,
                    "views": {
                        view_id: {"xyxy": box.as_list()}
                        for view_id, box in sorted(config.views.items())
                    },
                }
                for hand, config in sorted(
                    roi_config.hands.items(),
                    key=lambda item: item[0].value,
                )
            },
        }

    @staticmethod
    def _topology_contract(topology: CaptureTopology) -> dict[str, object]:
        """Snapshot dynamic rounds/slots/views while retaining the source digest."""
        return {
            "schema_version": topology.schema_version,
            "topology_id": topology.topology_id,
            "topology_source_sha256": topology.topology_sha256,
            "product": topology.product,
            "rounds": [
                {"round_id": item.round_id, "prompt": item.prompt}
                for item in topology.rounds
            ],
            "camera_slots": [
                {
                    "slot_id": item.slot_id,
                    "serial": item.serial,
                    "views": dict(item.views),
                }
                for item in topology.camera_slots
            ],
            "required_views": list(topology.required_views),
        }


def _read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise PublicationError(
                f"CSV header differs from frozen schema: {path.relative_to(path.parent.parent)}"
            )
        rows = list(reader)
    if any(set(row) != set(expected_columns) or any(value is None for value in row.values()) for row in rows):
        raise PublicationError(f"CSV contains malformed or extra cells: {path}")
    return rows


def _verify_release_content(root: Path, manifest: DatasetReleaseManifest) -> None:
    """Validate semantic hashes, crop hashes, adapter identity, and split isolation."""
    canonical_path = root / "canonical_manifest.csv"
    provenance_path = root / "capture_provenance.jsonl"
    bbox_path = root / "bbox_migration_audit.jsonl"
    split_path = root / "split_assignments.json"
    semantics_path = root / "canonical_semantics.json"
    calibration_targets_path = root / "calibration_targets.json"
    dataset_provenance_path = root / "dataset_provenance.json"
    roi_contract_path = root / "roi_contract.json"
    topology_contract_path = root / "topology_contract.json"
    if sha256_file(canonical_path) != manifest.canonical_manifest_sha256:
        raise PublicationError("canonical manifest SHA256 mismatch")
    if sha256_file(provenance_path) != manifest.capture_provenance_sha256:
        raise PublicationError("capture provenance SHA256 mismatch")
    if sha256_file(bbox_path) != manifest.bbox_audit_sha256:
        raise PublicationError("bbox audit SHA256 mismatch")
    if sha256_file(semantics_path) != manifest.canonical_semantics_sha256:
        raise PublicationError("canonical semantics SHA256 mismatch")
    if sha256_file(calibration_targets_path) != manifest.calibration_targets_sha256:
        raise PublicationError("calibration targets SHA256 mismatch")
    if sha256_file(dataset_provenance_path) != manifest.dataset_provenance_sha256:
        raise PublicationError("dataset provenance SHA256 mismatch")
    if sha256_file(split_path) != manifest.split_assignments_sha256:
        raise PublicationError("split assignments SHA256 mismatch")
    try:
        semantics_payload = json.loads(semantics_path.read_text(encoding="utf-8"))
        if not isinstance(semantics_payload, Mapping):
            raise ValueError("canonical semantics root must be an object")
        semantics_snapshot = CanonicalSemanticsSnapshot.from_mapping(semantics_payload)
        calibration_targets_payload = json.loads(
            calibration_targets_path.read_text(encoding="utf-8")
        )
        if not isinstance(calibration_targets_payload, Mapping):
            raise ValueError("calibration targets root must be an object")
        calibration_targets = CalibrationTargetsSnapshot.from_mapping(
            calibration_targets_payload
        )
        semantics_by_path = _semantics_capture_bindings(
            semantics_snapshot,
            manifest.required_views,
        )
        split_policy = SplitPolicy.from_mapping(manifest.split_policy)
        png_codec = CanonicalPngCodec.from_mapping(manifest.canonical_png_codec)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise PublicationError(f"dataset governance contract is malformed: {error}") from error
    if semantics_snapshot.canonical_bytes != semantics_path.read_bytes():
        raise PublicationError("canonical semantics snapshot bytes are not canonical JSON")
    if calibration_targets.canonical_bytes != calibration_targets_path.read_bytes():
        raise PublicationError("calibration targets bytes are not canonical JSON")
    if split_policy.sha256 != manifest.split_policy_sha256:
        raise PublicationError("dataset manifest split policy SHA256 mismatch")
    if png_codec.sha256 != manifest.canonical_png_codec_sha256:
        raise PublicationError("dataset manifest canonical PNG codec SHA256 mismatch")
    try:
        dataset_provenance = json.loads(
            dataset_provenance_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError("dataset provenance is not valid UTF-8 JSON") from error
    expected_dataset_provenance = {
        "schema_version": 2,
        "canonical_semantics": {
            "relative_path": "canonical_semantics.json",
            "sha256": manifest.canonical_semantics_sha256,
            "approval": semantics_snapshot.approval.as_dict(),
        },
        "calibration_targets": {
            "relative_path": "calibration_targets.json",
            "sha256": manifest.calibration_targets_sha256,
            "target_count": manifest.calibration_target_count,
            "approval": calibration_targets.approval.as_dict(),
        },
        "split": {
            "relative_path": "split_assignments.json",
            "artifact_sha256": manifest.split_assignments_sha256,
            "policy": split_policy.as_dict(),
            "policy_sha256": manifest.split_policy_sha256,
        },
        "canonical_png": {
            "codec": png_codec.as_dict(),
            "codec_sha256": manifest.canonical_png_codec_sha256,
        },
    }
    if (
        dataset_provenance != expected_dataset_provenance
        or dataset_provenance_path.read_bytes()
        != canonical_json_bytes(expected_dataset_provenance)
    ):
        raise PublicationError("dataset provenance differs from bound governance inputs")
    topology_contract = json.loads(topology_contract_path.read_text(encoding="utf-8"))
    if (
        topology_contract.get("schema_version"),
        topology_contract.get("product"),
        topology_contract.get("topology_id"),
        topology_contract.get("topology_source_sha256"),
        topology_contract.get("required_views"),
    ) != (
        1,
        "ZS32",
        manifest.topology_id,
        manifest.topology_sha256,
        list(manifest.required_views),
    ):
        raise PublicationError("topology contract differs from dataset release")
    try:
        round_ids = [item["round_id"] for item in topology_contract["rounds"]]
        slots = topology_contract["camera_slots"]
        slot_ids = [item["slot_id"] for item in slots]
        serials = [item["serial"] for item in slots]
        generated_views = [
            slot["views"][round_id]
            for round_id in round_ids
            for slot in slots
        ]
    except (KeyError, TypeError) as error:
        raise PublicationError("topology contract round/slot mapping is malformed") from error
    if (
        len(round_ids) != 2
        or len(slots) not in {3, 4, 5}
        or len(round_ids) != len(set(round_ids))
        or len(slot_ids) != len(set(slot_ids))
        or len(serials) != len(set(serials))
        or len(generated_views) != len(set(generated_views))
        or set(generated_views) != set(manifest.required_views)
    ):
        raise PublicationError("topology contract round/slot/view identity is ambiguous")
    roi_contract = json.loads(roi_contract_path.read_text(encoding="utf-8"))
    if (
        roi_contract.get("schema_version") != 2
        or roi_contract.get("product") != "ZS32"
        or roi_contract.get("coordinate_system") != "pixel_xyxy_half_open"
    ):
        raise PublicationError("ROI contract schema/product/coordinate system is invalid")
    if (
        roi_contract.get("roi_version"),
        roi_contract.get("roi_source_sha256"),
        roi_contract.get("topology_id"),
    ) != (manifest.roi_version, manifest.roi_sha256, manifest.topology_id):
        raise PublicationError("ROI contract identity differs from dataset release")
    for hand in manifest.hands:
        try:
            status = roi_contract["hands"][hand]["status"]
        except (KeyError, TypeError) as error:
            raise PublicationError(f"ROI contract is missing enabled hand: {hand}") from error
        if status != "ready":
            raise PublicationError(f"enabled dataset hand ROI is not ready: {hand}")
    rows = _read_csv(canonical_path, CANONICAL_MANIFEST_COLUMNS)
    if len(rows) != manifest.sample_count:
        raise PublicationError("canonical manifest sample_count mismatch")
    parsed_rows: list[CanonicalSampleRow] = []
    for row in rows:
        try:
            if row["is_synthetic"] not in {"true", "false"}:
                raise ValueError("is_synthetic must be exactly true or false")
            parsed_rows.append(
                CanonicalSampleRow(
                    dataset_release_id=row["dataset_release_id"],
                    part_instance_id=row["part_instance_id"],
                    capture_set_id=row["capture_set_id"],
                    hand=row["hand"],
                    view=row["view"],
                    source_path=row["source_path"],
                    source_sha256=row["source_sha256"],
                    crop_path=row["crop_path"],
                    crop_sha256=row["crop_sha256"],
                    roi_version=row["roi_version"],
                    roi_sha256=row["roi_sha256"],
                    roi_x1=int(row["roi_x1"]),
                    roi_y1=int(row["roi_y1"]),
                    roi_x2=int(row["roi_x2"]),
                    roi_y2=int(row["roi_y2"]),
                    crop_width=int(row["crop_width"]),
                    crop_height=int(row["crop_height"]),
                    label=row["label"],
                    defect_type=row["defect_type"],
                    split=row["split"],
                    is_synthetic=row["is_synthetic"] == "true",
                    annotation_state=row["annotation_state"],
                    source_label_sha256=row["source_label_sha256"],
                    crop_label_sha256=row["crop_label_sha256"],
                    source_box_count=int(row["source_box_count"]),
                    crop_box_count=int(row["crop_box_count"]),
                    clipped_box_count=int(row["clipped_box_count"]),
                    outside_roi_box_count=int(row["outside_roi_box_count"]),
                    dropped_box_count=int(row["dropped_box_count"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PublicationError(f"invalid canonical manifest row: {error}") from error
    if any(row.dataset_release_id != manifest.dataset_release_id for row in parsed_rows):
        raise PublicationError("canonical row dataset_release_id mismatch")
    try:
        validate_calibration_targets(
            calibration_targets,
            parsed_rows,
            dataset_release_id=manifest.dataset_release_id,
            topology_id=manifest.topology_id,
            roi_version=manifest.roi_version,
        )
    except ValueError as error:
        raise PublicationError(f"calibration target contract differs from canonical rows: {error}") from error
    try:
        canonical_paths = {
            _capture_path_from_source(
                row.source_path,
                capture_set_id=row.capture_set_id,
                view_id=row.view,
            )
            for row in parsed_rows
        }
    except ValueError as error:
        raise PublicationError(f"canonical source layout is invalid: {error}") from error
    if canonical_paths != set(semantics_by_path):
        raise PublicationError(
            "canonical semantics capture identities differ from canonical manifest"
        )
    semantics_source_path_by_identity: dict[tuple[str, str], str] = {}
    for row in parsed_rows:
        capture_path = _capture_path_from_source(
            row.source_path,
            capture_set_id=row.capture_set_id,
            view_id=row.view,
        )
        semantics = semantics_by_path[capture_path]
        annotations = cast(
            Mapping[str, Mapping[str, object]],
            semantics["annotations"],
        )
        annotation = annotations[row.view]
        if (
            row.label != semantics["label"]
            or row.defect_type != semantics["defect_type"]
            or row.annotation_state != annotation["state"]
        ):
            raise PublicationError(
                f"canonical semantics differs from canonical row: "
                f"{row.capture_set_id}/{row.view}"
            )
        if "text" in annotation and hashlib.sha256(
            cast(str, annotation["text"]).encode("utf-8")
        ).hexdigest() != row.source_label_sha256:
            raise PublicationError(
                f"inline canonical semantics label hash differs: "
                f"{row.capture_set_id}/{row.view}"
            )
        semantics_source_path_by_identity[
            (row.capture_set_id, row.view)
        ] = cast(str, annotation["source_path"])
    if any(
        row.roi_version != manifest.roi_version or row.roi_sha256 != manifest.roi_sha256
        for row in parsed_rows
    ):
        raise PublicationError("canonical row ROI identity mismatch")
    if {row.hand for row in parsed_rows} != set(manifest.hands):
        raise PublicationError("canonical row hands differ from release manifest")
    if len({row.part_instance_id for row in parsed_rows}) != manifest.part_count:
        raise PublicationError("dataset release part_count mismatch")
    if len({row.capture_set_id for row in parsed_rows}) != manifest.capture_set_count:
        raise PublicationError("dataset release capture_set_count mismatch")
    validate_part_grouped_splits((row["part_instance_id"], row["split"]) for row in rows)
    by_capture: dict[str, set[str]] = {}
    crop_identity: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        crop_path = root / _safe_relative_path(row["crop_path"], "crop_path")
        actual_crop_hash = sha256_file(crop_path)
        if actual_crop_hash != row["crop_sha256"]:
            raise PublicationError(f"canonical crop SHA256 mismatch: {row['crop_path']}")
        try:
            expected_xyxy = roi_contract["hands"][row["hand"]]["views"][row["view"]]["xyxy"]
        except (KeyError, TypeError) as error:
            raise PublicationError(
                f"canonical row has no authoritative ROI: {row['hand']}/{row['view']}"
            ) from error
        actual_xyxy = [
            int(row["roi_x1"]),
            int(row["roi_y1"]),
            int(row["roi_x2"]),
            int(row["roi_y2"]),
        ]
        if actual_xyxy != expected_xyxy:
            raise PublicationError(
                f"canonical row ROI differs from contract: {row['hand']}/{row['view']}"
            )
        by_capture.setdefault(row["capture_set_id"], set()).add(row["view"])
        identity = (row["capture_set_id"], row["view"])
        if identity in crop_identity:
            raise PublicationError(f"duplicate canonical crop identity: {identity}")
        crop_identity[identity] = (row["crop_path"], row["crop_sha256"])
    for capture_set_id, views in by_capture.items():
        if views != set(manifest.required_views):
            raise PublicationError(
                f"dataset release capture {capture_set_id} is incomplete: {sorted(views)}"
            )

    try:
        raw_provenance_rows = [
            json.loads(line)
            for line in provenance_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        provenance_rows = [
            CaptureProvenanceRow.from_mapping(row) for row in raw_provenance_rows
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise PublicationError(f"capture provenance artifact is malformed: {error}") from error
    if len(provenance_rows) != manifest.capture_set_count:
        raise PublicationError("capture provenance row count differs from release")
    if provenance_rows != sorted(provenance_rows, key=lambda row: row.capture_set_id):
        raise PublicationError("capture provenance rows are not in canonical capture order")
    if provenance_path.read_bytes() != _jsonl_bytes(
        [row.as_dict() for row in provenance_rows]
    ):
        raise PublicationError("capture provenance JSONL is not canonical")
    provenance_by_capture: dict[str, CaptureProvenanceRow] = {}
    for row in provenance_rows:
        if row.capture_set_id in provenance_by_capture:
            raise PublicationError(
                f"duplicate capture provenance identity: {row.capture_set_id}"
            )
        provenance_by_capture[row.capture_set_id] = row
    canonical_capture_identity: dict[str, tuple[str, str]] = {}
    for row in parsed_rows:
        identity = (row.part_instance_id, row.hand)
        previous = canonical_capture_identity.setdefault(row.capture_set_id, identity)
        if previous != identity:
            raise PublicationError(
                f"canonical capture has conflicting part/hand identity: {row.capture_set_id}"
            )
    if set(provenance_by_capture) != set(canonical_capture_identity):
        raise PublicationError("capture provenance identities differ from canonical rows")
    provenance_identity_by_hand: dict[str, tuple[object, ...]] = {}
    policy_identity: set[tuple[str, str]] = set()
    for capture_set_id, row in provenance_by_capture.items():
        if (row.part_instance_id, row.hand) != canonical_capture_identity[capture_set_id]:
            raise PublicationError(
                f"capture provenance part/hand differs from canonical rows: {capture_set_id}"
            )
        if row.policy_sha256 != manifest.capture_gate_policy_sha256:
            raise PublicationError(
                f"capture gate policy differs from release recipe: {capture_set_id}"
            )
        if row.topology_sha256 != manifest.topology_sha256:
            raise PublicationError(
                f"capture gate topology differs from release topology: {capture_set_id}"
            )
        if set(row.registration_reference_sha256_by_view) != set(
            manifest.required_views
        ):
            raise PublicationError(
                f"capture gate reference views differ from topology: {capture_set_id}"
            )
        policy_identity.add((row.policy_id, row.policy_sha256))
        hand_identity = (
            row.policy_id,
            row.policy_sha256,
            row.topology_sha256,
            row.acquisition_config_sha256,
            row.quality_profile_sha256,
            row.registration_profile_sha256,
            tuple(sorted(row.registration_reference_sha256_by_view.items())),
        )
        previous = provenance_identity_by_hand.setdefault(row.hand, hand_identity)
        if previous != hand_identity:
            raise PublicationError(
                f"capture gate provenance is mixed within hand {row.hand!r}"
            )
    if len(policy_identity) != 1:
        raise PublicationError("dataset release mixes capture gate policy identities")
    if set(provenance_identity_by_hand) != set(manifest.hands):
        raise PublicationError("capture provenance hands differ from release manifest")
    sample_id_by_identity: dict[tuple[str, str], str] = {}
    crop_label_path_by_identity: dict[tuple[str, str], str] = {}
    for adapter, expected_digest in manifest.adapter_manifest_sha256.items():
        path = root / "adapters" / adapter / "manifest.csv"
        if sha256_file(path) != expected_digest:
            raise PublicationError(f"{adapter} adapter manifest SHA256 mismatch")
        expected_columns = (
            YOLO_ADAPTER_COLUMNS if adapter == "yolo" else ADAPTER_BASE_COLUMNS
        )
        adapter_rows = _read_csv(path, expected_columns)
        if len(adapter_rows) != len(rows):
            raise PublicationError(f"{adapter} adapter sample_count mismatch")
        adapter_identities = {
            (row["capture_set_id"], row["view"])
            for row in adapter_rows
        }
        if len(adapter_identities) != len(adapter_rows) or adapter_identities != set(
            crop_identity
        ):
            raise PublicationError(f"{adapter} adapter identities are duplicated or incomplete")
        for row in adapter_rows:
            identity = (row["capture_set_id"], row["view"])
            if crop_identity.get(identity) != (
                row["canonical_crop_path"],
                row["canonical_crop_sha256"],
            ):
                raise PublicationError(
                    f"{adapter} adapter does not reference the canonical crop for {identity}"
                )
            if adapter == "yolo":
                sample_id_by_identity[identity] = row["sample_id"]
                crop_label_path_by_identity[identity] = row["crop_label_path"]
                label_path = root / _safe_relative_path(row["crop_label_path"], "crop_label_path")
                if sha256_file(label_path) != row["crop_label_sha256"]:
                    raise PublicationError(f"YOLO crop label SHA256 mismatch for {identity}")

    if set(sample_id_by_identity) != set(crop_identity):
        raise PublicationError("YOLO adapter sample identities differ from canonical release")

    audit_rows = [
        json.loads(line)
        for line in bbox_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(audit_rows) != len(rows):
        raise PublicationError("bbox audit row count differs from canonical manifest")
    audit_identities = {
        (row["capture_set_id"], row["view"])
        for row in audit_rows
    }
    if audit_identities != set(crop_identity):
        raise PublicationError("bbox audit identities differ from canonical manifest")
    canonical_by_identity = {
        (row["capture_set_id"], row["view"]): row
        for row in rows
    }
    for row in audit_rows:
        identity = (row["capture_set_id"], row["view"])
        canonical = canonical_by_identity[identity]
        audit = row.get("audit", {})
        expected = {
            "annotation_state": canonical["annotation_state"],
            "source_label_sha256": canonical["source_label_sha256"],
            "crop_label_sha256": canonical["crop_label_sha256"],
            "source_box_count": int(canonical["source_box_count"]),
            "crop_box_count": int(canonical["crop_box_count"]),
            "clipped_box_count": int(canonical["clipped_box_count"]),
            "outside_roi_box_count": int(canonical["outside_roi_box_count"]),
            "dropped_box_count": int(canonical["dropped_box_count"]),
        }
        if any(audit.get(field) != value for field, value in expected.items()):
            raise PublicationError(f"bbox audit summary differs from canonical row: {identity}")
        if audit.get("source_path") != semantics_source_path_by_identity[identity]:
            raise PublicationError(
                f"bbox audit source_path differs from canonical semantics: {identity}"
            )
        if len(audit.get("boxes", [])) != expected["source_box_count"]:
            raise PublicationError(f"bbox audit box rows differ from source count: {identity}")
        source_label_path = root / "labels" / "source" / f"{sample_id_by_identity[identity]}.txt"
        if sha256_file(source_label_path) != canonical["source_label_sha256"]:
            raise PublicationError(f"source YOLO label SHA256 mismatch for {identity}")
        try:
            source_label_text = source_label_path.read_bytes().decode("utf-8")
            source_size = roi_contract["source_image_size"]
            roi = RoiBox(
                int(canonical["roi_x1"]),
                int(canonical["roi_y1"]),
                int(canonical["roi_x2"]),
                int(canonical["roi_y2"]),
            )
            replayed = migrate_yolo_labels(
                LabelDocument(
                    state=canonical["annotation_state"],
                    text=source_label_text,
                    source_path=audit["source_path"],
                ),
                source_width=int(source_size["width"]),
                source_height=int(source_size["height"]),
                roi=roi,
            )
        except (KeyError, TypeError, ValueError, UnicodeError) as error:
            raise PublicationError(
                f"bbox migration replay failed for {identity}: {error}"
            ) from error
        crop_label_path = root / _safe_relative_path(
            crop_label_path_by_identity[identity],
            "crop_label_path",
        )
        if crop_label_path.read_bytes() != replayed.text.encode("utf-8"):
            raise PublicationError(
                f"replayed crop YOLO label differs from publication for {identity}"
            )
        if canonical_json_bytes(audit) != canonical_json_bytes(replayed.audit.as_dict()):
            raise PublicationError(
                f"replayed bbox audit differs from publication for {identity}"
            )

    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(split_payload, dict) or set(split_payload) != {
            "schema_version", "policy", "policy_sha256", "assignments"
        }:
            raise ValueError("split assignment fields differ from strict schema")
        if split_payload["schema_version"] != 2:
            raise ValueError("split assignment schema_version must be 2")
        if not isinstance(split_payload["policy"], Mapping):
            raise ValueError("split assignment policy must be an object")
        artifact_policy = SplitPolicy.from_mapping(split_payload["policy"])
        if (
            artifact_policy != split_policy
            or split_payload["policy_sha256"] != manifest.split_policy_sha256
            or artifact_policy.sha256 != manifest.split_policy_sha256
        ):
            raise ValueError("split assignment policy differs from dataset manifest")
        assignments = split_payload["assignments"]
        if not isinstance(assignments, list):
            raise ValueError("split assignments must be an array")
        parsed_assignments = tuple(
            SplitAssignment(
                part_instance_id=item["part_instance_id"],
                split=item["split"],
                stratum=item["stratum"],
            )
            for item in assignments
            if isinstance(item, dict)
            and set(item) == {"part_instance_id", "split", "stratum"}
        )
        if len(parsed_assignments) != len(assignments):
            raise ValueError("split assignment row fields differ from strict schema")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
        raise PublicationError(f"split assignment artifact is malformed: {error}") from error
    if split_path.read_bytes() != canonical_json_bytes(split_payload):
        raise PublicationError("split assignment artifact is not canonical JSON")
    if parsed_assignments != tuple(
        sorted(parsed_assignments, key=lambda item: item.part_instance_id)
    ):
        raise PublicationError("split assignments are not in canonical part order")
    assignment_ids = [item.part_instance_id for item in parsed_assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise PublicationError("split assignment artifact contains duplicate physical parts")
    split_from_artifact = validate_part_grouped_splits(
        (item.part_instance_id, item.split) for item in parsed_assignments
    )
    split_from_rows = validate_part_grouped_splits(
        (row["part_instance_id"], row["split"]) for row in rows
    )
    if split_from_artifact != split_from_rows:
        raise PublicationError("split assignment artifact differs from canonical manifest")
    strata_by_part: dict[str, str] = {}
    for row in parsed_rows:
        stratum = f"{row.hand}:{row.label}:{row.defect_type or 'none'}"
        previous = strata_by_part.setdefault(row.part_instance_id, stratum)
        if previous != stratum:
            raise PublicationError(
                f"canonical part {row.part_instance_id!r} has conflicting split strata"
            )
    if parsed_assignments != assign_grouped_splits(strata_by_part, policy=split_policy):
        raise PublicationError(
            "split assignments are not reproducible from the bound SplitPolicy"
        )


def verify_dataset_release(root: Path) -> DatasetReleaseManifest:
    """Re-verify an already published immutable release including tree checksums."""
    if root.is_symlink():
        raise PublicationError(f"dataset release root must not be a symlink: {root}")
    root = root.resolve()
    payload = json.loads((root / "dataset_release.json").read_text(encoding="utf-8"))
    expected_manifest_fields = {
        "schema_version", "dataset_release_id", "product", "topology_id",
        "topology_sha256", "roi_version", "roi_sha256",
        "capture_gate_policy_sha256", "bbox_migration_policy",
        "required_views", "hands", "sample_count", "part_count",
        "capture_set_count", "canonical_manifest_sha256",
        "capture_provenance_sha256", "bbox_audit_sha256",
        "canonical_semantics_sha256", "calibration_targets_sha256",
        "calibration_target_count", "dataset_provenance_sha256",
        "split_assignments_sha256", "split_policy", "split_policy_sha256",
        "canonical_png_codec", "canonical_png_codec_sha256",
        "adapter_manifest_sha256", "created_at",
    }
    if not isinstance(payload, dict) or set(payload) != expected_manifest_fields:
        raise PublicationError("dataset release manifest fields differ from strict schema")
    manifest = DatasetReleaseManifest(
        schema_version=payload["schema_version"],
        dataset_release_id=payload["dataset_release_id"],
        product=payload["product"],
        topology_id=payload["topology_id"],
        topology_sha256=payload["topology_sha256"],
        roi_version=payload["roi_version"],
        roi_sha256=payload["roi_sha256"],
        capture_gate_policy_sha256=payload["capture_gate_policy_sha256"],
        bbox_migration_policy=payload["bbox_migration_policy"],
        required_views=tuple(payload["required_views"]),
        hands=tuple(payload["hands"]),
        sample_count=payload["sample_count"],
        part_count=payload["part_count"],
        capture_set_count=payload["capture_set_count"],
        canonical_manifest_sha256=payload["canonical_manifest_sha256"],
        capture_provenance_sha256=payload["capture_provenance_sha256"],
        bbox_audit_sha256=payload["bbox_audit_sha256"],
        canonical_semantics_sha256=payload["canonical_semantics_sha256"],
        calibration_targets_sha256=payload["calibration_targets_sha256"],
        calibration_target_count=payload["calibration_target_count"],
        dataset_provenance_sha256=payload["dataset_provenance_sha256"],
        split_assignments_sha256=payload["split_assignments_sha256"],
        split_policy=payload["split_policy"],
        split_policy_sha256=payload["split_policy_sha256"],
        canonical_png_codec=payload["canonical_png_codec"],
        canonical_png_codec_sha256=payload["canonical_png_codec_sha256"],
        adapter_manifest_sha256=payload["adapter_manifest_sha256"],
        created_at=payload["created_at"],
    )
    if root.name != manifest.dataset_release_id:
        raise PublicationError("dataset release directory identity does not match manifest")
    _verify_release_content(root, manifest)
    checksum_bytes = (root / "checksums.sha256").read_bytes()
    publication_root = json.loads((root / "publication_root.json").read_text(encoding="utf-8"))
    if (
        publication_root.get("publication_id") != root.name
        or publication_root.get("algorithm") != "sha256(checksums.sha256 bytes)"
        or publication_root.get("root_sha256")
        != hashlib.sha256(checksum_bytes).hexdigest()
    ):
        raise PublicationError("dataset release publication root digest mismatch")
    recorded: dict[str, str] = {}
    for line in checksum_bytes.decode("utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        recorded[relative] = digest
    actual = tree_checksums(
        root,
        excluded=frozenset({"checksums.sha256", "publication_root.json"}),
    )
    if recorded != actual:
        raise PublicationError("dataset release tree checksums do not match")
    return manifest
