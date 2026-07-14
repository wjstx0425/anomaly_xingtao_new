"""Materialize a training-safe external YOLO dataset.

Calibration and held-out test rows never cross this trust boundary.  Model
validation is derived deterministically from physical parts already assigned
to the canonical ``train`` split and is bound by a canonical policy document.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    VerifiedAtomicPublication,
    canonical_json_bytes,
    sha256_file,
)

from .export_common import csv_bytes, read_adapter_rows, safe_source
from .manifests import YOLO_ADAPTER_COLUMNS


YOLO_TRAINING_EXPORT_ALGORITHM = "sha256_seed_stratum_part_v1"
YOLO_TRAINING_EXPORT_POLICY_PATH = "training_export_policy.json"
YOLO_TRAINING_DATA_YAML = (
    b"train: images/train\nval: images/model_val\n\nnames:\n  0: defect\n"
)
YOLO_EXPORT_COLUMNS = (
    *YOLO_ADAPTER_COLUMNS,
    "yolo_split",
    "image_path",
    "label_path",
)


@dataclass(frozen=True, slots=True)
class YoloTrainingExportPolicy:
    """Frozen derivation of model-val parts from canonical train parts only."""

    seed: int
    model_val_ratio: float

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("YOLO training export seed must be a non-negative integer")
        if (
            isinstance(self.model_val_ratio, bool)
            or not isinstance(self.model_val_ratio, (int, float))
            or not math.isfinite(self.model_val_ratio)
            or not 0 < float(self.model_val_ratio) < 1
        ):
            raise ValueError("YOLO model_val_ratio must be finite and in (0, 1)")
        object.__setattr__(self, "model_val_ratio", float(self.model_val_ratio))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "YoloTrainingExportPolicy":
        expected = {
            "schema",
            "schema_version",
            "algorithm",
            "seed",
            "model_val_ratio",
            "source_split",
            "excluded_source_splits",
            "assignment_unit",
            "stratification_fields",
        }
        if set(payload) != expected:
            raise ValueError(
                "YOLO training export policy fields differ from strict schema; "
                f"missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)}"
            )
        if (
            payload["schema"] != "zs32.yolo_training_export_policy"
            or payload["schema_version"] != 1
            or payload["algorithm"] != YOLO_TRAINING_EXPORT_ALGORITHM
            or payload["source_split"] != "train"
            or payload["excluded_source_splits"] != ["calibration", "test"]
            or payload["assignment_unit"] != "part_instance_id"
            or payload["stratification_fields"] != ["hand", "label", "defect_type"]
        ):
            raise ValueError("YOLO training export policy identity or isolation contract is invalid")
        return cls(seed=payload["seed"], model_val_ratio=payload["model_val_ratio"])  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "zs32.yolo_training_export_policy",
            "schema_version": 1,
            "algorithm": YOLO_TRAINING_EXPORT_ALGORITHM,
            "seed": self.seed,
            "model_val_ratio": self.model_val_ratio,
            "source_split": "train",
            "excluded_source_splits": ["calibration", "test"],
            "assignment_unit": "part_instance_id",
            "stratification_fields": ["hand", "label", "defect_type"],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def _part_assignments(
    rows: Sequence[Mapping[str, str]],
    policy: YoloTrainingExportPolicy,
) -> dict[str, str]:
    """Return deterministic train/model_val assignments without part leakage."""
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("YOLO training export may consume only canonical train rows")
    stratum_by_part: dict[str, str] = {}
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        part_id = row["part_instance_id"]
        if not part_id.strip() or not row["sample_id"].strip() or not row["view"].strip():
            raise ValueError("YOLO training export contains an empty sample/part/view identity")
        if row["hand"] not in {"left", "right"} or row["label"] not in {"normal", "defect"}:
            raise ValueError("YOLO training export contains an invalid hand or label")
        if (row["label"] == "normal" and row["defect_type"]) or (
            row["label"] == "defect" and not row["defect_type"]
        ):
            raise ValueError("YOLO training export label/defect_type semantics are invalid")
        stratum = f"{row['hand']}:{row['label']}:{row['defect_type'] or 'none'}"
        previous = stratum_by_part.setdefault(part_id, stratum)
        if previous != stratum:
            raise ValueError(f"YOLO training part {part_id!r} crosses semantic strata")
        grouped[stratum].add(part_id)

    assignments: dict[str, str] = {}
    for stratum, part_ids in sorted(grouped.items()):
        ordered = sorted(
            part_ids,
            key=lambda part_id: hashlib.sha256(
                f"{policy.seed}:{stratum}:{part_id}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ordered) < 2:
            model_val_count = 0
        else:
            model_val_count = math.floor(len(ordered) * policy.model_val_ratio + 0.5)
            model_val_count = min(len(ordered) - 1, max(1, model_val_count))
        model_val_parts = set(ordered[:model_val_count])
        for part_id in ordered:
            assignments[part_id] = "model_val" if part_id in model_val_parts else "train"
    if set(assignments.values()) != {"train", "model_val"}:
        raise ValueError(
            "YOLO training export needs at least two canonical train parts in a common "
            "hand/label/defect_type stratum to derive a non-empty model_val split"
        )
    return assignments


def export_yolo_dataset(
    *,
    release_root: Path,
    output_root: Path,
    export_id: str,
    policy: YoloTrainingExportPolicy,
) -> Path:
    """Publish only canonical train rows, with an independent model-val subset."""
    if not isinstance(policy, YoloTrainingExportPolicy):
        raise TypeError("YOLO export requires a structured training-safe policy")
    all_rows = read_adapter_rows(release_root, "yolo")
    rows = [row for row in all_rows if row["split"] == "train"]
    if not rows:
        raise ValueError("YOLO training export has no canonical train rows")
    assignments = _part_assignments(rows, policy)
    destinations: set[str] = set()
    output_rows: list[dict[str, str]] = []
    with AtomicDirectoryPublisher(output_root, export_id) as publisher:
        for row in rows:
            split = assignments[row["part_instance_id"]]
            image_path = f"images/{split}/{row['sample_id']}.png"
            label_path = f"labels/{split}/{row['sample_id']}.txt"
            if image_path in destinations or label_path in destinations:
                raise ValueError(f"YOLO export destination collision for {row['sample_id']}")
            destinations.update((image_path, label_path))
            publisher.copy_file(
                safe_source(release_root, row["canonical_crop_path"]),
                image_path,
                expected_sha256=row["canonical_crop_sha256"],
            )
            publisher.copy_file(
                safe_source(release_root, row["crop_label_path"]),
                label_path,
                expected_sha256=row["crop_label_sha256"],
            )
            output_rows.append(
                {**row, "yolo_split": split, "image_path": image_path, "label_path": label_path}
            )
        publisher.write_bytes("export_manifest.csv", csv_bytes(YOLO_EXPORT_COLUMNS, output_rows))
        publisher.write_bytes("data.yaml", YOLO_TRAINING_DATA_YAML)
        publisher.write_bytes(YOLO_TRAINING_EXPORT_POLICY_PATH, policy.canonical_bytes)
        release_manifest_path = safe_source(release_root, "dataset_release.json")
        publisher.copy_file(
            release_manifest_path,
            "provenance/dataset_release.json",
            expected_sha256=sha256_file(release_manifest_path),
        )
        destinations.update(
            (
                "export_manifest.csv",
                "data.yaml",
                YOLO_TRAINING_EXPORT_POLICY_PATH,
                "provenance/dataset_release.json",
            )
        )
        return publisher.finalize(
            validator=lambda staging: _validate_yolo_export(staging, output_rows, policy),
            required_paths=frozenset(destinations),
        )


def _parse_export_rows(content: bytes) -> list[dict[str, str]]:
    try:
        with io.StringIO(content.decode("utf-8"), newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != YOLO_EXPORT_COLUMNS:
                raise ValueError("YOLO export manifest header differs from training-safe schema")
            rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError(f"cannot parse YOLO training export manifest: {error}") from error
    if not rows or any(
        set(row) != set(YOLO_EXPORT_COLUMNS) or any(value is None for value in row.values())
        for row in rows
    ):
        raise ValueError("YOLO training export manifest is empty or malformed")
    if csv_bytes(YOLO_EXPORT_COLUMNS, rows) != content:
        raise ValueError("YOLO training export manifest bytes are not canonical CSV")
    return rows


def validate_yolo_training_export(publication: VerifiedAtomicPublication) -> YoloTrainingExportPolicy:
    """Rebuild isolation and the complete allowed file set before external import."""
    policy_content = publication.read_bytes(YOLO_TRAINING_EXPORT_POLICY_PATH)
    try:
        policy_payload = json.loads(policy_content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"YOLO training export policy is invalid JSON: {error}") from error
    if not isinstance(policy_payload, Mapping):
        raise ValueError("YOLO training export policy must be an object")
    policy = YoloTrainingExportPolicy.from_mapping(policy_payload)
    if policy_content != policy.canonical_bytes:
        raise ValueError("YOLO training export policy must use canonical JSON bytes")
    rows = _parse_export_rows(publication.read_bytes("export_manifest.csv"))
    assignments = _part_assignments(rows, policy)
    expected_paths = {
        "export_manifest.csv",
        "data.yaml",
        YOLO_TRAINING_EXPORT_POLICY_PATH,
        "provenance/dataset_release.json",
    }
    observed_by_part: dict[str, set[str]] = defaultdict(set)
    seen_samples: set[str] = set()
    seen_materialized_paths: set[str] = set()
    for row in rows:
        if row["split"] != "train" or row["yolo_split"] not in {"train", "model_val"}:
            raise ValueError("YOLO export exposes a non-train canonical split or invalid model split")
        if row["yolo_split"] != assignments[row["part_instance_id"]]:
            raise ValueError("YOLO export model_val assignment differs from its frozen policy")
        observed_by_part[row["part_instance_id"]].add(row["yolo_split"])
        expected_image = f"images/{row['yolo_split']}/{row['sample_id']}.png"
        expected_label = f"labels/{row['yolo_split']}/{row['sample_id']}.txt"
        if row["sample_id"] in seen_samples:
            raise ValueError(f"YOLO export repeats sample_id {row['sample_id']!r}")
        seen_samples.add(row["sample_id"])
        if expected_image in seen_materialized_paths or expected_label in seen_materialized_paths:
            raise ValueError("YOLO export contains a materialized path collision")
        seen_materialized_paths.update((expected_image, expected_label))
        if row["image_path"] != expected_image or row["label_path"] != expected_label:
            raise ValueError("YOLO export materialized path differs from its safe split identity")
        if publication.checksums.get(expected_image) != row["canonical_crop_sha256"]:
            raise ValueError("YOLO export image digest differs from canonical crop")
        if publication.checksums.get(expected_label) != row["crop_label_sha256"]:
            raise ValueError("YOLO export label digest differs from canonical ROI label")
        expected_paths.update((expected_image, expected_label))
    if any(len(splits) != 1 for splits in observed_by_part.values()):
        raise ValueError("one physical part crosses YOLO train/model_val splits")
    if set(publication.checksums) != expected_paths:
        raise ValueError("YOLO training export contains missing or unmanifested files")
    if publication.read_bytes("data.yaml") != YOLO_TRAINING_DATA_YAML:
        raise ValueError("YOLO data.yaml may expose only train and model_val directories")
    return policy


def _validate_yolo_export(
    staging: Path,
    rows: Sequence[Mapping[str, str]],
    policy: YoloTrainingExportPolicy,
) -> None:
    if (staging / YOLO_TRAINING_EXPORT_POLICY_PATH).read_bytes() != policy.canonical_bytes:
        raise PublicationError("YOLO training export policy changed in staging")
    if (staging / "data.yaml").read_bytes() != YOLO_TRAINING_DATA_YAML:
        raise PublicationError("YOLO data.yaml changed in staging")
    for row in rows:
        if row["split"] != "train":
            raise PublicationError("YOLO export attempted to publish calibration/test data")
        if sha256_file(staging / row["image_path"]) != row["canonical_crop_sha256"]:
            raise PublicationError(f"YOLO image changed: {row['image_path']}")
        if sha256_file(staging / row["label_path"]) != row["crop_label_sha256"]:
            raise PublicationError(f"YOLO label changed: {row['label_path']}")
