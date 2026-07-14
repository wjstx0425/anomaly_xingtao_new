"""Fail-closed definitions for the external YOLO training data boundary."""

from __future__ import annotations

from copy import deepcopy

import pytest

from zs32_inspection.data.yolo_export import (
    YOLO_EXPORT_COLUMNS,
    YOLO_TRAINING_DATA_YAML,
    YoloTrainingExportPolicy,
    _part_assignments,
    validate_yolo_training_export,
)
from zs32_inspection.data.export_common import csv_bytes
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    verify_atomic_publication,
)


def _row(part: str, sample: str, *, split: str = "train") -> dict[str, str]:
    crop_digest = ("a" if part == "part-a" else "b") * 64
    label_digest = ("c" if part == "part-a" else "d") * 64
    return {
        "sample_id": sample,
        "part_instance_id": part,
        "capture_set_id": f"capture-{part}",
        "hand": "right",
        "view": "front",
        "split": split,
        "label": "defect",
        "defect_type": "scratch",
        "canonical_crop_path": f"crops/{sample}.png",
        "canonical_crop_sha256": crop_digest,
        "crop_label_path": f"labels/{sample}.txt",
        "crop_label_sha256": label_digest,
    }


def _publication(tmp_path, *, rows=None, data_yaml=YOLO_TRAINING_DATA_YAML, extra=False):
    policy = YoloTrainingExportPolicy(seed=43, model_val_ratio=0.5)
    source_rows = rows or [_row("part-a", "sample-a"), _row("part-b", "sample-b")]
    assignments = _part_assignments(source_rows, policy)
    export_rows = []
    with AtomicDirectoryPublisher(tmp_path / "exports", "yolo-training-safe-v1") as publisher:
        for row in source_rows:
            split = assignments[row["part_instance_id"]]
            image_path = f"images/{split}/{row['sample_id']}.png"
            label_path = f"labels/{split}/{row['sample_id']}.txt"
            publisher.write_bytes(image_path, bytes.fromhex(row["canonical_crop_sha256"]))
            publisher.write_bytes(label_path, bytes.fromhex(row["crop_label_sha256"]))
            # Bind manifest digests to the actual synthetic file bytes.
            bound = deepcopy(row)
            import hashlib

            bound["canonical_crop_sha256"] = hashlib.sha256(
                bytes.fromhex(row["canonical_crop_sha256"])
            ).hexdigest()
            bound["crop_label_sha256"] = hashlib.sha256(
                bytes.fromhex(row["crop_label_sha256"])
            ).hexdigest()
            export_rows.append(
                {**bound, "yolo_split": split, "image_path": image_path, "label_path": label_path}
            )
        publisher.write_bytes("export_manifest.csv", csv_bytes(YOLO_EXPORT_COLUMNS, export_rows))
        publisher.write_bytes("data.yaml", data_yaml)
        publisher.write_bytes("training_export_policy.json", policy.canonical_bytes)
        publisher.write_bytes("provenance/dataset_release.json", b"{}\n")
        if extra:
            publisher.write_bytes("images/test/hidden.png", b"held-out-leak")
        path = publisher.finalize(
            validator=lambda _staging: None,
            required_paths=frozenset(
                {
                    "export_manifest.csv",
                    "data.yaml",
                    "training_export_policy.json",
                    "provenance/dataset_release.json",
                    *(row["image_path"] for row in export_rows),
                    *(row["label_path"] for row in export_rows),
                    *({"images/test/hidden.png"} if extra else set()),
                }
            ),
        )
    return verify_atomic_publication(path)


def test_training_export_policy_assigns_whole_parts_to_train_or_model_val() -> None:
    policy = YoloTrainingExportPolicy(seed=43, model_val_ratio=0.5)
    rows = [
        _row("part-a", "a-front"),
        {**_row("part-a", "a-back"), "view": "back"},
        _row("part-b", "b-front"),
        {**_row("part-b", "b-back"), "view": "back"},
    ]
    assignments = _part_assignments(rows, policy)
    assert set(assignments) == {"part-a", "part-b"}
    assert set(assignments.values()) == {"train", "model_val"}


def test_verified_training_export_contains_no_calibration_test_or_hidden_files(tmp_path) -> None:
    publication = _publication(tmp_path)
    policy = validate_yolo_training_export(publication)
    assert policy.seed == 43
    assert not any("/test/" in path or "/calibration/" in path for path in publication.checksums)


def test_training_export_rejects_a_canonical_calibration_row() -> None:
    policy = YoloTrainingExportPolicy(seed=43, model_val_ratio=0.5)
    with pytest.raises(ValueError, match="only canonical train rows"):
        _part_assignments(
            [_row("part-a", "sample-a", split="calibration")],
            policy,
        )


def test_training_export_rejects_unmanifested_heldout_file(tmp_path) -> None:
    publication = _publication(tmp_path, extra=True)
    with pytest.raises(ValueError, match="missing or unmanifested files"):
        validate_yolo_training_export(publication)


def test_training_export_data_yaml_cannot_expose_test(tmp_path) -> None:
    publication = _publication(
        tmp_path,
        data_yaml=(
            b"train: images/train\nval: images/model_val\ntest: images/test\n\n"
            b"names:\n  0: defect\n"
        ),
    )
    with pytest.raises(ValueError, match="may expose only train and model_val"):
        validate_yolo_training_export(publication)


def test_training_policy_schema_rejects_weakened_exclusion_list() -> None:
    payload = YoloTrainingExportPolicy(seed=43, model_val_ratio=0.2).as_dict()
    payload["excluded_source_splits"] = ["test"]
    with pytest.raises(ValueError, match="isolation contract"):
        YoloTrainingExportPolicy.from_mapping(payload)
