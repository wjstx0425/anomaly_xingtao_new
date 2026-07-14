"""Import and immutably snapshot one externally trained global YOLO bundle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from zs32_inspection.data.dataset_release import verify_dataset_release
from zs32_inspection.data.yolo_export import validate_yolo_training_export
from zs32_inspection.models.base import sha256_file
from zs32_inspection.models.yolo import YoloRuntimeSettings, import_yolo_bundle
from zs32_inspection.models.yolo_receipt import (
    YoloTrainingReceipt,
    load_yolo_training_receipt,
)
from zs32_inspection.runtime.publisher import (
    AtomicDirectoryPublisher,
    PublicationError,
    VerifiedAtomicPublication,
    verify_atomic_publication,
)

from ._common import command_error, command_result
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import external ZS32 YOLO best.pt as an immutable candidate"
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--import-id", required=True)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument("--yolo-export", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--candidate-conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser


def _verify_training_data(
    *,
    receipt: YoloTrainingReceipt,
    dataset_release: Path,
    yolo_export: Path,
) -> VerifiedAtomicPublication:
    """Bind the receipt to the exact canonical dataset and YOLO export publications."""
    dataset = verify_dataset_release(dataset_release)
    dataset_manifest = Path(dataset_release) / "dataset_release.json"
    dataset_manifest_digest = sha256_file(dataset_manifest)
    if (
        dataset.dataset_release_id != receipt.dataset_release_id
        or dataset_manifest_digest != receipt.dataset_manifest_sha256
    ):
        raise ValueError("training receipt differs from the verified dataset release")
    export = verify_atomic_publication(
        yolo_export,
        required_paths=frozenset(
            {
                "export_manifest.csv",
                "data.yaml",
                "training_export_policy.json",
                "provenance/dataset_release.json",
            }
        ),
        expected_publication_id=receipt.yolo_export_publication_id,
    )
    if (
        export.root_sha256 != receipt.yolo_export_root_sha256
        or export.checksums["export_manifest.csv"] != receipt.export_manifest_sha256
        or export.checksums["data.yaml"] != receipt.export_data_yaml_sha256
        or export.checksums["training_export_policy.json"]
        != receipt.export_policy_sha256
    ):
        raise ValueError("training receipt differs from the verified YOLO export publication")
    validate_yolo_training_export(export)
    if export.read_bytes("provenance/dataset_release.json") != dataset_manifest.read_bytes():
        raise ValueError("YOLO export embeds a different dataset release manifest")
    return export


def _run(argv: Sequence[str] | None) -> int:
    args = _parser().parse_args(argv)
    receipt = load_yolo_training_receipt(args.training_receipt)
    _verify_training_data(
        receipt=receipt,
        dataset_release=args.dataset_release,
        yolo_export=args.yolo_export,
    )
    runtime = YoloRuntimeSettings(
        imgsz=args.imgsz,
        candidate_conf=args.candidate_conf,
        iou=args.iou,
        max_det=args.max_det,
        single_class_name="defect",
    )
    spec = import_yolo_bundle(
        args.bundle_dir,
        training_receipt_path=args.training_receipt,
        runtime=runtime,
    )
    spec.verify()

    def validate(staging: Path) -> None:
        loaded = import_yolo_bundle(
            staging,
            training_receipt_path=staging / "training_receipt.json",
            runtime=runtime,
        )
        loaded.verify()
        if loaded.bundle_digest != spec.bundle_digest or loaded.model_digest != spec.model_digest:
            raise PublicationError("published YOLO descriptor differs from verified source bundle")

    with AtomicDirectoryPublisher(args.output_root, args.import_id) as publisher:
        publisher.copy_file(spec.weights.path, "best.pt", expected_sha256=spec.weights.sha256)
        publisher.copy_file(spec.args_yaml.path, "args.yaml", expected_sha256=spec.args_yaml.sha256)
        publisher.copy_file(spec.data_yaml.path, "data.yaml", expected_sha256=spec.data_yaml.sha256)
        publisher.copy_file(
            spec.class_names_yaml.path,
            "class_names.yaml",
            expected_sha256=spec.class_names_yaml.sha256,
        )
        publisher.copy_file(
            spec.training_receipt.path,
            "training_receipt.json",
            expected_sha256=spec.training_receipt.sha256,
        )
        publisher.write_json("metadata.json", spec.to_dict())
        published = publisher.finalize(
            validator=validate,
            required_paths=frozenset(
                {
                    "best.pt",
                    "args.yaml",
                    "data.yaml",
                    "class_names.yaml",
                    "training_receipt.json",
                    "metadata.json",
                }
            ),
        )
    command_result(
        "zs32-import-yolo",
        {
            "bundle_digest": spec.bundle_digest,
            "model_digest": spec.model_digest,
            "published_path": str(published),
            "training_seed": spec.provenance.training_seed,
            "run_id": spec.provenance.run_id,
            "training_receipt_sha256": spec.provenance.training_receipt_digest,
            "yolo_export_root_sha256": spec.provenance.yolo_export_root_sha256,
            "yolo_export_policy_sha256": spec.provenance.yolo_export_policy_digest,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-import-yolo", error)


if __name__ == "__main__":
    raise SystemExit(main())
