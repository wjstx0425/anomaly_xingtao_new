"""Build the immutable canonical ROI dataset and optional adapter exports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from zs32_inspection.capture import load_verified_capture_gate_publication
from zs32_inspection.capture.reader import load_capture_bundle
from zs32_inspection.config.loaders import load_recipe, load_roi_config, load_topology
from zs32_inspection.data import (
    CanonicalDatasetBuilder,
    CanonicalSemanticsSnapshot,
    CaptureSemantics,
    DatasetBuildSpec,
    LabelDocument,
    load_calibration_targets,
    SourceSample,
    SplitPolicy,
    YoloTrainingExportPolicy,
    SPLIT_ALGORITHM,
    assign_grouped_splits,
    export_anomalib_dataset,
    export_template_dataset,
    export_yolo_dataset,
    sources_from_capture_bundle,
)
from zs32_inspection.domain.identity import Hand
from zs32_inspection.domain.topology import CaptureTopology
from zs32_inspection.runtime.publisher import verify_atomic_publication

from ._common import (
    array_value,
    command_error,
    command_result,
    object_value,
    require_keys,
    string_value,
)
from ._environment import require_linux_nvidia


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one immutable ZS32 canonical ROI dataset")
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="bound recipe authorizing the exact capture gate policy used by all sources",
    )
    parser.add_argument(
        "--gate-publication",
        type=Path,
        required=True,
        help="immutable publication containing the exact policy/profile/reference assets",
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument(
        "--calibration-targets",
        type=Path,
        required=True,
        help="human-approved exact target for every capture/view/template|anomaly|yolo branch",
    )
    parser.add_argument("--dataset-release-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True, help="Explicit audited timestamp")
    parser.add_argument("--hand", action="append", choices=[item.value for item in Hand], required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--calibration-ratio", type=float, required=True)
    parser.add_argument("--test-ratio", type=float, required=True)
    parser.add_argument("--png-compression", type=int, default=1)
    parser.add_argument("--export-output-root", type=Path)
    parser.add_argument("--yolo-export-id")
    parser.add_argument(
        "--yolo-model-val-seed",
        type=int,
        help="deterministic physical-part seed for a training-only YOLO model_val split",
    )
    parser.add_argument(
        "--yolo-model-val-ratio",
        type=float,
        help="fraction of canonical train parts reserved for external YOLO model selection",
    )
    parser.add_argument("--anomalib-export-id")
    parser.add_argument("--template-export-id")
    return parser


def _annotation(
    payload: object,
    *,
    view: str,
    semantics_dir: Path,
) -> LabelDocument:
    item = object_value(payload, f"annotations.{view}")
    require_keys(
        item,
        required=("state", "source_path"),
        optional=("text", "label_path"),
        context=f"annotations.{view}",
    )
    if "text" in item and "label_path" in item:
        raise ValueError(f"annotations.{view} cannot contain both text and label_path")
    text = item.get("text")
    if "label_path" in item:
        label_relative = PurePosixPath(
            string_value(item["label_path"], f"annotations.{view}.label_path")
        )
        if label_relative.as_posix() in {"", "."} or label_relative.is_absolute() or ".." in label_relative.parts:
            raise ValueError(f"annotations.{view}.label_path must be a safe relative POSIX path")
        raw_label_path = semantics_dir.joinpath(*label_relative.parts)
        if raw_label_path.is_symlink() or not raw_label_path.is_file():
            raise ValueError(f"annotation label_path is not a regular file: {raw_label_path}")
        label_path = raw_label_path.resolve()
        try:
            label_path.relative_to(semantics_dir.resolve())
        except ValueError as error:
            raise ValueError(f"annotation label_path escapes semantics directory: {raw_label_path}") from error
        if not label_path.is_file():
            raise ValueError(f"annotation label_path is not a regular file: {label_path}")
        # Decode bytes directly so CRLF/LF differences remain part of the
        # source-label digest and can be replayed exactly from the release.
        text = label_path.read_bytes().decode("utf-8")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"annotations.{view}.text must be a string or null")
    return LabelDocument(
        state=string_value(item["state"], f"annotations.{view}.state"),
        text=text,
        source_path=string_value(item["source_path"], f"annotations.{view}.source_path"),
    )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"capture semantics contains duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _load_semantics_document(path: Path) -> dict[str, object]:
    if path.suffix.lower() != ".json":
        raise ValueError("capture semantics must be an explicit .json document")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"capture semantics must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load capture semantics {path}: {error}") from error
    return object_value(payload, "capture semantics")


def _sources(
    args: argparse.Namespace,
    topology: CaptureTopology,
) -> tuple[tuple[SourceSample, ...], CanonicalSemanticsSnapshot]:
    document = _load_semantics_document(args.semantics)
    require_keys(
        document,
        required=("schema_version", "approval", "captures"),
        context="capture semantics",
    )
    if document["schema_version"] != 2:
        raise ValueError("capture semantics schema_version must be 2")
    semantics_snapshot = CanonicalSemanticsSnapshot.from_mapping(document)
    capture_items = array_value(document["captures"], "capture semantics.captures")
    if not capture_items:
        raise ValueError("capture semantics must list at least one capture")
    output: list[SourceSample] = []
    seen_paths: set[Path] = set()
    for index, raw in enumerate(capture_items):
        item = object_value(raw, f"captures[{index}]")
        require_keys(
            item,
            required=("capture_set_path", "label", "defect_type", "annotations"),
            context=f"captures[{index}]",
        )
        capture_set_path = string_value(item["capture_set_path"], f"captures[{index}].capture_set_path")
        capture_relative = PurePosixPath(capture_set_path)
        if capture_relative.as_posix() in {"", "."} or capture_relative.is_absolute() or ".." in capture_relative.parts:
            raise ValueError(f"captures[{index}].capture_set_path must be a safe relative POSIX path")
        set_root = args.raw_root.joinpath(*capture_relative.parts).resolve()
        raw_root = args.raw_root.resolve()
        try:
            set_root.relative_to(raw_root)
        except ValueError as error:
            raise ValueError(f"capture_set_path escapes raw root: {item['capture_set_path']!r}") from error
        if set_root in seen_paths:
            raise ValueError(f"duplicate capture_set_path: {set_root}")
        seen_paths.add(set_root)
        annotations_raw = object_value(item["annotations"], f"captures[{index}].annotations")
        annotations = {
            view: _annotation(
                annotations_raw[view],
                view=view,
                semantics_dir=args.semantics.resolve().parent,
            )
            for view in topology.required_views
            if view in annotations_raw
        }
        if set(annotations_raw) != set(topology.required_views):
            raise ValueError(
                f"captures[{index}] annotation views must exactly match topology; "
                f"expected={sorted(topology.required_views)}, got={sorted(annotations_raw)}"
            )
        bundle = load_capture_bundle(set_root, topology)
        semantics = CaptureSemantics(
            label=string_value(item["label"], f"captures[{index}].label"),
            defect_type=string_value(
                item["defect_type"], f"captures[{index}].defect_type", allow_empty=True
            ),
            annotations=annotations,
        )
        output.extend(sources_from_capture_bundle(bundle, topology, semantics))
    return tuple(output), semantics_snapshot


def _run(argv: Sequence[str] | None) -> int:
    # OpenCV is imported only after main() has enforced Linux + NVIDIA.
    from zs32_inspection.data.image_codec import OpenCvPngCropper

    args = _parser().parse_args(argv)
    export_ids = {
        "yolo": args.yolo_export_id,
        "anomalib": args.anomalib_export_id,
        "template": args.template_export_id,
    }
    selected_export_ids = [value for value in export_ids.values() if value]
    if selected_export_ids and args.export_output_root is None:
        raise ValueError("--export-output-root is required when an adapter export ID is supplied")
    if len(selected_export_ids) != len(set(selected_export_ids)):
        raise ValueError("adapter export IDs must be unique")
    yolo_partition_arguments = (
        args.yolo_model_val_seed,
        args.yolo_model_val_ratio,
    )
    if args.yolo_export_id and any(value is None for value in yolo_partition_arguments):
        raise ValueError(
            "--yolo-model-val-seed and --yolo-model-val-ratio are required for a "
            "training-safe YOLO export"
        )
    if not args.yolo_export_id and any(value is not None for value in yolo_partition_arguments):
        raise ValueError("YOLO model_val arguments require --yolo-export-id")
    topology = load_topology(args.topology)
    roi = load_roi_config(args.roi)
    recipe = load_recipe(args.recipe)
    gate_publication = load_verified_capture_gate_publication(args.gate_publication)
    if gate_publication.topology.topology_sha256 != topology.topology_sha256:
        raise ValueError("gate publication topology differs from dataset topology")
    hands = tuple(Hand.parse(value) for value in args.hand)
    if len(hands) != len(set(hands)):
        raise ValueError("--hand values must be unique")
    sources, semantics_snapshot = _sources(args, topology)
    calibration_targets = load_calibration_targets(args.calibration_targets)
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
    split_policy = SplitPolicy(
        algorithm=SPLIT_ALGORITHM,
        seed=args.split_seed,
        calibration_ratio=args.calibration_ratio,
        test_ratio=args.test_ratio,
    )
    assignments = assign_grouped_splits(strata_by_part, policy=split_policy)
    cropper = OpenCvPngCropper(compression=args.png_compression)
    spec = DatasetBuildSpec(
        dataset_release_id=args.dataset_release_id,
        output_root=args.output_root,
        topology=topology,
        roi_config=roi,
        recipe=recipe,
        enabled_hands=hands,
        split_assignments=assignments,
        split_policy=split_policy,
        semantics_snapshot=semantics_snapshot,
        canonical_png_codec=cropper.canonical_png_codec,
        created_at=args.created_at,
        gate_publication=gate_publication,
        calibration_targets=calibration_targets,
    )
    release = CanonicalDatasetBuilder(cropper).build(spec, sources)
    exports: dict[str, str] = {}
    export_manifest_sha256: dict[str, str] = {}
    export_publication_root_sha256: dict[str, str] = {}
    export_data_yaml_sha256: dict[str, str] = {}
    export_policy_sha256: dict[str, str] = {}
    if args.yolo_export_id:
        yolo_export = export_yolo_dataset(
            release_root=release.path,
            output_root=args.export_output_root,
            export_id=args.yolo_export_id,
            policy=YoloTrainingExportPolicy(
                seed=args.yolo_model_val_seed,
                model_val_ratio=args.yolo_model_val_ratio,
            ),
        )
        verified_yolo_export = verify_atomic_publication(
            yolo_export,
            required_paths=frozenset(
                {
                    "export_manifest.csv",
                    "data.yaml",
                    "training_export_policy.json",
                    "provenance/dataset_release.json",
                }
            ),
            expected_publication_id=args.yolo_export_id,
        )
        exports["yolo"] = str(yolo_export)
        export_manifest_sha256["yolo"] = verified_yolo_export.checksums["export_manifest.csv"]
        export_publication_root_sha256["yolo"] = verified_yolo_export.root_sha256
        export_data_yaml_sha256["yolo"] = verified_yolo_export.checksums["data.yaml"]
        export_policy_sha256["yolo"] = (
            verified_yolo_export.checksums["training_export_policy.json"]
        )
    if args.anomalib_export_id:
        anomalib_export = export_anomalib_dataset(
            release_root=release.path,
            output_root=args.export_output_root,
            export_id=args.anomalib_export_id,
        )
        verified_anomalib_export = verify_atomic_publication(
            anomalib_export,
            expected_publication_id=args.anomalib_export_id,
        )
        exports["anomalib"] = str(anomalib_export)
        export_manifest_sha256["anomalib"] = verified_anomalib_export.checksums[
            "materialized_manifest.csv"
        ]
        export_publication_root_sha256["anomalib"] = (
            verified_anomalib_export.root_sha256
        )
    if args.template_export_id:
        template_export = export_template_dataset(
            release_root=release.path,
            output_root=args.export_output_root,
            export_id=args.template_export_id,
        )
        verified_template_export = verify_atomic_publication(
            template_export,
            expected_publication_id=args.template_export_id,
        )
        exports["template"] = str(template_export)
        export_manifest_sha256["template"] = verified_template_export.checksums[
            "materialized_manifest.csv"
        ]
        export_publication_root_sha256["template"] = (
            verified_template_export.root_sha256
        )
    command_result(
        "zs32-build-dataset",
        {
            "dataset_release_id": release.manifest.dataset_release_id,
            "dataset_release_path": str(release.path),
            "sample_count": release.manifest.sample_count,
            "part_count": release.manifest.part_count,
            "canonical_semantics_sha256": (
                release.manifest.canonical_semantics_sha256
            ),
            "calibration_targets_sha256": (
                release.manifest.calibration_targets_sha256
            ),
            "calibration_target_count": (
                release.manifest.calibration_target_count
            ),
            "dataset_provenance_sha256": (
                release.manifest.dataset_provenance_sha256
            ),
            "split_assignments_sha256": (
                release.manifest.split_assignments_sha256
            ),
            "split_policy_sha256": release.manifest.split_policy_sha256,
            "canonical_png_codec_sha256": (
                release.manifest.canonical_png_codec_sha256
            ),
            "exports": exports,
            "export_manifest_sha256": export_manifest_sha256,
            "export_publication_root_sha256": export_publication_root_sha256,
            "export_data_yaml_sha256": export_data_yaml_sha256,
            "export_policy_sha256": export_policy_sha256,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_linux_nvidia()
        return _run(argv)
    except Exception as error:
        return command_error("zs32-build-dataset", error)


if __name__ == "__main__":
    raise SystemExit(main())
