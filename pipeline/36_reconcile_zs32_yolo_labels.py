# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stage 36: reconcile ZS32 eight-view labels and finalize reviewed YOLO data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.reconcile_zs32_yolo_labels import (  # noqa: E402
    build_balanced_defect_normal_yolo_dataset,
    export_defect_label_studio_tasks,
    finalize_defect_yolo_export,
    finalize_reconciled_dataset,
    reconcile_zs32_yolo_dataset,
)

DEFAULT_PREPARED_ROOT = REPO_ROOT / "dataset" / "zs32_eight_view_yolo_reconciled_20260715_v3"


def build_parser() -> argparse.ArgumentParser:
    """Build the content-addressed ZS32 label reconciliation parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build review tasks and a non-trainable YOLO draft.")
    prepare.add_argument("--dataset-root", type=Path, default=Path("/home/yunjing/anomalib/dataset/zs32_new"))
    prepare.add_argument(
        "--roi-config",
        type=Path,
        default=Path("/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json"),
    )
    prepare.add_argument(
        "--labeling-root",
        type=Path,
        default=Path("/home/yunjing/anomalib/dataset/zs32_new_yolo_labeling"),
    )
    prepare.add_argument(
        "--existing-yolo-root",
        type=Path,
        default=Path("/home/yunjing/anomalib/dataset/zs32_eight_view_roi_yolo"),
    )
    prepare.add_argument("--legacy-repo-root", type=Path, default=Path("/home/yunjing/anomalib"))
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    prepare.add_argument("--val-ratio", type=float, default=0.15)
    prepare.add_argument("--test-ratio", type=float, default=0.15)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--perceptual-distance", type=int, default=6)
    prepare.add_argument("--workers", type=int, default=4)

    defect = subparsers.add_parser(
        "prepare-defect-840",
        help="Export all 840 raw defect images as independent Label Studio tasks.",
    )
    defect.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    defect.add_argument("--dataset-root", type=Path, default=Path("/home/yunjing/anomalib/dataset/zs32_new"))
    defect.add_argument(
        "--roi-config",
        type=Path,
        default=Path("/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json"),
    )
    defect.add_argument(
        "--labeling-root",
        type=Path,
        default=Path("/home/yunjing/anomalib/dataset/zs32_new_yolo_labeling"),
    )
    defect.add_argument("--legacy-repo-root", type=Path, default=Path("/home/yunjing/anomalib"))
    defect.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PREPARED_ROOT / "defect_label_studio_840",
    )

    finalize = subparsers.add_parser("finalize", help="Merge completed Label Studio JSON into final YOLO data.")
    finalize.add_argument("--prepared-root", type=Path, required=True)
    finalize.add_argument("--label-studio-export-json", type=Path, required=True)
    finalize.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "zs32_eight_view_yolo_reconciled_final_20260715",
    )

    finalize_defect = subparsers.add_parser(
        "finalize-defect-840-yolo",
        help="Finalize the defect-840 Label Studio YOLO export after explicit empty-task confirmation.",
    )
    finalize_defect.add_argument("--prepared-root", type=Path, required=True)
    finalize_defect.add_argument("--yolo-export-root", type=Path, required=True)
    finalize_defect.add_argument("--output-root", type=Path, required=True)
    finalize_defect.add_argument("--confirm-skipped-empty", action="store_true", required=True)

    balanced = subparsers.add_parser(
        "build-balanced-defect-normal",
        help="Exclude uncertain skipped defects and add a trusted right-normal session.",
    )
    balanced.add_argument("--defect-prepared-root", type=Path, required=True)
    balanced.add_argument("--yolo-export-root", type=Path, required=True)
    balanced.add_argument("--normal-mapping", type=Path, required=True)
    balanced.add_argument("--normal-session-id")
    balanced.add_argument("--max-normal-groups", type=int)
    balanced.add_argument("--output-root", type=Path, required=True)
    balanced.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Run preparation or post-review finalization."""
    args = build_parser().parse_args()
    if args.command == "prepare":
        summary = reconcile_zs32_yolo_dataset(
            dataset_root=args.dataset_root,
            roi_config=args.roi_config,
            labeling_root=args.labeling_root,
            existing_yolo_root=args.existing_yolo_root,
            output_root=args.output_root,
            legacy_repo_root=args.legacy_repo_root,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            perceptual_distance=args.perceptual_distance,
            workers=args.workers,
        )
        print(f"Prepared reconciliation root: {args.output_root}")
    elif args.command == "prepare-defect-840":
        summary = export_defect_label_studio_tasks(
            prepared_root=args.prepared_root,
            dataset_root=args.dataset_root,
            roi_config=args.roi_config,
            labeling_root=args.labeling_root,
            legacy_repo_root=args.legacy_repo_root,
            output_root=args.output_root,
        )
        print(f"Defect-only Label Studio tasks: {args.output_root}")
    elif args.command == "finalize":
        summary = finalize_reconciled_dataset(
            prepared_root=args.prepared_root,
            label_studio_json=args.label_studio_export_json,
            output_root=args.output_root,
        )
        print(f"Final YOLO dataset: {args.output_root}")
    elif args.command == "finalize-defect-840-yolo":
        summary = finalize_defect_yolo_export(
            prepared_root=args.prepared_root,
            yolo_export_root=args.yolo_export_root,
            output_root=args.output_root,
            confirm_skipped_empty=args.confirm_skipped_empty,
        )
        print(f"Final defect-840 YOLO dataset: {args.output_root}")
    else:
        summary = build_balanced_defect_normal_yolo_dataset(
            defect_prepared_root=args.defect_prepared_root,
            yolo_export_root=args.yolo_export_root,
            normal_manifest=args.normal_mapping,
            normal_session_id=args.normal_session_id,
            max_normal_groups=args.max_normal_groups,
            output_root=args.output_root,
            seed=args.seed,
        )
        print(f"Balanced defect+normal YOLO dataset: {args.output_root}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
