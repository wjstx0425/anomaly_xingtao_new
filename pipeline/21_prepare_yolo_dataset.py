# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 21: export C789 part crops and bbox annotations to YOLO format."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.prepare_yolo_dataset import build_parser, export_yolo_dataset  # noqa: E402


def main() -> None:
    """Run the C789 YOLO dataset export wrapper."""
    args = build_parser().parse_args()
    summary = export_yolo_dataset(
        manifest_path=args.manifest,
        annotations_path=args.annotations,
        output_root=args.output_root,
        positive_val_ratio=args.positive_val_ratio,
        positive_test_ratio=args.positive_test_ratio,
        seed=args.seed,
        normal_test_split=args.normal_test_split,
        on_unlabeled_defect=args.on_unlabeled_defect,
        preview_dir=args.preview_dir,
        overwrite=args.overwrite,
    )
    print(f"YOLO dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
