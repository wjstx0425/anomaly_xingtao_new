# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 26: build ROI-level C789 YOLO datasets."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.roi_yolo_dataset import build_parser, build_roi_yolo_dataset  # noqa: E402


def main() -> None:
    """Run the ROI YOLO dataset builder wrapper."""
    args = build_parser().parse_args()
    summary = build_roi_yolo_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        mode=args.mode,
        roi_size=args.roi_size,
        stride=args.stride,
        val_groups=args.val_groups,
        normal_source_root=args.normal_source_root,
        train_normal_ratio=args.train_normal_ratio,
        val_normal_limit=args.val_normal_limit,
        preview_limit=args.preview_limit,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"roi YOLO dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
