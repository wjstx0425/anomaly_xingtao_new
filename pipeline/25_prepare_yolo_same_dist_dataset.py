# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 25: build a same-distribution C789 YOLO dataset."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.yolo_same_dist_dataset import build_parser, build_same_dist_yolo_dataset  # noqa: E402


def main() -> None:
    """Run the same-distribution YOLO dataset builder wrapper."""
    args = build_parser().parse_args()
    summary = build_same_dist_yolo_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        normal_source_root=args.normal_source_root,
        val_groups=args.val_groups,
        seed=args.seed,
        preview_limit=args.preview_limit,
        overwrite=args.overwrite,
        train_normal_limit=args.train_normal_limit,
        val_normal_limit=args.val_normal_limit,
    )
    print(f"same-distribution YOLO dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
