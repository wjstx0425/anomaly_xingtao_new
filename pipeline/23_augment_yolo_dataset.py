# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 23: augment a YOLO bbox dataset for C789 defect training."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.augment_yolo_dataset import build_parser, augment_yolo_dataset  # noqa: E402


def main() -> None:
    """Run the YOLO dataset augmentation wrapper."""
    args = build_parser().parse_args()
    summary = augment_yolo_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        seed=args.seed,
        preview_limit=args.preview_limit,
        overwrite=args.overwrite,
    )
    print(f"augmented YOLO dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
