# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 28: build the complete ZS32 six-view YOLO dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.prepare_zs32_yolo_dataset import build_zs32_yolo_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the quick ZS32 YOLO dataset parser."""
    root = REPO_ROOT / "dataset" / "zs32_yolo_labeling"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--label-export-root", type=Path, default=root / "project-10-at-2026-07-12-12-29-b424b36b")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "dataset" / "zs32_six_view_yolo")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Build and report the ZS32 YOLO dataset."""
    args = build_parser().parse_args()
    summary = build_zs32_yolo_dataset(
        repo_root=REPO_ROOT,
        dataset_root=REPO_ROOT / "dataset",
        labeling_root=REPO_ROOT / "dataset" / "zs32_yolo_labeling",
        label_export_root=args.label_export_root,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"ZS32 YOLO dataset: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
