# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 27: prepare ZS32 defect images for Label Studio local-file labeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.prepare_zs32_label_studio import prepare_label_studio_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the ZS32 Label Studio preparation parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "zs32_yolo_labeling",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Prepare the ZS32 Label Studio Local Files staging directory."""
    args = build_parser().parse_args()
    summary = prepare_label_studio_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        repo_root=REPO_ROOT,
        overwrite=args.overwrite,
    )
    print(f"Label Studio directory: {args.output_root}")
    print(f"Manifest: {args.output_root / 'labeling_manifest.csv'}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
