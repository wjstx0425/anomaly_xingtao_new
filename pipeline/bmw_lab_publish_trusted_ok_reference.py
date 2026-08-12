#!/usr/bin/env python3
"""Publish immutable BMW trusted-OK references from explicit human decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bmw_inspection.lab.trusted_ok_reference import publish_trusted_reference_index  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the immutable approved trusted-OK reference publisher CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=REPO_ROOT / "dataset/bmw_trusted_ok_review/bmw_right_20260810_21_train_normal_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT / "dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v2"
        ),
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/rois/bmw_right_hdr_eight_view_v1.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Publish only APPROVED complete parts; never alter the review package."""
    args = build_parser().parse_args(argv)
    try:
        summary = publish_trusted_reference_index(args.review_dir, args.output, args.roi_config)
    except (FileExistsError, OSError, ValueError) as error:
        print(f"BMW可信OK参考发布失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "approved_part_count": summary.approved_part_count,
                "reference_count_by_view": summary.reference_count_by_view,
                "roi_config_sha256": summary.roi_config_sha256,
                "whitelist_sha256": summary.whitelist_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
