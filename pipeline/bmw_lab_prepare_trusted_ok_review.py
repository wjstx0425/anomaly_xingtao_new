#!/usr/bin/env python3
"""Create a strict, PENDING-only BMW trusted-OK manual review package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bmw_inspection.lab.trusted_ok_reference import (  # noqa: E402
    TRUSTED_OK_SESSION_ID,
    prepare_review_package,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the trusted-OK review package command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset/bmw_lab_prepared/bmw_right_batch_20260810_21_v1/manifests/dataset_manifest.csv"
        ),
    )
    parser.add_argument("--session-id", required=True, choices=(TRUSTED_OK_SESSION_ID,))
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dataset/bmw_trusted_ok_review/bmw_right_20260810_21_train_normal_v2",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Create the package and print only its immutable PENDING summary."""
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_review_package(args.manifest, args.output, session_id=args.session_id)
    except (FileExistsError, OSError, ValueError) as error:
        print(f"BMW可信OK人工复核包创建失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "session_id": summary.session_id,
                "candidate_part_count": summary.candidate_part_count,
                "candidate_image_count": summary.candidate_image_count,
                "pending_decision_count": summary.pending_decision_count,
                "contact_sheet_count": summary.contact_sheet_count,
                "automatic_approvals": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
