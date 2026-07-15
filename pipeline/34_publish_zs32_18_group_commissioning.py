# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Publish a Stage18-compatible ZS32 offline commissioning artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.zs32_18_group_commissioning import publish_commissioning_artifact  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit commissioning publisher CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPO_ROOT / "config/fusion/zs32_right_unified_roi_18_group_commissioning.json",
    )
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--template-model", type=Path, required=True)
    parser.add_argument("--template-patchcore-thresholds", type=Path, required=True)
    parser.add_argument("--yolo-auxiliary-thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--commissioning-only",
        action="store_true",
        required=True,
        help="Acknowledge that the commissioning artifact cannot authorize production release.",
    )
    parser.add_argument(
        "--allow-test-leakage",
        action="store_true",
        help="TEMPORARY: accept an explicitly marked YOLO source selected using test data.",
    )
    return parser


def main() -> None:
    """Validate all source contracts and publish the immutable artifact."""
    args = build_parser().parse_args()
    path = publish_commissioning_artifact(
        args.profile.resolve(),
        args.runtime_config.resolve(),
        args.template_model.resolve(),
        args.template_patchcore_thresholds.resolve(),
        args.yolo_auxiliary_thresholds.resolve(),
        args.output_dir.resolve(),
        allow_test_leakage=args.allow_test_leakage,
    )
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    print(f"Wrote {len(profile['expected_versions'])}-group commissioning thresholds: {path}")
    print("production_release_allowed: false")
    if args.allow_test_leakage:
        print("WARNING: data_leakage=true; YOLO test data was used for threshold selection")


if __name__ == "__main__":
    main()
