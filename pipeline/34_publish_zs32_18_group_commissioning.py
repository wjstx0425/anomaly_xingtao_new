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


def _patchcore_threshold_override(value: str) -> tuple[str, float, float]:
    """Parse ``VIEW:LOW:HIGH`` for one explicit demo threshold override."""
    parts = value.split(":")
    if len(parts) != 3 or not parts[0].strip():
        raise argparse.ArgumentTypeError("expected VIEW:LOW:HIGH")
    try:
        return parts[0].strip(), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("PatchCore LOW/HIGH must be numbers") from exc


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
    parser.add_argument(
        "--allow-yolo-model-rebind",
        action="store_true",
        help="DEMO ONLY: reuse per-view YOLO thresholds after replacing the bound YOLO weights.",
    )
    parser.add_argument(
        "--allow-patchcore-model-rebind",
        action="store_true",
        help="DEMO ONLY: reuse per-view PatchCore thresholds after replacing the bound checkpoints.",
    )
    parser.add_argument(
        "--allow-roi-version-rebind",
        action="store_true",
        help="DEMO ONLY: rebind ROI identity after byte-identical source/runtime ROI verification.",
    )
    parser.add_argument(
        "--source-roi-config",
        type=Path,
        help="ROI config used by the source thresholds; required with --allow-roi-version-rebind.",
    )
    parser.add_argument(
        "--yolo-threshold-override",
        type=float,
        help="DEMO ONLY: replace every required YOLO low/high threshold with this fixed score.",
    )
    parser.add_argument(
        "--patchcore-threshold-override",
        type=_patchcore_threshold_override,
        action="append",
        default=[],
        metavar="VIEW:LOW:HIGH",
        help="DEMO ONLY: override one required PatchCore view; repeat for multiple views.",
    )
    return parser


def main() -> None:
    """Validate all source contracts and publish the immutable artifact."""
    args = build_parser().parse_args()
    patchcore_overrides = {view: (low, high) for view, low, high in args.patchcore_threshold_override}
    if len(patchcore_overrides) != len(args.patchcore_threshold_override):
        raise ValueError("duplicate --patchcore-threshold-override view")
    path = publish_commissioning_artifact(
        args.profile.resolve(),
        args.runtime_config.resolve(),
        args.template_model.resolve(),
        args.template_patchcore_thresholds.resolve(),
        args.yolo_auxiliary_thresholds.resolve(),
        args.output_dir.resolve(),
        allow_test_leakage=args.allow_test_leakage,
        allow_yolo_model_rebind=args.allow_yolo_model_rebind,
        allow_patchcore_model_rebind=args.allow_patchcore_model_rebind,
        allow_roi_version_rebind=args.allow_roi_version_rebind,
        source_roi_config_path=args.source_roi_config.resolve() if args.source_roi_config is not None else None,
        yolo_threshold_override=args.yolo_threshold_override,
        patchcore_threshold_overrides=patchcore_overrides,
    )
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    print(f"Wrote {len(profile['expected_versions'])}-group commissioning thresholds: {path}")
    print("production_release_allowed: false")
    if args.allow_test_leakage:
        print("WARNING: data_leakage=true; YOLO test data was used for threshold selection")
    if args.allow_yolo_model_rebind:
        print("WARNING: YOLO thresholds were rebound from a different model version for demo use")
    if args.allow_patchcore_model_rebind:
        print("WARNING: PatchCore thresholds were rebound from different model versions for demo use")
    if args.allow_roi_version_rebind:
        print("WARNING: ROI version identities were rebound after byte-identical source/runtime ROI verification")
    if args.yolo_threshold_override is not None:
        print(f"WARNING: all YOLO thresholds were overridden to {args.yolo_threshold_override} for demo use")
    if patchcore_overrides:
        print(f"WARNING: PatchCore thresholds were overridden for {sorted(patchcore_overrides)}")


if __name__ == "__main__":
    main()
