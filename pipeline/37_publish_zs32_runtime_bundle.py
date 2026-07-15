#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Publish versioned ZS32 runtime assets and finalize calibrated bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from capture_data.zs32_runtime_bundle import finalize_runtime_bundle, publish_runtime_assets


def build_parser() -> argparse.ArgumentParser:
    """Build the two-phase Stage 37 command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish-assets", help="publish immutable model, ROI, template, and profile assets")
    publish.add_argument("--source", type=Path, required=True, help="checked-in runtime bundle source declaration")
    publish.add_argument("--output-dir", type=Path, required=True, help="new immutable assets publication directory")
    finalize = commands.add_parser("finalize", help="bind a Stage 33/34 threshold artifact to published assets")
    finalize.add_argument("--assets-manifest", type=Path, required=True, help="published assets_manifest.json")
    finalize.add_argument("--threshold-artifact", type=Path, required=True, help="exact-hash-bound thresholds.json")
    finalize.add_argument("--output-dir", type=Path, required=True, help="new immutable runtime bundle directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one publication phase."""
    args = build_parser().parse_args(argv)
    if args.command == "publish-assets":
        result = publish_runtime_assets(args.source, args.output_dir)
        print(result.runtime_assets)
        print(result.assets_manifest)
        print(result.fusion_profile)
        print(result.asset_set_sha256)
    else:
        print(finalize_runtime_bundle(args.assets_manifest, args.threshold_artifact, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
