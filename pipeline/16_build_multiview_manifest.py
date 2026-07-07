# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 16: build a multi-view inspection manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data import multiview_manifest as multiview  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the multi-view manifest CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True, help="Raw multi-view image root.")
    parser.add_argument("--profile", type=Path, help="Optional YAML/JSON inspection profile.")
    parser.add_argument("--manifest-csv", type=Path, help="Optional explicit input manifest CSV.")
    parser.add_argument("--filename-regex", help="Optional filename regex with part_id/side/view groups.")
    parser.add_argument("--required-side", action="append", default=[], help="Required side, e.g. top or bottom.")
    parser.add_argument("--required-view", action="append", default=[], help="Required view, e.g. uniform.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output manifest CSV.")
    return parser


def _load_profile(path: Path | None) -> dict[str, Any]:
    """Load an optional YAML/JSON profile."""
    if path is None:
        return {}
    if not path.is_file():
        msg = f"Missing inspection profile: {path}"
        raise FileNotFoundError(msg)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ModuleNotFoundError as error:
            msg = "YAML profiles require PyYAML; use JSON or install PyYAML."
            raise RuntimeError(msg) from error
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        msg = f"Inspection profile must contain a mapping: {path}"
        raise ValueError(msg)
    return data


def _merge_cli_config(args: argparse.Namespace, profile: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI overrides into profile config."""
    config = dict(profile)
    if args.filename_regex:
        config["filename_regex"] = args.filename_regex
    if args.required_side or args.required_view:
        ok_requires = dict(config.get("ok_requires", {}))
        if args.required_side:
            ok_requires["required_sides"] = args.required_side
        if args.required_view:
            ok_requires["required_views"] = args.required_view
        config["ok_requires"] = ok_requires
    return config


def build_multiview_manifest(args: argparse.Namespace) -> list[multiview.MultiViewPartRecord]:
    """Build and write a multi-view manifest."""
    config = _merge_cli_config(args, _load_profile(args.profile))
    records = multiview.load_multiview_manifest(args.input_root, config, manifest_csv=args.manifest_csv)
    multiview.write_multiview_manifest(records, args.output_csv)
    invalid = []
    for record in records:
        valid, missing = multiview.validate_required_views(record, config)
        if not valid:
            invalid.append((record.part_id, missing))
    if invalid:
        for part_id, missing in invalid:
            print(f"[invalid_capture] {part_id}: missing {', '.join(missing)}")
    return records


def main() -> None:
    """Run manifest generation."""
    args = build_parser().parse_args()
    records = build_multiview_manifest(args)
    image_count = sum(len(record.images) for record in records)
    print(f"Wrote {image_count} images across {len(records)} parts: {args.output_csv}")


if __name__ == "__main__":
    main()
