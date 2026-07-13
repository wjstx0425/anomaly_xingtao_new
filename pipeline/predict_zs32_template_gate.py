# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the first, fail-closed ZS32 whole-view template gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.zs32_template_gate import (  # noqa: E402
    DEFAULT_HANDS,
    ZS32_VIEWS,
    TemplateGateError,
    predict_template_gate,
)

EXIT_CODES = {"PASS": 0, "REVIEW": 10, "NG_TEMPLATE": 20, "INVALID_TEMPLATE_GATE": 2}


def build_parser() -> argparse.ArgumentParser:
    """Build the single-image template gate parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--hand", choices=DEFAULT_HANDS, required=True)
    parser.add_argument("--view", choices=ZS32_VIEWS, required=True)
    parser.add_argument("--output-json", type=Path, help="Optional evidence JSON output path.")
    return parser


def _publish(payload: dict[str, object], output_json: Path | None) -> None:
    """Print JSON and optionally publish the same evidence to a file."""
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_json.with_suffix(output_json.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output_json)
    print(encoded, end="")


def main(argv: list[str] | None = None) -> int:
    """Run inference, emit structured evidence, and encode the gate status as an exit code."""
    args = build_parser().parse_args(argv)
    try:
        payload = predict_template_gate(args.model_dir, args.image, hand=args.hand, view=args.view).to_dict()
    except TemplateGateError as exc:
        payload = exc.to_result()
    _publish(payload, args.output_json)
    return EXIT_CODES[str(payload["status"])]


if __name__ == "__main__":
    raise SystemExit(main())
