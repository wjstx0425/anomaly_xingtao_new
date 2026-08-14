#!/usr/bin/env python3
"""Publish the isolated BMW V6 Demo composite from approved 2026-08-14 assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.v6_demo_publisher import publish_v6_demo  # noqa: E402


DEFAULT_OUTPUT_RUN = REPO_ROOT / "results/bmw_lab_one_click/bmw_right_normal_20260814_v6_demo_v1"
DEFAULT_OUTPUT_CONFIG = REPO_ROOT / "configs/bmw/experiments/bmw_eight_view_demo_v6_right_normal_20260814_v1.json"
V6_LAUNCH_COMMAND = (
    "MPLCONFIGDIR=/tmp/bmw-mpl-cache /home/yunjing/anomaly_xingtao_new/.venv/bin/python "
    "pipeline/bmw_lab_eight_view_demo.py "
    "--config configs/bmw/experiments/bmw_eight_view_demo_v6_right_normal_20260814_v1.json "
    "--experiment-mode"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the no-overwrite V6 composite publisher CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-run", type=Path, default=DEFAULT_OUTPUT_RUN)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Publish V6 and print its immutable receipt plus the exact launch command."""
    args = build_parser().parse_args(argv)
    try:
        receipt = publish_v6_demo(
            args.repo_root,
            output_run=args.output_run,
            output_config=args.output_config,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    print(V6_LAUNCH_COMMAND)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
