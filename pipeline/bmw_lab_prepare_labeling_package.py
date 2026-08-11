#!/usr/bin/env python3
"""整理BMW八视图YOLO待标注ROI，生成可移交的Label Studio标注包。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.labeling_package import prepare_labeling_package  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the labeling-package CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=REPO_ROOT / "dataset/bmw_lab_training/bmw_hdr_roi_training_v1",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_labeling")
    parser.add_argument("--package-id", default="bmw_hdr_roi_yolo_248_refs8_v2")
    parser.add_argument(
        "--normal-references-per-view",
        type=int,
        default=8,
        help="每个视角复制多少张正常train零件ROI作为只读参考；设为0可关闭。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate and report one immutable labeling handoff package."""
    args = build_parser().parse_args(argv)
    try:
        report = prepare_labeling_package(
            training_root=args.training_root,
            output_root=args.output_root,
            package_id=args.package_id,
            normal_references_per_view=args.normal_references_per_view,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"BMW标注包生成失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
