#!/usr/bin/env python3
"""根据BMW八视图ROI生成EfficientAD、Template和YOLO训练数据目录。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.eight_view_training_data import materialize_training_data  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the ROI training-data materializer CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=REPO_ROOT / "dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/rois/bmw_hdr_eight_view_v1.json",
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_training")
    parser.add_argument("--training-id", default="bmw_hdr_roi_training_v1")
    parser.add_argument(
        "--yolo-label-root",
        type=Path,
        help="可选：248张待复核缺陷ROI对应的标准YOLO txt目录；允许空txt表示该视图确认无可见缺陷。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Materialize one immutable branch training-data release."""
    args = build_parser().parse_args(argv)
    try:
        report = materialize_training_data(
            prepared_root=args.prepared_root,
            roi_config_path=args.roi_config,
            output_root=args.output_root,
            training_id=args.training_id,
            yolo_label_root=args.yolo_label_root,
            dry_run=args.dry_run,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"BMW八视图训练数据生成失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
