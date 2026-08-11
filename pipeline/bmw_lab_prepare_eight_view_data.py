#!/usr/bin/env python3
"""从BMW四相机八视图HDR原始采集生成分支独立的数据清单。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.eight_view_dataset import prepare_eight_view_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the BMW eight-view data-preparation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_raw")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_prepared")
    parser.add_argument("--dataset-id", default="bmw_hdr_eight_view_v1")
    parser.add_argument("--hand", choices=("left", "right"), help="只准备指定采集范围的数据。")
    parser.add_argument(
        "--session-id",
        action="append",
        dest="session_ids",
        metavar="SESSION_ID",
        help="只准备指定采集会话；处理多个会话时可重复传入。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-image-hash",
        action="store_true",
        help="跳过源图SHA-256，仅用于快速实验；报告会标记弱完整性。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验和统计，不写发布目录。")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare one immutable eight-view BMW dataset release."""
    args = build_parser().parse_args(argv)
    try:
        report = prepare_eight_view_dataset(
            raw_root=args.raw_root,
            output_root=args.output_root,
            dataset_id=args.dataset_id,
            seed=args.seed,
            verify_image_hash=not args.skip_image_hash,
            dry_run=args.dry_run,
            capture_scope=args.hand,
            session_ids=args.session_ids,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"BMW八视图数据准备失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
