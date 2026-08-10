#!/usr/bin/env python3
"""联合标定BMW八视图EfficientAD整件误报约束阈值。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.efficientad_thresholds import fit_part_thresholds, read_part_scores_csv  # noqa: E402
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402

DEFAULT_SCORE_CSV = (
    REPO_ROOT / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/efficientad/score_analysis/efficientad_scores.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SCORE_CSV.parent


def build_parser() -> argparse.ArgumentParser:
    """Build the whole-part threshold calibration command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-part-fpr", type=float, default=0.05)
    parser.add_argument("--views", nargs="+", choices=VIEW_ORDER, default=list(VIEW_ORDER))
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate(args: argparse.Namespace) -> dict[str, object]:
    """Fit thresholds and write immutable, leakage-labelled JSON assets."""
    score_csv = Path(args.scores_csv).expanduser().resolve()
    if not score_csv.is_file():
        raise ValueError(f"EfficientAD score CSV does not exist: {score_csv}")
    target = float(args.target_part_fpr)
    if not math.isfinite(target):
        raise ValueError("--target-part-fpr must be finite")
    views = tuple(args.views)
    if views != VIEW_ORDER:
        raise ValueError(f"--views must contain the complete BMW eight-view order: {VIEW_ORDER}")
    rows = read_part_scores_csv(score_csv)
    fit = fit_part_thresholds(rows, views=views, target_part_fpr=target)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shared = {
        **fit.to_dict(),
        "source_csv": str(score_csv),
        "source_csv_sha256": _sha256(score_csv),
    }
    threshold_path = output_dir / "part_thresholds.json"
    report_path = output_dir / "part_threshold_report.json"
    threshold_path.write_text(json.dumps(shared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "complete",
        **shared,
        "threshold_asset": str(threshold_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**report, "report": str(report_path)}


def main(argv: list[str] | None = None) -> int:
    """Run calibration and print the published artifact map."""
    args = build_parser().parse_args(argv)
    try:
        report = calibrate(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW EfficientAD整件阈值标定失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
