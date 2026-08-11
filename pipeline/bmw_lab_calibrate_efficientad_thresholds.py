#!/usr/bin/env python3
"""联合标定BMW八视图EfficientAD整件误报约束阈值。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.efficientad_analysis import EfficientAdScore, render_score_distributions  # noqa: E402
from bmw_inspection.lab.efficientad_thresholds import (  # noqa: E402
    PartScore,
    evaluate_part_thresholds,
    fit_part_thresholds,
    read_part_scores_csv_snapshot,
)
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402

DEFAULT_SCORE_CSV = (
    REPO_ROOT / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/efficientad/score_analysis/efficientad_scores.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SCORE_CSV.parent
DEFAULT_MODEL_ROOT = DEFAULT_SCORE_CSV.parent.parent


def build_parser() -> argparse.ArgumentParser:
    """Build the whole-part threshold calibration command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--business-manifest", type=Path)
    parser.add_argument("--target-part-fpr", type=float, default=0.05)
    parser.add_argument("--views", nargs="+", choices=VIEW_ORDER, default=list(VIEW_ORDER))
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_hashes(model_root: Path, views: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for view in views:
        checkpoint = model_root / view / "model.ckpt"
        if not checkpoint.is_file():
            raise ValueError(f"EfficientAD checkpoint does not exist: {checkpoint}")
        hashes[view] = _sha256(checkpoint)
    return hashes


def _business_normal_metrics(
    rows: tuple[PartScore, ...],
    manifest_path: Path,
    thresholds: dict[str, float],
) -> dict[str, object]:
    """Report true business-normal FPR separately from branch negatives."""
    content = manifest_path.read_bytes()
    labels: dict[str, str] = {}
    with io.StringIO(content.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"physical_part_id", "business_label"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"business manifest must contain fields: {sorted(required)}")
        for row in reader:
            part_id = (row.get("physical_part_id") or "").strip()
            label = (row.get("business_label") or "").strip()
            if not part_id or label not in {"OK", "NG"}:
                raise ValueError("business manifest contains invalid identity or label")
            previous = labels.setdefault(part_id, label)
            if previous != label:
                raise ValueError(f"business manifest has conflicting labels for {part_id}")
    score_part_ids = {row.part_id for row in rows}
    missing = sorted(score_part_ids - labels.keys())
    if missing:
        raise ValueError(f"business manifest is missing score parts: {missing[:3]}")
    business_normal_ids = {part_id for part_id in score_part_ids if labels[part_id] == "OK"}
    business_normal_rows = tuple(row for row in rows if row.part_id in business_normal_ids)
    evaluation = evaluate_part_thresholds(business_normal_rows, thresholds)
    return {
        "business_manifest": str(manifest_path),
        "business_manifest_sha256": hashlib.sha256(content).hexdigest(),
        "business_normal_part_count": evaluation.normal_part_count,
        "business_normal_false_positive_count": evaluation.normal_false_positive_count,
        "business_normal_part_fpr": evaluation.observed_normal_part_fpr,
    }


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
    rows, source_csv_sha256 = read_part_scores_csv_snapshot(score_csv)
    fit = fit_part_thresholds(rows, views=views, target_part_fpr=target)
    model_root = Path(args.model_root).expanduser().resolve()
    checkpoint_sha256_by_view = _checkpoint_hashes(model_root, views)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    shared = {
        **fit.to_dict(),
        "calibrated_at_utc": calibrated_at_utc,
        "source_csv": str(score_csv),
        "source_csv_sha256": source_csv_sha256,
        "checkpoint_sha256_by_view": checkpoint_sha256_by_view,
        "branch_negative_part_count": fit.normal_part_count,
        "branch_negative_false_positive_count": fit.normal_false_positive_count,
        "branch_negative_part_fpr": fit.observed_normal_part_fpr,
    }
    if args.business_manifest is not None:
        business_manifest = Path(args.business_manifest).expanduser().resolve()
        if not business_manifest.is_file():
            raise ValueError(f"business manifest does not exist: {business_manifest}")
        shared.update(_business_normal_metrics(rows, business_manifest, dict(fit.thresholds)))
    threshold_path = output_dir / "part_thresholds.json"
    report_path = output_dir / "part_threshold_report.json"
    distribution_path = render_score_distributions(
        output_dir / "efficientad_calibrated_score_distributions.png",
        tuple(
            EfficientAdScore(
                view_id=row.view_id,
                label=row.label,
                image_path=row.image_path,
                score=row.score,
                predicted_anomalous=row.score >= fit.thresholds[row.view_id],
            )
            for row in rows
        ),
        views=views,
        threshold=fit.thresholds,
    )
    threshold_path.write_text(json.dumps(shared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "complete",
        **shared,
        "threshold_asset": str(threshold_path),
        "calibrated_score_distributions": str(distribution_path),
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
