# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate C789 slot-aware geometry templates and optionally fuse scores."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import geometry_shape as geometry


GEOMETRY_FIELDNAMES = [
    "source_path",
    "image_path",
    "slot",
    "dataset_label",
    "gt_label",
    "geometry_score",
    "geometry_pred_label",
    "geometry_type",
    "geometry_threshold",
    "threshold_source",
    "threshold_lookup_level",
    "geometry_missing_area",
    "geometry_extra_area",
    "geometry_region",
    "missing_region",
    "extra_region",
    "less_score",
    "more_score",
    "align_dx",
    "align_dy",
    "align_iou",
    "missing_component_count",
    "extra_component_count",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Image or directory to evaluate.")
    parser.add_argument(
        "--template-dir",
        type=Path,
        required=True,
        help="Directory created by build_geometry_templates.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for CSV reports.")
    parser.add_argument("--preset", choices=geometry.preset_choices(), default="c789_left_top_3x2", help="Slot preset.")
    parser.add_argument("--thresholds", type=Path, help="Optional slot threshold CSV.")
    parser.add_argument(
        "--calibrate-thresholds",
        action="store_true",
        help="Write slot thresholds from this normal set.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("region", "slot"),
        default="region",
        help="Calibrate thresholds per boundary region or per whole slot.",
    )
    parser.add_argument("--threshold-margin", type=float, default=0.05, help="Margin ratio for calibration.")
    parser.add_argument("--anomaly-predictions", type=Path, help="Optional AnomalyDINO predictions CSV to OR-fuse.")
    parser.add_argument("--search-radius", type=int, default=25, help="Alignment search radius in pixels.")
    parser.add_argument("--coarse-step", type=int, default=5, help="Coarse alignment step in pixels.")
    parser.add_argument("--min-component-area", type=int, default=64, help="Minimum connected component area.")
    parser.add_argument(
        "--tolerance-px",
        type=int,
        default=3,
        help="Pixel tolerance before less/more component scoring.",
    )
    parser.add_argument("--hole-dilation", type=int, default=32, help="Preset hole mask dilation in pixels.")
    parser.add_argument("--border-margin", type=int, default=8, help="Foreground mask border margin in pixels.")
    parser.add_argument(
        "--foreground-threshold-scale",
        type=float,
        default=0.45,
        help="Scale applied to Otsu threshold for foreground extraction.",
    )
    return parser


def _threshold_rows(
    scores: list[geometry.GeometryScore],
    thresholds: dict[geometry.ThresholdKey, float],
    margin_ratio: float,
) -> list[dict[str, str | int | float]]:
    """Return per-slot threshold rows."""
    rows = []
    grouped_region: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for score in scores:
        for key_text, area in score.region_scores.items():
            geometry_type, region = key_text.split(":", maxsplit=1)
            grouped_region[(score.slot, geometry_type, region)].append(float(area))
    for key, threshold in sorted(thresholds.items()):
        slot, geometry_type, region = key
        values = _matching_region_values(grouped_region, key)
        max_score = max(values) if values else 0.0
        is_fallback = "*" in key
        rows.append(
            {
                "slot": slot,
                "geometry_type": geometry_type,
                "geometry_region": region,
                "slot_id": slot,
                "defect_type": geometry_type,
                "region_id": region,
                "count": len(values),
                "n_normal": len(values),
                "max_geometry_score": max_score,
                "max_normal": max_score,
                "p99_normal": _percentile(values, 0.99),
                "p999_normal": _percentile(values, 0.999),
                "geometry_threshold": threshold,
                "threshold": threshold,
                "threshold_margin": margin_ratio,
                "threshold_source": "auto_fallback" if is_fallback else "calibrated_exact",
            },
        )
    return rows


def _percentile(values: list[float], quantile: float) -> float:
    """Return a simple nearest-rank percentile for diagnostic threshold CSV columns."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def _matching_region_values(
    grouped_region: dict[tuple[str, str, str], list[float]],
    key: geometry.ThresholdKey,
) -> list[float]:
    """Return calibration values covered by an exact or wildcard threshold key."""
    slot, geometry_type, region = key
    values: list[float] = []
    for (candidate_slot, candidate_type, candidate_region), candidate_values in grouped_region.items():
        if slot not in {"*", candidate_slot}:
            continue
        if geometry_type not in {"*", candidate_type}:
            continue
        if region not in {"*", candidate_region}:
            continue
        values.extend(candidate_values)
    return values


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def _path_keys(value: str | None) -> set[str]:
    """Return lookup keys for a path-like CSV value."""
    if not value:
        return set()
    path = Path(value)
    keys = {value, path.name}
    try:
        keys.add(str(path.resolve(strict=False)))
    except OSError:
        pass
    return {key for key in keys if key}


def _index_anomaly_predictions(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Build a path and basename index for anomaly prediction rows."""
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        for column in ("source_path", "processed_path", "image_path"):
            for key in _path_keys(row.get(column)):
                index.setdefault(key, row)
    return index


def _float_value(value: str | int | float | None, default: float = 0.0) -> float:
    """Parse a CSV float-like value."""
    if value in {None, ""}:
        return default
    return float(value)


def _int_label(value: str | int | float | bool | None) -> int:
    """Parse a CSV label-like value."""
    if isinstance(value, bool):
        return int(value)
    if value in {None, ""}:
        return 0
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no"}:
        return 0
    return int(float(text))


def _ratio(score: float, threshold: float | None) -> float:
    """Return score normalized by threshold."""
    if threshold is None:
        return 0.0
    if threshold <= 0:
        return 1_000_000_000.0 if score > 0 else 0.0
    return score / threshold


def _fuse_predictions(
    scores: list[geometry.GeometryScore],
    anomaly_predictions: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fuse geometry scores with an existing anomaly prediction CSV."""
    anomaly_rows, anomaly_fieldnames = _read_csv(anomaly_predictions)
    anomaly_index = _index_anomaly_predictions(anomaly_rows)
    fused_rows: list[dict[str, Any]] = []
    for score in scores:
        geometry_row = score.as_row()
        anomaly_row = None
        for key in _path_keys(score.source_path):
            anomaly_row = anomaly_index.get(key)
            if anomaly_row is not None:
                break

        anomaly_score = _float_value(anomaly_row.get("pred_score") if anomaly_row else None)
        anomaly_threshold = _float_value(anomaly_row.get("deploy_threshold") if anomaly_row else None, default=0.0)
        anomaly_pred_label = _int_label(anomaly_row.get("deploy_pred_label") if anomaly_row else 0)
        geometry_pred_label = int(score.geometry_pred_label or 0)
        final_pred_label = int(bool(anomaly_pred_label) or bool(geometry_pred_label))
        final_score = max(
            _ratio(anomaly_score, anomaly_threshold),
            _ratio(score.geometry_score, score.geometry_threshold),
        )

        row: dict[str, Any] = {}
        if anomaly_row:
            row.update(anomaly_row)
        row.update(geometry_row)
        row.update(
            {
                "anomaly_score": anomaly_score,
                "anomaly_threshold": anomaly_threshold,
                "anomaly_deploy_pred_label": anomaly_pred_label,
                "final_pred_label": final_pred_label,
                "final_score": final_score,
            },
        )
        fused_rows.append(row)

    fieldnames = list(
        dict.fromkeys(
            [
                *anomaly_fieldnames,
                *GEOMETRY_FIELDNAMES,
                "anomaly_score",
                "anomaly_threshold",
                "anomaly_deploy_pred_label",
                "final_pred_label",
                "final_score",
            ],
        ),
    )
    return fused_rows, fieldnames


def evaluate_geometry(args: argparse.Namespace) -> Path:
    """Evaluate geometry templates and write reports."""
    templates = geometry.load_templates(args.template_dir)
    image_paths = geometry.iter_image_paths(args.data_root)
    if not image_paths:
        msg = f"No images found under {args.data_root}"
        raise FileNotFoundError(msg)

    scores: list[geometry.GeometryScore] = []
    skipped = 0
    for image_path in image_paths:
        slot = geometry.slot_name_from_path(image_path)
        template = templates.get(slot or "")
        if template is None:
            skipped += 1
            continue
        scores.append(
            geometry.evaluate_image(
                image_path,
                template,
                preset=args.preset,
                search_radius=args.search_radius,
                coarse_step=args.coarse_step,
                min_component_area=args.min_component_area,
                tolerance_px=args.tolerance_px,
                hole_dilation=args.hole_dilation,
                border_margin=args.border_margin,
                foreground_threshold_scale=args.foreground_threshold_scale,
            ),
        )

    if not scores:
        msg = f"No images matched available geometry templates in {args.data_root}; skipped={skipped}"
        raise RuntimeError(msg)

    thresholds: dict[geometry.ThresholdKey, float] | None = None
    if args.thresholds:
        thresholds = geometry.load_thresholds(args.thresholds)
    if args.calibrate_thresholds:
        if args.threshold_mode == "slot":
            slot_thresholds = geometry.calibrate_thresholds(scores, margin_ratio=args.threshold_margin)
            thresholds = geometry.add_threshold_fallbacks(
                {(slot, "*", "*"): threshold for slot, threshold in slot_thresholds.items()},
            )
            threshold_rows = _threshold_rows(scores, thresholds, args.threshold_margin)
        else:
            thresholds = geometry.calibrate_region_thresholds(scores, margin_ratio=args.threshold_margin)
            threshold_rows = _threshold_rows(scores, thresholds, args.threshold_margin)
        geometry.write_csv(
            args.output_dir / "geometry_thresholds.csv",
            threshold_rows,
            [
                "slot",
                "geometry_type",
                "geometry_region",
                "slot_id",
                "defect_type",
                "region_id",
                "count",
                "n_normal",
                "max_geometry_score",
                "max_normal",
                "p99_normal",
                "p999_normal",
                "geometry_threshold",
                "threshold",
                "threshold_margin",
                "threshold_source",
            ],
        )
    if thresholds:
        geometry.apply_thresholds(scores, thresholds)

    prediction_rows = [score.as_row() for score in scores]
    predictions_path = args.output_dir / "geometry_predictions.csv"
    geometry.write_csv(predictions_path, prediction_rows, GEOMETRY_FIELDNAMES)

    if args.anomaly_predictions:
        fused_rows, fused_fieldnames = _fuse_predictions(scores, args.anomaly_predictions)
        geometry.write_csv(args.output_dir / "fused_predictions.csv", fused_rows, fused_fieldnames)

    print(f"Evaluated {len(scores)} images; skipped {skipped} without matching slot template")
    print(f"Wrote geometry predictions: {predictions_path}")
    if args.calibrate_thresholds:
        print(f"Wrote geometry thresholds: {args.output_dir / 'geometry_thresholds.csv'}")
    if args.anomaly_predictions:
        print(f"Wrote fused predictions: {args.output_dir / 'fused_predictions.csv'}")
    return predictions_path


def main() -> None:
    """Run geometry evaluation."""
    args = build_parser().parse_args()
    evaluate_geometry(args)


if __name__ == "__main__":
    main()
