# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 17: calibrate whole-image quality gate thresholds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data import quality_gate as quality  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the quality calibration CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--normal-root", type=Path, required=True, help="Clean normal image root.")
    parser.add_argument("--stress-root", type=Path, help="Stress-normal image root.")
    parser.add_argument("--invalid-root", type=Path, help="Optional invalid/bad-capture image root for reporting only.")
    parser.add_argument(
        "--profile",
        type=Path,
        help="Optional inspection profile path kept for workflow compatibility.",
    )
    parser.add_argument("--mode", choices=("warn", "fail"), default="warn", help="Calibrated quality gate mode.")
    parser.add_argument("--margin-ratio", type=float, default=0.05, help="Safety margin around normal thresholds.")
    parser.add_argument("--output-yaml", type=Path, required=True, help="Output calibrated quality gate YAML.")
    parser.add_argument("--output-report", type=Path, required=True, help="Output calibration markdown report.")
    parser.add_argument("--output-metrics-csv", type=Path, help="Optional quality metrics CSV path.")
    return parser


def _collect_metrics(root: Path | None, split: str) -> list[tuple[str, quality.ImageQualityMetrics]]:
    """Compute metrics for all images under one split root."""
    if root is None:
        return []
    rows = []
    for image_path in quality.iter_image_paths(root):
        rows.append((split, quality.compute_quality_metrics(image_path)))
    return rows


def _metrics_csv_path(args: argparse.Namespace) -> Path:
    """Resolve the output metrics CSV path."""
    if args.output_metrics_csv is not None:
        return args.output_metrics_csv
    return args.output_report.parent / "quality_metrics.csv"


def _write_metrics_csv(rows: list[tuple[str, quality.ImageQualityMetrics]], output_csv: Path) -> None:
    """Write per-image calibration metrics."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", *quality.QUALITY_FIELDNAMES]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split, metrics in rows:
            result = quality.QualityGateResult(quality.PASS_STATUS, [], metrics)
            row = quality.quality_result_row(result)
            row["split"] = split
            writer.writerow(row)


def _format_thresholds(config: dict[str, Any]) -> list[str]:
    """Return report lines for calibrated thresholds."""
    lines = ["| metric | min | max |", "| --- | ---: | ---: |"]
    for name, rule in sorted(config.get("metrics", {}).items()):
        lines.append(f"| `{name}` | {rule.get('min', '')} | {rule.get('max', '')} |")
    return lines


def _write_report(
    output_report: Path,
    *,
    normal_count: int,
    stress_count: int,
    invalid_count: int,
    config: dict[str, Any],
    metrics_csv: Path,
) -> None:
    """Write a compact calibration report."""
    output_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quality Gate Calibration Report",
        "",
        f"- mode: `{config.get('mode', 'warn')}`",
        f"- normal images: `{normal_count}`",
        f"- stress normal images: `{stress_count}`",
        f"- invalid images: `{invalid_count}`",
        f"- metrics csv: `{metrics_csv}`",
        "",
        "## Thresholds",
        "",
        *_format_thresholds(config),
        "",
        "Invalid images are reported for review only and are not used to fit OK thresholds.",
        "",
    ]
    output_report.write_text("\n".join(lines), encoding="utf-8")


def calibrate_quality_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Calibrate quality thresholds and write configured outputs."""
    normal_rows = _collect_metrics(args.normal_root, "normal")
    stress_rows = _collect_metrics(args.stress_root, "stress_normal")
    invalid_rows = _collect_metrics(args.invalid_root, "invalid")
    normal_metrics = [metrics for _split, metrics in [*normal_rows, *stress_rows]]
    config = quality.calibrate_quality_config(normal_metrics, mode=args.mode, margin_ratio=args.margin_ratio)
    metrics_csv = _metrics_csv_path(args)
    _write_metrics_csv([*normal_rows, *stress_rows, *invalid_rows], metrics_csv)
    quality.write_quality_config(config, args.output_yaml)
    _write_report(
        args.output_report,
        normal_count=len(normal_rows),
        stress_count=len(stress_rows),
        invalid_count=len(invalid_rows),
        config=config,
        metrics_csv=metrics_csv,
    )
    return config


def main() -> None:
    """Run quality gate calibration."""
    args = build_parser().parse_args()
    config = calibrate_quality_gate(args)
    print(f"Calibrated quality gate with {len(config.get('metrics', {}))} metrics")
    print(f"Wrote quality config: {args.output_yaml}")
    print(f"Wrote calibration report: {args.output_report}")


if __name__ == "__main__":
    main()
