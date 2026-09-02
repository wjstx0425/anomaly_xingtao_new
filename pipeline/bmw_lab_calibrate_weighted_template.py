#!/usr/bin/env python3
"""Calibrate critical-region weighted Template thresholds from normal images."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import EightViewDemoConfig, load_demo_config
from bmw_inspection.lab.eight_view_demo_models import (
    EightViewTemplatePredictor,
    _load_demo_ignore_masks,
    load_part_rois,
)
from bmw_inspection.lab.template_region_weighting import (
    load_template_weighted_regions,
    normal_envelope_threshold,
)


def summarize_view(
    calibration_scores: Sequence[float],
    final_test_scores: Sequence[float],
) -> dict[str, Any]:
    """Fit from calibration normals and report untouched final-test behavior."""
    if not calibration_scores:
        raise ValueError("calibration normal分数不能为空")
    threshold = normal_envelope_threshold(calibration_scores)
    final_values = [float(score) for score in final_test_scores]
    return {
        "calibration_scores": [float(score) for score in calibration_scores],
        "calibration_count": len(calibration_scores),
        "calibration_max": max(float(score) for score in calibration_scores),
        "threshold": threshold,
        "final_test_scores": final_values,
        "final_test_count": len(final_values),
        "final_test_max": max(final_values) if final_values else None,
        "final_test_false_reject_count": sum(score > threshold for score in final_values),
    }


def write_weighted_thresholds(
    config_path: Path,
    thresholds: Mapping[str, float],
) -> None:
    """Update only template.weighted_regions.thresholds in one editable config."""
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Demo配置必须是JSON对象")
    template = payload.get("template")
    weighted = template.get("weighted_regions") if isinstance(template, dict) else None
    if not isinstance(weighted, dict):
        raise ValueError("Demo配置缺少template.weighted_regions")
    if set(thresholds) != set(VIEW_ORDER):
        raise ValueError("Template加权阈值必须覆盖八个标准视角")
    weighted["thresholds"] = {view: float(thresholds[view]) for view in VIEW_ORDER}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normal_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[dict[str, str]] = []
    for row in rows:
        source_class = row.get("source_class", row.get("label", ""))
        business_label = row.get("business_label", "OK")
        if (
            source_class == "normal"
            and business_label == "OK"
            and row.get("split") in {"calibration", "final_test"}
            and row.get("view_id") in VIEW_ORDER
        ):
            selected.append(row)
    return selected


def _build_template_predictor(
    config: EightViewDemoConfig,
) -> tuple[
    EightViewTemplatePredictor,
    Mapping[str, tuple[int, int, int, int]],
    Mapping[str, tuple[object, ...]],
]:
    weighted = config.template_weighted_regions
    if weighted is None or not weighted.enabled:
        raise ValueError("Demo配置未启用template.weighted_regions")
    rois = load_part_rois(config.roi_config)
    expected_shapes = {
        view: (rois[view][3] - rois[view][1], rois[view][2] - rois[view][0])
        for view in VIEW_ORDER
    }
    regions = load_template_weighted_regions(
        weighted.roi_config,
        expected_shapes=expected_shapes,
    )
    masks = (
        None
        if config.template_ignore_mask_index is None
        else _load_demo_ignore_masks(config.template_ignore_mask_index, expected_shapes)
    )
    predictor = EightViewTemplatePredictor(
        config.template_models,
        thresholds=config.template_thresholds,
        ignore_masks=masks,
        weighted_regions=regions,
        weighted_region_weight=weighted.weight,
        weighted_outside_weight=weighted.outside_weight,
        weighted_thresholds=weighted.thresholds,
    )
    return predictor, rois, regions


def calibrate(
    config_path: Path,
    prepared_manifest: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Score one hand and return a report plus the proposed eight thresholds."""
    config = load_demo_config(config_path)
    weighted = config.template_weighted_regions
    if weighted is None:
        raise ValueError("Demo配置缺少template.weighted_regions")
    predictor, rois, regions = _build_template_predictor(config)
    rows = _normal_rows(prepared_manifest)
    if not rows:
        raise ValueError("prepared manifest没有calibration/final_test normal图像")

    scores: dict[str, dict[str, list[float]]] = {
        view: {"calibration": [], "final_test": []} for view in VIEW_ORDER
    }
    for row in rows:
        view = row["view_id"]
        if not regions[view]:
            continue
        source_path = Path(row["source_path"]).expanduser().resolve()
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Template标定图片无法读取：{source_path}")
        x1, y1, x2, y2 = rois[view]
        if x2 > image.shape[1] or y2 > image.shape[0]:
            raise ValueError(f"Template标定ROI越界：{view} {source_path}")
        output = predictor.predict(view, image[y1:y2, x1:x2])
        if output.score is None:
            raise ValueError(f"Template标定未返回分数：{view} {source_path}")
        scores[view][row["split"]].append(float(output.score))

    proposed = {view: float(config.template_thresholds[view]) for view in VIEW_ORDER}
    views_report: dict[str, Any] = {}
    for view in VIEW_ORDER:
        if not regions[view]:
            views_report[view] = {
                "mode": "legacy_empty_regions",
                "region_count": 0,
                "threshold": proposed[view],
            }
            continue
        summary = summarize_view(
            scores[view]["calibration"],
            scores[view]["final_test"],
        )
        proposed[view] = float(summary["threshold"])
        views_report[view] = {
            "mode": "weighted",
            "region_count": len(regions[view]),
            **summary,
        }
    report = {
        "config": str(Path(config_path).expanduser().resolve()),
        "prepared_manifest": str(Path(prepared_manifest).expanduser().resolve()),
        "roi_config": str(weighted.roi_config),
        "weight": weighted.weight,
        "outside_weight": weighted.outside_weight,
        "effective_region_ratio": weighted.weight / weighted.outside_weight,
        "threshold_rule": "calibration_normal_max_times_1.10",
        "views": views_report,
        "proposed_thresholds": proposed,
    }
    return report, proposed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report, thresholds = calibrate(args.config, args.prepared_manifest)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.write_config:
        write_weighted_thresholds(args.config, thresholds)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
