"""Train and calibrate BMW Template models from manually retained review crops."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import EightViewDemoConfig
from bmw_inspection.lab.eight_view_demo_models import (
    EightViewTemplatePredictor,
    _load_demo_ignore_masks,
    _prepare_template_image,
    load_part_rois,
)
from bmw_inspection.lab.template_region_weighting import (
    load_template_weighted_regions,
    normal_envelope_threshold,
)


def _review_rows(review_root: Path, hand: str) -> dict[str, list[dict[str, str]]]:
    root = Path(review_root).expanduser().resolve()
    with (root / "candidate_manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {view: [] for view in VIEW_ORDER}
    for row in rows:
        view = row.get("view_id", "")
        if row.get("hand") != hand or view not in grouped:
            continue
        review_path = root / row.get("review_image_path", "")
        if review_path.is_file():
            item = dict(row)
            item["resolved_review_path"] = str(review_path.resolve())
            grouped[view].append(item)
    for view in VIEW_ORDER:
        grouped[view].sort(key=lambda row: int(row["candidate_index"]))
        if not 3 <= len(grouped[view]) <= 40:
            raise ValueError(f"{hand}/{view} must retain at least 3 and at most 40 candidates")
    return grouped


def _load_roi_shapes(path: Path) -> dict[str, tuple[int, int]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    rois = payload.get("part_rois") if isinstance(payload, dict) else None
    if not isinstance(rois, dict) or set(rois) != set(VIEW_ORDER):
        raise ValueError("ROI config must contain all eight part_rois")
    shapes: dict[str, tuple[int, int]] = {}
    for view in VIEW_ORDER:
        roi = rois[view]
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError(f"invalid ROI for {view}")
        x1, y1, x2, y2 = roi
        if any(isinstance(value, bool) or not isinstance(value, int) for value in roi) or not (
            0 <= x1 < x2 and 0 <= y1 < y2
        ):
            raise ValueError(f"invalid ROI for {view}")
        shapes[view] = (y2 - y1, x2 - x1)
    return shapes


def write_reviewed_template_models(
    review_root: Path,
    hand: str,
    roi_config: Path,
    output_root: Path,
) -> dict[str, Path]:
    """Write every surviving review copy as a resident Template, without reselection."""
    if hand not in {"left", "right"}:
        raise ValueError("hand must be left or right")
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Template output already exists: {output}")
    grouped = _review_rows(review_root, hand)
    shapes = _load_roi_shapes(roi_config)

    models: dict[str, Path] = {}
    for view in VIEW_ORDER:
        view_root = output / "template" / view
        template_root = view_root / "templates"
        template_root.mkdir(parents=True)
        metadata = []
        input_height, input_width = shapes[view]
        for output_index, row in enumerate(grouped[view]):
            source = Path(row["resolved_review_path"])
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (input_height, input_width):
                raise ValueError(f"review crop is unreadable or has wrong size: {source}")
            prepared = _prepare_template_image(image, (512, 512))
            template_path = template_root / f"template_{output_index:02d}.png"
            if not cv2.imwrite(str(template_path), prepared):
                raise OSError(f"could not write Template image: {template_path}")
            metadata.append(
                {
                    "path": template_path.relative_to(view_root).as_posix(),
                    "sample_id": row["sample_id"],
                    "part_id": row["physical_part_id"],
                    "session_id": row["session_id"],
                    "review_image_path": str(source),
                }
            )
        model = {
            "schema_version": 1,
            "view_id": view,
            "input_width": input_width,
            "input_height": input_height,
            "preprocess": {
                "grayscale": True,
                "target_width": 512,
                "target_height": 512,
                "blur_kernel": 3,
                "match_method": "TM_CCOEFF_NORMED",
                "padding": "BORDER_REFLECT_101",
                "max_shift": 12,
            },
            "threshold": 0.0,
            "threshold_fit": {"status": "pending_calibration"},
            "templates": metadata,
        }
        model_path = view_root / "model.json"
        model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        models[view] = model_path
    return models


def calibration_summary(
    calibration_scores: Sequence[float],
    final_test_scores: Sequence[float],
) -> dict[str, Any]:
    """Fit from calibration normals only and report final-test false rejects."""
    threshold = normal_envelope_threshold(calibration_scores)
    final_values = [float(value) for value in final_test_scores]
    return {
        "calibration_count": len(calibration_scores),
        "calibration_max": max(float(value) for value in calibration_scores),
        "threshold": threshold,
        "final_test_count": len(final_values),
        "final_test_max": max(final_values) if final_values else None,
        "final_test_false_reject_count": sum(value > threshold for value in final_values),
    }


def _normal_score_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row
        for row in rows
        if row.get("view_id") in VIEW_ORDER
        and row.get("split") in {"calibration", "final_test"}
        and row.get("source_class") == "normal"
        and row.get("business_label") == "OK"
    ]


def calibrate_reviewed_template_models(
    model_paths: Mapping[str, Path],
    config: EightViewDemoConfig,
    prepared_manifest: Path,
) -> dict[str, Any]:
    """Score 0823 normal calibration/final-test images with ordinary and weighted risks."""
    if tuple(model_paths) != VIEW_ORDER:
        raise ValueError("reviewed Template models must cover eight views in canonical order")
    rois = load_part_rois(config.roi_config)
    expected_shapes = {
        view: (rois[view][3] - rois[view][1], rois[view][2] - rois[view][0]) for view in VIEW_ORDER
    }
    masks = (
        None
        if config.template_ignore_mask_index is None
        else _load_demo_ignore_masks(config.template_ignore_mask_index, expected_shapes)
    )
    weighted = config.template_weighted_regions
    if weighted is None or not weighted.enabled:
        raise ValueError("active Template weighted regions are required for dual calibration")
    regions = load_template_weighted_regions(weighted.roi_config, expected_shapes=expected_shapes)
    ones = {view: 1.0 for view in VIEW_ORDER}
    predictor = EightViewTemplatePredictor(
        model_paths,
        thresholds=ones,
        ignore_masks=masks,
        weighted_regions=regions,
        weighted_region_weight=weighted.weight,
        weighted_outside_weight=weighted.outside_weight,
        weighted_thresholds=ones,
    )
    scores = {
        view: {
            "ordinary": {"calibration": [], "final_test": []},
            "weighted": {"calibration": [], "final_test": []},
        }
        for view in VIEW_ORDER
    }
    for row in _normal_score_rows(prepared_manifest):
        view = row["view_id"]
        image_path = Path(row["source_path"]).expanduser().resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"calibration image cannot be read: {image_path}")
        x1, y1, x2, y2 = rois[view]
        if x2 > image.shape[1] or y2 > image.shape[0]:
            raise ValueError(f"calibration ROI is out of bounds: {view} {image_path}")
        output = predictor.predict(view, image[y1:y2, x1:x2])
        if output.score is None:
            raise ValueError(f"Template calibration returned no score: {view} {image_path}")
        legacy = output.details.get("legacy_risk")
        ordinary = float(output.score if legacy is None else legacy)
        weighted_score = float(output.score) if regions[view] else ordinary
        scores[view]["ordinary"][row["split"]].append(ordinary)
        scores[view]["weighted"][row["split"]].append(weighted_score)

    views: dict[str, Any] = {}
    ordinary_thresholds: dict[str, float] = {}
    weighted_thresholds: dict[str, float] = {}
    for view in VIEW_ORDER:
        ordinary = calibration_summary(
            scores[view]["ordinary"]["calibration"],
            scores[view]["ordinary"]["final_test"],
        )
        weighted_summary = calibration_summary(
            scores[view]["weighted"]["calibration"],
            scores[view]["weighted"]["final_test"],
        )
        ordinary_thresholds[view] = float(ordinary["threshold"])
        weighted_thresholds[view] = float(weighted_summary["threshold"])
        views[view] = {
            "template_count": len(json.loads(Path(model_paths[view]).read_text(encoding="utf-8"))["templates"]),
            "weighted_region_count": len(regions[view]),
            "ordinary": ordinary,
            "weighted": weighted_summary,
        }

    for view in VIEW_ORDER:
        model_path = Path(model_paths[view])
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        payload["threshold"] = ordinary_thresholds[view]
        payload["threshold_fit"] = {
            "metric": "calibration_normal_max_times_1.10",
            "split": "calibration",
            "final_test_used": False,
            **views[view]["ordinary"],
        }
        model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "threshold_rule": "calibration_normal_max_times_1.10",
        "prepared_manifest": str(Path(prepared_manifest).expanduser().resolve()),
        "region_weight": weighted.weight,
        "template_thresholds": ordinary_thresholds,
        "weighted_thresholds": weighted_thresholds,
        "views": views,
    }


def write_reviewed_demo_config(
    base_config: Path,
    output_config: Path,
    *,
    model_paths: Mapping[str, Path],
    calibration_report: Mapping[str, Any],
    prepared_manifest: Path,
    result_root: Path,
) -> Path:
    """Create a new candidate config while leaving the rollback config untouched."""
    base = Path(base_config).expanduser().resolve()
    output = Path(output_config).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"candidate Demo config already exists: {output}")
    payload = json.loads(base.read_text(encoding="utf-8"))
    template = payload.get("template")
    if not isinstance(template, dict) or not isinstance(template.get("weighted_regions"), dict):
        raise ValueError("base config lacks template.weighted_regions")
    template["models"] = {view: str(Path(model_paths[view]).expanduser().resolve()) for view in VIEW_ORDER}
    template["thresholds"] = {
        view: float(calibration_report["template_thresholds"][view]) for view in VIEW_ORDER
    }
    template["threshold_source"] = "reviewed_0823_calibration_normal_max_plus_10pct"
    template["weighted_regions"]["thresholds"] = {
        view: float(calibration_report["weighted_thresholds"][view]) for view in VIEW_ORDER
    }
    payload["prepared_manifest"] = str(Path(prepared_manifest).expanduser().resolve())
    payload["result_root"] = str(Path(result_root).expanduser().resolve())
    payload["demo_id"] = f"{payload.get('demo_id', 'bmw-eight-view')}-template40"
    notes = payload.setdefault("experiment_notes", {})
    if isinstance(notes, dict):
        notes["template_model"] = "0823 manually reviewed up-to-40 templates per view"
        notes["template_thresholds"] = "0823 calibration normal max plus 10 percent; rollback config unchanged"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = [
    "calibrate_reviewed_template_models",
    "calibration_summary",
    "write_reviewed_demo_config",
    "write_reviewed_template_models",
]
