#!/usr/bin/env python3
"""Build review-required BMW masks and run an immutable EfficientAD same-image A/B."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.efficientad_analysis import fixed_scale_heatmap  # noqa: E402
from bmw_inspection.lab.efficientad_mask_ab import (  # noqa: E402
    aggregate_anomaly_map,
    apply_fixed_fill,
    build_candidate_mask,
    classify_diagnostic_hotspot,
    load_foreground_mask_asset,
)
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER, _atomic_publish_noreplace  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json"
DEFAULT_REFERENCE = (
    REPO_ROOT / "dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v2/reference_index.json"
)
DEFAULT_INSPECTIONS = REPO_ROOT / "results/bmw_eight_view_demo_v3_ng_evidence_v1"


def build_parser() -> argparse.ArgumentParser:
    """Build the two-stage offline experiment CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    masks = subparsers.add_parser("build-masks", help="build review-required candidates from trusted OK medians")
    masks.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    masks.add_argument("--reference-index", type=Path, default=DEFAULT_REFERENCE)
    masks.add_argument("--output-dir", type=Path, required=True)
    masks.add_argument("--working-size", type=int, default=512)
    masks.add_argument("--erosion-px", type=int, default=3)
    masks.add_argument("--fixed-fill-value", type=int, default=0)

    evaluate = subparsers.add_parser("evaluate", help="re-infer saved ROI images and publish map A/B evidence")
    evaluate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate.add_argument("--inspection-root", type=Path, default=DEFAULT_INSPECTIONS)
    evaluate.add_argument("--mask-index", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--capture-id", action="append", default=[])
    evaluate.add_argument("--quantile", type=float, default=0.995)
    evaluate.add_argument("--top-k-fraction", type=float, default=0.001)
    evaluate.add_argument("--component-threshold", type=float, default=0.5)
    evaluate.add_argument("--minimum-component-area", type=int, default=8)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(config_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (config_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _config_assets(config_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    roi_path = _resolve(path, payload["roi_config"])
    threshold_path = _resolve(path, payload["efficientad"]["threshold_artifact"])
    training_run = _resolve(path, payload["training_run"])
    if _sha256(threshold_path) != payload["efficientad"]["threshold_artifact_sha256"]:
        raise ValueError("EfficientAD threshold artifact SHA mismatch")
    return payload, roi_path, threshold_path, training_run


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"cannot write image: {path}")


def _mask_overlay(median: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = median.copy()
    overlay[mask == 0] = (overlay[mask == 0].astype(np.float32) * 0.18).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), max(2, round(max(mask.shape) / 512 * 2)))
    return overlay


def build_masks(args: argparse.Namespace) -> Path:
    """Publish eight generated candidates plus SHA-bound index and review overlays."""
    _config, roi_path, _threshold_path, _training_run = _config_assets(args.config)
    roi_sha = _sha256(roi_path)
    reference_index = Path(args.reference_index).expanduser().resolve()
    reference_sha = _sha256(reference_index)
    reference_payload = json.loads(reference_index.read_text(encoding="utf-8"))
    if reference_payload.get("roi_config_sha256") != roi_sha:
        raise ValueError("trusted OK reference ROI SHA does not match active public ROI")
    references = reference_payload.get("references")
    if not isinstance(references, list):
        raise ValueError("trusted OK reference index has no references")
    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        views: dict[str, dict[str, Any]] = {}
        for view in VIEW_ORDER:
            rows = [row for row in references if row.get("view_id") == view]
            if len(rows) != 50:
                raise ValueError(f"trusted OK references must contain 50 rows for {view}")
            images = []
            for row in rows:
                image_path = (reference_index.parent / row["roi_image_path"]).resolve()
                if _sha256(image_path) != row["roi_image_sha256"]:
                    raise ValueError(f"trusted OK ROI SHA mismatch: {image_path}")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"trusted OK ROI is unreadable: {image_path}")
                images.append(image)
            mask, median = build_candidate_mask(
                images,
                working_size=args.working_size,
                erosion_px=args.erosion_px,
            )
            mask_path = staging / "masks" / f"{view}.png"
            median_path = staging / "medians" / f"{view}.png"
            overlay_path = staging / "review_overlays" / f"{view}.png"
            _write_image(mask_path, mask)
            _write_image(median_path, median)
            _write_image(overlay_path, _mask_overlay(median, mask))
            height, width = mask.shape
            views[view] = {
                "mask_path": str(mask_path.relative_to(staging)),
                "mask_sha256": _sha256(mask_path),
                "median_path": str(median_path.relative_to(staging)),
                "median_sha256": _sha256(median_path),
                "review_overlay_path": str(overlay_path.relative_to(staging)),
                "review_overlay_sha256": _sha256(overlay_path),
                "roi_height": height,
                "roi_width": width,
                "foreground_fraction": float(np.mean(mask == 255)),
                "reference_count": len(rows),
            }
        index_payload = {
            "schema_version": "bmw.efficientad_foreground_masks/1.0",
            "asset_id": destination.name,
            "candidate_only": True,
            "review_status": "generated_requires_human_visual_review",
            "generator": "trusted_ok_median_grabcut_v1",
            "generator_parameters": {
                "working_size": args.working_size,
                "erosion_px": args.erosion_px,
            },
            "public_roi_config_path": str(roi_path),
            "public_roi_config_sha256": roi_sha,
            "trusted_ok_reference_index_path": str(reference_index),
            "trusted_ok_reference_index_sha256": reference_sha,
            "fixed_fill_value": args.fixed_fill_value,
            "views": views,
        }
        (staging / "foreground_mask_index.json").write_text(
            json.dumps(index_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _atomic_publish_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "foreground_mask_index.json"


def _array(value: object) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np.asarray(current)


def _normalize_prediction_batches(
    batches: object,
) -> dict[Path, tuple[float, bool, np.ndarray]]:
    """Normalize Anomalib batches while retaining exact requested path identity."""
    normalized: dict[Path, tuple[float, bool, np.ndarray]] = {}
    for batch in batches or []:
        paths = tuple(Path(path).resolve() for path in batch.image_path)
        scores = _array(batch.pred_score).reshape(-1)
        labels = _array(batch.pred_label).reshape(-1)
        maps = _array(batch.anomaly_map)
        while maps.ndim > 3 and maps.shape[1] == 1:
            maps = maps[:, 0]
        if maps.ndim == 2:
            maps = maps[np.newaxis, ...]
        if not (len(paths) == len(scores) == len(labels) == len(maps)):
            raise ValueError("EfficientAD prediction batch lengths do not match")
        for path, score, label, anomaly_map in zip(paths, scores, labels, maps, strict=True):
            if path in normalized:
                raise ValueError(f"duplicate EfficientAD prediction path: {path}")
            map_value = np.asarray(anomaly_map, dtype=np.float32)
            value = float(score)
            if map_value.ndim != 2 or not math.isfinite(value) or not np.isfinite(map_value).all():
                raise ValueError(f"invalid EfficientAD prediction: {path}")
            normalized[path] = (value, bool(label), map_value)
    return normalized


def _predict_paths(model: object, engine: object, paths: list[Path]) -> dict[Path, tuple[float, bool, np.ndarray]]:
    from anomalib.data import PredictDataset

    resolved = [path.resolve() for path in paths]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("prediction paths must be non-empty and unique")
    dataset = PredictDataset(resolved[0])
    dataset.image_filenames = resolved
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=None, return_predictions=True)
    normalized = _normalize_prediction_batches(predictions)
    if set(normalized) != set(resolved):
        raise ValueError("EfficientAD returned paths outside the requested set")
    return normalized


def _inspection_rows(root: Path, capture_ids: list[str]) -> list[tuple[str, Path, dict[str, Any]]]:
    selected = set(capture_ids)
    if len(selected) != len(capture_ids):
        raise ValueError("capture IDs must be unique")
    rows = []
    for inspection_path in sorted(Path(root).expanduser().resolve().glob("*/inspection.json")):
        payload = json.loads(inspection_path.read_text(encoding="utf-8"))
        capture_id = payload.get("capture_id")
        if not isinstance(capture_id, str):
            raise ValueError(f"inspection has no capture_id: {inspection_path}")
        if not selected or capture_id in selected:
            rows.append((capture_id, inspection_path.parent, payload))
    found = {row[0] for row in rows}
    if selected and found != selected:
        raise ValueError(f"capture IDs were not found: {sorted(selected - found)}")
    if not rows:
        raise ValueError("no inspections selected")
    return rows


def _recorded_efficientad(payload: dict[str, Any], view: str) -> dict[str, Any]:
    rows = [
        row
        for row in payload.get("results", [])
        if row.get("branch") == "efficientad" and row.get("view_id") == view
    ]
    if len(rows) != 1:
        raise ValueError(f"inspection must contain one EfficientAD row for {view}")
    return rows[0]


def _masked_heatmap(image: np.ndarray, anomaly_map: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized_mask = cv2.resize(mask, (anomaly_map.shape[1], anomaly_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    masked_map = anomaly_map.copy()
    masked_map[resized_mask == 0] = 0
    heatmap = fixed_scale_heatmap(masked_map)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0.0)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), max(2, round(max(mask.shape) / 512)))
    return heatmap, overlay


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    summaries = []
    for identity, group in sorted(grouped.items()):
        summaries.append(
            {
                key: identity,
                "row_count": len(group),
                "recorded_ng_count": sum(row["recorded_status"] == "NG" for row in group),
                "rerun_ng_count": sum(row["rerun_pred_score"] >= row["deployment_threshold"] for row in group),
                "filled_input_ng_count": sum(
                    row["filled_input_pred_score"] >= row["deployment_threshold"] for row in group
                ),
                "background_hotspot_count": sum(row["unmasked_hotspot_region"] == "background" for row in group),
                "mean_background_activation_fraction": float(
                    np.mean([row["unmasked_background_activation_fraction"] for row in group])
                ),
                "mean_background_max_advantage": float(
                    np.mean([row["unmasked_background_max_advantage"] for row in group])
                ),
                "max_background_max_advantage": float(
                    np.max([row["unmasked_background_max_advantage"] for row in group])
                ),
            }
        )
    return summaries


def evaluate(args: argparse.Namespace) -> Path:
    """Re-infer selected saved ROIs and atomically publish all score/evidence domains."""
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    config, roi_path, threshold_path, training_run = _config_assets(args.config)
    roi_sha = _sha256(roi_path)
    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    masks = load_foreground_mask_asset(
        args.mask_index,
        expected_views=VIEW_ORDER,
        expected_roi_sha256=roi_sha,
    )
    inspections = _inspection_rows(args.inspection_root, args.capture_id)
    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for view in VIEW_ORDER:
            _write_image(staging / "masks" / f"{view}.png", masks.masks[view])
        path_identity: dict[Path, tuple[str, str, str]] = {}
        input_paths: dict[str, dict[str, list[Path]]] = {
            view: {"unmasked": [], "filled": []} for view in VIEW_ORDER
        }
        recorded_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
        for capture_id, record_dir, payload in inspections:
            for view in VIEW_ORDER:
                roi_path_saved = (record_dir / payload["views"][view]["roi"]).resolve()
                image = cv2.imread(str(roi_path_saved), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"saved ROI is unreadable: {roi_path_saved}")
                mask = masks.masks[view]
                filled = apply_fixed_fill(image, mask, masks.fixed_fill_value)
                evidence_dir = staging / "evidence" / capture_id / view
                original_copy = evidence_dir / "original_roi.png"
                filled_path = evidence_dir / "fixed_fill_roi.png"
                _write_image(original_copy, image)
                _write_image(filled_path, filled)
                input_paths[view]["unmasked"].append(original_copy)
                input_paths[view]["filled"].append(filled_path)
                path_identity[original_copy.resolve()] = (capture_id, view, "unmasked")
                path_identity[filled_path.resolve()] = (capture_id, view, "filled")
                recorded_by_identity[(capture_id, view)] = _recorded_efficientad(payload, view)

        predictions: dict[tuple[str, str, str], tuple[float, bool, np.ndarray]] = {}
        for view in VIEW_ORDER:
            checkpoint = (training_run / "efficientad" / view / "model.ckpt").resolve()
            expected_checkpoint_sha = threshold_payload["checkpoint_sha256_by_view"][view]
            if _sha256(checkpoint) != expected_checkpoint_sha:
                raise ValueError(f"EfficientAD checkpoint SHA mismatch: {view}")
            model = EfficientAd.load_from_checkpoint(
                checkpoint,
                map_location="cpu",
                weights_only=False,
                visualizer=False,
            )
            engine = Engine(logger=False)
            for mode in ("unmasked", "filled"):
                for path, prediction in _predict_paths(model, engine, input_paths[view][mode]).items():
                    predictions[path_identity[path]] = prediction
            del engine, model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        result_rows: list[dict[str, Any]] = []
        for capture_id, _record_dir, _payload in inspections:
            for view in VIEW_ORDER:
                evidence_dir = staging / "evidence" / capture_id / view
                image = cv2.imread(str(evidence_dir / "original_roi.png"), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("staged original ROI disappeared")
                recorded = recorded_by_identity[(capture_id, view)]
                rerun_score, rerun_label, unmasked_map = predictions[(capture_id, view, "unmasked")]
                filled_score, filled_label, filled_map = predictions[(capture_id, view, "filled")]
                unmasked_aggregates = aggregate_anomaly_map(
                    unmasked_map,
                    masks.masks[view],
                    quantile=args.quantile,
                    top_k_fraction=args.top_k_fraction,
                    component_threshold=args.component_threshold,
                    minimum_component_area=args.minimum_component_area,
                )
                filled_aggregates = aggregate_anomaly_map(
                    filled_map,
                    masks.masks[view],
                    quantile=args.quantile,
                    top_k_fraction=args.top_k_fraction,
                    component_threshold=args.component_threshold,
                    minimum_component_area=args.minimum_component_area,
                )
                business_category, category_provenance = classify_diagnostic_hotspot(
                    unmasked_aggregates.hotspot_region
                )
                np.save(evidence_dir / "raw_anomaly_map_unmasked.npy", unmasked_map, allow_pickle=False)
                np.save(evidence_dir / "raw_anomaly_map_fixed_fill.npy", filled_map, allow_pickle=False)
                heatmap, overlay = _masked_heatmap(image, unmasked_map, masks.masks[view])
                _write_image(evidence_dir / "masked_heatmap.png", heatmap)
                _write_image(evidence_dir / "masked_heatmap_overlay.png", overlay)
                row: dict[str, Any] = {
                    "capture_id": capture_id,
                    "capture_identity_status": "capture_id_not_proven_physical_part_id",
                    "view": view,
                    "business_category": business_category,
                    "category_provenance": category_provenance,
                    "recorded_status": recorded["status"],
                    "recorded_pred_score": float(recorded["score"]),
                    "deployment_threshold": float(recorded["threshold"]),
                    "rerun_pred_score": rerun_score,
                    "rerun_raw_pred_label": rerun_label,
                    "filled_input_pred_score": filled_score,
                    "filled_input_raw_pred_label": filled_label,
                    "pred_score_domain_note": "pred_score_is_not_directly_comparable_to_map_aggregates",
                    "fixed_fill_distribution_note": "current_checkpoint_was_not_trained_with_fixed_fill",
                }
                row.update({f"unmasked_{key}": value for key, value in unmasked_aggregates.to_dict().items()})
                row.update({f"filled_{key}": value for key, value in filled_aggregates.to_dict().items()})
                result_rows.append(row)
                (evidence_dir / "score_components.json").write_text(
                    json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
        per_capture = _summary_rows(result_rows, "capture_id")
        per_view = _summary_rows(result_rows, "view")
        _write_csv(staging / "per_image_scores.csv", result_rows)
        _write_csv(staging / "per_capture_summary.csv", per_capture)
        _write_csv(staging / "per_view_summary.csv", per_view)
        report = {
            "schema_version": "bmw.efficientad_mask_ab/1.0",
            "experiment_id": destination.name,
            "candidate_only": True,
            "diagnostic_hot_regions_are_not_defect_segmentation": True,
            "business_labels": {
                "status": "fixture_background_spatial_only_other_categories_unavailable",
                "allowed_values": [
                    "fixture_or_background",
                    "acceptable_stain",
                    "visible_defect",
                    "pose_or_light_drift",
                    "uncertain",
                ],
            },
            "selection": {
                "capture_count": len(inspections),
                "view_row_count": len(result_rows),
                "capture_ids": [row[0] for row in inspections],
            },
            "assets": {
                "config_path": str(Path(args.config).resolve()),
                "config_sha256": _sha256(Path(args.config).resolve()),
                "public_roi_config_path": str(roi_path),
                "public_roi_config_sha256": roi_sha,
                "threshold_artifact_path": str(threshold_path),
                "threshold_artifact_sha256": _sha256(threshold_path),
                "foreground_mask_index_path": str(masks.index_path),
                "foreground_mask_index_sha256": masks.index_sha256,
                "mask_sha256_by_view": dict(masks.mask_sha256_by_view),
            },
            "aggregators": {
                "quantile": args.quantile,
                "top_k_fraction": args.top_k_fraction,
                "component_threshold": args.component_threshold,
                "minimum_component_area": args.minimum_component_area,
                "connectivity": 8,
                "component_score": "maximum_component_mean",
            },
            "score_domain_warning": (
                "Engine pred_score and returned anomaly_map use separate post-processing parameters; map aggregators "
                "must be calibrated independently and must not use the V3 pred_score threshold."
            ),
            "fixed_fill_warning": (
                "The current checkpoint was not trained with fixed outside fill; filled-input predictions "
                "are diagnostic "
                "distribution-shift evidence, not V4 deployment scores."
            ),
            "aggregate": {
                "background_hotspot_rows": sum(row["unmasked_hotspot_region"] == "background" for row in result_rows),
                "business_category_counts": {
                    category: sum(row["business_category"] == category for row in result_rows)
                    for category in (
                        "fixture_or_background",
                        "acceptable_stain",
                        "visible_defect",
                        "pose_or_light_drift",
                        "uncertain",
                    )
                },
                "mean_background_activation_fraction": float(
                    np.mean([row["unmasked_background_activation_fraction"] for row in result_rows])
                ),
                "rows_with_background_max_advantage": sum(
                    row["unmasked_background_max_advantage"] > 0 for row in result_rows
                ),
            },
            "per_capture": per_capture,
            "per_view": per_view,
        }
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            "# BMW EfficientAD masked-map A/B\n\n"
            f"- Captures: {len(inspections)}\n"
            f"- View rows: {len(result_rows)}\n"
            f"- Background diagnostic hotspots: {report['aggregate']['background_hotspot_rows']}\n"
            "- Mean background activation fraction: "
            f"{report['aggregate']['mean_background_activation_fraction']:.6f}\n\n"
            "Only candidate-mask-excluded hotspots are classified as `fixture_or_background`; all foreground semantics "
            "remain `uncertain` until human labels are supplied. The V3 pred-score threshold "
            "does not apply to map aggregators. Fixed-fill inference is diagnostic because these checkpoints were not "
            "trained with fixed fill. EfficientAD heatmaps are diagnostic hot regions only.\n",
            encoding="utf-8",
        )
        _atomic_publish_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main() -> int:
    """Run one explicit offline stage."""
    args = build_parser().parse_args()
    output = build_masks(args) if args.command == "build-masks" else evaluate(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
