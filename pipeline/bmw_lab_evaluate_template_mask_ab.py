#!/usr/bin/env python3
"""Calibrate and evaluate BMW Template matching with the manual ignore mask."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

from bmw_inspection.lab.efficientad_ignore_mask import load_ignore_mask_asset
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import load_demo_config
from bmw_inspection.lab.eight_view_demo_models import EightViewTemplatePredictor
from bmw_inspection.lab.template_mask_ab import fit_masked_thresholds, summarize_decisions


EXPECTED_FIRST41_INDEX_PREFIX_SHA256 = "c4707c81b1bd145dd8b108b297394face234b7e5f5298af9faff663438deaf63"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_image(path: Path) -> Any:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(rows: Sequence[Mapping[str, Any]], status_field: str) -> dict[str, Any]:
    normal = [row for row in rows if row.get("label") == "normal"]
    defect = [row for row in rows if row.get("label") == "defect"]
    false_rejects = sum(row[status_field] == "NG" for row in normal)
    false_accepts = sum(row[status_field] == "PASS" for row in defect)
    balanced_accuracy = None
    if normal and defect:
        balanced_accuracy = (
            1.0 - false_rejects / len(normal) + 1.0 - false_accepts / len(defect)
        ) / 2.0
    return {
        "row_count": len(rows),
        "normal_count": len(normal),
        "defect_count": len(defect),
        "normal_false_rejects": false_rejects,
        "defect_false_accepts": false_accepts,
        "balanced_accuracy": balanced_accuracy,
    }


def _score_manifest(
    manifest: Path,
    predictor: EightViewTemplatePredictor,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "part_id", "view_id", "image_path", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Template trainer manifest fields are incomplete")
        for source in reader:
            if source["split"] not in {"calibration", "final_test"}:
                continue
            view = source["view_id"]
            output = predictor.predict(view, _read_image(Path(source["image_path"])))
            records.append(
                {
                    "sample_id": source["sample_id"],
                    "part_id": source["part_id"],
                    "view_id": view,
                    "split": source["split"],
                    "label": source["label"],
                    "image_path": source["image_path"],
                    "raw_risk": float(output.details["raw_unmasked_risk"]),
                    "masked_risk": float(output.details["masked_risk"]),
                    "raw_shift_x": int(output.details["raw_unmasked_best_shift_x"]),
                    "raw_shift_y": int(output.details["raw_unmasked_best_shift_y"]),
                    "masked_shift_x": int(output.details["best_shift_x"]),
                    "masked_shift_y": int(output.details["best_shift_y"]),
                    "valid_target_pixel_fraction": float(output.details["valid_target_pixel_fraction"]),
                }
            )
    return records


def _first41_rows(index_path: Path) -> list[dict[str, str]]:
    lines = index_path.read_bytes().splitlines(keepends=True)
    if len(lines) < 42:
        raise ValueError("inspection index does not contain the pinned first 41 rows")
    prefix_sha = hashlib.sha256(b"".join(lines[:42])).hexdigest()
    if prefix_sha != EXPECTED_FIRST41_INDEX_PREFIX_SHA256:
        raise ValueError(f"first-41 inspection prefix SHA256 drifted: {prefix_sha}")
    decoded = b"".join(lines[:42]).decode("utf-8-sig").splitlines()
    return list(csv.DictReader(decoded))


def _score_first41(
    result_root: Path,
    predictor: EightViewTemplatePredictor,
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index_row in _first41_rows(result_root / "inspection_index.csv"):
        capture_id = index_row["capture_id"]
        directory = result_root / index_row["result_path"]
        inspection = json.loads((directory / "inspection.json").read_text(encoding="utf-8"))
        saved = {
            row["view_id"]: row
            for row in inspection["results"]
            if row["branch"] == "template"
        }
        if tuple(saved) != VIEW_ORDER:
            raise ValueError(f"{capture_id} does not contain eight ordered Template results")
        for view in VIEW_ORDER:
            source = saved[view]
            output = predictor.predict(view, _read_image(directory / "rois" / f"{view}.png"))
            raw_risk = float(output.details["raw_unmasked_risk"])
            if abs(raw_risk - float(source["score"])) > 2e-5:
                raise ValueError(
                    f"{capture_id}/{view} raw replay drift: {raw_risk} vs {source['score']}"
                )
            masked_risk = float(output.details["masked_risk"])
            records.append(
                {
                    "capture_id": capture_id,
                    "view_id": view,
                    "roi_path": str(directory / "rois" / f"{view}.png"),
                    "old_status": source["status"],
                    "old_risk": float(source["score"]),
                    "old_threshold": float(source["threshold"]),
                    "masked_status": "PASS" if masked_risk <= thresholds[view] else "NG",
                    "masked_risk": masked_risk,
                    "masked_threshold": thresholds[view],
                    "risk_change": masked_risk - float(source["score"]),
                    "raw_shift_x": int(output.details["raw_unmasked_best_shift_x"]),
                    "raw_shift_y": int(output.details["raw_unmasked_best_shift_y"]),
                    "masked_shift_x": int(output.details["best_shift_x"]),
                    "masked_shift_y": int(output.details["best_shift_y"]),
                    "valid_target_pixel_fraction": float(output.details["valid_target_pixel_fraction"]),
                }
            )
    return records


def _capture_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["capture_id"])].append(row)
    records = [
        {
            "capture_id": capture_id,
            "old_status": "NG" if any(row["old_status"] == "NG" for row in items) else "PASS",
            "masked_status": "NG" if any(row["masked_status"] == "NG" for row in items) else "PASS",
        }
        for capture_id, items in grouped.items()
    ]
    return {**summarize_decisions(records), "records": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bmw/experiments/bmw_eight_view_demo_v4_manual_ignore_mask_v3.json"),
    )
    parser.add_argument(
        "--trainer-manifest",
        type=Path,
        default=Path(
            "dataset/bmw_lab_training/bmw_right_multisource_left_yolo_v1/template/trainer_manifest.csv"
        ),
    )
    parser.add_argument(
        "--inspection-root",
        type=Path,
        default=Path("results/bmw_eight_view_demo_v3_ng_evidence_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/bmw_template_manual_ignore_ab/bmw_right_manual_ignore_v3_template_ab_v1"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_demo_config(args.config)
    manifest = args.trainer_manifest.expanduser().resolve()
    inspection_root = args.inspection_root.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"A/B output already exists: {destination}")
    if config.efficientad_ignore_mask_index is None:
        raise ValueError("config does not contain the approved manual ignore mask")
    models = {
        view: json.loads(config.template_models[view].read_text(encoding="utf-8"))
        for view in VIEW_ORDER
    }
    expected_shapes = {
        view: (int(models[view]["input_height"]), int(models[view]["input_width"]))
        for view in VIEW_ORDER
    }
    mask_asset = load_ignore_mask_asset(
        config.efficientad_ignore_mask_index,
        expected_views=VIEW_ORDER,
        expected_roi_config_sha256=_sha256(config.roi_config),
        expected_shapes=expected_shapes,
    )
    predictor = EightViewTemplatePredictor(
        config.template_models,
        ignore_masks=mask_asset.masks,
        ignore_mask_index_sha256=mask_asset.index_sha256,
        masked_thresholds={view: 1.0 for view in VIEW_ORDER},
        masked_threshold_artifact_sha256="0" * 64,
    )
    manifest_rows = _score_manifest(manifest, predictor)
    thresholds, calibration_fits = fit_masked_thresholds(manifest_rows)
    base_thresholds = {view: float(models[view]["threshold"]) for view in VIEW_ORDER}
    for row in manifest_rows:
        view = str(row["view_id"])
        row["old_status"] = "PASS" if float(row["raw_risk"]) <= base_thresholds[view] else "NG"
        row["old_threshold"] = base_thresholds[view]
        row["masked_status"] = "PASS" if float(row["masked_risk"]) <= thresholds[view] else "NG"
        row["masked_threshold"] = thresholds[view]
    first41 = _score_first41(inspection_root, predictor, thresholds)

    destination.mkdir(parents=True)
    _write_csv(destination / "manifest_template_ab.csv", manifest_rows)
    _write_csv(destination / "first41_template_ab.csv", first41)
    per_view: dict[str, Any] = {}
    for view in VIEW_ORDER:
        view_rows = [row for row in manifest_rows if row["view_id"] == view]
        per_view[view] = {
            "model_json": str(config.template_models[view]),
            "model_json_sha256": _sha256(config.template_models[view]),
            "base_threshold": base_thresholds[view],
            "masked_threshold": thresholds[view],
            "calibration_fit": calibration_fits[view],
            "calibration_old_metrics": _metrics(
                [row for row in view_rows if row["split"] == "calibration"], "old_status"
            ),
            "calibration_masked_metrics": _metrics(
                [row for row in view_rows if row["split"] == "calibration"], "masked_status"
            ),
            "final_test_old_metrics": _metrics(
                [row for row in view_rows if row["split"] == "final_test"], "old_status"
            ),
            "final_test_masked_metrics": _metrics(
                [row for row in view_rows if row["split"] == "final_test"], "masked_status"
            ),
            "first41_decisions": summarize_decisions([row for row in first41 if row["view_id"] == view]),
        }
    threshold_artifact = {
        "schema_version": "bmw.template_manual_ignore_thresholds/1.0",
        "candidate_only": True,
        "demo_only": True,
        "selection_split": "calibration",
        "final_test_used_for_selection": False,
        "manual_ignore_mask_index": str(mask_asset.index_path),
        "manual_ignore_mask_index_sha256": mask_asset.index_sha256,
        "public_roi_config": str(config.roi_config),
        "public_roi_config_sha256": _sha256(config.roi_config),
        "trainer_manifest": str(manifest),
        "trainer_manifest_sha256": _sha256(manifest),
        "first41_inspection_root": str(inspection_root),
        "first41_index_prefix_sha256": EXPECTED_FIRST41_INDEX_PREFIX_SHA256,
        "thresholds": thresholds,
        "views": per_view,
    }
    _write_json(destination / "template_masked_thresholds.json", threshold_artifact)
    report = {
        "schema_version": "bmw.template_manual_ignore_ab/1.0",
        "threshold_artifact": str(destination / "template_masked_thresholds.json"),
        "manifest_row_count": len(manifest_rows),
        "manifest_calibration_summary": {
            "old": _metrics([row for row in manifest_rows if row["split"] == "calibration"], "old_status"),
            "masked": _metrics(
                [row for row in manifest_rows if row["split"] == "calibration"], "masked_status"
            ),
        },
        "manifest_final_test_summary": {
            "old": _metrics([row for row in manifest_rows if row["split"] == "final_test"], "old_status"),
            "masked": _metrics(
                [row for row in manifest_rows if row["split"] == "final_test"], "masked_status"
            ),
        },
        "first41_view_decisions": summarize_decisions(first41),
        "first41_capture_decisions": _capture_summary(first41),
        "sentinels": {
            capture_id: [row for row in first41 if row["capture_id"] == capture_id]
            for capture_id in (
                "bmw_demo_20260812_170451",
                "bmw_demo_20260812_171815",
                "bmw_demo_20260812_173050",
                "bmw_demo_20260812_205326",
            )
        },
        "per_view": per_view,
    }
    _write_json(destination / "report.json", report)
    print(json.dumps(report["first41_view_decisions"], ensure_ascii=False))
    print(json.dumps(report["first41_capture_decisions"], ensure_ascii=False))
    print(destination / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
