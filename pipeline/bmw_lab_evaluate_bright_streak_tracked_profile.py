#!/usr/bin/env python3
"""Fit and evaluate the immutable BMW tracked-profile bright-streak v3 candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.bright_streak_tracked_profile import (  # noqa: E402
    TRACKED_PROFILE_ROI_SHAPE,
    TrackedProfileGeometry,
    TrackedProfileMetrics,
    TrackedProfileThresholds,
    analyze_tracked_profile,
    classify_tracked_profile,
    fit_tracked_profile_thresholds,
)


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "dataset/bmw_lab_prepared/bmw_right_batch_20260810_21_v1/manifests/bright_streak.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v3_tracked"
)
DEFAULT_ROI_XYXY = [1792, 1180, 1873, 1793]
DEFAULT_V2_REPORT = (
    REPO_ROOT
    / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v2_roi_corrected/"
    "report.json"
)
DEFAULT_V2_METRICS = DEFAULT_V2_REPORT.parent / "metrics.csv"
_ACCEPTED_RECORD_FILES = (
    Path("inspection.json"),
    Path("images/front_left_hdr.png"),
    Path("images/front_left_short.png"),
    Path("images/front_left_long.png"),
)
GEOMETRY_CANDIDATE_WIDTHS = (5, 7, 9)


def _absolute(path: Path) -> Path:
    return Path(path).expanduser().absolute()


def _validate_accepted_record(record_path: Path) -> None:
    for relative_path in _ACCEPTED_RECORD_FILES:
        candidate = record_path / relative_path
        if not candidate.is_file():
            raise ValueError(
                f"accepted normal record is missing required file: {candidate}"
            )
    inspection_path = record_path / "inspection.json"
    try:
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid accepted normal inspection: {inspection_path}") from error
    if inspection.get("capture_id") != record_path.name:
        raise ValueError(
            "accepted normal inspection capture_id must match record directory: "
            f"{record_path}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _read_roi(image_path: Path, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode bright-streak image: {image_path}")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("bright-streak source image must be grayscale, BGR, or BGRA")
    x1, y1, x2, y2 = roi_xyxy
    roi = gray[y1:y2, x1:x2]
    if roi.shape != TRACKED_PROFILE_ROI_SHAPE:
        raise ValueError("bright-streak config ROI must remain 81x613 for tracked-profile v3")
    return roi


def _provisional_thresholds() -> TrackedProfileThresholds:
    return TrackedProfileThresholds(
        strong_row_score=0.0,
        weak_row_score=0.0,
        min_presence_coverage_ratio=0.0,
        min_longest_run_ratio=0.0,
        max_gap_ratio=1.0,
        max_gap_count=TRACKED_PROFILE_ROI_SHAPE[0],
    )


def _analyze(
    image_path: Path,
    roi_xyxy: tuple[int, int, int, int],
    geometry: TrackedProfileGeometry,
    thresholds: TrackedProfileThresholds,
) -> tuple[TrackedProfileMetrics, float]:
    roi = _read_roi(image_path, roi_xyxy)
    started = time.process_time_ns()
    metrics = analyze_tracked_profile(roi, geometry, thresholds)
    elapsed_ms = (time.process_time_ns() - started) / 1_000_000.0
    return metrics, elapsed_ms


def _manifest_records(
    manifest_path: Path,
    roi_xyxy: tuple[int, int, int, int],
    geometry: TrackedProfileGeometry,
    *,
    splits: frozenset[str] = frozenset({"calibration", "final_test"}),
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "split", "expected_status", "source_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("bright-streak manifest is missing required fields")
        for row_number, row in enumerate(reader, start=2):
            if row["split"] not in splits:
                continue
            if row["expected_status"] not in {"OK", "NG_NO_STREAK"}:
                continue
            source_path = Path(row["source_path"]).expanduser()
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            source_path = source_path.resolve()
            if not source_path.is_file():
                raise ValueError(f"bright-streak source image does not exist: {source_path}")
            source_sha256 = _sha256(source_path)
            declared_sha256 = row.get("source_sha256", "").strip()
            if declared_sha256 and declared_sha256 != source_sha256:
                raise ValueError(f"bright-streak source SHA-256 mismatch: {source_path}")
            metrics, elapsed_ms = _analyze(
                source_path,
                roi_xyxy,
                geometry,
                _provisional_thresholds(),
            )
            records.append(
                {
                    "record_id": f"manifest-row-{row_number:04d}",
                    "row_number": row_number,
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "expected_status": row["expected_status"],
                    "label": "normal" if row["expected_status"] == "OK" else "no_streak",
                    "provenance_kind": "manifest",
                    "source_path": source_path,
                    "source_sha256": source_sha256,
                    "path_scores": metrics.path_scores,
                    "initial_cpu_ms": elapsed_ms,
                }
            )
    if not records:
        raise ValueError("bright-streak manifest has no calibration or final_test decision rows")
    return records


def _selection_accepted_rows(
    accepted_paths: Sequence[Path],
    roi_xyxy: tuple[int, int, int, int],
    geometry: TrackedProfileGeometry,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record_path in accepted_paths:
        source_paths = _live_source_paths(record_path)
        metrics, _ = _analyze(
            source_paths["front_left_hdr.png"], roi_xyxy, geometry, _provisional_thresholds()
        )
        rows.append(
            {
                "label": "normal",
                "path_scores": metrics.path_scores,
                "source_path": source_paths["front_left_hdr.png"],
            }
        )
    return rows


def _select_geometry(
    manifest_path: Path,
    accepted_paths: Sequence[Path],
    roi_xyxy: tuple[int, int, int, int],
) -> tuple[TrackedProfileGeometry, dict[str, object]]:
    """Select geometry using only calibration and explicitly confirmed normal inputs."""
    candidates: list[dict[str, object]] = []
    candidate_objects: dict[int, TrackedProfileGeometry] = {}
    for candidate_width in GEOMETRY_CANDIDATE_WIDTHS:
        geometry = TrackedProfileGeometry(candidate_width=candidate_width)
        candidate_objects[candidate_width] = geometry
        calibration = _manifest_records(
            manifest_path,
            roi_xyxy,
            geometry,
            splits=frozenset({"calibration"}),
        )
        accepted = _selection_accepted_rows(accepted_paths, roi_xyxy, geometry)
        fit_rows = [*calibration, *accepted]
        thresholds = fit_tracked_profile_thresholds(
            [
                {
                    "label": row["label"],
                    "path_scores": row["path_scores"],
                    "split": "calibration",
                    "final_test": False,
                }
                for row in fit_rows
            ],
            geometry,
        )
        normal_metrics: list[TrackedProfileMetrics] = []
        normal_errors = no_streak_errors = accepted_errors = 0
        for row in calibration:
            metrics, _ = _analyze(Path(row["source_path"]), roi_xyxy, geometry, thresholds)
            status = classify_tracked_profile(metrics, thresholds)
            if row["label"] == "normal":
                normal_metrics.append(metrics)
                normal_errors += status != "OK"
            else:
                no_streak_errors += status != "NG_NO_STREAK"
        for row in accepted:
            metrics, _ = _analyze(Path(row["source_path"]), roi_xyxy, geometry, thresholds)
            normal_metrics.append(metrics)
            accepted_errors += classify_tracked_profile(metrics, thresholds) != "OK"
        candidates.append(
            {
                "candidate_width": candidate_width,
                "calibration_normal_false_rejects": normal_errors,
                "calibration_no_streak_errors": no_streak_errors,
                "accepted_live_false_rejects": accepted_errors,
                "weakest_normal_coverage_ratio": min(
                    item.coverage_ratio for item in normal_metrics
                ),
                "normal_max_gap_ratio": max(item.max_gap_ratio for item in normal_metrics),
                "normal_max_gap_count": max(item.gap_count for item in normal_metrics),
                "strong_row_score": thresholds.strong_row_score,
                "weak_row_score": thresholds.weak_row_score,
            }
        )
    valid = [
        item
        for item in candidates
        if item["calibration_normal_false_rejects"] == 0
        and item["calibration_no_streak_errors"] == 0
        and item["accepted_live_false_rejects"] == 0
    ]
    if not valid:
        raise RuntimeError("no tracked-profile geometry candidate passes calibration acceptance")
    selected = max(
        valid,
        key=lambda item: (
            item["weakest_normal_coverage_ratio"],
            -item["normal_max_gap_count"],
            -item["normal_max_gap_ratio"],
            -abs(item["candidate_width"] - TrackedProfileGeometry().candidate_width),
        ),
    )
    selection_inputs = {
        "manifest_sha256": _sha256(manifest_path),
        "roi_xyxy": list(roi_xyxy),
        "candidate_widths": list(GEOMETRY_CANDIDATE_WIDTHS),
        "algorithm_source_sha256": _sha256(
            SRC_ROOT / "bmw_inspection/lab/bright_streak_tracked_profile.py"
        ),
        "accepted_files": {
            path.name: {
                name: _sha256(file_path)
                for name, file_path in _live_source_paths(path).items()
            }
            for path in accepted_paths
        },
    }
    evidence: dict[str, object] = {
        "fit_data": "calibration_plus_user_confirmed_live_normal",
        "final_test_used_for_selection": False,
        "candidate_widths": list(GEOMETRY_CANDIDATE_WIDTHS),
        "selection_rule": (
            "require zero calibration/accepted errors; maximize weakest normal coverage; "
            "then minimize max gap count, max gap ratio, and distance from Task-1 default"
        ),
        "candidates": candidates,
        "selected_candidate_width": selected["candidate_width"],
        "selection_input_sha256": _json_sha256(selection_inputs),
    }
    evidence["selection_evidence_sha256"] = _json_sha256(evidence)
    return candidate_objects[int(selected["candidate_width"])], evidence


def _live_source_paths(record_path: Path) -> dict[str, Path]:
    return {
        "inspection.json": record_path / "inspection.json",
        "front_left_hdr.png": record_path / "images/front_left_hdr.png",
        "front_left_short.png": record_path / "images/front_left_short.png",
        "front_left_long.png": record_path / "images/front_left_long.png",
    }


def _inspection_v2_status(inspection_path: Path) -> str | None:
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    for result in inspection.get("results", []):
        if result.get("branch") == "bright_streak" and result.get("view_id") == "front_left":
            decision = result.get("details", {}).get("decision")
            if decision in {"OK", "NG_NO_STREAK", "NG_BROKEN"}:
                return decision
            return "OK" if result.get("status") == "PASS" else None
    return None


def _recent_live_records(accepted_records: Sequence[Path]) -> list[Path]:
    candidates: dict[Path, int] = {}
    for accepted in accepted_records:
        for candidate in accepted.parent.iterdir():
            inspection = candidate / "inspection.json"
            if candidate.is_dir() and inspection.is_file():
                candidates[candidate.resolve()] = inspection.stat().st_mtime_ns
    return [item[0] for item in sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:20]]


def _load_v2_outcomes() -> dict[str, str]:
    outcomes: dict[str, str] = {}
    try:
        if DEFAULT_V2_METRICS.is_file():
            with DEFAULT_V2_METRICS.open("r", newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if row.get("sample_id") and row.get("raw_profile_predicted_status"):
                        outcomes[row["sample_id"]] = row["raw_profile_predicted_status"]
        if DEFAULT_V2_REPORT.is_file():
            report = json.loads(DEFAULT_V2_REPORT.read_text(encoding="utf-8"))
            outcomes.update(
                {
                    str(item["sample_id"]): str(item["predicted_status"])
                    for item in report.get("raw_profile_v2", {}).get("outcomes", [])
                }
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return outcomes


def _save_profile(path: Path, metrics: TrackedProfileMetrics) -> None:
    np.savez_compressed(
        path,
        response_map=metrics.response_map,
        path_x=metrics.path_x,
        path_scores=metrics.path_scores,
        strong_mask=metrics.strong_mask,
        accepted_mask=metrics.accepted_mask,
        bridged_mask=metrics.bridged_mask,
    )


def _metric_row(
    record: Mapping[str, object],
    metrics: TrackedProfileMetrics,
    predicted_status: str,
    profile_path: Path,
    cpu_ms: float,
    v2_status: str | None,
) -> dict[str, object]:
    return {
        "record_id": record["record_id"],
        "sample_id": record["sample_id"],
        "split": record["split"],
        "expected_status": record["expected_status"],
        "truth": record["truth"],
        "provenance_kind": record["provenance_kind"],
        "source_path": str(record["source_path"]),
        "source_sha256": record["source_sha256"],
        "profile_npz": str(profile_path),
        "v2_predicted_status": v2_status or "",
        "tracked_profile_v3_predicted_status": predicted_status,
        "v2_to_v3_changed": v2_status is not None and v2_status != predicted_status,
        "path_score_max": float(metrics.path_scores.max()),
        "coverage_ratio": metrics.coverage_ratio,
        "longest_run_px": metrics.longest_run_px,
        "longest_run_ratio": metrics.longest_run_ratio,
        "max_gap_px": metrics.max_gap_px,
        "max_gap_ratio": metrics.max_gap_ratio,
        "gap_count": metrics.gap_count,
        "strong_rows": int(metrics.strong_mask.sum()),
        "accepted_rows": int(metrics.accepted_mask.sum()),
        "bridged_rows": int(metrics.bridged_mask.sum()),
        "cpu_ms": cpu_ms,
    }


def _comparison_groups(
    manifest_outcomes: Sequence[Mapping[str, object]],
    v2_outcomes: Mapping[str, str],
) -> dict[str, object]:
    comparable = [row for row in manifest_outcomes if row["sample_id"] in v2_outcomes]
    groups: dict[str, object] = {}
    for split in ("calibration", "final_test"):
        split_rows = [row for row in comparable if row["split"] == split]
        normal_rows = [row for row in split_rows if row["expected_status"] == "OK"]
        groups[split] = {
            "count": len(split_rows),
            "normal_count": len(normal_rows),
            "v2_normal_false_rejects": sum(
                v2_outcomes[str(row["sample_id"])] != "OK" for row in normal_rows
            ),
            "v3_normal_false_rejects": sum(
                row["predicted_status"] != "OK" for row in normal_rows
            ),
            "changes": [
                {
                    "sample_id": row["sample_id"],
                    "expected_status": row["expected_status"],
                    "v2_predicted_status": v2_outcomes[str(row["sample_id"])],
                    "v3_predicted_status": row["predicted_status"],
                }
                for row in split_rows
                if v2_outcomes[str(row["sample_id"])] != row["predicted_status"]
            ],
        }
    return {"comparable_manifest_count": len(comparable), **groups}


def _enforce_acceptance_gate(
    accepted_outcomes: Sequence[Mapping[str, object]],
    no_streak_outcomes: Sequence[Mapping[str, object]],
    comparison: Mapping[str, object],
) -> dict[str, object]:
    rejected_accepted = [
        str(row.get("capture_id"))
        for row in accepted_outcomes
        if row.get("predicted_status") != "OK"
    ]
    if rejected_accepted:
        raise RuntimeError(
            "confirmed live normal acceptance gate failed: " + ", ".join(rejected_accepted)
        )
    bad_no_streak = [
        str(row.get("sample_id"))
        for row in no_streak_outcomes
        if row.get("predicted_status") != "NG_NO_STREAK"
    ]
    if bad_no_streak:
        raise RuntimeError("no-streak acceptance gate failed: " + ", ".join(bad_no_streak))
    for split in ("calibration", "final_test"):
        group = comparison.get(split)
        if not isinstance(group, Mapping) or not group.get("normal_count"):
            continue
        if group["v3_normal_false_rejects"] > group["v2_normal_false_rejects"]:
            raise RuntimeError(f"normal false rejects are worse than v2 for {split}")
    return {
        "passed": True,
        "confirmed_live_normal_count": len(accepted_outcomes),
        "no_streak_count": len(no_streak_outcomes),
        "normal_not_worse_than_v2_splits": [
            split
            for split in ("calibration", "final_test")
            if isinstance(comparison.get(split), Mapping)
            and comparison[split].get("normal_count")
        ],
    }


def evaluate_bright_streak_tracked_profile(
    manifest: Path,
    output_dir: Path,
    accepted_normal_records: Sequence[Path],
    roi_xyxy: tuple[int, int, int, int],
) -> Mapping[str, object]:
    """Fit and publish one provenance-bound tracked-profile v3 evaluation."""
    output_path = _absolute(output_dir)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"output path already exists: {output_path}")
    manifest_path = _absolute(manifest)
    if not manifest_path.is_file():
        raise ValueError(f"bright-streak manifest does not exist: {manifest_path}")
    if (
        len(roi_xyxy) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in roi_xyxy)
        or (roi_xyxy[3] - roi_xyxy[1], roi_xyxy[2] - roi_xyxy[0])
        != TRACKED_PROFILE_ROI_SHAPE
    ):
        raise ValueError("bright-streak config ROI must remain 81x613 for tracked-profile v3")

    accepted_paths = tuple(_absolute(record).resolve() for record in accepted_normal_records)
    if len(set(accepted_paths)) != len(accepted_paths):
        raise ValueError("accepted normal records must be unique")
    for record in accepted_paths:
        _validate_accepted_record(record)

    geometry, geometry_selection = _select_geometry(
        manifest_path, accepted_paths, roi_xyxy
    )
    manifest_rows = _manifest_records(manifest_path, roi_xyxy, geometry)
    accepted_rows: list[dict[str, object]] = []
    initial_cpu_times = [float(row["initial_cpu_ms"]) for row in manifest_rows]
    for record_path in accepted_paths:
        source_paths = _live_source_paths(record_path)
        metrics, elapsed_ms = _analyze(
            source_paths["front_left_hdr.png"],
            roi_xyxy,
            geometry,
            _provisional_thresholds(),
        )
        initial_cpu_times.append(elapsed_ms)
        accepted_rows.append(
            {
                "record_id": f"accepted-live-{record_path.name}",
                "sample_id": record_path.name,
                "capture_id": record_path.name,
                "split": "calibration",
                "expected_status": "OK",
                "label": "normal",
                "truth": "normal",
                "provenance_kind": "user_confirmed_live_normal",
                "record_path": record_path,
                "source_path": source_paths["front_left_hdr.png"],
                "source_sha256": _sha256(source_paths["front_left_hdr.png"]),
                "file_sha256": {name: _sha256(path) for name, path in source_paths.items()},
                "path_scores": metrics.path_scores,
                "initial_cpu_ms": elapsed_ms,
            }
        )

    calibration_rows = [row for row in manifest_rows if row["split"] == "calibration"]
    final_test_rows = [row for row in manifest_rows if row["split"] == "final_test"]
    fit_rows = [*calibration_rows, *accepted_rows]
    thresholds = fit_tracked_profile_thresholds(
        [
            {
                "label": row["label"],
                "path_scores": row["path_scores"],
                "split": "calibration",
                "final_test": False,
            }
            for row in fit_rows
        ],
        geometry,
    )

    v2_outcomes = _load_v2_outcomes()
    preflight_manifest: list[dict[str, object]] = []
    preflight_accepted: list[dict[str, object]] = []
    for row in manifest_rows:
        metrics, _ = _analyze(Path(row["source_path"]), roi_xyxy, geometry, thresholds)
        preflight_manifest.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "expected_status": row["expected_status"],
                "predicted_status": classify_tracked_profile(metrics, thresholds),
            }
        )
    for row in accepted_rows:
        metrics, _ = _analyze(Path(row["source_path"]), roi_xyxy, geometry, thresholds)
        preflight_accepted.append(
            {
                "capture_id": row["capture_id"],
                "predicted_status": classify_tracked_profile(metrics, thresholds),
            }
        )
    preflight_comparison = _comparison_groups(preflight_manifest, v2_outcomes)
    acceptance_gate = _enforce_acceptance_gate(
        preflight_accepted,
        [row for row in preflight_manifest if row["expected_status"] == "NG_NO_STREAK"],
        preflight_comparison,
    )

    # Reserve the destination only after all fit and real acceptance gates pass.
    output_path.mkdir(parents=True)
    profiles_dir = output_path / "profiles"
    profiles_dir.mkdir()
    metrics_path = output_path / "metrics.csv"
    replay_path = output_path / "replay_summary.json"
    report_path = output_path / "report.json"
    metric_rows: list[dict[str, object]] = []
    final_outcomes: list[dict[str, object]] = []
    manifest_outcomes: list[dict[str, object]] = []
    accepted_outcomes: list[dict[str, object]] = []
    metrics_by_source: dict[Path, tuple[TrackedProfileMetrics, str, Path, float]] = {}

    evaluation_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        evaluation_rows.append({**row, "truth": row["label"]})
    evaluation_rows.extend(accepted_rows)
    for index, row in enumerate(evaluation_rows, start=1):
        source_path = Path(row["source_path"])
        metrics, cpu_ms = _analyze(source_path, roi_xyxy, geometry, thresholds)
        predicted_status = classify_tracked_profile(metrics, thresholds)
        profile_path = profiles_dir / f"profile_{index:04d}.npz"
        _save_profile(profile_path, metrics)
        metrics_by_source[source_path.resolve()] = (
            metrics,
            predicted_status,
            profile_path,
            cpu_ms,
        )
        v2_status = v2_outcomes.get(str(row["sample_id"]))
        metric_rows.append(
            _metric_row(row, metrics, predicted_status, profile_path, cpu_ms, v2_status)
        )
        outcome = {
            "sample_id": row["sample_id"],
            "expected_status": row["expected_status"],
            "predicted_status": predicted_status,
            "correct": predicted_status == row["expected_status"],
        }
        if row["split"] == "final_test":
            final_outcomes.append(outcome)
        if row["provenance_kind"] == "manifest":
            manifest_outcomes.append({**outcome, "split": row["split"]})
        if row["provenance_kind"] == "user_confirmed_live_normal":
            accepted_outcomes.append(
                {
                    "capture_id": row["capture_id"],
                    "record_path": str(row["record_path"]),
                    "provenance_kind": row["provenance_kind"],
                    "file_sha256": row["file_sha256"],
                    "predicted_status": predicted_status,
                }
            )

    replay_outcomes: list[dict[str, object]] = []
    accepted_set = set(accepted_paths)
    for record_path in _recent_live_records(accepted_paths):
        try:
            _validate_accepted_record(record_path)
        except ValueError:
            continue
        source_paths = _live_source_paths(record_path)
        source_path = source_paths["front_left_hdr.png"].resolve()
        cached = metrics_by_source.get(source_path)
        if cached is None:
            metrics, cpu_ms = _analyze(source_path, roi_xyxy, geometry, thresholds)
            initial_cpu_times.append(cpu_ms)
            predicted_status = classify_tracked_profile(metrics, thresholds)
            profile_path = profiles_dir / f"profile_{len(metric_rows) + 1:04d}.npz"
            _save_profile(profile_path, metrics)
            row = {
                "record_id": f"live-replay-{record_path.name}",
                "sample_id": record_path.name,
                "split": "live_replay",
                "expected_status": "",
                "truth": "unknown",
                "provenance_kind": "unconfirmed_live_replay",
                "source_path": source_path,
                "source_sha256": _sha256(source_path),
            }
            v2_status = _inspection_v2_status(source_paths["inspection.json"])
            metric_rows.append(
                _metric_row(row, metrics, predicted_status, profile_path, cpu_ms, v2_status)
            )
        else:
            metrics, predicted_status, profile_path, cpu_ms = cached
            v2_status = _inspection_v2_status(source_paths["inspection.json"])
        confirmed = record_path in accepted_set
        replay_outcome: dict[str, object] = {
            "capture_id": record_path.name,
            "record_path": str(record_path),
            "truth": "normal" if confirmed else "unknown",
            "included_in_accuracy": confirmed,
            "v2_predicted_status": v2_status,
            "v3_predicted_status": predicted_status,
            "v2_to_v3_changed": v2_status is not None and v2_status != predicted_status,
        }
        if confirmed:
            replay_outcome["correct"] = predicted_status == "OK"
        replay_outcomes.append(replay_outcome)

    if metric_rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)

    replay_summary: dict[str, object] = {
        "limit": 20,
        "count": len(replay_outcomes),
        "known_truth_count": sum(item["truth"] != "unknown" for item in replay_outcomes),
        "unknown_truth_count": sum(item["truth"] == "unknown" for item in replay_outcomes),
        "accuracy_denominator": sum(item["included_in_accuracy"] for item in replay_outcomes),
        "outcomes": replay_outcomes,
    }
    _write_json(replay_path, replay_summary)
    artifact_identities = {
        "metrics_csv_sha256": _sha256(metrics_path),
        "replay_summary_sha256": _sha256(replay_path),
        "profile_npz_sha256": {
            path.name: _sha256(path) for path in sorted(profiles_dir.glob("*.npz"))
        },
    }

    normal_final = [row for row in final_outcomes if row["expected_status"] == "OK"]
    no_streak_final = [
        row for row in final_outcomes if row["expected_status"] == "NG_NO_STREAK"
    ]
    comparison_groups = _comparison_groups(manifest_outcomes, v2_outcomes)
    comparable_count = int(comparison_groups["comparable_manifest_count"])
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "algorithm": "tracked_profile_v3",
        "fit_split": "calibration",
        "final_test_used_for_fit": False,
        "real_broken_samples": 0,
        "manifest": str(manifest_path),
        "roi_xyxy": list(roi_xyxy),
        "geometry": asdict(geometry),
        "geometry_selection": geometry_selection,
        "thresholds": asdict(thresholds),
        "calibration_counts": {
            "manifest_normal": sum(row["label"] == "normal" for row in calibration_rows),
            "manifest_no_streak": sum(row["label"] == "no_streak" for row in calibration_rows),
            "accepted_live_normal": len(accepted_rows),
            "fit_total": len(fit_rows),
        },
        "accepted_live_normals": accepted_outcomes,
        "acceptance_gate": acceptance_gate,
        "comparison_to_v2": {
            "available": bool(comparable_count),
            "baseline_report": str(DEFAULT_V2_REPORT) if comparable_count else None,
            "baseline_report_sha256": (
                _sha256(DEFAULT_V2_REPORT) if comparable_count else None
            ),
            "baseline_metrics": str(DEFAULT_V2_METRICS) if comparable_count else None,
            "baseline_metrics_sha256": (
                _sha256(DEFAULT_V2_METRICS) if comparable_count else None
            ),
            **comparison_groups,
        },
        "final_test": {
            "count": len(final_outcomes),
            "normal_count": len(normal_final),
            "no_streak_count": len(no_streak_final),
            "normal_false_rejects": sum(not row["correct"] for row in normal_final),
            "no_streak_false_accepts": sum(not row["correct"] for row in no_streak_final),
            "outcomes": final_outcomes,
        },
        "replay": replay_summary,
        "artifact_identities": artifact_identities,
        "cpu_per_image_ms": {
            "scope": "offline_process_cpu_not_hardware_cycle",
            "count": len(initial_cpu_times),
            "p50": float(np.median(initial_cpu_times)),
            "max": max(initial_cpu_times),
        },
        "metrics_csv": str(metrics_path),
        "profiles_dir": str(profiles_dir),
        "replay_summary_json": str(replay_path),
        "report_json": str(report_path),
        "identities": {
            "manifest_sha256": _sha256(manifest_path),
            "roi_config_sha256": _json_sha256({"roi_xyxy": list(roi_xyxy)}),
            "algorithm_source_sha256": _sha256(
                SRC_ROOT / "bmw_inspection/lab/bright_streak_tracked_profile.py"
            ),
            "evaluator_source_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the no-overwrite tracked-profile v3 evaluator parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--accepted-normal-record",
        action="append",
        type=Path,
        default=[],
        help="Repeatable user-confirmed live normal capture directory.",
    )
    parser.add_argument(
        "--roi-xyxy",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=DEFAULT_ROI_XYXY,
        help="Fixed front-left tracked-profile ROI; default remains 81x613.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one immutable tracked-profile v3 evaluation."""
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_bright_streak_tracked_profile(
            args.manifest,
            args.output_dir,
            args.accepted_normal_record,
            tuple(args.roi_xyxy),
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"BMW tracked-profile bright-streak evaluation failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
