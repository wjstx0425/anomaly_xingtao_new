#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train the isolated BMW 21-point Template candidate with morning thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402
from bmw_inspection.lab.template import (  # noqa: E402
    train_template_group_fixed_threshold,
    validate_template_source_sessions,
)


ALLOWED_SESSION_IDS = (
    "20260810_210030_527506",
    "20260810_213407_600611",
    "20260810_213852_756399",
    "20260810_214505_826120",
    "20260810_214747_085800",
)
_MANIFEST_FIELDS = ("sample_id", "part_id", "view_id", "image_path", "split", "label")
_SPLITS = frozenset({"train", "calibration", "final_test"})
_LABELS = frozenset({"normal", "defect"})


class CandidateView(str, Enum):
    """Eight fixed views used only by the offline BMW candidate."""

    FRONT = "front"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    FRONT_SECONDARY = "front_secondary"
    BACK = "back"
    BACK_LEFT = "back_left"
    BACK_RIGHT = "back_right"
    BACK_SECONDARY = "back_secondary"


@dataclass(frozen=True, slots=True)
class CandidateTemplateSample:
    """Manifest row compatible with the existing one-view Template trainer."""

    sample_id: str
    part_id: str
    view_id: CandidateView
    image_path: Path
    split: str
    label: str


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_manifest(path: Path) -> tuple[CandidateTemplateSample, ...]:
    manifest = Path(path).expanduser().resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _MANIFEST_FIELDS:
            raise ValueError("Template candidate manifest header differs from the frozen schema")
        rows: list[CandidateTemplateSample] = []
        identities: set[tuple[str, CandidateView]] = set()
        splits_by_part: dict[str, str] = {}
        for number, raw in enumerate(reader, start=2):
            try:
                view_id = CandidateView(raw["view_id"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"unknown Template view at row {number}: {raw.get('view_id')!r}") from error
            image_path = Path(raw["image_path"])
            image_path = (image_path if image_path.is_absolute() else manifest.parent / image_path).resolve()
            if not all(isinstance(raw.get(field), str) and raw[field] for field in _MANIFEST_FIELDS):
                raise ValueError(f"Template candidate row {number} contains an empty field")
            if not image_path.is_file():
                raise ValueError(f"Template candidate image does not exist: {image_path}")
            if raw["split"] not in _SPLITS or raw["label"] not in _LABELS:
                raise ValueError(f"Template candidate row {number} has an unknown split or label")
            identity = (raw["sample_id"], view_id)
            if identity in identities:
                raise ValueError(f"duplicate Template candidate identity: {identity[0]}/{view_id.value}")
            identities.add(identity)
            previous_split = splits_by_part.setdefault(raw["part_id"], raw["split"])
            if previous_split != raw["split"]:
                raise ValueError(f"part_id {raw['part_id']} crosses splits: {previous_split}, {raw['split']}")
            rows.append(
                CandidateTemplateSample(
                    sample_id=raw["sample_id"],
                    part_id=raw["part_id"],
                    view_id=view_id,
                    image_path=image_path,
                    split=raw["split"],
                    label=raw["label"],
                )
            )
    if not rows or {row.view_id.value for row in rows} != set(VIEW_ORDER):
        raise ValueError("Template candidate manifest must contain exactly the canonical eight views")
    return tuple(rows)


def _baseline_thresholds(root: Path) -> dict[str, float]:
    baseline_root = Path(root).expanduser().resolve()
    thresholds: dict[str, float] = {}
    for view in VIEW_ORDER:
        model_path = baseline_root / view / "model.json"
        try:
            threshold = json.loads(model_path.read_text(encoding="utf-8"))["threshold"]
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"baseline Template threshold is unavailable for {view}: {model_path}") from error
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or threshold < 0
        ):
            raise ValueError(f"baseline Template threshold is invalid for {view}: {model_path}")
        thresholds[view] = float(threshold)
    return thresholds


def _normal_part_pass_rate(rows_by_view: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    by_part: dict[str, dict[str, Mapping[str, Any]]] = {}
    for view, rows in rows_by_view.items():
        for row in rows:
            if row.get("label") != "normal":
                raise ValueError("candidate report may contain only normal scoring rows")
            part_rows = by_part.setdefault(str(row["part_id"]), {})
            if view in part_rows:
                raise ValueError(f"duplicate normal Template score for {row['part_id']}/{view}")
            part_rows[view] = row
    expected_views = set(VIEW_ORDER)
    if any(set(part_rows) != expected_views for part_rows in by_part.values()):
        raise ValueError("normal Template part report requires every canonical view")
    passed = sum(all(row["prediction"] == "normal" for row in part_rows.values()) for part_rows in by_part.values())
    count = len(by_part)
    return {"part_count": count, "passed_part_count": passed, "pass_rate": passed / count if count else 0.0}


def run_fixed_threshold_candidate(
    *,
    manifest: Path,
    baseline_template_root: Path,
    output_root: Path,
    allowed_session_ids: Sequence[str] = ALLOWED_SESSION_IDS,
) -> dict[str, Any]:
    """Publish one no-overwrite 21-point Template candidate and normal-only report."""
    rows = _load_manifest(manifest)
    source_sessions = validate_template_source_sessions(rows, allowed_session_ids)  # type: ignore[arg-type]
    thresholds = _baseline_thresholds(baseline_template_root)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refuse to overwrite existing Template candidate directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        per_view: dict[str, dict[str, Any]] = {}
        calibration_rows: dict[str, Sequence[Mapping[str, Any]]] = {}
        final_rows: dict[str, Sequence[Mapping[str, Any]]] = {}
        for view in VIEW_ORDER:
            group = [row for row in rows if row.view_id.value == view]
            first = group[0]
            image = cv2.imread(str(first.image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"cannot decode Template candidate input: {first.image_path}")
            height, width = image.shape[:2]
            model_path = train_template_group_fixed_threshold(  # type: ignore[arg-type]
                group,
                (0, 0, width, height),
                staging / "template" / view,
                threshold=thresholds[view],
                target_size=(512, 512),
                max_shift=12,
                template_count=5,
            )
            metrics = json.loads((model_path.parent / "metrics.json").read_text(encoding="utf-8"))
            per_view[view] = {
                "threshold": metrics["threshold"],
                "calibration": metrics["calibration"],
                "final_test": metrics["final_test"],
            }
            calibration_rows[view] = metrics["calibration_rows"]
            final_rows[view] = metrics["final_test_rows"]
        report = {
            "schema_version": 1,
            "candidate": "bmw_21point_template_fixed_thresholds",
            "source_sessions": list(source_sessions),
            "parameters": {"target_size": [512, 512], "max_shift": 12, "template_count": 5},
            "per_view": per_view,
            "normal_part_pass_rates": {
                "calibration": _normal_part_pass_rate(calibration_rows),
                "final_test": _normal_part_pass_rate(final_rows),
            },
        }
        _write_json(staging / "template_report.json", report)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the strict, offline-only 21-point Template candidate CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "dataset/bmw_lab_training/bmw_right_batch_20260810_21_roi_v1/template/trainer_manifest.csv",
    )
    parser.add_argument(
        "--baseline-template-root",
        type=Path,
        default=REPO_ROOT / "results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1/template",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_template_only_v1",
    )
    parser.add_argument("--allowed-session-id", action="append", help="Optional strict source-session allowlist.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train and report the candidate without changing any Demo configuration."""
    args = build_parser().parse_args(argv)
    try:
        report = run_fixed_threshold_candidate(
            manifest=args.manifest,
            baseline_template_root=args.baseline_template_root,
            output_root=args.output_root,
            allowed_session_ids=tuple(args.allowed_session_id or ALLOWED_SESSION_IDS),
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"BMW fixed-threshold Template candidate failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
