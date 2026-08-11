#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train one immutable global one-class YOLO version for all BMW views."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bmw_inspection.lab.config import load_experiment_config  # noqa: E402
from bmw_inspection.lab.contracts import ViewId  # noqa: E402
from bmw_inspection.lab.yolo import YoloCalibrationSample, train_versioned_yolo  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1")
    return parsed


def _load_calibration_samples(path: Path) -> tuple[YoloCalibrationSample, ...]:
    manifest = Path(path).expanduser().resolve()
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "view_id", "split", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("YOLO calibration manifest is missing required fields")
        samples: list[YoloCalibrationSample] = []
        for row_number, row in enumerate(reader, start=2):
            if row["split"] != "calibration":
                continue
            # Calibration must follow the runtime contract: full frame -> configured ROI once.
            # The exported image is already cropped and would be cropped a second time here.
            raw_path = row.get("image_path") or row.get("exported_image_path")
            if not raw_path:
                raise ValueError(f"row {row_number} lacks exported_image_path/image_path")
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            samples.append(
                YoloCalibrationSample(
                    sample_id=row["sample_id"],
                    view_id=ViewId(row["view_id"]),
                    image_path=image_path,
                    split=row["split"],
                    label=row["label"],
                )
            )
    if not samples:
        raise ValueError("YOLO calibration manifest has no calibration rows")
    return tuple(samples)


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit versioned training interface."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Generated one-class YOLO data.yaml.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/bmw/experiments/bmw_lab_v1.json",
        help="Profile supplying six part ROIs and the low candidate floor.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--version", required=True, help="New immutable version directory name.")
    parser.add_argument("--candidate-conf", type=_probability, help="Override the profile candidate floor.")
    parser.add_argument("--imgsz", type=_positive_int, default=1280)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train without modifying or activating the experiment profile."""
    args = build_parser().parse_args(argv)
    try:
        config = load_experiment_config(args.config)
        samples = _load_calibration_samples(args.calibration_manifest)
        run = train_versioned_yolo(
            data_yaml=args.data,
            base_checkpoint=args.base_checkpoint,
            output_root=args.output_root,
            version=args.version,
            calibration_samples=samples,
            part_rois=config.part_rois,
            candidate_conf=(config.yolo.candidate_conf if args.candidate_conf is None else args.candidate_conf),
            imgsz=args.imgsz,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW YOLO training failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "TRAINED_NOT_ACTIVATED",
                "output_dir": str(run.output_dir),
                "best_checkpoint": str(run.best_checkpoint),
                "final_threshold": run.final_threshold,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
