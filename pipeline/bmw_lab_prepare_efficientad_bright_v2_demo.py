#!/usr/bin/env python3
"""Prepare the isolated BMW Demo run for candidate EfficientAD and bright v2."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VIEW_ORDER = (
    "front",
    "front_left",
    "front_right",
    "front_secondary",
    "back",
    "back_left",
    "back_right",
    "back_secondary",
)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")


def prepare_efficientad_candidate_run(
    candidate_run: Path,
    baseline_run: Path,
    output_run: Path,
) -> dict[str, Any]:
    """Atomically compose baseline Template/YOLO with candidate EfficientAD."""
    candidate = Path(candidate_run).expanduser().resolve()
    baseline = Path(baseline_run).expanduser().resolve()
    output = Path(output_run).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuse to overwrite existing composite Demo run: {output}")
    for view in VIEW_ORDER:
        _require_file(baseline / "template" / view / "model.json", f"baseline Template {view}")
        _require_file(candidate / "efficientad" / view / "model.ckpt", f"candidate EfficientAD {view}")
    _require_file(baseline / "yolo/train/weights/best.pt", "baseline YOLO")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (staging / "template").symlink_to(baseline / "template", target_is_directory=True)
        (staging / "efficientad").symlink_to(candidate / "efficientad", target_is_directory=True)
        (staging / "yolo").symlink_to(baseline / "yolo", target_is_directory=True)
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "changed_branches": ["efficientad", "bright_streak"],
            "candidate_run": str(candidate),
            "baseline_run": str(baseline),
            "output_run": str(output),
            "links": {
                "template": str((baseline / "template").resolve()),
                "efficientad": str((candidate / "efficientad").resolve()),
                "yolo": str((baseline / "yolo").resolve()),
            },
        }
        (staging / "composition.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-run",
        type=Path,
        default=REPO_ROOT / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_efficientad_v1",
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=REPO_ROOT / "results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1",
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=REPO_ROOT
        / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_efficientad_bright_v2_demo_v1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = prepare_efficientad_candidate_run(args.candidate_run, args.baseline_run, args.output_run)
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"BMW第二版Demo准备失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
