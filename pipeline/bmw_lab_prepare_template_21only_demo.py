#!/usr/bin/env python3
"""Prepare an isolated BMW Demo run that changes only Template models."""

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


def prepare_template_only_run(
    candidate_template_root: Path,
    baseline_run: Path,
    output_run: Path,
) -> dict[str, Any]:
    """Atomically create a composite run with candidate Template only."""
    candidate = Path(candidate_template_root).expanduser().resolve()
    baseline = Path(baseline_run).expanduser().resolve()
    output = Path(output_run).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuse to overwrite existing composite Demo run: {output}")
    for view in VIEW_ORDER:
        _require_file(candidate / view / "model.json", f"candidate Template {view}")
        _require_file(baseline / "efficientad" / view / "model.ckpt", f"baseline EfficientAD {view}")
    _require_file(baseline / "yolo/train/weights/best.pt", "baseline YOLO")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (staging / "template").symlink_to(candidate, target_is_directory=True)
        (staging / "efficientad").symlink_to(baseline / "efficientad", target_is_directory=True)
        (staging / "yolo").symlink_to(baseline / "yolo", target_is_directory=True)
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "changed_branch": "template",
            "candidate_template_root": str(candidate),
            "baseline_run": str(baseline),
            "output_run": str(output),
            "links": {
                "template": str(candidate),
                "efficientad": str((baseline / "efficientad").resolve()),
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
    """Build the no-overwrite composite-run preparation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-template-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_template_only_v1/template"
        ),
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=REPO_ROOT / "results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1",
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=(
            REPO_ROOT
            / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_template_demo_v1"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare the isolated run and print its exact branch identities."""
    args = build_parser().parse_args(argv)
    try:
        report = prepare_template_only_run(
            args.candidate_template_root,
            args.baseline_run,
            args.output_run,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"BMW Template单变量Demo准备失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
