#!/usr/bin/env python3
"""Prepare the immutable third BMW Demo composition and threshold asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
THRESHOLD_MARGIN = 0.05
THRESHOLD_ARTIFACT_NAME = "efficientad_thresholds_deployment_v3.json"


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deployment_threshold_payload(base_threshold_artifact: Path) -> dict[str, Any]:
    """Copy the v2 evidence and raise each deployment threshold by the fixed margin."""
    try:
        payload = json.loads(base_threshold_artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load base EfficientAD threshold artifact: {base_threshold_artifact}") from error
    if not isinstance(payload, dict):
        raise ValueError("base EfficientAD threshold artifact must be a JSON object")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(VIEW_ORDER):
        raise ValueError("base EfficientAD threshold artifact must cover the canonical eight views")
    base_thresholds: dict[str, float] = {}
    for view in VIEW_ORDER:
        value = thresholds[view]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"base EfficientAD threshold for {view} must be finite")
        base_thresholds[view] = float(value)
    checkpoint_sha256 = payload.get("checkpoint_sha256_by_view")
    if not isinstance(checkpoint_sha256, dict) or set(checkpoint_sha256) != set(VIEW_ORDER):
        raise ValueError("base EfficientAD threshold artifact must keep checkpoint SHA256 for every view")
    for field in ("source_csv", "source_csv_sha256"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"base EfficientAD threshold artifact must keep {field}")

    deployment = dict(payload)
    deployment["base_thresholds"] = base_thresholds
    deployment["threshold_margin"] = THRESHOLD_MARGIN
    deployment["thresholds"] = {view: base_thresholds[view] + THRESHOLD_MARGIN for view in VIEW_ORDER}
    deployment["source_threshold_artifact"] = str(base_threshold_artifact)
    deployment["source_threshold_artifact_sha256"] = _sha256(base_threshold_artifact)
    return deployment


def prepare_v3_ng_evidence_run(
    template_root: Path,
    efficientad_candidate_run: Path,
    yolo_baseline_run: Path,
    base_threshold_artifact: Path,
    output_run: Path,
) -> dict[str, Any]:
    """Atomically compose the requested v3 model branches without overwriting an older run."""
    template = Path(template_root).expanduser().resolve()
    efficientad_candidate = Path(efficientad_candidate_run).expanduser().resolve()
    yolo_baseline = Path(yolo_baseline_run).expanduser().resolve()
    base_thresholds = Path(base_threshold_artifact).expanduser().resolve()
    output = Path(output_run).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuse to overwrite existing composite Demo run: {output}")
    for view in VIEW_ORDER:
        _require_file(template / view / "model.json", f"21:00 Template {view}")
        _require_file(efficientad_candidate / "efficientad" / view / "model.ckpt", f"v2 EfficientAD {view}")
    _require_file(yolo_baseline / "yolo/train/weights/best.pt", "morning baseline YOLO")
    _require_file(base_thresholds, "v2 EfficientAD threshold artifact")
    threshold_payload = _deployment_threshold_payload(base_thresholds)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (staging / "template").symlink_to(template, target_is_directory=True)
        (staging / "efficientad").symlink_to(efficientad_candidate / "efficientad", target_is_directory=True)
        (staging / "yolo").symlink_to(yolo_baseline / "yolo", target_is_directory=True)
        threshold_artifact = staging / THRESHOLD_ARTIFACT_NAME
        threshold_artifact.write_text(
            json.dumps(threshold_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "template_source": "21:00_template_only",
            "efficientad_source": "v2_candidate",
            "yolo_source": "morning_baseline",
            "template_root": str(template),
            "efficientad_candidate_run": str(efficientad_candidate),
            "yolo_baseline_run": str(yolo_baseline),
            "base_threshold_artifact": str(base_thresholds),
            "base_threshold_artifact_sha256": _sha256(base_thresholds),
            "threshold_artifact": str(output / THRESHOLD_ARTIFACT_NAME),
            "threshold_artifact_sha256": _sha256(threshold_artifact),
            "links": {
                "template": str(template),
                "efficientad": str((efficientad_candidate / "efficientad").resolve()),
                "yolo": str((yolo_baseline / "yolo").resolve()),
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
        "--template-root",
        type=Path,
        default=REPO_ROOT / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_template_only_v1/template",
    )
    parser.add_argument(
        "--efficientad-candidate-run",
        type=Path,
        default=REPO_ROOT / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_efficientad_v1",
    )
    parser.add_argument(
        "--yolo-baseline-run",
        type=Path,
        default=REPO_ROOT / "results/bmw_lab_one_click/bmw_right_multisource_left_yolo_v1",
    )
    parser.add_argument(
        "--base-threshold-artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_efficientad_v1"
            / "efficientad/score_analysis/part_thresholds.json"
        ),
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=REPO_ROOT / "results/bmw_lab_one_click/bmw_right_batch_20260810_21_v3_ng_evidence_demo_v1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = prepare_v3_ng_evidence_run(
            args.template_root,
            args.efficientad_candidate_run,
            args.yolo_baseline_run,
            args.base_threshold_artifact,
            args.output_run,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"BMW第三版NG证据Demo准备失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
