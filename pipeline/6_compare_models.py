# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run PatchCore, EfficientAD, and AnomalyDINO for side-by-side comparison."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CHOICES = ("patchcore", "efficient_ad", "anomaly_dino")
EFFICIENTAD_MIN_IMAGE_SIDE = 256
IMAGE_SIZE_MULTIPLE = 32


@dataclass(frozen=True)
class WorkflowCommand:
    """One comparison workflow command."""

    name: str
    args: list[str]


def _extend_option(command: list[str], flag: str, values: list[Any]) -> None:
    """Append an option only when values are present."""
    if values:
        command.extend([flag, *(str(value) for value in values)])


def _parse_image_size(value: str) -> tuple[int, int]:
    """Parse an image size in height,width format."""
    try:
        height_text, width_text = value.split(",", maxsplit=1)
        height = int(height_text)
        width = int(width_text)
    except ValueError as error:
        msg = "Image size must be provided as height,width."
        raise argparse.ArgumentTypeError(msg) from error
    if height <= 0 or width <= 0:
        msg = "Image size values must be positive."
        raise argparse.ArgumentTypeError(msg)
    return height, width


def _ceil_to_multiple(value: float, multiple: int) -> int:
    """Round a positive value up to a multiple."""
    return int(math.ceil(value / multiple) * multiple)


def _format_image_size(image_size: tuple[int, int]) -> str:
    """Format an image size as height,width."""
    return f"{image_size[0]},{image_size[1]}"


def effective_image_size(args: argparse.Namespace) -> str:
    """Return a model-safe image size for the selected comparison models."""
    height, width = _parse_image_size(args.image_size)
    if "efficient_ad" not in args.models:
        return args.image_size
    if height >= EFFICIENTAD_MIN_IMAGE_SIDE and width >= EFFICIENTAD_MIN_IMAGE_SIDE:
        return args.image_size

    scale = max(EFFICIENTAD_MIN_IMAGE_SIDE / height, EFFICIENTAD_MIN_IMAGE_SIDE / width)
    safe_size = (
        _ceil_to_multiple(height * scale, IMAGE_SIZE_MULTIPLE),
        _ceil_to_multiple(width * scale, IMAGE_SIZE_MULTIPLE),
    )
    return _format_image_size(safe_size)


def build_workflow_args(command: str, args: argparse.Namespace) -> list[str]:
    """Build arguments for pipeline/3_train_model.py."""
    workflow_args = [
        command,
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(args.output_root),
        "--models",
        *args.models,
        "--roi",
        args.roi,
        "--image-size",
        effective_image_size(args),
        "--accelerator",
        args.accelerator,
        "--devices",
        str(args.devices),
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--deploy-fpr",
        str(args.deploy_fpr),
        "--patchcore-batch-size",
        str(args.patchcore_batch_size),
        "--patchcore-backbone",
        args.patchcore_backbone,
        "--patchcore-coreset-ratio",
        str(args.patchcore_coreset_ratio),
        "--patchcore-num-neighbors",
        str(args.patchcore_num_neighbors),
        "--patchcore-precision",
        args.patchcore_precision,
        "--efficientad-batch-size",
        str(args.efficientad_batch_size),
        "--efficientad-epochs",
        str(args.efficientad_epochs),
        "--anomaly-dino-batch-size",
        str(args.anomaly_dino_batch_size),
        "--anomaly-dino-encoder",
        args.anomaly_dino_encoder,
        "--anomaly-dino-neighbors",
        str(args.anomaly_dino_neighbors),
        "--anomaly-dino-sampling-ratio",
        str(args.anomaly_dino_sampling_ratio),
    ]
    _extend_option(workflow_args, "--views", args.views)
    _extend_option(workflow_args, "--patchcore-layers", args.patchcore_layers)

    if args.visualizer_field_size is not None:
        workflow_args.extend(["--visualizer-field-size", args.visualizer_field_size])
    if args.skip_blue_removal:
        workflow_args.append("--skip-blue-removal")
    if args.anomaly_dino_masking:
        workflow_args.append("--anomaly-dino-masking")
    if args.anomaly_dino_coreset_subsampling:
        workflow_args.append("--anomaly-dino-coreset-subsampling")
    if args.skip_missing_efficientad_assets:
        workflow_args.append("--skip-missing-efficientad-assets")
    return workflow_args


def build_commands(args: argparse.Namespace) -> list[WorkflowCommand]:
    """Build comparison workflow commands."""
    if args.evaluate_only:
        return [WorkflowCommand("evaluate", build_workflow_args("evaluate", args))]
    if args.skip_preprocess:
        return [
            WorkflowCommand("train", build_workflow_args("train", args)),
            WorkflowCommand("evaluate", build_workflow_args("evaluate", args)),
        ]
    return [WorkflowCommand("all", build_workflow_args("all", args))]


def run_command(command: WorkflowCommand, dry_run: bool) -> int:
    """Run one comparison command."""
    script = REPO_ROOT / "pipeline" / "3_train_model.py"
    process_args = [sys.executable, str(script), *command.args]
    print()
    print(f"=== compare:{command.name} ===")
    print(" ".join(shlex.quote(part) for part in process_args))
    if dry_run:
        return 0
    return subprocess.run(process_args, cwd=REPO_ROOT, check=False).returncode


def _float_text(value: str | None) -> str:
    """Return a compact numeric table cell."""
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return str(value)


def _sort_key(row: dict[str, str]) -> tuple[float, float, float]:
    """Sort rows by the most useful comparison metrics first."""
    def value(name: str) -> float:
        try:
            return float(row.get(name, "0") or 0)
        except ValueError:
            return 0.0

    return (value("sample_f1"), value("image_f1"), value("image_recall"))


def write_comparison_report(output_root: Path) -> Path:
    """Write a compact model-comparison Markdown table from summary.csv."""
    summary_path = output_root / "reports" / "summary.csv"
    if not summary_path.is_file():
        msg = f"Missing summary file: {summary_path}"
        raise FileNotFoundError(msg)

    with summary_path.open(newline="", encoding="utf-8") as file:
        rows = sorted(csv.DictReader(file), key=_sort_key, reverse=True)

    report_path = output_root / "reports" / "model_comparison.md"
    lines = [
        "# Model Comparison",
        "",
        "| model | view | threshold | target_fpr | image_fpr | image_acc | image_recall | image_f1 | "
        "sample_fpr | sample_acc | sample_recall | sample_f1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {view} | {threshold} | {target_fpr} | {image_fpr} | {image_acc} | "
            "{image_recall} | {image_f1} | {sample_fpr} | {sample_acc} | {sample_recall} | {sample_f1} |".format(
                model=row.get("model", ""),
                view=row.get("view", ""),
                threshold=_float_text(row.get("deploy_threshold")),
                target_fpr=_float_text(row.get("deploy_target_fpr")),
                image_fpr=_float_text(row.get("image_fpr")),
                image_acc=_float_text(row.get("image_accuracy")),
                image_recall=_float_text(row.get("image_recall")),
                image_f1=_float_text(row.get("image_f1")),
                sample_fpr=_float_text(row.get("sample_fpr")),
                sample_acc=_float_text(row.get("sample_accuracy")),
                sample_recall=_float_text(row.get("sample_recall")),
                sample_f1=_float_text(row.get("sample_f1")),
            ),
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Single-part dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Workflow output root for all models.")
    parser.add_argument("--views", nargs="+", required=True, help="Views to compare, e.g. no_hand_top.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
        help="Models to run.",
    )
    parser.add_argument("--roi", default="full", help="Workflow ROI. Use full for already-cropped single-part images.")
    parser.add_argument("--image-size", default="392,784", help="Model input size as H,W.")
    parser.add_argument("--visualizer-field-size", help="Visualization field size as W,H.")
    parser.add_argument(
        "--skip-blue-removal",
        action="store_true",
        help="Skip HSV blue-mark removal during preprocessing.",
    )
    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="gpu", help="Lightning accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of accelerator devices.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--deploy-fpr", type=float, default=0.05, help="Allowed normal_test false-positive rate.")

    parser.add_argument("--patchcore-batch-size", type=int, default=1, help="PatchCore training batch size.")
    parser.add_argument("--patchcore-backbone", default="wide_resnet50_2", help="PatchCore timm backbone.")
    parser.add_argument("--patchcore-layers", nargs="+", default=["layer2"], help="PatchCore feature layers.")
    parser.add_argument("--patchcore-coreset-ratio", type=float, default=0.1, help="PatchCore coreset ratio.")
    parser.add_argument("--patchcore-num-neighbors", type=int, default=1, help="PatchCore nearest-neighbor count.")
    parser.add_argument(
        "--patchcore-precision",
        choices=("float32", "float16"),
        default="float16",
        help="PatchCore precision.",
    )

    parser.add_argument("--efficientad-batch-size", type=int, default=1, help="EfficientAD training batch size.")
    parser.add_argument("--efficientad-epochs", type=int, default=20, help="EfficientAD max epochs.")
    parser.add_argument(
        "--skip-missing-efficientad-assets",
        action="store_true",
        default=True,
        help="Skip EfficientAD if teacher weights or ImageNette data are unavailable.",
    )
    parser.add_argument(
        "--require-efficientad-assets",
        action="store_false",
        dest="skip_missing_efficientad_assets",
        help="Fail instead of skipping when EfficientAD assets are missing.",
    )

    parser.add_argument("--anomaly-dino-batch-size", type=int, default=1, help="AnomalyDINO training batch size.")
    parser.add_argument("--anomaly-dino-encoder", default="dinov2_vit_small_14", help="AnomalyDINO DINOv2 encoder.")
    parser.add_argument("--anomaly-dino-neighbors", type=int, default=1, help="AnomalyDINO nearest-neighbor count.")
    parser.add_argument("--anomaly-dino-masking", action="store_true", help="Enable AnomalyDINO masking.")
    parser.add_argument(
        "--no-anomaly-dino-coreset-subsampling",
        action="store_false",
        dest="anomaly_dino_coreset_subsampling",
        help="Disable AnomalyDINO coreset subsampling.",
    )
    parser.set_defaults(anomaly_dino_coreset_subsampling=True)
    parser.add_argument("--anomaly-dino-sampling-ratio", type=float, default=0.03, help="AnomalyDINO coreset ratio.")

    parser.add_argument("--skip-preprocess", action="store_true", help="Reuse existing preprocessed images.")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only regenerate reports from existing checkpoints.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def main() -> None:
    """Run the model comparison workflow."""
    args = build_parser().parse_args()
    if args.evaluate_only and args.skip_preprocess:
        raise SystemExit("--evaluate-only already skips preprocessing; do not combine it with --skip-preprocess.")

    for command in build_commands(args):
        code = run_command(command, args.dry_run)
        if code:
            raise SystemExit(code)

    if not args.dry_run:
        report_path = write_comparison_report(args.output_root.resolve())
        print()
        print(f"comparison_report: {report_path}")


if __name__ == "__main__":
    main()
