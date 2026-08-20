# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train selected anomaly models on a user-provided dataset.

This is a generic launcher around ``pipeline/3_train_model.py``. It runs
preprocessing once, trains each selected model in its own process, evaluates
successful runs, and writes an aggregate comparison report.

Example:
    .venv/bin/python pipeline/8_train_custom_models.py \
      --data-root /DATA/ljl/c789_100_left_top_parts \
      --output-root results/custom_left_top \
      --views left_top \
      --models patchcore efficient_ad anomaly_dino \
      --gpus 0 \
      --roi full \
      --image-size 392,784 \
      --patchcore-batch-size 4 \
      --patchcore-coreset-ratio 0.1 \
      --efficientad-batch-size 1 \
      --efficientad-epochs 20 \
      --anomaly-dino-batch-size 4 \
      --anomaly-dino-sampling-ratio 0.1
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CHOICES = ("patchcore", "efficient_ad", "anomaly_dino")
VIEW_CHOICES = (
    "left_top",
    "left_bottom",
    "right_top",
    "right_bottom",
    "no_hand_top",
    "no_hand_bottom",
    "right_front",
    "right_front_left",
    "right_front_right",
    "right_front_secondary",
    "right_back",
    "right_back_left",
    "right_back_right",
    "right_back_secondary",
)


@dataclass(frozen=True)
class CommandJob:
    """One subprocess command plus reporting metadata."""

    name: str
    command: list[str]
    log_path: Path
    model_name: str | None = None
    experiment_name: str | None = None


@dataclass(frozen=True)
class CommandResult:
    """Finished subprocess result."""

    name: str
    code: int
    log_path: Path
    model_name: str | None
    experiment_name: str | None
    gpu: str
    elapsed_seconds: float


def _repo_path(path: Path) -> Path:
    """Resolve relative paths against the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def _workflow_script() -> Path:
    """Return the stage-3 training wrapper path."""
    return REPO_ROOT / "pipeline" / "3_train_model.py"


def _quote_command(command: Sequence[str]) -> str:
    """Return a shell-readable command line."""
    return " ".join(shlex.quote(part) for part in command)


def _format_duration(seconds: float) -> str:
    """Return a compact duration string."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _split_gpu_values(values: Sequence[str]) -> list[str]:
    """Normalize --gpus values from comma or space separated input."""
    gpu_ids = []
    for value in values:
        gpu_ids.extend(part.strip() for part in value.split(",") if part.strip())
    if not gpu_ids:
        msg = "At least one GPU id must be provided."
        raise argparse.ArgumentTypeError(msg)
    return gpu_ids


def _parse_ratio(value: str) -> float:
    """Parse a ratio in the open-closed range (0, 1]."""
    try:
        ratio = float(value)
    except ValueError as error:
        msg = "Ratio must be a number."
        raise argparse.ArgumentTypeError(msg) from error
    if ratio <= 0 or ratio > 1:
        msg = "Ratio must satisfy 0 < value <= 1."
        raise argparse.ArgumentTypeError(msg)
    return ratio


def _parse_normal_test_ratio(value: str) -> float:
    """Parse an optional normal holdout ratio in the half-open range [0, 1)."""
    try:
        ratio = float(value)
    except ValueError as error:
        msg = "Normal-test ratio must be a number."
        raise argparse.ArgumentTypeError(msg) from error
    if ratio < 0 or ratio >= 1:
        msg = "Normal-test ratio must satisfy 0 <= value < 1."
        raise argparse.ArgumentTypeError(msg)
    return ratio


def _parse_run_suffix(value: str) -> str:
    """Parse a safe optional suffix for model run names."""
    if not value:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        msg = "Run suffix may contain only letters, numbers, underscore, dash, and dot."
        raise argparse.ArgumentTypeError(msg)
    return value


def _experiment_name(model_name: str, suffix: str) -> str:
    """Return the output/report name for one model run."""
    return f"{model_name}_{suffix}" if suffix else model_name


def _common_workflow_args(args: argparse.Namespace, model_name: str | None = None) -> list[str]:
    """Build workflow arguments shared by preprocess, train, and evaluate."""
    efficientad_batch_size = 1 if model_name == "efficient_ad" else args.efficientad_batch_size
    command = [
        "--data-root",
        str(_repo_path(args.data_root)),
        "--output-root",
        str(_repo_path(args.output_root)),
        "--views",
        *args.views,
        "--roi",
        args.roi,
        "--image-size",
        args.image_size,
        "--accelerator",
        args.accelerator,
        "--devices",
        str(args.devices_per_job),
        "--num-workers",
        str(args.num_workers),
        "--train-sampling-ratio",
        str(args.train_sampling_ratio),
        "--normal-test-ratio",
        str(args.normal_test_ratio),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--deploy-fpr",
        str(args.deploy_fpr),
        "--seed",
        str(args.seed),
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
        "--patchcore-layers",
        *args.patchcore_layers,
        "--efficientad-batch-size",
        str(efficientad_batch_size),
        "--efficientad-epochs",
        str(args.efficientad_epochs),
        "--efficientad-model-size",
        args.efficientad_model_size,
        "--imagenet-dir",
        str(_repo_path(args.imagenet_dir)),
        "--anomaly-dino-batch-size",
        str(args.anomaly_dino_batch_size),
        "--anomaly-dino-encoder",
        args.anomaly_dino_encoder,
        "--anomaly-dino-neighbors",
        str(args.anomaly_dino_neighbors),
        "--anomaly-dino-sampling-ratio",
        str(args.anomaly_dino_sampling_ratio),
    ]
    if args.skip_blue_removal:
        command.append("--skip-blue-removal")
    if args.visualizer_field_size is not None:
        command.extend(["--visualizer-field-size", args.visualizer_field_size])
    if args.anomaly_dino_masking:
        command.append("--anomaly-dino-masking")
    if args.anomaly_dino_coreset_subsampling:
        command.append("--anomaly-dino-coreset-subsampling")
    if args.skip_missing_efficientad_assets:
        command.append("--skip-missing-efficientad-assets")
    return command


def _workflow_command(
    step: str,
    args: argparse.Namespace,
    models: Sequence[str] = (),
    model_run_suffix: str = "",
    reports_dir_name: Path | None = None,
) -> list[str]:
    """Build a pipeline/3_train_model.py command."""
    model_name = models[0] if len(models) == 1 else None
    command = [
        sys.executable,
        str(_workflow_script()),
        step,
        *_common_workflow_args(args, model_name),
    ]
    if models:
        command.extend(["--models", *models])
    if model_run_suffix:
        command.extend(["--model-run-suffix", model_run_suffix])
    if reports_dir_name is not None:
        command.extend(["--reports-dir-name", str(reports_dir_name)])
    return command


def _log_tail(log_path: Path, line_count: int = 30) -> str:
    """Return the last lines of a log file."""
    if not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _run_command(job: CommandJob, gpu_queue: Queue[str], args: argparse.Namespace) -> CommandResult:
    """Run one command on the next available GPU."""
    gpu_id = gpu_queue.get()
    try:
        env = os.environ.copy()
        if args.accelerator == "gpu":
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        if args.cuda_alloc_conf:
            env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

        printable = _quote_command(job.command)
        print(f"[start] {job.name} on GPU {gpu_id}", flush=True)
        print(f"        log: {job.log_path}", flush=True)
        print(f"        cmd: {printable}", flush=True)

        start = time.monotonic()
        if args.dry_run:
            code = 0
        else:
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            with job.log_path.open("w", encoding="utf-8") as log_file:
                prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if args.accelerator == "gpu" else ""
                log_file.write(f"$ {prefix}{printable}\n\n")
                log_file.flush()
                process = subprocess.run(
                    job.command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            code = process.returncode
        elapsed = time.monotonic() - start

        status = "ok" if code == 0 else f"failed:{code}"
        print(f"[done]  {job.name} on GPU {gpu_id}: {status} in {_format_duration(elapsed)}", flush=True)
        if code != 0 and not args.dry_run:
            tail = _log_tail(job.log_path)
            if tail:
                print(f"[tail]  {job.name}\n{tail}", flush=True)

        return CommandResult(
            name=job.name,
            code=code,
            log_path=job.log_path,
            model_name=job.model_name,
            experiment_name=job.experiment_name,
            gpu=gpu_id,
            elapsed_seconds=elapsed,
        )
    finally:
        gpu_queue.put(gpu_id)


def _run_parallel_jobs(jobs: Sequence[CommandJob], args: argparse.Namespace) -> list[CommandResult]:
    """Run jobs with at most one process per selected GPU."""
    if not jobs:
        return []

    selected_gpus = args.gpus[: args.max_parallel]
    gpu_queue: Queue[str] = Queue()
    for gpu_id in selected_gpus:
        gpu_queue.put(gpu_id)

    results = []
    with ThreadPoolExecutor(max_workers=len(selected_gpus)) as executor:
        futures = [executor.submit(_run_command, job, gpu_queue, args) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_serial_job(job: CommandJob, args: argparse.Namespace, gpu_id: str) -> CommandResult:
    """Run a single command on a chosen GPU."""
    gpu_queue: Queue[str] = Queue()
    gpu_queue.put(gpu_id)
    return _run_command(job, gpu_queue, args)


def _output_root(args: argparse.Namespace) -> Path:
    """Return the resolved output root."""
    return _repo_path(args.output_root)


def _preprocess(args: argparse.Namespace) -> None:
    """Run preprocessing once for the selected dataset and views."""
    output_root = _output_root(args)
    job = CommandJob(
        name="preprocess",
        command=_workflow_command("preprocess", args),
        log_path=output_root / "logs" / "preprocess.log",
    )
    result = _run_serial_job(job, args, args.gpus[0])
    if result.code != 0:
        raise SystemExit(result.code)


def _train_jobs(args: argparse.Namespace) -> list[CommandJob]:
    """Create selected model training jobs."""
    output_root = _output_root(args)
    jobs = []
    for model_name in args.models:
        experiment_name = _experiment_name(model_name, args.model_run_suffix)
        jobs.append(
            CommandJob(
                name=f"train:{experiment_name}",
                command=_workflow_command(
                    "train",
                    args,
                    models=[model_name],
                    model_run_suffix=args.model_run_suffix,
                ),
                log_path=output_root / "logs" / f"train_{experiment_name}.log",
                model_name=model_name,
                experiment_name=experiment_name,
            ),
        )
    return jobs


def _successful_models(
    train_results: Sequence[CommandResult],
    evaluate_only: bool,
    args: argparse.Namespace,
) -> list[str]:
    """Return models that should be evaluated."""
    if evaluate_only:
        return list(args.models)
    return [result.model_name for result in train_results if result.code == 0 and result.model_name is not None]


def _evaluate_jobs(model_names: Sequence[str], args: argparse.Namespace) -> list[CommandJob]:
    """Create selected model evaluation jobs."""
    output_root = _output_root(args)
    jobs = []
    for model_name in model_names:
        experiment_name = _experiment_name(model_name, args.model_run_suffix)
        jobs.append(
            CommandJob(
                name=f"evaluate:{experiment_name}",
                command=_workflow_command(
                    "evaluate",
                    args,
                    models=[model_name],
                    model_run_suffix=args.model_run_suffix,
                    reports_dir_name=Path("reports") / experiment_name,
                ),
                log_path=output_root / "logs" / f"evaluate_{experiment_name}.log",
                model_name=model_name,
                experiment_name=experiment_name,
            ),
        )
    return jobs


def _summary_sort_key(row: dict[str, str]) -> tuple[float, float, float]:
    """Sort rows by the most useful metrics first."""
    def value(name: str) -> float:
        try:
            return float(row.get(name, "0") or 0)
        except ValueError:
            return 0.0

    return (value("sample_f1"), value("image_f1"), value("image_recall"))


def _float_text(value: str | float | None) -> str:
    """Return a compact numeric table cell."""
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return str(value)


def _read_summary(args: argparse.Namespace, model_name: str) -> list[dict[str, str]]:
    """Read one per-model summary and add experiment metadata."""
    experiment_name = _experiment_name(model_name, args.model_run_suffix)
    summary_path = _output_root(args) / "reports" / experiment_name / "summary.csv"
    if not summary_path.is_file():
        msg = f"Missing summary for {experiment_name}: {summary_path}"
        raise FileNotFoundError(msg)

    with summary_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["experiment"] = experiment_name
    return rows


def _write_aggregate_report(args: argparse.Namespace, model_names: Sequence[str]) -> None:
    """Write aggregate CSV and Markdown reports."""
    if args.dry_run or not model_names:
        return

    rows = []
    for model_name in model_names:
        rows.extend(_read_summary(args, model_name))
    rows = sorted(rows, key=_summary_sort_key, reverse=True)

    reports_dir = _output_root(args) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model",
        "view",
        "deploy_threshold",
        "deploy_target_fpr",
        "image_fpr",
        "image_accuracy",
        "image_recall",
        "image_f1",
        "sample_fpr",
        "sample_accuracy",
        "sample_recall",
        "sample_f1",
    ]
    extra_fields = sorted({key for row in rows for key in row} - set(fieldnames))
    with (reports_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*fieldnames, *extra_fields])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Model Comparison",
        "",
        "| experiment | model | view | threshold | target_fpr | image_fpr | image_acc | image_recall | "
        "image_f1 | sample_fpr | sample_acc | sample_recall | sample_f1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {model} | {view} | {threshold} | {target_fpr} | {image_fpr} | {image_acc} | "
            "{image_recall} | {image_f1} | {sample_fpr} | {sample_acc} | {sample_recall} | {sample_f1} |".format(
                experiment=row.get("experiment", ""),
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
    (reports_dir / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Raw or single-part dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for this training run.")
    parser.add_argument("--views", nargs="+", choices=VIEW_CHOICES, required=True, help="Views to train.")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES), help="Models.")
    parser.add_argument("--model-run-suffix", type=_parse_run_suffix, default="", help="Optional run suffix.")

    parser.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids, space or comma separated.")
    parser.add_argument("--max-parallel", type=int, help="Maximum concurrent model jobs.")
    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="gpu", help="Lightning accelerator.")
    parser.add_argument("--devices-per-job", type=int, default=1, help="Lightning devices inside each process.")
    parser.add_argument("--cuda-alloc-conf", default="expandable_segments:True", help="PYTORCH_CUDA_ALLOC_CONF value.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per process.")

    parser.add_argument("--roi", default="full", help="Workflow ROI. Use full for already-cropped part images.")
    parser.add_argument("--image-size", default="392,784", help="Model input size as H,W.")
    parser.add_argument("--visualizer-field-size", help="Visualization field size as W,H.")
    parser.add_argument(
        "--blue-removal",
        action="store_false",
        dest="skip_blue_removal",
        help="Enable HSV blue-mark removal. Default skips it for pre-cropped parts.",
    )
    parser.set_defaults(skip_blue_removal=True)
    parser.add_argument("--train-sampling-ratio", type=_parse_ratio, default=1.0, help="Normal-train sampling ratio.")
    parser.add_argument(
        "--normal-test-ratio",
        type=_parse_normal_test_ratio,
        default=0.2,
        help="Normal capture-group fraction held out when normal_test is absent.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--deploy-fpr", type=float, default=0.05, help="Allowed normal_test false-positive rate.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset split and sampling seed.")

    parser.add_argument("--patchcore-batch-size", type=int, default=1, help="PatchCore training batch size.")
    parser.add_argument("--patchcore-backbone", default="wide_resnet50_2", help="PatchCore timm backbone.")
    parser.add_argument("--patchcore-layers", nargs="+", default=["layer2"], help="PatchCore feature layers.")
    parser.add_argument("--patchcore-coreset-ratio", type=_parse_ratio, default=0.1, help="PatchCore coreset ratio.")
    parser.add_argument("--patchcore-num-neighbors", type=int, default=1, help="PatchCore nearest-neighbor count.")
    parser.add_argument(
        "--patchcore-precision",
        choices=("float32", "float16"),
        default="float16",
        help="PatchCore feature precision.",
    )

    parser.add_argument("--efficientad-batch-size", type=int, default=1, help="EfficientAD train batch size.")
    parser.add_argument("--efficientad-epochs", type=int, default=20, help="EfficientAD max epochs.")
    parser.add_argument(
        "--efficientad-model-size",
        choices=("small", "medium"),
        default="medium",
        help="EfficientAD model size.",
    )
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        default=Path(".cache/anomalib/imagenette"),
        help="EfficientAD ImageNette directory.",
    )
    parser.add_argument(
        "--skip-missing-efficientad-assets",
        action="store_true",
        help="Skip EfficientAD if teacher weights or ImageNette data are missing.",
    )

    parser.add_argument("--anomaly-dino-batch-size", type=int, default=1, help="AnomalyDINO training batch size.")
    parser.add_argument("--anomaly-dino-encoder", default="dinov2_vit_small_14", help="AnomalyDINO encoder.")
    parser.add_argument("--anomaly-dino-neighbors", type=int, default=1, help="AnomalyDINO nearest-neighbor count.")
    parser.add_argument("--anomaly-dino-masking", action="store_true", help="Enable AnomalyDINO foreground masking.")
    parser.add_argument(
        "--no-anomaly-dino-coreset-subsampling",
        action="store_false",
        dest="anomaly_dino_coreset_subsampling",
        help="Disable AnomalyDINO coreset subsampling.",
    )
    parser.set_defaults(anomaly_dino_coreset_subsampling=True)
    parser.add_argument(
        "--anomaly-dino-sampling-ratio",
        type=_parse_ratio,
        default=0.1,
        help="AnomalyDINO coreset sampling ratio.",
    )

    parser.add_argument("--skip-preprocess", action="store_true", help="Reuse existing preprocessed images.")
    parser.add_argument("--train-only", action="store_true", help="Stop after training.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip preprocess/train and evaluate checkpoints.")
    parser.add_argument("--keep-going", action="store_true", help="Evaluate successful models if a model fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def _normalize_args(args: argparse.Namespace) -> None:
    """Normalize and validate parsed arguments in place."""
    args.gpus = _split_gpu_values(args.gpus)
    if args.max_parallel is None:
        args.max_parallel = len(args.gpus) if args.accelerator == "gpu" else 1
    args.max_parallel = min(args.max_parallel, len(args.gpus), len(args.models))
    if args.max_parallel <= 0:
        raise SystemExit("--max-parallel must be positive.")
    if args.evaluate_only and args.train_only:
        raise SystemExit("--evaluate-only and --train-only cannot be used together.")
    if "efficient_ad" in args.models and args.efficientad_batch_size != 1:
        print("EfficientAD train batch size is forced to 1 by the model.", flush=True)


def main() -> None:
    """Run the custom model training workflow."""
    args = build_parser().parse_args()
    _normalize_args(args)

    print("Custom anomaly model training", flush=True)
    print(f"Data root: {_repo_path(args.data_root)}", flush=True)
    print(f"Output root: {_repo_path(args.output_root)}", flush=True)
    print(f"Views: {', '.join(args.views)}", flush=True)
    print(f"Models: {', '.join(args.models)}", flush=True)
    print(f"GPUs: {', '.join(args.gpus[: args.max_parallel])}", flush=True)

    if not args.evaluate_only and not args.skip_preprocess:
        _preprocess(args)

    train_results: list[CommandResult] = []
    if not args.evaluate_only:
        train_results = _run_parallel_jobs(_train_jobs(args), args)
        failures = [result for result in train_results if result.code != 0]
        if failures:
            print("\nTraining failures:", flush=True)
            for result in failures:
                print(f"  - {result.name}: {result.log_path}", flush=True)
            if not args.keep_going:
                raise SystemExit(1)

    if args.train_only:
        print("\nTraining complete. Evaluation was skipped by --train-only.", flush=True)
        return

    model_names = _successful_models(train_results, args.evaluate_only, args)
    if not model_names:
        print("\nNo models available for evaluation.", flush=True)
        raise SystemExit(1)

    eval_results = _run_parallel_jobs(_evaluate_jobs(model_names, args), args)
    eval_failures = [result for result in eval_results if result.code != 0]
    if eval_failures:
        print("\nEvaluation failures:", flush=True)
        for result in eval_failures:
            print(f"  - {result.name}: {result.log_path}", flush=True)
        raise SystemExit(1)

    _write_aggregate_report(args, model_names)
    reports_dir = _output_root(args) / "reports"
    print("\nReports:", flush=True)
    print(f"  - {reports_dir / 'model_comparison.md'}", flush=True)
    print(f"  - {reports_dir / 'model_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
