# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the C789-100 top/bottom model comparison across four GPUs.

The script trains PatchCore, EfficientAD, and AnomalyDINO on:

* /DATA/ljl/c789_100_left_top_parts
* /DATA/ljl/c789_100_left_bottom_parts

It runs one model/dataset experiment per process and binds each process to one
GPU via CUDA_VISIBLE_DEVICES. This is usually more reliable for these
memory-bank style anomaly models than launching every experiment as multi-GPU
DDP.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CHOICES = ("patchcore", "efficient_ad", "anomaly_dino")
LABELS = ("normal", "normal_test", "defect")
IMAGE_EXTENSIONS = (".png",)
RESOURCE_FAILURE_MARKERS = (
    "cannot send a request",
    "client has been closed",
    "hf_hub_download",
    "huggingface_hub",
    "load_state_dict_from_hf",
    "tried to download a pretrained resource",
    "tried to download required assets",
)


@dataclass(frozen=True)
class DatasetSpec:
    """One single-part C789 dataset to benchmark."""

    name: str
    view: str
    data_root: Path


@dataclass(frozen=True)
class ExperimentSpec:
    """One model configuration to train/evaluate on a dataset."""

    name: str
    dataset: DatasetSpec
    model_name: str
    model_run_suffix: str = ""
    patchcore_coreset_ratio: float | None = None
    efficientad_train_sampling_ratio: float | None = None
    anomaly_dino_sampling_ratio: float | None = None
    sampling_type: str = ""
    sampling_ratio: float | None = None


@dataclass(frozen=True)
class CommandJob:
    """One subprocess command plus metadata for reporting."""

    name: str
    command: list[str]
    log_path: Path
    dataset_name: str
    model_name: str | None = None
    experiment_name: str | None = None


@dataclass(frozen=True)
class CommandResult:
    """Finished subprocess result."""

    name: str
    code: int
    log_path: Path
    dataset_name: str
    model_name: str | None
    experiment_name: str | None
    gpu: str
    elapsed_seconds: float


def _repo_path(path: Path) -> Path:
    """Resolve relative paths against the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def _workflow_script() -> Path:
    """Return the stage-3 workflow wrapper path."""
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
    """Normalize --gpus values from either comma or space separated input."""
    gpu_ids = []
    for value in values:
        gpu_ids.extend(part.strip() for part in value.split(",") if part.strip())
    if not gpu_ids:
        msg = "At least one GPU id must be provided."
        raise argparse.ArgumentTypeError(msg)
    return gpu_ids


def _parse_ratio(value: str) -> float:
    """Parse a coreset ratio in the open-closed range (0, 1]."""
    try:
        ratio = float(value)
    except ValueError as error:
        msg = "Ratio must be a number."
        raise argparse.ArgumentTypeError(msg) from error
    if ratio <= 0 or ratio > 1:
        msg = "Ratio must satisfy 0 < value <= 1."
        raise argparse.ArgumentTypeError(msg)
    return ratio


def _ratio_token(value: float) -> str:
    """Return a stable filename-safe token for a ratio."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _float_text(value: str | float | None) -> str:
    """Return a compact numeric table cell."""
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return str(value)


def _candidate_view_dirs(dataset: DatasetSpec) -> list[Path]:
    """Return raw view directories accepted by the workflow."""
    if dataset.view == "left_top":
        return [dataset.data_root / "left" / "top"]
    if dataset.view == "left_bottom":
        return [
            dataset.data_root / "left" / "bottom_ZS32",
            dataset.data_root / "left" / "bottom",
        ]
    msg = f"Unsupported C789 view: {dataset.view}"
    raise ValueError(msg)


def _view_dir_has_images(view_dir: Path) -> bool:
    """Return whether a raw view directory contains workflow-readable PNGs."""
    for label in LABELS:
        label_dir = view_dir / label
        if any(label_dir.glob("*/images/*.png")):
            return True
        if any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in label_dir.glob("*")):
            return True
    return False


def _validate_datasets(datasets: Sequence[DatasetSpec], dry_run: bool) -> None:
    """Fail early when a dataset does not match the workflow layout."""
    if dry_run:
        return

    missing = []
    for dataset in datasets:
        view_dirs = _candidate_view_dirs(dataset)
        if not any(_view_dir_has_images(view_dir) for view_dir in view_dirs):
            missing.append((dataset.name, view_dirs))

    if not missing:
        return

    lines = ["No workflow-readable PNG images found for these C789 part datasets:"]
    for name, paths in missing:
        lines.append(f"  - {name}:")
        for path in paths:
            lines.append(f"      checked {path}")
    lines.append("")
    lines.append("For left_bottom the workflow accepts both left/bottom_ZS32 and left/bottom.")
    lines.append(
        "Expected label folders are normal, normal_test, and defect, with either label/*.png "
        "or label/<sample>/images/*.png.",
    )
    raise SystemExit("\n".join(lines))


def _dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    """Build selected dataset specs from CLI arguments."""
    roots = {
        "left_top": _repo_path(args.top_data_root),
        "left_bottom": _repo_path(args.bottom_data_root),
    }
    return [DatasetSpec(name=name, view=name, data_root=roots[name]) for name in args.datasets]


def _output_root(args: argparse.Namespace, dataset: DatasetSpec) -> Path:
    """Return the output root for one dataset."""
    return _repo_path(args.output_root) / dataset.name


def _experiment_specs(datasets: Sequence[DatasetSpec], args: argparse.Namespace) -> list[ExperimentSpec]:
    """Build the selected model-configuration experiments."""
    experiments = []
    patchcore_sweep = len(args.patchcore_coreset_ratios) > 1
    efficientad_sweep = len(args.efficientad_train_sampling_ratios) > 1
    anomaly_dino_sweep = len(args.anomaly_dino_sampling_ratios) > 1
    for dataset in datasets:
        for model_name in args.models:
            if model_name == "patchcore":
                for ratio in args.patchcore_coreset_ratios:
                    suffix = f"coreset_{_ratio_token(ratio)}" if patchcore_sweep else ""
                    name = f"patchcore_{suffix}" if suffix else "patchcore"
                    experiments.append(
                        ExperimentSpec(
                            name=name,
                            dataset=dataset,
                            model_name=model_name,
                            model_run_suffix=suffix,
                            patchcore_coreset_ratio=ratio,
                            sampling_type="patchcore_coreset",
                            sampling_ratio=ratio,
                        ),
                    )
                continue

            if model_name == "efficient_ad":
                for ratio in args.efficientad_train_sampling_ratios:
                    suffix = f"train_{_ratio_token(ratio)}" if efficientad_sweep else ""
                    name = f"efficient_ad_{suffix}" if suffix else "efficient_ad"
                    experiments.append(
                        ExperimentSpec(
                            name=name,
                            dataset=dataset,
                            model_name=model_name,
                            model_run_suffix=suffix,
                            efficientad_train_sampling_ratio=ratio,
                            sampling_type="efficientad_train",
                            sampling_ratio=ratio,
                        ),
                    )
                continue

            if model_name == "anomaly_dino":
                for ratio in args.anomaly_dino_sampling_ratios:
                    suffix = f"sampling_{_ratio_token(ratio)}" if anomaly_dino_sweep else ""
                    name = f"anomaly_dino_{suffix}" if suffix else "anomaly_dino"
                    experiments.append(
                        ExperimentSpec(
                            name=name,
                            dataset=dataset,
                            model_name=model_name,
                            model_run_suffix=suffix,
                            anomaly_dino_sampling_ratio=ratio,
                            sampling_type="anomaly_dino_coreset",
                            sampling_ratio=ratio,
                        ),
                    )
                continue

            experiments.append(ExperimentSpec(name=model_name, dataset=dataset, model_name=model_name))
    return experiments


def _experiment_reports_dir(experiment: ExperimentSpec) -> Path:
    """Return the reports directory name for a single experiment."""
    return Path("reports") / experiment.name


def _common_workflow_args(
    args: argparse.Namespace,
    dataset: DatasetSpec,
    model_name: str | None = None,
    patchcore_coreset_ratio: float | None = None,
    efficientad_train_sampling_ratio: float | None = None,
    anomaly_dino_sampling_ratio: float | None = None,
) -> list[str]:
    """Build common arguments accepted by the workflow and comparison wrappers."""
    coreset_ratio = patchcore_coreset_ratio
    if coreset_ratio is None:
        coreset_ratio = args.patchcore_coreset_ratios[0]
    train_sampling_ratio = efficientad_train_sampling_ratio
    if train_sampling_ratio is None:
        train_sampling_ratio = 1.0
    dino_sampling_ratio = anomaly_dino_sampling_ratio
    if dino_sampling_ratio is None:
        dino_sampling_ratio = args.anomaly_dino_sampling_ratios[0]
    efficientad_batch_size = 1 if model_name == "efficient_ad" else args.efficientad_batch_size

    command = [
        "--data-root",
        str(dataset.data_root),
        "--output-root",
        str(_output_root(args, dataset)),
        "--views",
        dataset.view,
        "--roi",
        args.roi,
        "--image-size",
        args.image_size,
        "--accelerator",
        args.accelerator,
        "--devices",
        "1",
        "--num-workers",
        str(args.num_workers),
        "--train-sampling-ratio",
        str(train_sampling_ratio),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--deploy-fpr",
        str(args.deploy_fpr),
        "--patchcore-batch-size",
        str(args.patchcore_batch_size),
        "--patchcore-backbone",
        args.patchcore_backbone,
        "--patchcore-coreset-ratio",
        str(coreset_ratio),
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
        "--anomaly-dino-batch-size",
        str(args.anomaly_dino_batch_size),
        "--anomaly-dino-encoder",
        args.anomaly_dino_encoder,
        "--anomaly-dino-neighbors",
        str(args.anomaly_dino_neighbors),
        "--anomaly-dino-sampling-ratio",
        str(dino_sampling_ratio),
    ]
    if args.skip_blue_removal:
        command.append("--skip-blue-removal")
    if args.visualizer_field_size is not None:
        command.extend(["--visualizer-field-size", args.visualizer_field_size])
    if args.anomaly_dino_masking:
        command.append("--anomaly-dino-masking")
    if args.anomaly_dino_coreset_subsampling:
        command.append("--anomaly-dino-coreset-subsampling")
    return command


def _workflow_command(
    step: str,
    args: argparse.Namespace,
    dataset: DatasetSpec,
    models: Sequence[str],
    model_run_suffix: str = "",
    reports_dir_name: Path | None = None,
    patchcore_coreset_ratio: float | None = None,
    efficientad_train_sampling_ratio: float | None = None,
    anomaly_dino_sampling_ratio: float | None = None,
) -> list[str]:
    """Build a pipeline/3_train_model.py command."""
    command = [
        sys.executable,
        str(_workflow_script()),
        step,
        *_common_workflow_args(
            args,
            dataset,
            models[0] if len(models) == 1 else None,
            patchcore_coreset_ratio,
            efficientad_train_sampling_ratio,
            anomaly_dino_sampling_ratio,
        ),
    ]
    if models:
        command.extend(["--models", *models])
    if model_run_suffix:
        command.extend(["--model-run-suffix", model_run_suffix])
    if reports_dir_name is not None:
        command.extend(["--reports-dir-name", str(reports_dir_name)])
    if args.skip_missing_efficientad_assets:
        command.append("--skip-missing-efficientad-assets")
    return command


def _log_tail(log_path: Path, line_count: int = 30) -> str:
    """Return the last lines of a log file."""
    if not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _log_has_shared_resource_failure(log_path: Path) -> bool:
    """Return whether a log shows a shared pretrained-resource failure."""
    log_tail = _log_tail(log_path, line_count=80).lower()
    return any(marker in log_tail for marker in RESOURCE_FAILURE_MARKERS)


def _run_command(
    job: CommandJob,
    gpu_queue: Queue[str],
    args: argparse.Namespace,
    efficientad_lock: Lock,
    patchcore_lock: Lock,
    blocked_models: dict[str, Path],
    blocked_models_lock: Lock,
) -> CommandResult:
    """Run one command on the next available GPU."""
    lock_context: Any
    if job.model_name == "efficient_ad" and not args.parallel_efficientad:
        lock_context = efficientad_lock
    elif job.model_name == "patchcore" and not args.parallel_patchcore:
        lock_context = patchcore_lock
    else:
        lock_context = nullcontext()

    with lock_context:
        if job.model_name is not None:
            with blocked_models_lock:
                blocked_by = blocked_models.get(job.model_name)
            if blocked_by is not None:
                print(
                    f"[skip]  {job.name}: previous {job.model_name} pretrained-resource failure: {blocked_by}",
                    flush=True,
                )
                return CommandResult(
                    name=job.name,
                    code=99,
                    log_path=job.log_path,
                    dataset_name=job.dataset_name,
                    model_name=job.model_name,
                    experiment_name=job.experiment_name,
                    gpu="-",
                    elapsed_seconds=0.0,
                )

        gpu_id = gpu_queue.get()
        try:
            env = os.environ.copy()
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
                    log_file.write(f"$ CUDA_VISIBLE_DEVICES={gpu_id} {printable}\n\n")
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
                if job.model_name is not None and _log_has_shared_resource_failure(job.log_path):
                    with blocked_models_lock:
                        blocked_models.setdefault(job.model_name, job.log_path)

            return CommandResult(
                name=job.name,
                code=code,
                log_path=job.log_path,
                dataset_name=job.dataset_name,
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

    efficientad_lock = Lock()
    patchcore_lock = Lock()
    blocked_models: dict[str, Path] = {}
    blocked_models_lock = Lock()
    results = []
    with ThreadPoolExecutor(max_workers=len(selected_gpus)) as executor:
        futures = [
            executor.submit(
                _run_command,
                job,
                gpu_queue,
                args,
                efficientad_lock,
                patchcore_lock,
                blocked_models,
                blocked_models_lock,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_serial_job(job: CommandJob, args: argparse.Namespace, gpu_id: str) -> CommandResult:
    """Run a single command on a chosen GPU."""
    gpu_queue: Queue[str] = Queue()
    gpu_queue.put(gpu_id)
    return _run_command(job, gpu_queue, args, Lock(), Lock(), {}, Lock())


def _preprocess(datasets: Sequence[DatasetSpec], args: argparse.Namespace) -> None:
    """Run preprocessing once per dataset."""
    for dataset in datasets:
        output_root = _output_root(args, dataset)
        job = CommandJob(
            name=f"preprocess:{dataset.name}",
            command=_workflow_command("preprocess", args, dataset, models=()),
            log_path=output_root / "logs" / "preprocess.log",
            dataset_name=dataset.name,
        )
        result = _run_serial_job(job, args, args.gpus[0])
        if result.code != 0:
            raise SystemExit(result.code)


def _train_jobs(experiments: Sequence[ExperimentSpec], args: argparse.Namespace) -> list[CommandJob]:
    """Create all selected model/dataset training jobs."""
    jobs = []
    for experiment in experiments:
        dataset = experiment.dataset
        output_root = _output_root(args, dataset)
        jobs.append(
            CommandJob(
                name=f"train:{dataset.name}:{experiment.name}",
                command=_workflow_command(
                    "train",
                    args,
                    dataset,
                    models=[experiment.model_name],
                    model_run_suffix=experiment.model_run_suffix,
                    patchcore_coreset_ratio=experiment.patchcore_coreset_ratio,
                    efficientad_train_sampling_ratio=experiment.efficientad_train_sampling_ratio,
                    anomaly_dino_sampling_ratio=experiment.anomaly_dino_sampling_ratio,
                ),
                log_path=output_root / "logs" / f"train_{experiment.name}.log",
                dataset_name=dataset.name,
                model_name=experiment.model_name,
                experiment_name=experiment.name,
            ),
        )
    return jobs


def _successful_experiments_by_dataset(
    experiments: Sequence[ExperimentSpec],
    train_results: Sequence[CommandResult],
    evaluate_only: bool,
) -> dict[str, list[ExperimentSpec]]:
    """Return experiments that should be evaluated for each dataset."""
    by_dataset: dict[str, list[ExperimentSpec]] = {}
    for experiment in experiments:
        by_dataset.setdefault(experiment.dataset.name, [])

    if evaluate_only:
        for experiment in experiments:
            by_dataset[experiment.dataset.name].append(experiment)
        return by_dataset

    successful_names = {
        (result.dataset_name, result.experiment_name)
        for result in train_results
        if result.code == 0 and result.experiment_name is not None
    }
    for experiment in experiments:
        if (experiment.dataset.name, experiment.name) in successful_names:
            by_dataset[experiment.dataset.name].append(experiment)
    return by_dataset


def _evaluate_jobs(
    args: argparse.Namespace,
    experiments_by_dataset: dict[str, list[ExperimentSpec]],
) -> list[CommandJob]:
    """Create final evaluation/report jobs."""
    jobs = []
    for dataset_name, experiments in experiments_by_dataset.items():
        if not experiments:
            print(f"[skip]  evaluate:{dataset_name}: no successful experiments", flush=True)
            continue
        for experiment in experiments:
            dataset = experiment.dataset
            output_root = _output_root(args, dataset)
            jobs.append(
                CommandJob(
                    name=f"evaluate:{dataset.name}:{experiment.name}",
                    command=_workflow_command(
                        "evaluate",
                        args,
                        dataset,
                        models=[experiment.model_name],
                        model_run_suffix=experiment.model_run_suffix,
                        reports_dir_name=_experiment_reports_dir(experiment),
                        patchcore_coreset_ratio=experiment.patchcore_coreset_ratio,
                        efficientad_train_sampling_ratio=experiment.efficientad_train_sampling_ratio,
                        anomaly_dino_sampling_ratio=experiment.anomaly_dino_sampling_ratio,
                    ),
                    log_path=output_root / "logs" / f"evaluate_{experiment.name}.log",
                    dataset_name=dataset.name,
                    model_name=experiment.model_name,
                    experiment_name=experiment.name,
                ),
            )
    return jobs


def _summary_sort_key(row: dict[str, str]) -> tuple[float, float, float]:
    """Sort rows by the most useful comparison metrics first."""
    def value(name: str) -> float:
        try:
            return float(row.get(name, "0") or 0)
        except ValueError:
            return 0.0

    return (value("sample_f1"), value("image_f1"), value("image_recall"))


def _read_experiment_summary(args: argparse.Namespace, experiment: ExperimentSpec) -> list[dict[str, str]]:
    """Read one experiment summary and add sweep metadata columns."""
    summary_path = _output_root(args, experiment.dataset) / _experiment_reports_dir(experiment) / "summary.csv"
    if not summary_path.is_file():
        msg = f"Missing summary for {experiment.dataset.name}/{experiment.name}: {summary_path}"
        raise FileNotFoundError(msg)

    with summary_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row["experiment"] = experiment.name
        row["sampling_type"] = experiment.sampling_type
        row["sampling_ratio"] = f"{experiment.sampling_ratio:g}" if experiment.sampling_ratio is not None else ""
        row["patchcore_coreset_ratio"] = (
            f"{experiment.patchcore_coreset_ratio:g}" if experiment.patchcore_coreset_ratio is not None else ""
        )
        row["efficientad_train_sampling_ratio"] = (
            f"{experiment.efficientad_train_sampling_ratio:g}"
            if experiment.efficientad_train_sampling_ratio is not None
            else ""
        )
        row["anomaly_dino_sampling_ratio"] = (
            f"{experiment.anomaly_dino_sampling_ratio:g}" if experiment.anomaly_dino_sampling_ratio is not None else ""
        )
    return rows


def _write_aggregate_report(
    args: argparse.Namespace,
    dataset: DatasetSpec,
    experiments: Sequence[ExperimentSpec],
) -> None:
    """Write aggregate CSV and Markdown reports for one dataset."""
    output_root = _output_root(args, dataset)
    rows = []
    for experiment in experiments:
        rows.extend(_read_experiment_summary(args, experiment))

    rows = sorted(rows, key=_summary_sort_key, reverse=True)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment",
        "model",
        "view",
        "sampling_type",
        "sampling_ratio",
        "patchcore_coreset_ratio",
        "efficientad_train_sampling_ratio",
        "anomaly_dino_sampling_ratio",
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
        "| experiment | model | view | sampling_type | sampling_ratio | threshold | target_fpr | image_fpr | "
        "image_acc | image_recall | image_f1 | sample_fpr | sample_acc | sample_recall | sample_f1 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {model} | {view} | {sampling_type} | {sampling_ratio} | {threshold} | {target_fpr} | "
            "{image_fpr} | {image_acc} | {image_recall} | {image_f1} | {sample_fpr} | {sample_acc} | "
            "{sample_recall} | {sample_f1} |".format(
                experiment=row.get("experiment", ""),
                model=row.get("model", ""),
                view=row.get("view", ""),
                sampling_type=row.get("sampling_type", ""),
                sampling_ratio=row.get("sampling_ratio", ""),
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


def _write_aggregate_reports(
    datasets: Sequence[DatasetSpec],
    args: argparse.Namespace,
    experiments_by_dataset: dict[str, list[ExperimentSpec]],
) -> None:
    """Write aggregate reports for all datasets."""
    if args.dry_run:
        return
    for dataset in datasets:
        experiments = experiments_by_dataset.get(dataset.name, [])
        if experiments:
            _write_aggregate_report(args, dataset, experiments)


def _print_outputs(datasets: Sequence[DatasetSpec], args: argparse.Namespace) -> None:
    """Print report locations for each dataset."""
    print("\nReports:", flush=True)
    for dataset in datasets:
        output_root = _output_root(args, dataset)
        print(f"  - {dataset.name}: {output_root / 'reports' / 'model_comparison.md'}", flush=True)
        print(f"    csv: {output_root / 'reports' / 'model_comparison.csv'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--top-data-root",
        type=Path,
        default=Path("/DATA/ljl/c789_100_left_top_parts"),
        help="C789 left/top single-part dataset root.",
    )
    parser.add_argument(
        "--bottom-data-root",
        type=Path,
        default=Path("/DATA/ljl/c789_100_left_bottom_parts"),
        help="C789 left/bottom single-part dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/c789_100_compare_4gpu"),
        help="Base output directory. Each dataset gets its own subdirectory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("left_top", "left_bottom"),
        default=["left_top", "left_bottom"],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
        help="Models to train and compare.",
    )

    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"], help="GPU ids, space or comma separated.")
    parser.add_argument("--max-parallel", type=int, help="Maximum concurrent GPU jobs. Defaults to number of GPUs.")
    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="gpu", help="Lightning accelerator.")
    parser.add_argument(
        "--cuda-alloc-conf",
        default="expandable_segments:True",
        help="PYTORCH_CUDA_ALLOC_CONF value. Use an empty string to disable.",
    )
    parser.add_argument(
        "--parallel-efficientad",
        action="store_true",
        help="Allow both EfficientAD jobs to run at once.",
    )
    parser.add_argument(
        "--parallel-patchcore",
        action="store_true",
        help="Allow PatchCore jobs to run at once after pretrained weights are cached.",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per process.")

    parser.add_argument("--roi", default="full", help="Workflow ROI. Use full for already-cropped part images.")
    parser.add_argument("--image-size", default="392,784", help="Model input size as H,W.")
    parser.add_argument("--visualizer-field-size", help="Visualization field size as W,H.")
    parser.add_argument(
        "--blue-removal",
        action="store_false",
        dest="skip_blue_removal",
        help="Enable HSV blue-mark removal. The default skips it for pre-cropped C789 parts.",
    )
    parser.set_defaults(skip_blue_removal=True)
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--deploy-fpr", type=float, default=0.05, help="Allowed normal_test false-positive rate.")

    parser.add_argument("--patchcore-batch-size", type=int, default=1, help="PatchCore training batch size.")
    parser.add_argument("--patchcore-backbone", default="wide_resnet50_2", help="PatchCore timm backbone.")
    parser.add_argument("--patchcore-layers", nargs="+", default=["layer2"], help="PatchCore feature layers.")
    parser.add_argument(
        "--patchcore-coreset-ratio",
        type=_parse_ratio,
        help="Single PatchCore coreset ratio. Kept for quick one-ratio runs.",
    )
    parser.add_argument(
        "--patchcore-coreset-ratios",
        nargs="+",
        type=_parse_ratio,
        help="PatchCore coreset ratios to sweep.",
    )
    parser.add_argument("--patchcore-num-neighbors", type=int, default=1, help="PatchCore nearest-neighbor count.")
    parser.add_argument(
        "--patchcore-precision",
        choices=("float32", "float16"),
        default="float16",
        help="PatchCore feature precision.",
    )

    parser.add_argument("--efficientad-batch-size", type=int, default=1, help="EfficientAD training batch size.")
    parser.add_argument("--efficientad-epochs", type=int, default=20, help="EfficientAD max epochs.")
    parser.add_argument(
        "--efficientad-train-sampling-ratio",
        type=_parse_ratio,
        help="Single EfficientAD normal-train sampling ratio.",
    )
    parser.add_argument(
        "--efficientad-train-sampling-ratios",
        nargs="+",
        type=_parse_ratio,
        help="EfficientAD normal-train sampling ratios to sweep.",
    )
    parser.add_argument(
        "--skip-missing-efficientad-assets",
        action="store_true",
        help="Skip EfficientAD inside the workflow if local teacher/ImageNette assets are missing.",
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
        help="Single AnomalyDINO coreset sampling ratio.",
    )
    parser.add_argument(
        "--anomaly-dino-sampling-ratios",
        nargs="+",
        type=_parse_ratio,
        help="AnomalyDINO coreset sampling ratios to sweep.",
    )

    parser.add_argument("--skip-preprocess", action="store_true", help="Reuse existing preprocessed images.")
    parser.add_argument("--train-only", action="store_true", help="Stop after training and skip final reports.")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip preprocess/train and regenerate reports only.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Evaluate successful models even if one or more training jobs fail.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def _normalize_args(args: argparse.Namespace) -> None:
    """Normalize and validate parsed arguments in place."""
    args.gpus = _split_gpu_values(args.gpus)
    if args.patchcore_coreset_ratios is None:
        args.patchcore_coreset_ratios = (
            [args.patchcore_coreset_ratio] if args.patchcore_coreset_ratio is not None else [0.1, 0.2, 0.3]
        )
    if args.efficientad_train_sampling_ratios is None:
        args.efficientad_train_sampling_ratios = (
            [args.efficientad_train_sampling_ratio]
            if args.efficientad_train_sampling_ratio is not None
            else [0.1, 0.2, 0.4]
        )
    if args.anomaly_dino_sampling_ratios is None:
        args.anomaly_dino_sampling_ratios = (
            [args.anomaly_dino_sampling_ratio] if args.anomaly_dino_sampling_ratio is not None else [0.1, 0.2, 0.4]
        )
    if args.max_parallel is None:
        args.max_parallel = len(args.gpus)
    args.max_parallel = min(args.max_parallel, len(args.gpus))
    if args.max_parallel <= 0:
        raise SystemExit("--max-parallel must be positive.")
    if args.evaluate_only and args.train_only:
        raise SystemExit("--evaluate-only and --train-only cannot be used together.")


def main() -> None:
    """Run the C789 four-GPU comparison workflow."""
    args = build_parser().parse_args()
    _normalize_args(args)

    datasets = _dataset_specs(args)
    _validate_datasets(datasets, args.dry_run)
    experiments = _experiment_specs(datasets, args)

    print("C789-100 model comparison", flush=True)
    print(f"Datasets: {', '.join(dataset.name for dataset in datasets)}", flush=True)
    print(f"Models: {', '.join(args.models)}", flush=True)
    if "patchcore" in args.models:
        ratios = ", ".join(f"{ratio:g}" for ratio in args.patchcore_coreset_ratios)
        print(f"PatchCore coreset ratios: {ratios}", flush=True)
        if not args.parallel_patchcore:
            print("PatchCore jobs: serialized by default to avoid first-run pretrained-weight cache races", flush=True)
    if "efficient_ad" in args.models:
        ratios = ", ".join(f"{ratio:g}" for ratio in args.efficientad_train_sampling_ratios)
        print(f"EfficientAD train sampling ratios: {ratios}", flush=True)
        if args.efficientad_batch_size != 1:
            print("EfficientAD train batch size: forced to 1 by the model", flush=True)
    if "anomaly_dino" in args.models:
        ratios = ", ".join(f"{ratio:g}" for ratio in args.anomaly_dino_sampling_ratios)
        print(f"AnomalyDINO coreset ratios: {ratios}", flush=True)
    print(f"Experiments: {len(experiments)}", flush=True)
    print(f"GPUs: {', '.join(args.gpus[: args.max_parallel])}", flush=True)
    print(f"Output root: {_repo_path(args.output_root)}", flush=True)

    if not args.evaluate_only and not args.skip_preprocess:
        _preprocess(datasets, args)

    train_results: list[CommandResult] = []
    if not args.evaluate_only:
        train_results = _run_parallel_jobs(_train_jobs(experiments, args), args)
        failures = [result for result in train_results if result.code != 0]
        if failures:
            print("\nTraining failures:", flush=True)
            for result in failures:
                print(f"  - {result.name}: {result.log_path}", flush=True)
            if not args.keep_going:
                raise SystemExit(1)

    if args.train_only:
        print("\nTraining complete. Final evaluation was skipped by --train-only.", flush=True)
        return

    experiments_by_dataset = _successful_experiments_by_dataset(experiments, train_results, args.evaluate_only)
    eval_results = _run_parallel_jobs(_evaluate_jobs(args, experiments_by_dataset), args)
    eval_failures = [result for result in eval_results if result.code != 0]
    if eval_failures:
        print("\nEvaluation failures:", flush=True)
        for result in eval_failures:
            print(f"  - {result.name}: {result.log_path}", flush=True)
        raise SystemExit(1)

    _write_aggregate_reports(datasets, args, experiments_by_dataset)
    _print_outputs(datasets, args)


if __name__ == "__main__":
    main()
