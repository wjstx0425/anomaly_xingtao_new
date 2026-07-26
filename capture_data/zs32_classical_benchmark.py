# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Immutable offline benchmark for ZS32 classical diagnostic operators.

This module is deliberately independent from the production inspection graph.
It publishes continuous operator evidence only and never creates a final
inspection decision.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import random
import resource
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from capture_data.zs32_classical_operators import (
    OperatorParameters,
    OperatorResult,
    detect_pits_spots,
    detect_thin_lines,
    render_overlay,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

PRIMARY_VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")
_VALID_LABELS = frozenset({"normal", "defect"})
_PREDICTION_FIELDS = (
    "case_id",
    "session_id",
    "view",
    "label",
    "defect_type",
    "source_path",
    "source_sha256",
    "operator",
    "score",
    "component_count",
    "components_json",
    "original_shape",
    "processed_shape",
    "resize_scale",
    "mask_path",
    "overlay_path",
    "decode_resize_seconds",
    "operator_seconds",
    "evidence_render_write_seconds",
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One validated ROI selected from a crop manifest."""

    case_id: str
    image_path: Path
    manifest_output_path: str
    view: str
    label: str
    defect_type: str
    session_id: str


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for one immutable diagnostic generation."""

    manifest_path: Path
    output_dir: Path
    views: tuple[str, ...] = PRIMARY_VIEWS
    max_normal_per_view: int | None = None
    max_defect_per_view: int | None = None
    seed: int = 0
    resize_scale: float = 1.0
    warmup_rounds: int = 1
    timing_rounds: int = 30
    timing_samples_per_view: int = 1
    parallel_workers: int = 2
    save_overlays: bool = True
    parameters: OperatorParameters = field(default_factory=OperatorParameters)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_manifest_output(manifest_path: Path, raw_path: str) -> Path:
    """Resolve an absolute path or search ancestors for a repo-relative path."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        candidates = (path,)
    else:
        candidates = tuple(parent / path for parent in (manifest_path.parent, *manifest_path.parents))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.R_OK):
            return candidate.resolve()
    msg = f"Manifest output_path is not a readable file: {raw_path}"
    raise FileNotFoundError(msg)


def load_crop_manifest(path: Path, views: tuple[str, ...]) -> tuple[BenchmarkCase, ...]:
    """Load validated cases for the requested canonical primary views."""
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        msg = f"Crop manifest does not exist: {manifest_path}"
        raise FileNotFoundError(msg)
    if not views:
        msg = "At least one primary view is required"
        raise ValueError(msg)
    unsupported = sorted(set(views) - set(PRIMARY_VIEWS))
    if unsupported:
        msg = f"Unsupported view(s); expected canonical primary views: {unsupported}"
        raise ValueError(msg)
    requested = set(views)
    cases: list[BenchmarkCase] = []
    identities: set[str] = set()
    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"output_path", "resolved_view", "label", "session_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            msg = f"Crop manifest missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        for line_number, row in enumerate(reader, start=2):
            view = (row.get("resolved_view") or "").strip()
            if view not in requested:
                continue
            label = (row.get("label") or "").strip()
            if label not in _VALID_LABELS:
                msg = f"Invalid label at manifest line {line_number}: {label!r}"
                raise ValueError(msg)
            session_id = (row.get("session_id") or "").strip()
            if not session_id:
                msg = f"Missing session_id at manifest line {line_number}"
                raise ValueError(msg)
            raw_output = (row.get("output_path") or "").strip()
            image_path = _resolve_manifest_output(manifest_path, raw_output)
            identity = f"{session_id}:{image_path.name}"
            if identity in identities:
                msg = f"Duplicate case identity at manifest line {line_number}: {identity}"
                raise ValueError(msg)
            identities.add(identity)
            case_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    image_path=image_path,
                    manifest_output_path=raw_output,
                    view=view,
                    label=label,
                    defect_type=(row.get("defect_type") or "").strip(),
                    session_id=session_id,
                ),
            )
    return tuple(sorted(cases, key=lambda case: (case.view, case.label, case.case_id)))


def _validate_config(config: BenchmarkConfig) -> None:
    if config.output_dir.exists():
        msg = f"Output generation already exists: {config.output_dir}"
        raise FileExistsError(msg)
    if config.resize_scale <= 0.0:
        msg = "resize_scale must be positive"
        raise ValueError(msg)
    if len(config.views) != len(set(config.views)):
        msg = "config.views must not contain duplicate views"
        raise ValueError(msg)
    integer_values = {
        "warmup_rounds": config.warmup_rounds,
        "timing_rounds": config.timing_rounds,
        "timing_samples_per_view": config.timing_samples_per_view,
        "parallel_workers": config.parallel_workers,
    }
    for name, value in integer_values.items():
        minimum = 1 if name in {"timing_rounds", "timing_samples_per_view", "parallel_workers"} else 0
        if value < minimum:
            msg = f"{name} must be at least {minimum}"
            raise ValueError(msg)
    for name, value in (
        ("max_normal_per_view", config.max_normal_per_view),
        ("max_defect_per_view", config.max_defect_per_view),
    ):
        if value is not None and value < 0:
            msg = f"{name} must be non-negative or omitted"
            raise ValueError(msg)


def _select_cases(cases: Sequence[BenchmarkCase], config: BenchmarkConfig) -> tuple[BenchmarkCase, ...]:
    selected: list[BenchmarkCase] = []
    limits = {
        "normal": config.max_normal_per_view,
        "defect": config.max_defect_per_view,
    }
    for view in config.views:
        for label in ("normal", "defect"):
            group = [case for case in cases if case.view == view and case.label == label]
            # The seed exists to reproduce benchmark sampling, not for security.
            random.Random(f"{config.seed}:{view}:{label}").shuffle(group)  # noqa: S311
            limit = limits[label]
            selected.extend(group if limit is None else group[:limit])
    return tuple(selected)


def _shape_text(image: np.ndarray) -> str:
    return "x".join(str(dimension) for dimension in image.shape)


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    if math.isclose(scale, 1.0):
        return image
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _write_png(path: Path, image: np.ndarray) -> None:
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        msg = f"Could not encode evidence PNG: {path}"
        raise OSError(msg)
    _write_bytes(path, payload.tobytes())


def _write_predictions(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())


def timing_statistics(samples: Sequence[float]) -> dict[str, int | float]:
    """Summarize measured formal samples; warmups must not be passed here."""
    if not samples:
        msg = "At least one formal timing sample is required"
        raise ValueError(msg)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": len(samples),
        "p50_seconds": float(np.percentile(values, 50)),
        "p95_seconds": float(np.percentile(values, 95)),
        "p99_seconds": float(np.percentile(values, 99)),
        "max_seconds": float(np.max(values)),
    }


def _invoke_operators(
    image: np.ndarray,
    parameters: OperatorParameters,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[tuple[OperatorResult, float], ...]:
    measured: list[tuple[OperatorResult, float]] = []
    for operator in (detect_thin_lines, detect_pits_spots):
        started = clock()
        result = operator(image, parameters)
        measured.append((result, clock() - started))
    return tuple(measured)


def measure_timing(
    images: Sequence[np.ndarray],
    config: BenchmarkConfig,
    *,
    operator_runner: Callable[[np.ndarray], dict[str, float]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = time.process_time,
    executor_factory: Callable[..., Any] = ThreadPoolExecutor,
    peak_rss_reader: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Measure warmup-excluded serial, parallel, operator, CPU, and RSS timing."""
    def default_runner(image: np.ndarray) -> dict[str, float]:
        return {
            result.operator: elapsed
            for result, elapsed in _invoke_operators(image, config.parameters, clock)
        }

    run_one = operator_runner or default_runner
    for _ in range(config.warmup_rounds):
        for image in images:
            run_one(image)

    pure_samples: dict[str, list[float]] = {"thin_line": [], "pit_spot": []}
    serial_wall: list[float] = []
    parallel_wall: list[float] = []
    cpu_started = cpu_clock()
    for _ in range(config.timing_rounds):
        started = clock()
        for image in images:
            for operator, elapsed in run_one(image).items():
                pure_samples[operator].append(elapsed)
        serial_wall.append(clock() - started)

        started = clock()
        with executor_factory(max_workers=config.parallel_workers) as executor:
            list(executor.map(run_one, images))
        parallel_wall.append(clock() - started)
    process_cpu_seconds = cpu_clock() - cpu_started
    if peak_rss_reader is None:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = int(peak_rss) if sys.platform == "darwin" else int(peak_rss * 1024)
    else:
        peak_rss_bytes = peak_rss_reader()
    return {
        "formal_rounds": config.timing_rounds,
        "warmup_rounds": config.warmup_rounds,
        "timing_sample_count": len(images),
        "pure_operator": {
            operator: timing_statistics(samples)
            for operator, samples in pure_samples.items()
        },
        "serial_wall": timing_statistics(serial_wall),
        "bounded_parallel_wall": timing_statistics(parallel_wall),
        "process_cpu_seconds": process_cpu_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "parallel_workers": config.parallel_workers,
        "opencv_threads": cv2.getNumThreads(),
    }


def _publish_case_evidence(
    staging: Path,
    case: BenchmarkCase,
    original: np.ndarray,
    processed: np.ndarray,
    decode_resize_seconds: float,
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_hash = _sha256_file(case.image_path)
    for result, operator_seconds in _invoke_operators(processed, config.parameters):
        stem = f"{case.case_id}_{case.view}_{result.operator}"
        mask_path = Path("masks") / f"{stem}.png"
        overlay_path = Path("overlays") / f"{stem}.png" if config.save_overlays else None
        write_started = time.perf_counter()
        _write_png(staging / mask_path, result.mask)
        if overlay_path is not None:
            _write_png(staging / overlay_path, render_overlay(processed, result))
        evidence_render_write_seconds = time.perf_counter() - write_started
        rows.append(
            {
                "case_id": case.case_id,
                "session_id": case.session_id,
                "view": case.view,
                "label": case.label,
                "defect_type": case.defect_type,
                "source_path": str(case.image_path),
                "source_sha256": source_hash,
                "operator": result.operator,
                "score": result.score,
                "component_count": len(result.components),
                "components_json": json.dumps(
                    [component.to_dict() for component in result.components],
                    separators=(",", ":"),
                ),
                "original_shape": _shape_text(original),
                "processed_shape": _shape_text(processed),
                "resize_scale": config.resize_scale,
                "mask_path": mask_path.as_posix(),
                "overlay_path": overlay_path.as_posix() if overlay_path is not None else "",
                "decode_resize_seconds": decode_resize_seconds,
                "operator_seconds": operator_seconds,
                "evidence_render_write_seconds": evidence_render_write_seconds,
            },
        )
    return rows


def _stream_cases(
    staging: Path,
    cases: Iterable[BenchmarkCase],
    config: BenchmarkConfig,
) -> tuple[list[dict[str, Any]], tuple[np.ndarray, ...]]:
    """Process cases one-by-one and retain only a bounded timing subset."""
    rows: list[dict[str, Any]] = []
    timing_by_view: dict[str, list[np.ndarray]] = {view: [] for view in config.views}
    for case in cases:
        started = time.perf_counter()
        original = cv2.imread(str(case.image_path), cv2.IMREAD_COLOR)
        if original is None:
            msg = f"Could not decode ROI image: {case.image_path}"
            raise ValueError(msg)
        processed = _resize(original, config.resize_scale)
        decode_resize_seconds = time.perf_counter() - started
        rows.extend(
            _publish_case_evidence(
                staging,
                case,
                original,
                processed,
                decode_resize_seconds,
                config,
            ),
        )
        retained = timing_by_view[case.view]
        if len(retained) < config.timing_samples_per_view:
            retained.append(processed)
    timing_images = tuple(image for view in config.views for image in timing_by_view[view])
    if not timing_images:
        msg = "No decoded timing samples were selected"
        raise ValueError(msg)
    return rows, timing_images


def publish_directory_noreplace(staging: Path, target: Path) -> None:
    """Atomically publish a directory without replacing an existing target.

    Linux uses renameat2(RENAME_NOREPLACE), which closes the check/rename race.
    Other platforms lack a stdlib equivalent and use a documented best-effort
    existence check followed by same-filesystem rename.
    """
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(staging),
            -100,
            os.fsencode(target),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            msg = f"Output generation already exists: {target}"
            raise FileExistsError(error_number, msg, target)
        raise OSError(error_number, os.strerror(error_number), target)
    if target.exists():
        msg = f"Output generation already exists: {target}"
        raise FileExistsError(msg)
    staging.rename(target)


def _build_generation(
    staging: Path,
    cases: Sequence[BenchmarkCase],
    manifest_path: Path,
    config: BenchmarkConfig,
) -> None:
    """Build every artifact in a private staging directory."""
    rows, timing_images = _stream_cases(staging, cases, config)
    benchmark = measure_timing(timing_images, config)
    benchmark["resize_scale"] = config.resize_scale
    _write_predictions(staging / "predictions.csv", rows)
    _write_json(staging / "parameters.json", config.parameters.to_dict())
    _write_json(staging / "benchmark.json", benchmark)
    _write_json(
        staging / "run_manifest.json",
        {
            "diagnostic_only": True,
            "integrated_with_stage32": False,
            "integrated_with_stage18": False,
            "integrated_with_dashboard": False,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "case_count": len(cases),
            "prediction_count": len(rows),
            "views": list(config.views),
            "selection": {
                "max_normal_per_view": config.max_normal_per_view,
                "max_defect_per_view": config.max_defect_per_view,
                "seed": config.seed,
            },
            "resize_scale": config.resize_scale,
            "save_overlays": config.save_overlays,
        },
    )


def _publish_generation(staging: Path, output_dir: Path) -> None:
    """Atomically publish a completed staging directory without clobbering."""
    publish_directory_noreplace(staging, output_dir)


def run_benchmark(config: BenchmarkConfig) -> Path:
    """Create and atomically publish one diagnostic-only benchmark generation."""
    _validate_config(config)
    manifest_path = config.manifest_path.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    cases = _select_cases(load_crop_manifest(manifest_path, config.views), config)
    if not cases:
        msg = "No crop-manifest cases matched the benchmark selection"
        raise ValueError(msg)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        _build_generation(staging, cases, manifest_path, config)
        _publish_generation(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir
