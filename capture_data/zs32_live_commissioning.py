# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Capture one complete ZS32 part and run the single Demo inference path."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import cv2

from zs32_inspection.capture.contracts import utc_now
from zs32_inspection.config.loaders import load_topology
from zs32_inspection.dashboard.contracts import ProgressRecord
from zs32_inspection.dashboard.control import write_progress
from zs32_inspection.dashboard.inference_worker import InferenceWorkerClient
from zs32_inspection.domain.views import VIEW_ORDER


@dataclass(frozen=True)
class CapturedSample:
    """One verified, complete topology-bound eight-view capture."""

    part_id: str
    capture_session: str
    group_id: str
    images: Mapping[str, Path]
    manifest_path: Path


@dataclass(frozen=True)
class LiveRunConfig:
    """Inputs required by one live Demo capture and inference run."""

    repo_root: Path
    part_id: str
    run_id: str
    capture_root: Path
    output_root: Path
    demo_config: Path
    topology_path: Path
    progress_json: Path | None = None
    control_json: Path | None = None
    inference_socket: Path | None = None


@dataclass(frozen=True)
class LiveRunResult:
    """Validated Demo result returned to Stage35 and the dashboard."""

    sample: CapturedSample
    output_dir: Path
    runtime_manifest_path: Path
    machine_status: str
    inspection_complete: bool
    errors: tuple[str, ...]


class LiveCommissioningError(RuntimeError):
    """Raised when capture, inference, or an emitted contract is invalid."""


CommandRunner = Callable[..., object]


def build_capture_command(config: LiveRunConfig, capture_run_root: Path) -> list[str]:
    """Build the one-sample topology-driven two-round capture command."""
    command = [
        sys.executable,
        str((config.repo_root / "pipeline/zs32_bootstrap_capture.py").expanduser().resolve()),
        "--topology",
        str(config.topology_path.expanduser().resolve()),
        "--root",
        str(capture_run_root.expanduser().resolve()),
        "--capture-session",
        config.run_id,
        "--legacy-layout",
        "--hand",
        "right",
        "--label",
        "normal",
        "--part-id",
        config.part_id,
        "--group-count",
        "1",
        "--images-per-group",
        "1",
        "--manual-load",
        "--hdr",
        "--short-exposure",
        "1500",
        "--long-exposure",
        "5500",
        "--gain",
        "0",
        "--capture-interval",
        "0.2",
        "--hdr-settle-frames",
        "1",
        "--timeout-ms",
        "2000",
        "--short-dark-threshold",
        "80",
        "--long-clip-threshold",
        "245",
        "--blend-width",
        "50",
        "--blur-size",
        "101",
        "--hdr-max-retries",
        "0",
        "--hdr-max-clip-pct",
        "5",
        "--no-align-hdr",
        "--timing-json",
        str(capture_run_root / "timing.json"),
    ]
    if config.progress_json is not None and config.control_json is not None:
        command.extend(
            (
                "--progress-json",
                str(config.progress_json),
                "--control-json",
                str(config.control_json),
            ),
        )
    return command


def build_demo_inference_command(
    config: LiveRunConfig,
    sample: CapturedSample,
    output_dir: Path,
) -> list[str]:
    """Build the only online inference command from one Demo config."""
    command = [
        sys.executable,
        str((config.repo_root / "pipeline/zs32_demo_inference.py").expanduser().resolve()),
        "--demo-config",
        str(config.demo_config.expanduser().resolve()),
        "--part-id",
        sample.part_id,
        "--capture-session",
        sample.capture_session,
        "--group-id",
        sample.group_id,
    ]
    for view in VIEW_ORDER:
        command.extend(
            (
                f"--{view.replace('_', '-')}-image",
                str(sample.images[view].expanduser().resolve()),
            ),
        )
    command.extend(("--output-dir", str(output_dir.expanduser().resolve())))
    if config.progress_json is not None:
        command.extend(("--progress-json", str(config.progress_json)))
    return command


def _validate_path_component(value: str, name: str) -> None:
    if not value.strip() or value in {".", ".."} or Path(value).is_absolute() or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be a safe non-empty path component")


def _run_child(command_runner: CommandRunner, command: list[str], name: str) -> None:
    try:
        completed = command_runner(command, check=False)
    except Exception as error:
        raise LiveCommissioningError(
            f"{name} command could not run: {type(error).__name__}: {error}",
        ) from error
    returncode = getattr(completed, "returncode", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise LiveCommissioningError(f"{name} command returned no integer return code")
    if returncode != 0:
        raise LiveCommissioningError(f"{name} command failed with return code {returncode}")


def _preflight(config: LiveRunConfig) -> None:
    try:
        topology = load_topology(config.topology_path.expanduser().resolve())
        if len(topology.camera_slots) != 4 or tuple(topology.required_views) != VIEW_ORDER:
            raise ValueError("capture topology must define exactly four cameras and the canonical eight views")
        if not config.demo_config.expanduser().resolve().is_file():
            raise FileNotFoundError(f"Demo config does not exist: {config.demo_config.expanduser().resolve()}")
    except (OSError, TypeError, ValueError) as error:
        raise LiveCommissioningError(f"Demo preflight failed: {error}") from error


def _validate_captured_sample(config: LiveRunConfig, sample: CapturedSample) -> None:
    _validate_path_component(sample.part_id, "manifest part_id")
    _validate_path_component(sample.capture_session, "manifest capture_session")
    _validate_path_component(sample.group_id, "manifest group_id")
    if sample.group_id != "group001":
        raise ValueError(f"manifest group_id must be group001, found {sample.group_id!r}")
    expected_part_id = f"{config.part_id}_group001_000001"
    if sample.part_id != expected_part_id:
        raise ValueError(f"manifest part_id must be {expected_part_id!r}, found {sample.part_id!r}")


def _load_verified_sample(config: LiveRunConfig, capture_run_root: Path) -> CapturedSample:
    try:
        sample = load_complete_sample(
            find_single_manifest(capture_run_root),
            config.topology_path,
            validate_decode=False,
        )
        _validate_captured_sample(config, sample)
    except (OSError, UnicodeError, ValueError) as error:
        raise LiveCommissioningError(f"capture manifest contract failed: {error}") from error
    return sample


def _load_runtime_manifest(path: Path, sample: CapturedSample) -> tuple[str, bool, tuple[str, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LiveCommissioningError(f"runtime manifest contract failed for {path}: {error}") from error
    if not isinstance(payload, dict):
        raise LiveCommissioningError("runtime manifest contract failed: expected a JSON object")
    expected_identity = {
        "part_id": sample.part_id,
        "capture_session": sample.capture_session,
        "group_id": sample.group_id,
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise LiveCommissioningError(
                f"runtime manifest contract failed: {field} does not match captured sample",
            )
    machine_status = payload.get("machine_status")
    if not isinstance(machine_status, str) or not machine_status.strip():
        raise LiveCommissioningError("runtime manifest contract failed: machine_status must be a non-empty string")
    inspection_complete = payload.get("inspection_complete")
    if not isinstance(inspection_complete, bool):
        raise LiveCommissioningError("runtime manifest contract failed: inspection_complete must be bool")
    raw_errors = payload.get("errors")
    if not isinstance(raw_errors, list) or any(not isinstance(error, str) or not error.strip() for error in raw_errors):
        raise LiveCommissioningError("runtime manifest contract failed: errors must be a list of non-empty strings")
    errors = tuple(raw_errors)
    if errors and (inspection_complete or machine_status.strip().upper() == "OK"):
        raise LiveCommissioningError(
            "runtime manifest contract failed: errored inference cannot be complete or OK",
        )
    return machine_status, inspection_complete, errors


def run_live_commissioning(
    config: LiveRunConfig,
    command_runner: CommandRunner = subprocess.run,
) -> LiveRunResult:
    """Run capture and Demo inference while publishing the real failure reason."""
    try:
        try:
            _validate_path_component(config.part_id, "part_id")
            _validate_path_component(config.run_id, "run_id")
            if (config.progress_json is None) != (config.control_json is None):
                raise ValueError("progress_json and control_json must be provided together")
        except ValueError as error:
            raise LiveCommissioningError(str(error)) from error
        return _run_live_commissioning(config, command_runner)
    except (Exception, KeyboardInterrupt) as error:
        if config.progress_json is not None:
            try:
                write_progress(
                    config.progress_json,
                    ProgressRecord(
                        config.part_id,
                        None,
                        "failed",
                        "inspection failed",
                        utc_now(),
                        error=f"{type(error).__name__}: {error}",
                    ),
                )
            except Exception as progress_error:
                raise LiveCommissioningError(
                    f"inspection failed and failed progress could not be written: {progress_error}",
                ) from error
        raise


def _run_live_commissioning(
    config: LiveRunConfig,
    command_runner: CommandRunner,
) -> LiveRunResult:
    _preflight(config)
    capture_root = config.capture_root.expanduser().resolve()
    capture_run_root = (capture_root / config.run_id).resolve()
    if capture_run_root.exists():
        raise LiveCommissioningError(f"capture run root already exists: {capture_run_root}")
    try:
        capture_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise LiveCommissioningError(f"could not create capture root {capture_root}: {error}") from error

    _run_child(command_runner, build_capture_command(config, capture_run_root), "capture")
    sample = _load_verified_sample(config, capture_run_root)
    output_dir = (config.output_root.expanduser().resolve() / sample.capture_session / sample.part_id).resolve()
    if output_dir.exists():
        raise LiveCommissioningError(f"output directory already exists: {output_dir}")

    inference_command = build_demo_inference_command(config, sample, output_dir)
    if config.inference_socket is None:
        _run_child(command_runner, inference_command, "Demo inference")
    else:
        try:
            response = InferenceWorkerClient(config.inference_socket, timeout=300.0).execute(
                f"{sample.capture_session}:{sample.part_id}",
                inference_command[2:],
            )
        except Exception as error:
            raise LiveCommissioningError(
                f"Demo inference worker failed: {type(error).__name__}: {error}",
            ) from error
        if response.error is not None or response.returncode != 0:
            detail = response.error or f"return code {response.returncode}"
            raise LiveCommissioningError(f"Demo inference worker failed: {detail}")

    runtime_manifest_path = output_dir / "runtime_manifest.json"
    machine_status, inspection_complete, errors = _load_runtime_manifest(runtime_manifest_path, sample)
    result = LiveRunResult(
        sample=sample,
        output_dir=output_dir,
        runtime_manifest_path=runtime_manifest_path,
        machine_status=machine_status,
        inspection_complete=inspection_complete,
        errors=errors,
    )
    if config.progress_json is not None:
        write_progress(
            config.progress_json,
            ProgressRecord(
                sample.part_id,
                sample.capture_session,
                "complete",
                str(output_dir),
                utc_now(),
            ),
        )
    return result


def find_single_manifest(root: Path) -> Path:
    """Return the only manifest CSV beneath one capture root."""
    resolved_root = root.expanduser().resolve()
    manifest_dir = resolved_root / "manifests"
    search_dir = manifest_dir if manifest_dir.is_dir() else resolved_root
    manifests = sorted(path.resolve() for path in search_dir.glob("*.csv") if path.is_file())
    if len(manifests) != 1:
        raise ValueError(f"expected exactly one manifest in {search_dir}, found {len(manifests)}")
    return manifests[0]


def _shared_identity(rows: list[dict[str, str]], field: str) -> str:
    values = {row.get(field, "").strip() for row in rows}
    if "" in values or len(values) != 1:
        raise ValueError(f"manifest rows must share one non-empty {field}")
    return values.pop()


def load_complete_sample(
    path: Path,
    topology_path: Path | None = None,
    *,
    validate_decode: bool = True,
) -> CapturedSample:
    """Load exactly eight decodable images with one shared capture identity."""
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"manifest is not a file: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"manifest has no rows: {manifest_path}")
    if any(row.get("sample_status", "").strip() != "complete" for row in rows):
        raise ValueError("every manifest row must have sample_status=complete")

    topology = load_topology(
        topology_path
        or Path(__file__).resolve().parents[1] / "configs/zs32/topology/zs32_4cam_double_side_v1.json",
    )
    if len(topology.camera_slots) != 4 or tuple(topology.required_views) != VIEW_ORDER:
        raise ValueError("capture topology must define exactly four cameras and the canonical eight views")

    image_rows = [row for row in rows if row.get("record_type", "").strip() == "image"]
    sample_rows = [row for row in rows if row.get("record_type", "").strip() == "sample"]
    if len(image_rows) != len(VIEW_ORDER) or len(sample_rows) != 1 or len(rows) != len(image_rows) + 1:
        raise ValueError("manifest must contain eight image rows and exactly one sample row")

    capture_session = _shared_identity(rows, "session_id")
    part_id = _shared_identity(rows, "sample_id")
    group_id = _shared_identity(rows, "group_id")
    views = [row.get("view", "").strip() for row in image_rows]
    if len(set(views)) != len(VIEW_ORDER) or set(views) != set(VIEW_ORDER):
        raise ValueError(f"image rows must contain exactly the canonical views: {VIEW_ORDER}")

    rows_by_view = {row["view"].strip(): row for row in image_rows}
    images: dict[str, Path] = {}
    capture_root = manifest_path.parent.parent
    for view in VIEW_ORDER:
        expected_round, _slot_id, expected_serial = topology.binding_for_view(view)
        row = rows_by_view[view]
        if row.get("round", "").strip() != expected_round:
            raise ValueError(f"manifest round for {view} must be {expected_round!r}")
        if row.get("camera_serial", "").strip() != expected_serial:
            raise ValueError(f"manifest camera_serial for {view} must be {expected_serial!r}")
        image_path = Path(row.get("file", "").strip()).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise ValueError(f"image file for {view} does not exist: {image_path}")
        expected_directory = (capture_root / "right" / view / "normal" / capture_session / "images").resolve()
        expected_prefix = f"right_{view}_normal_"
        if (
            image_path.parent != expected_directory
            or not image_path.name.startswith(expected_prefix)
            or image_path.suffix.lower() != ".png"
        ):
            raise ValueError(f"image file for {view} does not match the right-hand capture layout: {image_path}")
        if validate_decode and cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
            raise ValueError(f"image file for {view} could not decode: {image_path}")
        images[view] = image_path

    return CapturedSample(
        part_id=part_id,
        capture_session=capture_session,
        group_id=group_id,
        images=MappingProxyType(images),
        manifest_path=manifest_path,
    )
