# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed manifest and audit contracts for live ZS32 commissioning."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType

from capture_data.zs32_18_group_commissioning import validate_commissioning_source_assets
from capture_data.zs32_inspection_orchestrator import CANONICAL_VIEWS
from capture_data.zs32_runtime_bundle import RuntimeBundle, load_runtime_bundle
from zs32_inspection.capture.contracts import utc_now
from zs32_inspection.dashboard.contracts import ProgressRecord
from zs32_inspection.dashboard.control import write_progress
from zs32_inspection.config.loaders import load_topology
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
    """Locked bundle, topology, and identities for one live run."""

    repo_root: Path
    part_id: str
    run_id: str
    capture_root: Path
    output_root: Path
    runtime_config: Path
    topology_path: Path
    accelerator: str = "gpu"
    devices: int = 1
    yolo_device: str = "0"
    diagnostic_skip_template: bool = False
    progress_json: Path | None = None
    control_json: Path | None = None


@dataclass(frozen=True)
class LiveRunResult:
    """Verified successful execution and business result from one live run."""

    sample: CapturedSample
    capture_run_root: Path
    output_dir: Path
    runtime_summary_path: Path
    audit_path: Path | None
    machine_status: str
    inspection_complete: bool
    commissioning_only: bool
    production_release_allowed: bool
    audit_report: str
    diagnostic_skip_template: bool = False
    patchcore_csv: Path | None = None
    yolo_csv: Path | None = None


class LiveCommissioningError(RuntimeError):
    """Raised when execution or a child artifact violates the live contract."""


CommandRunner = Callable[..., object]


def build_capture_command(config: LiveRunConfig, capture_run_root: Path) -> list[str]:
    """Build the locked one-sample topology-driven capture command."""
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
    ]
    if config.progress_json is not None and config.control_json is not None:
        command.extend(("--progress-json", str(config.progress_json), "--control-json", str(config.control_json)))
    return command


def build_stage32_command(config: LiveRunConfig, sample: CapturedSample, output_dir: Path) -> list[str]:
    """Build the exact-eight Stage32 command from the finalized runtime bundle."""
    bundle = load_runtime_bundle(config.runtime_config)
    command = [
        sys.executable,
        str((config.repo_root / "pipeline/32_run_zs32_multimodel_inference.py").expanduser().resolve()),
    ]
    if config.diagnostic_skip_template:
        command.extend(("infer", "--diagnostic-skip-template"))
    else:
        command.extend(("fuse", "--fusion-config", str(bundle.fusion_profile)))
    command.extend(
        (
            "--part-id",
            sample.part_id,
            "--capture-session",
            sample.capture_session,
            "--group-id",
            sample.group_id,
            "--hand",
            "right",
        ),
    )
    for view in VIEW_ORDER:
        command.extend((f"--{view.replace('_', '-')}-image", str(sample.images[view].resolve())))
    command.extend(
        (
            "--runtime-config",
            str(bundle.runtime_assets),
        ),
    )
    if not config.diagnostic_skip_template:
        command.extend(
            (
                "--template-model-dir",
                str(bundle.template_model_dir),
                "--threshold-artifact",
                str(bundle.threshold_artifact),
            ),
        )
    command.extend(
        (
            "--accelerator",
            config.accelerator,
            "--devices",
            str(config.devices),
            "--yolo-device",
            config.yolo_device,
            "--output-dir",
            str(output_dir.expanduser().resolve()),
        ),
    )
    if config.progress_json is not None:
        command.extend(("--progress-json", str(config.progress_json)))
    return command


def _validate_path_component(value: str, name: str) -> None:
    """Reject an empty or path-like identity component."""
    if not value.strip() or value in {".", ".."} or Path(value).is_absolute() or "/" in value or "\\" in value:
        msg = f"{name} must be a safe non-empty path component"
        raise ValueError(msg)


def _run_child(command_runner: CommandRunner, command: list[str], name: str) -> None:
    """Run one child and translate execution failures into the live error contract."""
    try:
        completed = command_runner(command, check=False)
    except Exception as error:
        msg = f"{name} command could not run: {type(error).__name__}: {error}"
        raise LiveCommissioningError(msg) from error
    returncode = getattr(completed, "returncode", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        msg = f"{name} command returned no integer return code"
        raise LiveCommissioningError(msg)
    if returncode != 0:
        msg = f"{name} command failed with return code {returncode}"
        raise LiveCommissioningError(msg)


def _load_result_json(path: Path) -> dict[str, object]:
    """Load one required result JSON mapping."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        msg = f"result JSON contract failed for {path}: {error}"
        raise LiveCommissioningError(msg) from error
    if not isinstance(payload, dict):
        msg = f"result JSON contract failed for {path}: expected a JSON object"
        raise LiveCommissioningError(msg)
    return payload


def _validated_result(
    sample: CapturedSample,
    summary: Mapping[str, object],
    audit: Mapping[str, object],
) -> tuple[str, bool, bool, bool, str]:
    """Validate Stage32/Stage18 agreement and return typed result values."""
    _validate_summary_identity(sample, summary)
    fusion_policy = audit.get("fusion_policy")
    if not isinstance(fusion_policy, Mapping):
        msg = "result contract failed: audit fusion_policy must be a mapping"
        raise LiveCommissioningError(msg)
    audit_contract = {
        "machine_status": audit.get("machine_status"),
        "inspection_complete": audit.get("inspection_complete"),
        "commissioning_only": fusion_policy.get("commissioning_only"),
        "production_release_allowed": fusion_policy.get("production_release_allowed"),
    }
    for field, audit_value in audit_contract.items():
        if summary.get(field) != audit_value:
            msg = f"result contract failed: summary/audit {field} mismatch"
            raise LiveCommissioningError(msg)

    machine_status = summary.get("machine_status")
    if not isinstance(machine_status, str) or not (
        machine_status in {"OK", "REVIEW"} or machine_status.startswith("NG_")
    ):
        msg = f"result contract failed: invalid machine_status {machine_status!r}"
        raise LiveCommissioningError(msg)
    inspection_complete = summary.get("inspection_complete")
    commissioning_only = summary.get("commissioning_only")
    production_release_allowed = summary.get("production_release_allowed")
    if not isinstance(inspection_complete, bool):
        msg = "result contract failed: inspection_complete must be bool"
        raise LiveCommissioningError(msg)
    if commissioning_only is not True or production_release_allowed is not False:
        msg = "result contract failed: commissioning policy flags are not locked"
        raise LiveCommissioningError(msg)

    expected_identity = {
        "part_id": sample.part_id,
        "capture_session": sample.capture_session,
        "group_id": sample.group_id,
        "hand": "right",
    }
    for field, expected in expected_identity.items():
        if audit.get(field) != expected:
            msg = f"result contract failed: audit {field} does not match captured sample"
            raise LiveCommissioningError(msg)
    try:
        audit_report = format_audit_report(audit)
    except ValueError as error:
        msg = f"result contract failed: invalid audit evidence: {error}"
        raise LiveCommissioningError(msg) from error
    return machine_status, inspection_complete, True, False, audit_report


def _validate_summary_identity(sample: CapturedSample, summary: Mapping[str, object]) -> None:
    """Require a runtime summary to identify the captured right-hand sample."""
    expected_identity = {
        "part_id": sample.part_id,
        "capture_session": sample.capture_session,
        "group_id": sample.group_id,
        "hand": "right",
    }
    for field, expected in expected_identity.items():
        if summary.get(field) != expected:
            msg = f"result contract failed: summary {field} does not match captured sample"
            raise LiveCommissioningError(msg)


def _format_template_stop_report(summary: Mapping[str, object], machine_status: str) -> str:
    """Validate and format template rows from a Stage32 short circuit."""
    results = summary.get("template_results")
    if not isinstance(results, list) or not results or not all(isinstance(row, Mapping) for row in results):
        msg = "result contract failed: template_results must be a non-empty list of mappings"
        raise LiveCommissioningError(msg)
    views = [row.get("view") for row in results]
    if views != list(CANONICAL_VIEWS[: len(views)]):
        msg = "result contract failed: template_results views must be a canonical prefix"
        raise LiveCommissioningError(msg)
    evaluated_views = summary.get("evaluated_views")
    if evaluated_views is not None and evaluated_views != views:
        msg = "result contract failed: evaluated_views does not match template_results"
        raise LiveCommissioningError(msg)

    lines: list[str] = []
    for index, row in enumerate(results):
        status = row.get("status")
        if status not in {"PASS", "REVIEW", "NG_TEMPLATE"}:
            msg = f"result contract failed: invalid template status {status!r}"
            raise LiveCommissioningError(msg)
        if index < len(results) - 1 and status not in {"PASS", "REVIEW"}:
            msg = "result contract failed: only the final template result may stop execution"
            raise LiveCommissioningError(msg)
        reason_value = row.get("reason")
        reason = reason_value.strip() if isinstance(reason_value, str) else ""
        numeric_values = tuple(row.get(field) for field in ("score", "low_threshold", "high_threshold"))
        numeric_missing = all(value is None for value in numeric_values)
        numeric_valid = all(
            not isinstance(value, bool) and isinstance(value, Real) and isfinite(value) for value in numeric_values
        )
        missing_numeric_allowed = status == "REVIEW" and numeric_missing and bool(reason)
        if not numeric_valid and not missing_numeric_allowed:
            msg = f"result contract failed: template numeric evidence is invalid for {views[index]}"
            raise LiveCommissioningError(msg)
        if status != "PASS" and not reason:
            msg = f"result contract failed: template reason is required for {views[index]}/{status}"
            raise LiveCommissioningError(msg)
        lines.append(
            f"{views[index]} | status={status} | score={numeric_values[0]} | "
            f"low={numeric_values[1]} | high={numeric_values[2]} | reason={reason_value}",
        )
    if results[-1].get("status") != machine_status:
        msg = "result contract failed: final template status does not match machine_status"
        raise LiveCommissioningError(msg)
    return "\n".join(lines)


def _validated_short_result(
    sample: CapturedSample,
    summary: Mapping[str, object],
) -> tuple[str, bool, bool, bool, str]:
    """Validate a real template-only Stage32 stop without requiring fusion audit."""
    _validate_summary_identity(sample, summary)
    machine_status = summary.get("machine_status")
    if machine_status not in {"NG_TEMPLATE", "REVIEW"}:
        msg = f"result contract failed: invalid short-circuit machine_status {machine_status!r}"
        raise LiveCommissioningError(msg)
    if summary.get("inspection_complete") is not False:
        msg = "result contract failed: short-circuit inspection_complete must be false"
        raise LiveCommissioningError(msg)
    if summary.get("commissioning_only") is not True or summary.get("production_release_allowed") is not False:
        msg = "result contract failed: commissioning policy flags are not locked"
        raise LiveCommissioningError(msg)
    audit_report = _format_template_stop_report(summary, machine_status)
    return machine_status, False, True, False, audit_report


def _normalize_diagnostic_rows(
    rows: list[dict[str, str | None]],
    required_fields: set[str],
    family: str,
) -> list[dict[str, str]]:
    """Require and strip every diagnostic cell before semantic validation."""
    normalized_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows, start=1):
        normalized_row: dict[str, str] = {}
        for field in required_fields:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                msg = f"diagnostic CSV contract failed for {family}: required {field} cell is empty in row {row_index}"
                raise LiveCommissioningError(msg)
            normalized_row[field] = value.strip()
        normalized_rows.append(normalized_row)
    return normalized_rows


def _validate_diagnostic_csv(path: Path, sample: CapturedSample, family: str) -> list[str]:
    """Validate one complete six-view diagnostic model-family CSV."""
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        msg = f"diagnostic CSV contract failed for {family}: {error}"
        raise LiveCommissioningError(msg) from error

    required_fields = {
        "part_id",
        "capture_session",
        "group_id",
        "hand",
        "view",
        "branch",
        "score",
        "source_path",
        "evidence_path",
        "manifest_identity",
    }
    missing_fields = required_fields - fieldnames
    if missing_fields:
        msg = f"diagnostic CSV contract failed for {family}: missing fields {sorted(missing_fields)}"
        raise LiveCommissioningError(msg)
    normalized_rows = _normalize_diagnostic_rows(rows, required_fields, family)
    views = [row["view"] for row in normalized_rows]
    if len(rows) != len(CANONICAL_VIEWS) or len(set(views)) != len(views) or set(views) != set(CANONICAL_VIEWS):
        msg = f"diagnostic CSV contract failed for {family}: views must be exactly {CANONICAL_VIEWS}"
        raise LiveCommissioningError(msg)

    expected_identity = {
        "part_id": sample.part_id,
        "capture_session": sample.capture_session,
        "group_id": sample.group_id,
        "hand": "right",
    }
    report_lines: list[str] = []
    rows_by_view = {row["view"]: row for row in normalized_rows}
    for view in CANONICAL_VIEWS:
        row = rows_by_view[view]
        for field, expected in expected_identity.items():
            if row[field] != expected:
                msg = f"diagnostic CSV contract failed for {family}: {field} mismatch for {view}"
                raise LiveCommissioningError(msg)
        expected_branch = f"anomaly_{view}" if family == "patchcore" else "yolo"
        if row["branch"] != expected_branch:
            msg = f"diagnostic CSV contract failed for {family}: branch mismatch for {view}"
            raise LiveCommissioningError(msg)
        try:
            score = float(row["score"])
        except ValueError as error:
            msg = f"diagnostic CSV contract failed for {family}: score is not numeric for {view}"
            raise LiveCommissioningError(msg) from error
        if not isfinite(score):
            msg = f"diagnostic CSV contract failed for {family}: score is not finite for {view}"
            raise LiveCommissioningError(msg)
        for field in ("source_path", "evidence_path"):
            artifact = Path(row[field]).expanduser()
            if not artifact.is_absolute() or not artifact.is_file():
                msg = f"diagnostic CSV contract failed for {family}: {field} is not an existing file for {view}"
                raise LiveCommissioningError(msg)
            if field == "source_path" and artifact.resolve() != sample.images[view].resolve():
                msg = f"diagnostic CSV contract failed for {family}: source_path mismatch for {view}"
                raise LiveCommissioningError(msg)
        expected_manifest_identity = f"{sample.part_id}:right:{view}"
        if row["manifest_identity"] != expected_manifest_identity:
            msg = f"diagnostic CSV contract failed for {family}: manifest_identity mismatch for {view}"
            raise LiveCommissioningError(msg)
        report_lines.append(f"{view} | {expected_branch} | score={score}")
    return report_lines


def _reject_forbidden_diagnostic_artifacts(output_dir: Path) -> None:
    """Reject any template, fusion, or audit artifact from diagnostic infer."""
    try:
        artifacts = tuple(output_dir.rglob("*"))
    except OSError as error:
        msg = f"diagnostic result contract failed: could not inspect output artifacts: {error}"
        raise LiveCommissioningError(msg) from error
    for artifact in artifacts:
        relative_parts = artifact.relative_to(output_dir).parts
        if any(token in part.lower() for part in relative_parts for token in ("template", "fusion", "audit")):
            msg = f"forbidden diagnostic artifact: {artifact}"
            raise LiveCommissioningError(msg)


def _validated_diagnostic_result(
    sample: CapturedSample,
    summary: Mapping[str, object],
    output_dir: Path,
) -> tuple[str, bool, bool, bool, str, Path, Path]:
    """Validate explicit diagnostic policy and both complete model CSVs."""
    expected_summary = {
        "machine_status": "REVIEW",
        "inspection_complete": False,
        "strict_fusion": False,
        "commissioning_only": True,
        "production_release_allowed": False,
        "diagnostic_skip_template": True,
        "part_id": sample.part_id,
        "capture_session": sample.capture_session,
        "group_id": sample.group_id,
        "hand": "right",
    }
    for field, expected in expected_summary.items():
        actual = summary.get(field)
        value_matches = actual is expected if isinstance(expected, bool) else actual == expected
        if not value_matches:
            msg = f"diagnostic result contract failed: {field} must be {expected!r}"
            raise LiveCommissioningError(msg)
    if summary.get("errors") != []:
        msg = "diagnostic result contract failed: errors must be an empty list"
        raise LiveCommissioningError(msg)
    missing_required_evidence = summary.get("missing_required_evidence")
    if not isinstance(missing_required_evidence, list) or "complete_model_evidence" in missing_required_evidence:
        msg = (
            "diagnostic result contract failed: missing_required_evidence must be a list without "
            "complete_model_evidence"
        )
        raise LiveCommissioningError(msg)

    csv_paths: dict[str, Path] = {}
    for family in ("patchcore", "yolo"):
        field = f"{family}_csv"
        value = summary.get(field)
        if not isinstance(value, str) or not value.strip():
            msg = f"diagnostic result contract failed: {field} must be an absolute path"
            raise LiveCommissioningError(msg)
        path = Path(value).expanduser()
        expected_path = (output_dir / f"{family}.csv").resolve()
        if not path.is_absolute() or path.resolve() != expected_path:
            msg = f"diagnostic result contract failed: {field} must be {expected_path}"
            raise LiveCommissioningError(msg)
        csv_paths[family] = expected_path

    _reject_forbidden_diagnostic_artifacts(output_dir)
    report_lines = [
        *_validate_diagnostic_csv(csv_paths["patchcore"], sample, "patchcore"),
        *_validate_diagnostic_csv(csv_paths["yolo"], sample, "yolo"),
    ]
    return "REVIEW", False, True, False, "\n".join(report_lines), csv_paths["patchcore"], csv_paths["yolo"]


def _validate_captured_sample(config: LiveRunConfig, sample: CapturedSample) -> None:
    """Require the manifest identity to be the exact requested first sample."""
    _validate_path_component(sample.part_id, "manifest part_id")
    _validate_path_component(sample.capture_session, "manifest capture_session")
    _validate_path_component(sample.group_id, "manifest group_id")
    if sample.group_id != "group001":
        msg = f"manifest group_id must be group001, found {sample.group_id!r}"
        raise ValueError(msg)
    expected_part_id = f"{config.part_id}_group001_000001"
    if sample.part_id != expected_part_id:
        msg = f"manifest part_id must be {expected_part_id!r}, found {sample.part_id!r}"
        raise ValueError(msg)


def _load_verified_sample(config: LiveRunConfig, capture_run_root: Path) -> CapturedSample:
    """Load one manifest and translate all filesystem and identity errors."""
    try:
        sample = load_complete_sample(find_single_manifest(capture_run_root), config.topology_path)
        _validate_captured_sample(config, sample)
    except (OSError, UnicodeError, ValueError) as error:
        msg = f"capture manifest contract failed: {error}"
        raise LiveCommissioningError(msg) from error
    return sample


def _preflight_runtime(config: LiveRunConfig) -> RuntimeBundle:
    """Validate the exact topology and bundle bindings before opening cameras."""
    try:
        topology = load_topology(config.topology_path.expanduser().resolve())
        if len(topology.camera_slots) != 4 or tuple(topology.required_views) != VIEW_ORDER:
            raise ValueError("capture topology must define exactly four cameras and the canonical eight views")
        bundle = load_runtime_bundle(config.runtime_config)
        validate_commissioning_source_assets(
            bundle.threshold_artifact,
            bundle.runtime_assets,
            bundle.template_model_dir / "model.json",
        )
    except (OSError, TypeError, ValueError) as error:
        msg = f"runtime bundle preflight failed: {error}"
        raise LiveCommissioningError(msg) from error
    return bundle


def run_live_commissioning(
    config: LiveRunConfig,
    command_runner: CommandRunner = subprocess.run,
) -> LiveRunResult:
    """Capture and inspect one right-hand part using fail-closed child contracts."""
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
                    f"inspection failed and failed progress could not be written: {progress_error}"
                ) from error
        raise


def _run_live_commissioning(
    config: LiveRunConfig,
    command_runner: CommandRunner,
) -> LiveRunResult:
    """Run the live child sequence after the fail-closed progress wrapper."""
    _preflight_runtime(config)
    capture_root = config.capture_root.expanduser().resolve()
    capture_run_root = (capture_root / config.run_id).resolve()
    if capture_run_root.exists():
        msg = f"capture run root already exists: {capture_run_root}"
        raise LiveCommissioningError(msg)
    try:
        capture_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        msg = f"could not create capture root {capture_root}: {error}"
        raise LiveCommissioningError(msg) from error

    _run_child(command_runner, build_capture_command(config, capture_run_root), "capture")
    sample = _load_verified_sample(config, capture_run_root)

    output_dir = (config.output_root.expanduser().resolve() / sample.capture_session / sample.part_id).resolve()
    if output_dir.exists():
        msg = f"output directory already exists: {output_dir}"
        raise LiveCommissioningError(msg)
    _run_child(command_runner, build_stage32_command(config, sample, output_dir), "Stage32")

    runtime_summary_path = output_dir / "runtime_summary.json"
    audit_path = output_dir / "fusion" / "audit" / f"{sample.part_id}.json"
    summary = _load_result_json(runtime_summary_path)
    if config.diagnostic_skip_template:
        (
            machine_status,
            inspection_complete,
            commissioning_only,
            production_release_allowed,
            audit_report,
            patchcore_csv,
            yolo_csv,
        ) = _validated_diagnostic_result(sample, summary, output_dir)
        result_audit_path = None
        diagnostic_skip_template = True
    elif summary.get("short_circuited") is True:
        machine_status, inspection_complete, commissioning_only, production_release_allowed, audit_report = (
            _validated_short_result(sample, summary)
        )
        result_audit_path = None
        diagnostic_skip_template = False
        patchcore_csv = None
        yolo_csv = None
    else:
        audit = _load_result_json(audit_path)
        machine_status, inspection_complete, commissioning_only, production_release_allowed, audit_report = (
            _validated_result(sample, summary, audit)
        )
        result_audit_path = audit_path
        diagnostic_skip_template = False
        patchcore_csv = None
        yolo_csv = None
    result = LiveRunResult(
        sample=sample,
        capture_run_root=capture_run_root,
        output_dir=output_dir,
        runtime_summary_path=runtime_summary_path,
        audit_path=result_audit_path,
        machine_status=machine_status,
        inspection_complete=inspection_complete,
        commissioning_only=commissioning_only,
        production_release_allowed=production_release_allowed,
        audit_report=audit_report,
        diagnostic_skip_template=diagnostic_skip_template,
        patchcore_csv=patchcore_csv,
        yolo_csv=yolo_csv,
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
    """Find the only CSV manifest below a capture root.

    Args:
        root (Path): Capture root containing a ``manifests`` directory, or the
            manifest directory itself.

    Returns:
        Path: Absolute path to the only manifest.

    Raises:
        ValueError: If the root does not contain exactly one CSV manifest.
    """
    resolved_root = root.expanduser().resolve()
    manifest_dir = resolved_root / "manifests"
    search_dir = manifest_dir if manifest_dir.is_dir() else resolved_root
    manifests = sorted(path.resolve() for path in search_dir.glob("*.csv") if path.is_file())
    if len(manifests) != 1:
        msg = f"expected exactly one manifest in {search_dir}, found {len(manifests)}"
        raise ValueError(msg)
    return manifests[0]


def _shared_identity(rows: list[dict[str, str]], field: str) -> str:
    """Return one non-empty identity shared by every manifest row."""
    values = {row.get(field, "").strip() for row in rows}
    if "" in values or len(values) != 1:
        msg = f"manifest rows must share one non-empty {field}"
        raise ValueError(msg)
    return values.pop()


def load_complete_sample(path: Path, topology_path: Path | None = None) -> CapturedSample:
    """Load one complete topology-bound eight-view sample from a capture manifest.

    Args:
        path (Path): CSV manifest path.

    Returns:
        CapturedSample: Verified identities and absolute image paths.

    Raises:
        ValueError: If any row, identity, view, or file is ambiguous or
            incomplete.
    """
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        msg = f"manifest is not a file: {manifest_path}"
        raise ValueError(msg)
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        msg = f"manifest has no rows: {manifest_path}"
        raise ValueError(msg)
    if any(row.get("sample_status", "").strip() != "complete" for row in rows):
        msg = "every manifest row must have sample_status=complete"
        raise ValueError(msg)

    topology = load_topology(
        topology_path
        or Path(__file__).resolve().parents[1] / "configs/zs32/topology/zs32_4cam_double_side_v1.json",
    )
    if len(topology.camera_slots) != 4 or tuple(topology.required_views) != VIEW_ORDER:
        msg = "capture topology must define exactly four cameras and the canonical eight views"
        raise ValueError(msg)

    image_rows = [row for row in rows if row.get("record_type", "").strip() == "image"]
    sample_rows = [row for row in rows if row.get("record_type", "").strip() == "sample"]
    if len(image_rows) != len(VIEW_ORDER) or len(sample_rows) != 1 or len(rows) != len(image_rows) + 1:
        msg = "manifest must contain eight image rows and exactly one sample row"
        raise ValueError(msg)

    capture_session = _shared_identity(rows, "session_id")
    part_id = _shared_identity(rows, "sample_id")
    group_id = _shared_identity(rows, "group_id")

    views = [row.get("view", "").strip() for row in image_rows]
    if len(set(views)) != len(VIEW_ORDER) or set(views) != set(VIEW_ORDER):
        msg = f"image rows must contain exactly the canonical views: {VIEW_ORDER}"
        raise ValueError(msg)

    rows_by_view = {row["view"].strip(): row for row in image_rows}
    images: dict[str, Path] = {}
    capture_root = manifest_path.parent.parent
    for view in VIEW_ORDER:
        expected_round, _slot_id, expected_serial = topology.binding_for_view(view)
        row = rows_by_view[view]
        if row.get("round", "").strip() != expected_round:
            msg = f"manifest round for {view} must be {expected_round!r}"
            raise ValueError(msg)
        if row.get("camera_serial", "").strip() != expected_serial:
            msg = f"manifest camera_serial for {view} must be {expected_serial!r}"
            raise ValueError(msg)
        file_value = rows_by_view[view].get("file", "").strip()
        image_path = Path(file_value).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            msg = f"image file for {view} does not exist: {image_path}"
            raise ValueError(msg)
        expected_directory = (capture_root / "right" / view / "normal" / capture_session / "images").resolve()
        expected_prefix = f"right_{view}_normal_"
        if (
            image_path.parent != expected_directory
            or not image_path.name.startswith(expected_prefix)
            or image_path.suffix != ".png"
        ):
            msg = f"image file for {view} does not match the right-hand capture layout: {image_path}"
            raise ValueError(msg)
        images[view] = image_path

    return CapturedSample(
        part_id=part_id,
        capture_session=capture_session,
        group_id=group_id,
        images=MappingProxyType(images),
        manifest_path=manifest_path,
    )


def format_audit_report(audit: Mapping[str, object]) -> str:
    """Format Stage-18 audit evidence in canonical view and branch order.

    Args:
        audit (Mapping[str, object]): Stage-18 audit containing a ``views``
            mapping of branch rows.

    Returns:
        str: One stable, pipe-delimited line per view and branch.

    Raises:
        ValueError: If the audit omits, duplicates, or adds a required view or
            branch.
    """
    views = audit.get("views")
    if not isinstance(views, Mapping) or set(views) != set(CANONICAL_VIEWS):
        msg = f"audit views must be exactly the canonical views: {CANONICAL_VIEWS}"
        raise ValueError(msg)

    lines: list[str] = []
    for view in CANONICAL_VIEWS:
        raw_rows = views[view]
        if not isinstance(raw_rows, list) or not all(isinstance(row, Mapping) for row in raw_rows):
            msg = f"audit rows for {view} must be a list of mappings"
            raise ValueError(msg)
        rows_by_branch = {str(row.get("branch", "")): row for row in raw_rows}
        branch_order = ("template_match", f"anomaly_{view}", "yolo")
        if len(raw_rows) != len(branch_order) or set(rows_by_branch) != set(branch_order):
            msg = f"audit branches for {view} must be exactly {branch_order}"
            raise ValueError(msg)
        for branch in branch_order:
            row = rows_by_branch[branch]
            for field in ("score", "low_threshold", "high_threshold"):
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                    msg = f"audit {field} for {view}/{branch} must be a finite non-bool number"
                    raise ValueError(msg)
            evidence_level = row.get("computed_evidence_level")
            if evidence_level not in {"CLEAR", "GRAY", "STRONG"}:
                msg = f"audit computed_evidence_level for {view}/{branch} must be CLEAR, GRAY, or STRONG"
                raise ValueError(msg)
            fields = " | ".join(
                f"{field}={row.get(field)}"
                for field in (
                    "score",
                    "low_threshold",
                    "high_threshold",
                    "computed_evidence_level",
                )
            )
            lines.append(f"{view} | {branch} | {fields}")
    return "\n".join(lines)
