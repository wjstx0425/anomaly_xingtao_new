# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for live ZS32 manifest loading and audit formatting."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import capture_data.zs32_live_commissioning as live_commissioning
import pytest
from capture_data.zs32_inspection_orchestrator import CANONICAL_VIEWS
from capture_data.zs32_live_commissioning import (
    LiveCommissioningError,
    LiveRunConfig,
    LiveRunResult,
    build_capture_command,
    build_stage32_command,
    find_single_manifest,
    format_audit_report,
    load_complete_sample,
    run_live_commissioning,
)
from capture_data.zs32_runtime_bundle import load_runtime_bundle
from zs32_inspection.config.loaders import load_topology
from zs32_inspection.domain.views import VIEW_ORDER

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def complete_manifest(tmp_path: Path) -> Path:
    """Write one complete topology-bound eight-view capture manifest."""
    manifest = tmp_path / "manifests" / "20260714_120000_000001.csv"
    manifest.parent.mkdir()
    rows: list[dict[str, str]] = []
    topology = load_topology(Path(__file__).parents[3] / "configs/zs32/topology/zs32_4cam_double_side_v1.json")
    for view in VIEW_ORDER:
        round_id, _slot_id, serial = topology.binding_for_view(view)
        image = (
            tmp_path
            / "right"
            / view
            / "normal"
            / "20260714_120000_000001"
            / "images"
            / f"right_{view}_normal_live_part_001_group001_000001_single.png"
        )
        image.parent.mkdir(parents=True)
        image.write_bytes(view.encode())
        rows.append(
            {
                "record_type": "image",
                "session_id": "20260714_120000_000001",
                "sample_id": "live_part_001_group001_000001",
                "group_id": "group001",
                "round": round_id,
                "view": view,
                "camera_serial": serial,
                "file": str(image),
                "sample_status": "complete",
            },
        )
    rows.append(
        {
            "record_type": "sample",
            "session_id": "20260714_120000_000001",
            "sample_id": "live_part_001_group001_000001",
            "group_id": "group001",
            "view": "",
            "file": "",
            "sample_status": "complete",
        },
    )
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


@pytest.fixture
def live_config(tmp_path: Path) -> LiveRunConfig:
    """Return one fully locked live commissioning configuration."""
    repo_root = Path(__file__).parents[3].resolve()
    return LiveRunConfig(
        repo_root=repo_root,
        part_id="live_part_001",
        run_id="20260714_120000_000001",
        capture_root=tmp_path / "captures",
        output_root=tmp_path / "outputs",
        runtime_config=repo_root / "results/zs32_runtime_bundle_eight_view_v2/runtime_bundle.json",
        topology_path=repo_root / "configs/zs32/topology/zs32_4cam_double_side_v1.json",
    )


def test_build_capture_command_is_exact_and_uses_locked_topology(live_config: LiveRunConfig) -> None:
    """Capture uses the topology bootstrap with no serial override surface."""
    capture_run_root = (live_config.capture_root / live_config.run_id).resolve()

    command = build_capture_command(live_config, capture_run_root)

    assert command == [
        sys.executable,
        str((live_config.repo_root / "pipeline/zs32_bootstrap_capture.py").resolve()),
        "--topology",
        str(live_config.topology_path.resolve()),
        "--root",
        str(capture_run_root),
        "--capture-session",
        live_config.run_id,
        "--legacy-layout",
        "--hand",
        "right",
        "--label",
        "normal",
        "--part-id",
        "live_part_001",
        "--group-count",
        "1",
        "--images-per-group",
        "1",
        "--manual-load",
        "--hdr",
    ]


def test_build_stage32_command_is_exact_without_production_only_inputs(
    live_config: LiveRunConfig,
    complete_manifest: Path,
) -> None:
    """Stage32 receives all eight images and only assets resolved from the bundle."""
    sample = load_complete_sample(complete_manifest)
    output_dir = (live_config.output_root / sample.capture_session / sample.part_id).resolve()

    command = build_stage32_command(live_config, sample, output_dir)

    expected = [
        sys.executable,
        str((live_config.repo_root / "pipeline/32_run_zs32_multimodel_inference.py").resolve()),
        "fuse",
        "--fusion-config",
        str(load_runtime_bundle(live_config.runtime_config).fusion_profile),
        "--part-id",
        sample.part_id,
        "--capture-session",
        sample.capture_session,
        "--group-id",
        sample.group_id,
        "--hand",
        "right",
    ]
    for view in VIEW_ORDER:
        expected.extend((f"--{view.replace('_', '-')}-image", str(sample.images[view])))
    expected.extend(
        (
            "--runtime-config",
            str(load_runtime_bundle(live_config.runtime_config).runtime_assets),
            "--template-model-dir",
            str(load_runtime_bundle(live_config.runtime_config).template_model_dir),
            "--threshold-artifact",
            str(load_runtime_bundle(live_config.runtime_config).threshold_artifact),
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--yolo-device",
            "0",
            "--output-dir",
            str(output_dir),
        ),
    )
    assert command == expected
    assert not {"--quality-csv", "--registration-csv", "--geometry-csv"}.intersection(command)


def test_build_stage32_command_is_exact_for_skip_template_diagnostic(
    live_config: LiveRunConfig,
    complete_manifest: Path,
) -> None:
    """Diagnostic Stage32 still uses the bundle's eight-view runtime assets."""
    config = replace(live_config, diagnostic_skip_template=True)
    sample = load_complete_sample(complete_manifest)
    output_dir = (config.output_root / sample.capture_session / sample.part_id).resolve()

    command = build_stage32_command(config, sample, output_dir)

    expected = [
        sys.executable,
        str((config.repo_root / "pipeline/32_run_zs32_multimodel_inference.py").resolve()),
        "infer",
        "--diagnostic-skip-template",
        "--part-id",
        sample.part_id,
        "--capture-session",
        sample.capture_session,
        "--group-id",
        sample.group_id,
        "--hand",
        "right",
    ]
    bundle = load_runtime_bundle(config.runtime_config)
    for view in VIEW_ORDER:
        expected.extend((f"--{view.replace('_', '-')}-image", str(sample.images[view])))
    expected.extend(
        (
            "--runtime-config",
            str(bundle.runtime_assets),
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--yolo-device",
            "0",
            "--output-dir",
            str(output_dir),
        ),
    )
    assert command == expected
    assert not {
        "fuse",
        "--fusion-profile",
        "--template-model-dir",
        "--threshold-artifact",
    }.intersection(command)


def _command_value(command: list[str], option: str) -> str:
    """Return the value following one command option."""
    return command[command.index(option) + 1]


def _write_live_manifest(
    capture_root: Path,
    *,
    sample_id: str = "live_part_001_group001_000001",
    session_id: str = "20260714_121500_000001",
    group_id: str = "group001",
) -> Path:
    """Write the complete manifest emitted by the fake capture child."""
    manifest = capture_root / "manifests" / "20260714_121500_000001.csv"
    manifest.parent.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    topology = load_topology(Path(__file__).parents[3] / "configs/zs32/topology/zs32_4cam_double_side_v1.json")
    for view in VIEW_ORDER:
        round_id, _slot_id, serial = topology.binding_for_view(view)
        image_dir = capture_root / "right" / view / "normal" / session_id / "images"
        image = image_dir / f"right_{view}_normal_capture_single.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(view.encode())
        rows.append(
            {
                "record_type": "image",
                "session_id": session_id,
                "sample_id": sample_id,
                "group_id": group_id,
                "round": round_id,
                "view": view,
                "camera_serial": serial,
                "file": str(image),
                "sample_status": "complete",
            },
        )
    rows.append(
        {
            "record_type": "sample",
            "session_id": session_id,
            "sample_id": sample_id,
            "group_id": group_id,
            "view": "",
            "file": "",
            "sample_status": "complete",
        },
    )
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _valid_live_outputs(command: list[str], status: str = "OK") -> tuple[Path, Path]:
    """Write mutually agreeing Stage32 summary and Stage18 audit JSON."""
    output_dir = Path(_command_value(command, "--output-dir"))
    output_dir.mkdir(parents=True)
    shared = {
        "machine_status": status,
        "inspection_complete": True,
        "commissioning_only": True,
        "production_release_allowed": False,
        "part_id": _command_value(command, "--part-id"),
        "capture_session": _command_value(command, "--capture-session"),
        "group_id": _command_value(command, "--group-id"),
        "hand": "right",
    }
    summary_path = output_dir / "runtime_summary.json"
    summary_path.write_text(json.dumps(shared), encoding="utf-8")
    audit = {
        **_complete_audit(),
        "machine_status": status,
        "inspection_complete": True,
        "hand": "right",
        "fusion_policy": {
            "commissioning_only": True,
            "production_release_allowed": False,
        },
        "part_id": _command_value(command, "--part-id"),
        "capture_session": _command_value(command, "--capture-session"),
        "group_id": _command_value(command, "--group-id"),
    }
    audit_path = output_dir / "fusion" / "audit" / f"{audit['part_id']}.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return summary_path, audit_path


def _valid_short_output(
    command: list[str],
    status: str,
    *,
    result: dict[str, object] | None = None,
) -> Path:
    """Write the real Stage32 template-short-circuit summary with no fusion audit."""
    output_dir = Path(_command_value(command, "--output-dir"))
    output_dir.mkdir(parents=True)
    if result is None:
        result = {
            "view": "front",
            "status": status,
            "score": 0.9,
            "low_threshold": 0.2,
            "high_threshold": 0.8,
            "reason": "template threshold stop",
        }
    summary = {
        "machine_status": status,
        "inspection_complete": False,
        "short_circuited": True,
        "stopped_after": "template_match",
        "part_id": _command_value(command, "--part-id"),
        "capture_session": _command_value(command, "--capture-session"),
        "group_id": _command_value(command, "--group-id"),
        "hand": "right",
        "template_results": [result],
        "commissioning_only": True,
        "production_release_allowed": False,
    }
    summary_path = output_dir / "runtime_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def _write_diagnostic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write one diagnostic branch CSV using the fields Stage35 validates."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _valid_diagnostic_outputs(
    command: list[str],
) -> tuple[Path, Path, Path, dict[str, list[dict[str, str]]]]:
    """Write the diagnostic summary and complete PatchCore/YOLO evidence CSVs."""
    output_dir = Path(_command_value(command, "--output-dir"))
    output_dir.mkdir(parents=True)
    part_id = _command_value(command, "--part-id")
    capture_session = _command_value(command, "--capture-session")
    group_id = _command_value(command, "--group-id")
    rows_by_family: dict[str, list[dict[str, str]]] = {"patchcore": [], "yolo": []}
    for family in rows_by_family:
        for index, view in enumerate(CANONICAL_VIEWS):
            evidence_path = output_dir / "evidence" / family / f"{view}.png"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_bytes(f"{family}:{view}".encode())
            rows_by_family[family].append(
                {
                    "part_id": part_id,
                    "capture_session": capture_session,
                    "group_id": group_id,
                    "hand": "right",
                    "view": view,
                    "branch": f"anomaly_{view}" if family == "patchcore" else "yolo",
                    "score": str(index / 10),
                    "source_path": _command_value(command, f"--{view.replace('_', '-')}-image"),
                    "evidence_path": str(evidence_path.resolve()),
                    "manifest_identity": f"{part_id}:right:{view}",
                },
            )
    patchcore_csv = (output_dir / "patchcore.csv").resolve()
    yolo_csv = (output_dir / "yolo.csv").resolve()
    _write_diagnostic_csv(patchcore_csv, rows_by_family["patchcore"])
    _write_diagnostic_csv(yolo_csv, rows_by_family["yolo"])
    summary = {
        "machine_status": "REVIEW",
        "inspection_complete": False,
        "strict_fusion": False,
        "commissioning_only": True,
        "production_release_allowed": False,
        "diagnostic_skip_template": True,
        "part_id": part_id,
        "capture_session": capture_session,
        "group_id": group_id,
        "hand": "right",
        "patchcore_csv": str(patchcore_csv),
        "yolo_csv": str(yolo_csv),
        "errors": [],
        "missing_required_evidence": ["template_match", "strict_fusion_not_run"],
    }
    summary_path = output_dir / "runtime_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, patchcore_csv, yolo_csv, rows_by_family


def _run_diagnostic(
    live_config: LiveRunConfig,
    mutate: Callable[[Path, Path, Path, dict[str, list[dict[str, str]]]], None] | None = None,
) -> LiveRunResult:
    """Run diagnostic commissioning with optional post-child artifact corruption."""
    invocation = 0

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            _write_live_manifest(Path(_command_value(command, "--root")))
        else:
            outputs = _valid_diagnostic_outputs(command)
            if mutate is not None:
                mutate(*outputs)
        return SimpleNamespace(returncode=0)

    return run_live_commissioning(replace(live_config, diagnostic_skip_template=True), runner)


def test_run_rejects_capture_child_failure(live_config: LiveRunConfig) -> None:
    """A capture process failure is distinct from every downstream contract error."""
    commands: list[list[str]] = []

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        commands.append(command)
        assert check is False
        return SimpleNamespace(returncode=23)

    with pytest.raises(LiveCommissioningError, match=r"capture command failed.*23"):
        run_live_commissioning(live_config, runner)
    assert len(commands) == 1


def test_run_rejects_bundle_preflight_before_capture(
    live_config: LiveRunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundle drift must stop Stage35 before a camera command can run."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        live_commissioning,
        "validate_commissioning_source_assets",
        lambda *_args: (_ for _ in ()).throw(ValueError("bundle drift")),
    )

    with pytest.raises(LiveCommissioningError, match=r"runtime bundle preflight failed.*bundle drift"):
        run_live_commissioning(
            live_config,
            lambda command, **_kwargs: calls.append(command) or SimpleNamespace(returncode=0),
        )

    assert calls == []


@pytest.mark.parametrize("field", ["part_id", "run_id"])
@pytest.mark.parametrize("value", ["", ".", "..", "nested/id", "nested\\id"])
def test_run_rejects_unsafe_config_identity(live_config: LiveRunConfig, field: str, value: str) -> None:
    """Configured identities must be safe non-empty path components."""
    unsafe = replace(live_config, **{field: value})

    with pytest.raises(LiveCommissioningError, match=field):
        run_live_commissioning(unsafe, lambda _command, **_kwargs: SimpleNamespace(returncode=0))


def test_run_rejects_preexisting_capture_run_root(live_config: LiveRunConfig) -> None:
    """A run ID cannot reuse or overwrite an existing capture directory."""
    (live_config.capture_root / live_config.run_id).mkdir(parents=True)

    with pytest.raises(LiveCommissioningError, match="capture run root already exists"):
        run_live_commissioning(live_config, lambda _command, **_kwargs: SimpleNamespace(returncode=0))


def test_run_rejects_invalid_capture_manifest(live_config: LiveRunConfig) -> None:
    """A zero capture return code cannot bypass the complete-manifest contract."""
    with pytest.raises(LiveCommissioningError, match="capture manifest contract failed"):
        run_live_commissioning(live_config, lambda _command, **_kwargs: SimpleNamespace(returncode=0))


def test_run_wraps_manifest_unicode_error(live_config: LiveRunConfig) -> None:
    """Undecodable capture manifests remain a live contract error."""

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        manifest = Path(_command_value(command, "--root")) / "manifests" / "bad.csv"
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(b"\xff")
        return SimpleNamespace(returncode=0)

    with pytest.raises(LiveCommissioningError, match="capture manifest contract failed"):
        run_live_commissioning(live_config, runner)


def test_run_wraps_manifest_os_error(live_config: LiveRunConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifest filesystem failures are translated into the public live error."""

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        _write_live_manifest(Path(_command_value(command, "--root")))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        live_commissioning,
        "load_complete_sample",
        lambda _path, _topology: (_ for _ in ()).throw(OSError("io")),
    )

    with pytest.raises(LiveCommissioningError, match=r"capture manifest contract failed.*io"):
        run_live_commissioning(live_config, runner)


@pytest.mark.parametrize(
    ("field", "value"),
    [("sample_id", "../escape"), ("session_id", "../escape"), ("group_id", "../escape")],
)
def test_run_rejects_path_like_manifest_identity(
    live_config: LiveRunConfig,
    field: str,
    value: str,
) -> None:
    """Verified rows cannot inject path components into runtime output or audit paths."""
    identities = {
        "sample_id": "live_part_001_group001_000001",
        "session_id": "20260714_121500_000001",
        "group_id": "group001",
        field: value,
    }

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        _write_live_manifest(Path(_command_value(command, "--root")), **identities)
        return SimpleNamespace(returncode=0)

    with pytest.raises(LiveCommissioningError, match="capture manifest contract failed"):
        run_live_commissioning(live_config, runner)


@pytest.mark.parametrize(
    ("sample_id", "group_id"),
    [
        ("live_part_001_group002_000001", "group002"),
        ("another_part_group001_000001", "group001"),
    ],
)
def test_run_rejects_manifest_identity_that_does_not_match_capture_request(
    live_config: LiveRunConfig,
    sample_id: str,
    group_id: str,
) -> None:
    """One-row manifest identity must be the exact requested group001 sample."""

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        _write_live_manifest(Path(_command_value(command, "--root")), sample_id=sample_id, group_id=group_id)
        return SimpleNamespace(returncode=0)

    with pytest.raises(LiveCommissioningError, match="capture manifest contract failed"):
        run_live_commissioning(live_config, runner)


def test_run_rejects_preexisting_runtime_output(live_config: LiveRunConfig) -> None:
    """A live run never overwrites an earlier physical-part result."""
    existing = live_config.output_root / "20260714_121500_000001" / "live_part_001_group001_000001"
    existing.mkdir(parents=True)
    commands: list[list[str]] = []

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        commands.append(command)
        _write_live_manifest(Path(_command_value(command, "--root")))
        return SimpleNamespace(returncode=0)

    with pytest.raises(LiveCommissioningError, match="output directory already exists"):
        run_live_commissioning(live_config, runner)
    assert len(commands) == 1


def test_run_rejects_stage32_child_failure(live_config: LiveRunConfig) -> None:
    """A Stage32 process failure remains distinct from capture failures."""
    commands: list[list[str]] = []

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        commands.append(command)
        if len(commands) == 1:
            _write_live_manifest(Path(_command_value(command, "--root")))
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=37)

    with pytest.raises(LiveCommissioningError, match=r"Stage32 command failed.*37"):
        run_live_commissioning(live_config, runner)
    assert len(commands) == 2


@pytest.mark.parametrize(
    "bad_result",
    [
        "missing_summary",
        "invalid_summary",
        "undecodable_summary",
        "missing_audit",
        "invalid_audit",
        "undecodable_audit",
    ],
)
def test_run_rejects_missing_or_invalid_result_json(live_config: LiveRunConfig, bad_result: str) -> None:
    """Both child result JSON documents must exist and decode to mappings."""
    invocation = 0

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            _write_live_manifest(Path(_command_value(command, "--root")))
        else:
            summary_path, audit_path = _valid_live_outputs(command)
            if bad_result == "missing_summary":
                summary_path.unlink()
            elif bad_result == "invalid_summary":
                summary_path.write_text("[]", encoding="utf-8")
            elif bad_result == "undecodable_summary":
                summary_path.write_bytes(b"\xff")
            elif bad_result == "missing_audit":
                audit_path.unlink()
            elif bad_result == "invalid_audit":
                audit_path.write_text("{invalid", encoding="utf-8")
            else:
                audit_path.write_bytes(b"\xff")
        return SimpleNamespace(returncode=0)

    with pytest.raises(LiveCommissioningError, match="result JSON contract failed"):
        run_live_commissioning(live_config, runner)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("summary", "machine_status", "REVIEW"),
        ("summary", "part_id", "another-part"),
        ("summary", "capture_session", "another-session"),
        ("summary", "group_id", "another-group"),
        ("summary", "hand", "left"),
        ("audit", "inspection_complete", False),
        ("audit", "commissioning_only", False),
        ("audit", "production_release_allowed", True),
        ("audit", "part_id", "another-part"),
        ("audit", "capture_session", "another-session"),
        ("audit", "group_id", "another-group"),
        ("audit", "hand", "left"),
    ],
)
def test_run_rejects_summary_audit_or_sample_contract_mismatch(
    live_config: LiveRunConfig,
    target: str,
    field: str,
    value: object,
) -> None:
    """Results must agree with one another and with the captured sample identity."""
    invocation = 0

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            _write_live_manifest(Path(_command_value(command, "--root")))
        else:
            summary_path, audit_path = _valid_live_outputs(command)
            path = summary_path if target == "summary" else audit_path
            payload = json.loads(path.read_text(encoding="utf-8"))
            if target == "audit" and field in {"commissioning_only", "production_release_allowed"}:
                payload["fusion_policy"][field] = value
            else:
                payload[field] = value
            path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    with pytest.raises(LiveCommissioningError, match="result contract failed"):
        run_live_commissioning(live_config, runner)


@pytest.mark.parametrize("status", ["OK", "NG_PATCHCORE", "REVIEW"])
def test_run_preserves_valid_business_statuses(live_config: LiveRunConfig, status: str) -> None:
    """Business decisions return successfully and are never converted to execution failures."""
    invocation = 0

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            _write_live_manifest(Path(_command_value(command, "--root")))
        else:
            _valid_live_outputs(command, status)
        return SimpleNamespace(returncode=0)

    result = run_live_commissioning(live_config, runner)

    assert isinstance(result, LiveRunResult)
    assert result.machine_status == status
    assert result.inspection_complete is True
    assert result.commissioning_only is True
    assert result.production_release_allowed is False
    assert result.sample.part_id == "live_part_001_group001_000001"
    assert result.capture_run_root == (live_config.capture_root / live_config.run_id).resolve()
    assert (
        result.output_dir == (live_config.output_root / result.sample.capture_session / result.sample.part_id).resolve()
    )
    assert result.runtime_summary_path == result.output_dir / "runtime_summary.json"
    assert result.audit_path == result.output_dir / "fusion" / "audit" / f"{result.sample.part_id}.json"
    assert len(result.audit_report.splitlines()) == len(CANONICAL_VIEWS) * 3


def test_run_publishes_valid_skip_template_diagnostic_result(live_config: LiveRunConfig) -> None:
    """A complete diagnostic run is valid REVIEW evidence without template or fusion output."""
    result = _run_diagnostic(live_config)

    assert result.machine_status == "REVIEW"
    assert result.inspection_complete is False
    assert result.commissioning_only is True
    assert result.production_release_allowed is False
    assert result.diagnostic_skip_template is True
    assert result.audit_path is None
    assert result.patchcore_csv == (result.output_dir / "patchcore.csv").resolve()
    assert result.yolo_csv == (result.output_dir / "yolo.csv").resolve()
    assert result.patchcore_csv.is_absolute()
    assert result.yolo_csv.is_absolute()


@pytest.mark.parametrize(
    "mutate_rows",
    [
        pytest.param(lambda rows: rows.pop(), id="missing"),
        pytest.param(
            lambda rows: rows.append({**rows[0], "view": "extra", "branch": "anomaly_extra"}),
            id="extra",
        ),
        pytest.param(
            lambda rows: rows.__setitem__(1, {**rows[1], "view": rows[0]["view"], "branch": rows[0]["branch"]}),
            id="duplicate",
        ),
    ],
)
def test_run_rejects_diagnostic_csv_with_noncanonical_views(
    live_config: LiveRunConfig,
    mutate_rows: Callable[[list[dict[str, str]]], object],
) -> None:
    """Each diagnostic CSV must contain every canonical view exactly once."""

    def mutate(
        _summary: Path,
        patchcore_csv: Path,
        _yolo_csv: Path,
        rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        mutate_rows(rows_by_family["patchcore"])
        _write_diagnostic_csv(patchcore_csv, rows_by_family["patchcore"])

    with pytest.raises(LiveCommissioningError, match=r"diagnostic CSV contract failed.*views"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize("field", ["part_id", "capture_session", "group_id", "hand"])
def test_run_rejects_diagnostic_csv_with_wrong_sample_identity(
    live_config: LiveRunConfig,
    field: str,
) -> None:
    """Every diagnostic row must share the captured sample identity."""

    def mutate(
        _summary: Path,
        patchcore_csv: Path,
        _yolo_csv: Path,
        rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        rows_by_family["patchcore"][0][field] = "wrong"
        _write_diagnostic_csv(patchcore_csv, rows_by_family["patchcore"])

    with pytest.raises(LiveCommissioningError, match=rf"diagnostic CSV contract failed.*{field}"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize(("family", "branch"), [("patchcore", "yolo"), ("yolo", "anomaly_front")])
def test_run_rejects_diagnostic_csv_with_wrong_branch(
    live_config: LiveRunConfig,
    family: str,
    branch: str,
) -> None:
    """PatchCore rows are view-specific anomaly branches and YOLO rows are yolo."""

    def mutate(
        _summary: Path,
        patchcore_csv: Path,
        yolo_csv: Path,
        rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        rows_by_family[family][0]["branch"] = branch
        _write_diagnostic_csv(patchcore_csv if family == "patchcore" else yolo_csv, rows_by_family[family])

    with pytest.raises(LiveCommissioningError, match=r"diagnostic CSV contract failed.*branch"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize("score", ["nan", "inf", "-inf", "not-a-number"])
def test_run_rejects_diagnostic_csv_with_invalid_score(live_config: LiveRunConfig, score: str) -> None:
    """Diagnostic scores must parse as finite numbers."""

    def mutate(
        _summary: Path,
        patchcore_csv: Path,
        _yolo_csv: Path,
        rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        rows_by_family["patchcore"][0]["score"] = score
        _write_diagnostic_csv(patchcore_csv, rows_by_family["patchcore"])

    with pytest.raises(LiveCommissioningError, match=r"diagnostic CSV contract failed.*score"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize("missing_cell", ["view", "score"])
def test_run_wraps_truncated_diagnostic_csv_row(
    live_config: LiveRunConfig,
    missing_cell: str,
) -> None:
    """Header-complete CSVs with truncated required cells remain live contract errors."""

    def mutate(
        _summary: Path,
        patchcore_csv: Path,
        _yolo_csv: Path,
        _rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        with patchcore_csv.open(newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
        cutoff = rows[0].index(missing_cell)
        rows[1] = rows[1][:cutoff]
        with patchcore_csv.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows(rows)

    with pytest.raises(LiveCommissioningError, match=r"diagnostic CSV contract failed"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize("field", ["source_path", "evidence_path"])
def test_run_rejects_diagnostic_csv_with_missing_artifact(live_config: LiveRunConfig, field: str) -> None:
    """Every diagnostic row must retain an existing source and evidence file."""

    def mutate(
        _summary: Path,
        _patchcore_csv: Path,
        _yolo_csv: Path,
        rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        Path(rows_by_family["patchcore"][0][field]).unlink()

    with pytest.raises(LiveCommissioningError, match=rf"diagnostic CSV contract failed.*{field}"):
        _run_diagnostic(live_config, mutate)


def test_run_rejects_diagnostic_csv_with_wrong_manifest_identity(live_config: LiveRunConfig) -> None:
    """Manifest identity must bind the part, right hand, and current view."""

    def mutate(
        _summary: Path,
        patchcore_csv: Path,
        _yolo_csv: Path,
        rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        rows_by_family["patchcore"][0]["manifest_identity"] = "wrong:right:front"
        _write_diagnostic_csv(patchcore_csv, rows_by_family["patchcore"])

    with pytest.raises(LiveCommissioningError, match=r"diagnostic CSV contract failed.*manifest_identity"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize("family", ["patchcore", "yolo"])
def test_run_rejects_missing_diagnostic_csv(live_config: LiveRunConfig, family: str) -> None:
    """Both model-family CSVs are mandatory diagnostic evidence."""

    def mutate(
        _summary: Path,
        patchcore_csv: Path,
        yolo_csv: Path,
        _rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        (patchcore_csv if family == "patchcore" else yolo_csv).unlink()

    with pytest.raises(LiveCommissioningError, match=r"diagnostic CSV contract failed"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize("artifact", ["template", "fusion", "audit"])
def test_run_rejects_forbidden_diagnostic_artifact(live_config: LiveRunConfig, artifact: str) -> None:
    """Diagnostic mode may not publish template, fusion, or audit artifacts."""

    def mutate(
        summary_path: Path,
        _patchcore_csv: Path,
        _yolo_csv: Path,
        _rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        output_dir = summary_path.parent
        forbidden = {
            "template": output_dir / "template_match.csv",
            "fusion": output_dir / "fusion" / "result.csv",
            "audit": output_dir / "diagnostic_audit.json",
        }[artifact]
        forbidden.parent.mkdir(parents=True, exist_ok=True)
        forbidden.write_text("forbidden", encoding="utf-8")

    with pytest.raises(LiveCommissioningError, match="forbidden diagnostic artifact"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("machine_status", "OK"),
        ("inspection_complete", True),
        ("inspection_complete", 0),
        ("strict_fusion", True),
        ("strict_fusion", 0),
        ("commissioning_only", False),
        ("commissioning_only", 1),
        ("production_release_allowed", True),
        ("production_release_allowed", 0),
        ("diagnostic_skip_template", 1),
    ],
)
def test_run_rejects_wrong_diagnostic_policy_flag(
    live_config: LiveRunConfig,
    field: str,
    value: object,
) -> None:
    """The diagnostic summary policy is fixed to REVIEW and non-production."""

    def mutate(
        summary_path: Path,
        _patchcore_csv: Path,
        _yolo_csv: Path,
        _rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload[field] = value
        summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LiveCommissioningError, match=rf"diagnostic result contract failed.*{field}"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize(("field", "value"), [("errors", ["model failed"]), ("errors", None)])
def test_run_rejects_diagnostic_summary_errors(
    live_config: LiveRunConfig,
    field: str,
    value: object,
) -> None:
    """Complete diagnostic CSV evidence cannot coexist with runtime errors."""

    def mutate(
        summary_path: Path,
        _patchcore_csv: Path,
        _yolo_csv: Path,
        _rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload[field] = value
        summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LiveCommissioningError, match=r"diagnostic result contract failed.*errors"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize(
    "value",
    ["template_match", None, ["template_match", "complete_model_evidence"]],
)
def test_run_rejects_invalid_diagnostic_missing_evidence_summary(
    live_config: LiveRunConfig,
    value: object,
) -> None:
    """Missing-evidence metadata is a list and cannot report incomplete model evidence."""

    def mutate(
        summary_path: Path,
        _patchcore_csv: Path,
        _yolo_csv: Path,
        _rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["missing_required_evidence"] = value
        summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        LiveCommissioningError,
        match=r"diagnostic result contract failed.*missing_required_evidence",
    ):
        _run_diagnostic(live_config, mutate)


def test_run_rejects_non_diagnostic_summary_in_diagnostic_mode(live_config: LiveRunConfig) -> None:
    """The explicit Stage35 diagnostic request requires an explicit Stage32 diagnostic summary."""

    def mutate(
        summary_path: Path,
        _patchcore_csv: Path,
        _yolo_csv: Path,
        _rows_by_family: dict[str, list[dict[str, str]]],
    ) -> None:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        del payload["diagnostic_skip_template"]
        summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LiveCommissioningError, match=r"diagnostic result contract failed.*diagnostic_skip_template"):
        _run_diagnostic(live_config, mutate)


@pytest.mark.parametrize(
    ("status", "template_result"),
    [
        (
            "NG_TEMPLATE",
            {
                "view": "front",
                "status": "NG_TEMPLATE",
                "score": 0.9,
                "low_threshold": 0.2,
                "high_threshold": 0.8,
                "reason": "risk reached threshold",
            },
        ),
        (
            "REVIEW",
            {
                "view": "front",
                "status": "REVIEW",
                "reason": "template gate exception: ValueError: broken",
            },
        ),
    ],
)
def test_run_preserves_real_template_short_circuit_without_audit(
    live_config: LiveRunConfig,
    status: str,
    template_result: dict[str, object],
) -> None:
    """Real template stops return a verified short report without requiring a fusion audit."""
    invocation = 0

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            _write_live_manifest(Path(_command_value(command, "--root")))
        else:
            _valid_short_output(command, status, result=template_result)
        return SimpleNamespace(returncode=0)

    result = run_live_commissioning(live_config, runner)

    assert result.machine_status == status
    assert result.inspection_complete is False
    assert result.commissioning_only is True
    assert result.production_release_allowed is False
    assert result.audit_path is None
    assert result.audit_report == (
        f"front | status={status} | score={template_result.get('score')} | "
        f"low={template_result.get('low_threshold')} | high={template_result.get('high_threshold')} | "
        f"reason={template_result['reason']}"
    )


@pytest.mark.parametrize(
    "template_result",
    [
        {"view": "front", "status": "NG_TEMPLATE", "reason": "missing numeric"},
        {"view": "front", "status": "REVIEW", "reason": ""},
        {"view": "front", "status": "REVIEW"},
        {"view": "front", "status": "INVALID_CAPTURE", "reason": "not emitted by Stage32 template gate"},
        {"view": "front", "status": "NG_TEMPLATE", "score": True, "low_threshold": 0.2, "high_threshold": 0.8},
    ],
)
def test_run_rejects_invalid_template_short_report(
    live_config: LiveRunConfig,
    template_result: dict[str, object],
) -> None:
    """Short-circuit evidence remains fail-closed for numeric and exception records."""
    invocation = 0

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            _write_live_manifest(Path(_command_value(command, "--root")))
        else:
            _valid_short_output(command, str(template_result["status"]), result=template_result)
        return SimpleNamespace(returncode=0)

    with pytest.raises(LiveCommissioningError, match="result contract failed"):
        run_live_commissioning(live_config, runner)


def _rewrite(manifest: Path, mutate: Callable[[list[dict[str, str]]], object]) -> None:
    """Rewrite a fixture manifest after applying a row mutation callback."""
    with manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fieldnames = tuple(rows[0])
    mutate(rows)
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_find_single_manifest_and_load_exact_identity(complete_manifest: Path) -> None:
    """A complete manifest maps identities and canonical views exactly once."""
    manifest = find_single_manifest(complete_manifest.parents[1])

    sample = load_complete_sample(manifest)

    assert sample.part_id == "live_part_001_group001_000001"
    assert sample.capture_session == "20260714_120000_000001"
    assert sample.group_id == "group001"
    assert tuple(sample.images) == CANONICAL_VIEWS
    assert all(path.is_absolute() for path in sample.images.values())
    assert sample.manifest_path == complete_manifest.resolve()


@pytest.mark.parametrize("bad_view", [None, "front"])
def test_load_rejects_missing_or_duplicate_views(complete_manifest: Path, bad_view: str | None) -> None:
    """Every canonical view must occur exactly once."""
    if bad_view is None:
        _rewrite(complete_manifest, lambda rows: rows.pop(0))
    else:
        _rewrite(complete_manifest, lambda rows: rows.__setitem__(1, {**rows[1], "view": bad_view}))

    with pytest.raises(ValueError, match=r"eight image rows|canonical views"):
        load_complete_sample(complete_manifest)


@pytest.mark.parametrize("field", ["round", "camera_serial"])
def test_load_rejects_manifest_row_outside_topology_identity(complete_manifest: Path, field: str) -> None:
    """Each view must retain the round and physical serial bound by topology."""
    _rewrite(complete_manifest, lambda rows: rows.__setitem__(0, {**rows[0], field: "wrong"}))

    with pytest.raises(ValueError, match=field):
        load_complete_sample(complete_manifest)


@pytest.mark.parametrize("row_index", [0, 6])
def test_load_rejects_incomplete_image_or_sample_rows(complete_manifest: Path, row_index: int) -> None:
    """No row from an incomplete capture may enter live inference."""
    _rewrite(
        complete_manifest,
        lambda rows: rows.__setitem__(row_index, {**rows[row_index], "sample_status": "incomplete"}),
    )

    with pytest.raises(ValueError, match="complete"):
        load_complete_sample(complete_manifest)


@pytest.mark.parametrize("field", ["session_id", "sample_id", "group_id"])
def test_load_rejects_inconsistent_identities(complete_manifest: Path, field: str) -> None:
    """Image and sample rows must carry one shared non-empty identity."""
    _rewrite(complete_manifest, lambda rows: rows.__setitem__(0, {**rows[0], field: "different"}))

    with pytest.raises(ValueError, match=field):
        load_complete_sample(complete_manifest)


def test_load_rejects_absent_image_file(complete_manifest: Path) -> None:
    """Manifest presence alone cannot make a sample complete."""
    _rewrite(
        complete_manifest,
        lambda rows: rows.__setitem__(0, {**rows[0], "file": str(complete_manifest.parent / "missing.png")}),
    )

    with pytest.raises(ValueError, match="image file"):
        load_complete_sample(complete_manifest)


def test_load_rejects_left_hand_image_path(complete_manifest: Path) -> None:
    """Live commissioning cannot silently consume a left-hand capture."""
    with complete_manifest.open(newline="", encoding="utf-8") as file:
        first_row = next(csv.DictReader(file))
    right_image = Path(first_row["file"])
    capture_root = complete_manifest.parents[1]
    relative_parts = right_image.relative_to(capture_root).parts
    left_image = capture_root / "left" / Path(*relative_parts[1:])
    left_image.parent.mkdir(parents=True)
    left_image.write_bytes(right_image.read_bytes())
    _rewrite(complete_manifest, lambda rows: rows.__setitem__(0, {**rows[0], "file": str(left_image)}))

    with pytest.raises(ValueError, match="right-hand capture layout"):
        load_complete_sample(complete_manifest)


def test_loaded_images_mapping_is_read_only(complete_manifest: Path) -> None:
    """Callers cannot replace a verified image after manifest validation."""
    sample = load_complete_sample(complete_manifest)

    with pytest.raises(TypeError):
        sample.images["front"] = complete_manifest


def test_find_single_manifest_rejects_multiple_manifests(complete_manifest: Path) -> None:
    """Manifest discovery fails closed when capture identity is ambiguous."""
    duplicate = complete_manifest.with_name("another.csv")
    duplicate.write_bytes(complete_manifest.read_bytes())

    with pytest.raises(ValueError, match="exactly one manifest"):
        find_single_manifest(complete_manifest.parents[1])


def _complete_audit() -> dict[str, object]:
    """Return valid Stage-18 view evidence in deliberately reversed order."""
    views = {}
    for view_index, view in reversed(tuple(enumerate(CANONICAL_VIEWS))):
        branches = []
        for branch_index, branch in enumerate(("yolo", f"anomaly_{view}", "template_match")):
            branches.append(
                {
                    "branch": branch,
                    "score": view_index + branch_index / 10,
                    "low_threshold": 0.25,
                    "high_threshold": 0.75,
                    "computed_evidence_level": ("STRONG", "GRAY", "CLEAR")[branch_index],
                },
            )
        views[view] = branches
    return {"views": views}


def test_format_audit_report_uses_canonical_view_and_branch_order() -> None:
    """Audit text is stable regardless of mapping and list insertion order."""
    report = format_audit_report(_complete_audit())

    lines = report.splitlines()
    assert [line.split(" | ", 1)[0] for line in lines] == [view for view in CANONICAL_VIEWS for _ in range(3)]
    assert [line.split(" | ")[1] for line in lines] == [
        branch for view in CANONICAL_VIEWS for branch in ("template_match", f"anomaly_{view}", "yolo")
    ]
    assert lines[0] == (
        "front | template_match | score=0.2 | low_threshold=0.25 | high_threshold=0.75 | computed_evidence_level=CLEAR"
    )


@pytest.mark.parametrize("field", ["score", "low_threshold", "high_threshold", "computed_evidence_level"])
def test_format_audit_report_rejects_missing_evidence_field(field: str) -> None:
    """Every report line must contain the complete Stage-18 evidence contract."""
    audit = _complete_audit()
    del audit["views"]["front"][0][field]  # type: ignore[index]

    with pytest.raises(ValueError, match=field):
        format_audit_report(audit)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", True),
        ("score", math.nan),
        ("low_threshold", math.inf),
        ("high_threshold", "0.75"),
    ],
)
def test_format_audit_report_rejects_non_finite_or_non_numeric_evidence(field: str, value: object) -> None:
    """Threshold evidence is finite numeric data, never bool or text."""
    audit = _complete_audit()
    audit["views"]["front"][0][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=field):
        format_audit_report(audit)


@pytest.mark.parametrize("level", ["WEAK", "INVALID", "clear", None])
def test_format_audit_report_rejects_unknown_evidence_level(level: object) -> None:
    """Only Stage-18 CLEAR, GRAY, and STRONG evidence levels are reportable."""
    audit = _complete_audit()
    audit["views"]["front"][0]["computed_evidence_level"] = level  # type: ignore[index]

    with pytest.raises(ValueError, match="computed_evidence_level"):
        format_audit_report(audit)
