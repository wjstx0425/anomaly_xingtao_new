# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Stage35 live ZS32 commissioning CLI."""

# Numbered pipeline modules require an importlib loader in tests.
# ruff: noqa: SLF001

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from capture_data.zs32_live_commissioning import LiveCommissioningError

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "pipeline/35_run_zs32_live_commissioning.py"


def _load_stage35(name: str = "pipeline_zs32_live_commissioning") -> ModuleType:
    """Load the numbered Stage35 module."""
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"could not load {SCRIPT}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_requires_part_id_except_for_device_listing() -> None:
    """A live run needs an identity, while discovery must remain standalone."""
    stage35 = _load_stage35()

    with pytest.raises(SystemExit):
        stage35.build_parser().parse_args([])

    args = stage35.build_parser().parse_args(["--list-devices"])
    assert args.list_devices is True
    assert args.part_id is None


def test_parser_uses_approved_paths_capture_values_and_locked_surface() -> None:
    """Minimal live operation must resolve to the reviewed right-hand contract."""
    stage35 = _load_stage35("pipeline_zs32_live_defaults")

    args = stage35.build_parser().parse_args(["--part-id", "live_part_001"])

    assert args.capture_root == Path("/home/yunjing/anomalib/results/zs32_live_capture")
    assert args.output_root == Path("/home/yunjing/anomalib/results/zs32_live_runtime")
    assert args.runtime_config == REPO_ROOT / "results/zs32_runtime_bundle_eight_view_v2/runtime_bundle.json"
    assert args.topology == REPO_ROOT / "configs/zs32/topology/zs32_4cam_double_side_v1.json"
    assert args.diagnostic_skip_template is False

    option_destinations = {action.dest for action in stage35.build_parser()._actions}
    assert option_destinations.isdisjoint(
        {
            "fusion_profile",
            "hand",
            "group_count",
            "images_per_group",
            "front_serial",
            "left_serial",
            "right_serial",
            "template_model_dir",
            "threshold_artifact",
        },
    )


def test_parser_propagates_diagnostic_skip_template_to_live_config() -> None:
    """The explicit diagnostic flag must reach the live orchestration contract."""
    stage35 = _load_stage35("pipeline_zs32_live_diagnostic_config")

    args = stage35.build_parser().parse_args(["--part-id", "diagnostic-001", "--diagnostic-skip-template"])
    config = stage35._config_from_args(args)

    assert args.diagnostic_skip_template is True
    assert config.diagnostic_skip_template is True


def test_parser_accepts_bundle_topology_and_output_overrides(tmp_path: Path) -> None:
    """Operators may select only a finalized bundle and topology as runtime assets."""
    stage35 = _load_stage35("pipeline_zs32_live_overrides")

    args = stage35.build_parser().parse_args(
        [
            "--part-id",
            "part-9",
            "--capture-root",
            str(tmp_path / "capture"),
            "--output-root",
            str(tmp_path / "output"),
            "--runtime-config",
            str(tmp_path / "runtime_bundle.json"),
            "--topology",
            str(tmp_path / "topology.json"),
        ],
    )

    assert args.capture_root == tmp_path / "capture"
    assert args.output_root == tmp_path / "output"
    assert args.runtime_config == tmp_path / "runtime_bundle.json"
    assert args.topology == tmp_path / "topology.json"


def test_device_listing_uses_current_interpreter_and_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery must delegate only to the existing SDK collector."""
    stage35 = _load_stage35("pipeline_zs32_live_devices")
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        calls.append((command, check))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(stage35.subprocess, "run", fake_run)

    assert stage35.main(["--list-devices"]) == 7
    assert calls == [
        (
            [
                sys.executable,
                str((REPO_ROOT / "capture_data/collect_multicamera_dataset.py").resolve()),
                "--list-devices",
            ],
            False,
        ),
    ]


def _successful_result(
    tmp_path: Path,
    *,
    audit_path: Path | None,
    diagnostic_skip_template: bool = False,
) -> SimpleNamespace:
    """Return one CLI-facing successful core result."""
    return SimpleNamespace(
        audit_report="front | template_match | score=0.1 | level=CLEAR",
        machine_status="OK",
        inspection_complete=True,
        commissioning_only=True,
        production_release_allowed=False,
        sample=SimpleNamespace(manifest_path=tmp_path / "capture" / "manifests" / "run.csv"),
        runtime_summary_path=tmp_path / "output" / "runtime_summary.json",
        audit_path=audit_path,
        output_dir=tmp_path / "output",
        diagnostic_skip_template=diagnostic_skip_template,
        patchcore_csv=tmp_path / "output" / "patchcore.csv",
        yolo_csv=tmp_path / "output" / "yolo.csv",
    )


def test_live_run_builds_safe_config_and_prints_result_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI must pass approved identities to the core and print its auditable result."""
    stage35 = _load_stage35("pipeline_zs32_live_success")
    captured_configs: list[object] = []
    result = _successful_result(tmp_path, audit_path=tmp_path / "output" / "fusion" / "audit" / "part.json")

    class FixedDateTime:
        """Deterministic clock for run identity verification."""

        @classmethod
        def now(cls) -> datetime:
            return datetime.fromisoformat("2026-07-14T15:16:17.123456+08:00")

    def fake_live_run(config: object) -> SimpleNamespace:
        captured_configs.append(config)
        return result

    monkeypatch.setattr(stage35, "datetime", FixedDateTime)
    monkeypatch.setattr(stage35, "run_live_commissioning", fake_live_run)

    assert stage35.main(["--part-id", "live_part_001"]) == 0

    config = captured_configs[0]
    assert config.repo_root == REPO_ROOT
    assert config.part_id == "live_part_001"
    assert config.run_id == "20260714_151617_123456_live_part_001"
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", config.run_id)
    output = capsys.readouterr().out
    assert result.audit_report in output
    assert "machine_status: OK" in output
    assert "inspection_complete: true" in output
    assert "commissioning_only: true" in output
    assert "production_release_allowed: false" in output
    assert f"capture_manifest: {result.sample.manifest_path.resolve()}" in output
    assert f"runtime_summary: {result.runtime_summary_path.resolve()}" in output
    assert f"audit: {result.audit_path.resolve()}" in output
    assert f"output_dir: {result.output_dir.resolve()}" in output


def test_template_short_circuit_prints_explicit_missing_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A valid template short-circuit must not invent a Stage18 audit path."""
    stage35 = _load_stage35("pipeline_zs32_live_short_circuit")
    monkeypatch.setattr(
        stage35,
        "run_live_commissioning",
        lambda _config: _successful_result(tmp_path, audit_path=None),
    )

    assert stage35.main(["--part-id", "part-001"]) == 0
    assert "audit: none (template short-circuit; Stage18 audit was not generated)" in capsys.readouterr().out


def test_diagnostic_skip_template_prints_explicit_nonproduction_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Diagnostic infer must name skipped stages, evidence CSVs, and its release boundary."""
    stage35 = _load_stage35("pipeline_zs32_live_diagnostic_output")
    result = _successful_result(tmp_path, audit_path=None, diagnostic_skip_template=True)
    result.audit_report = ""
    result.machine_status = "REVIEW"
    result.inspection_complete = False
    monkeypatch.setattr(stage35, "run_live_commissioning", lambda _config: result)

    assert stage35.main(["--part-id", "part-001", "--diagnostic-skip-template"]) == 0

    output = capsys.readouterr().out
    assert "template: skipped (diagnostic)" in output
    assert "stage18_audit: none (diagnostic infer; fusion was not run)" in output
    assert f"patchcore_csv: {result.patchcore_csv.resolve()}" in output
    assert f"yolo_csv: {result.yolo_csv.resolve()}" in output
    assert "machine_status: REVIEW" in output
    assert "production_release_allowed: false" in output


@pytest.mark.parametrize("missing_field", ["patchcore_csv", "yolo_csv"])
def test_diagnostic_result_missing_csv_path_returns_two(
    missing_field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An inconsistent diagnostic result must fail with a stable CLI error."""
    stage35 = _load_stage35(f"pipeline_zs32_live_missing_{missing_field}")
    result = _successful_result(tmp_path, audit_path=None, diagnostic_skip_template=True)
    setattr(result, missing_field, None)
    monkeypatch.setattr(stage35, "run_live_commissioning", lambda _config: result)

    assert stage35.main(["--part-id", "part-001", "--diagnostic-skip-template"]) == 2
    assert f"ERROR: diagnostic result is missing {missing_field}" in capsys.readouterr().err


def test_live_contract_error_prints_error_and_returns_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Core contract failures must become a stable operator-facing error code."""
    stage35 = _load_stage35("pipeline_zs32_live_error")

    def fail(_config: object) -> None:
        message = "capture manifest contract failed"
        raise LiveCommissioningError(message)

    monkeypatch.setattr(stage35, "run_live_commissioning", fail)

    assert stage35.main(["--part-id", "part-001"]) == 2
    assert "ERROR: capture manifest contract failed" in capsys.readouterr().err
