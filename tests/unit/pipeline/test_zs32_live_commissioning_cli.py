# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Stage35 live ZS32 Demo CLI."""

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


def _load_stage35(name: str = "pipeline_zs32_live_demo") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_requires_part_id_except_for_device_listing() -> None:
    stage35 = _load_stage35()

    with pytest.raises(SystemExit):
        stage35.build_parser().parse_args([])

    args = stage35.build_parser().parse_args(["--list-devices"])
    assert args.list_devices is True
    assert args.part_id is None


def test_parser_exposes_only_demo_config_for_online_inference() -> None:
    stage35 = _load_stage35("pipeline_zs32_live_demo_surface")

    args = stage35.build_parser().parse_args(["--part-id", "live_part_001"])

    assert args.demo_config == REPO_ROOT / "configs/zs32/zs32_demo.json"
    assert args.capture_root == Path("/home/yunjing/anomalib/results/zs32_live_capture")
    assert args.output_root == Path("/home/yunjing/anomalib/results/zs32_live_runtime")
    option_destinations = {action.dest for action in stage35.build_parser()._actions}
    assert "demo_config" in option_destinations
    assert "topology" not in option_destinations
    assert "runtime_config" not in option_destinations
    assert "diagnostic_skip_template" not in option_destinations

    with pytest.raises(SystemExit):
        stage35.build_parser().parse_args(
            ["--part-id", "live_part_001", "--runtime-config", "old.json"],
        )
    with pytest.raises(SystemExit):
        stage35.build_parser().parse_args(
            ["--part-id", "live_part_001", "--diagnostic-skip-template"],
        )


def test_parser_propagates_demo_config_to_live_config(tmp_path: Path) -> None:
    stage35 = _load_stage35("pipeline_zs32_live_demo_config")
    demo_config = tmp_path / "demo.json"
    topology = tmp_path / "topology.json"
    stage35.load_demo_config = lambda path: SimpleNamespace(
        path=Path(path).resolve(),
        topology=topology.resolve(),
    )

    args = stage35.build_parser().parse_args(
        ["--part-id", "part-9", "--demo-config", str(demo_config)],
    )
    config = stage35._config_from_args(args)

    assert config.demo_config == demo_config.resolve()
    assert config.topology_path == topology.resolve()
    assert not hasattr(config, "runtime_config")
    assert not hasattr(config, "diagnostic_skip_template")


def test_device_listing_uses_current_interpreter_and_collector(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_live_run_builds_safe_config_and_prints_demo_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stage35 = _load_stage35("pipeline_zs32_live_demo_success")
    captured_configs: list[object] = []
    result = SimpleNamespace(
        machine_status="OK",
        inspection_complete=True,
        errors=(),
        sample=SimpleNamespace(manifest_path=tmp_path / "capture" / "manifests" / "run.csv"),
        runtime_manifest_path=tmp_path / "output" / "runtime_manifest.json",
        output_dir=tmp_path / "output",
    )

    class FixedDateTime:
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
    assert "machine_status: OK" in output
    assert "inspection_complete: true" in output
    assert "errors: none" in output
    assert f"capture_manifest: {result.sample.manifest_path.resolve()}" in output
    assert f"runtime_manifest: {result.runtime_manifest_path.resolve()}" in output
    assert f"output_dir: {result.output_dir.resolve()}" in output
    assert "audit" not in output
    assert "commissioning_only" not in output


def test_live_contract_error_prints_error_and_returns_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stage35 = _load_stage35("pipeline_zs32_live_demo_error")

    def fail(_config: object) -> None:
        raise LiveCommissioningError("capture manifest contract failed")

    monkeypatch.setattr(stage35, "run_live_commissioning", fail)

    assert stage35.main(["--part-id", "part-001"]) == 2
    assert "ERROR: capture manifest contract failed" in capsys.readouterr().err
