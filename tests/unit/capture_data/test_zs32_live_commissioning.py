# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the minimal live ZS32 Demo orchestration."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import capture_data.zs32_live_commissioning as live_commissioning
from capture_data.zs32_live_commissioning import (
    LiveCommissioningError,
    LiveRunConfig,
    LiveRunResult,
    build_capture_command,
    find_single_manifest,
    load_complete_sample,
    run_live_commissioning,
)
from zs32_inspection.config.loaders import load_topology
from zs32_inspection.domain.views import VIEW_ORDER


@pytest.fixture
def live_config(tmp_path: Path) -> LiveRunConfig:
    repo_root = Path(__file__).parents[3].resolve()
    demo_config = tmp_path / "zs32_demo.json"
    demo_config.write_text("{}", encoding="utf-8")
    return LiveRunConfig(
        repo_root=repo_root,
        part_id="live_part_001",
        run_id="20260714_120000_000001",
        capture_root=tmp_path / "captures",
        output_root=tmp_path / "outputs",
        demo_config=demo_config,
        topology_path=repo_root / "configs/zs32/topology/zs32_4cam_double_side_v1.json",
    )


def _write_png(path: Path, value: int = 127) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 12, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_manifest(
    capture_root: Path,
    *,
    topology_path: Path,
    sample_id: str = "live_part_001_group001_000001",
    session_id: str = "20260714_121500_000001",
    group_id: str = "group001",
) -> Path:
    manifest = capture_root / "manifests" / f"{session_id}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    topology = load_topology(topology_path)
    rows: list[dict[str, str]] = []
    for index, view in enumerate(VIEW_ORDER):
        round_id, _slot_id, serial = topology.binding_for_view(view)
        image = (
            capture_root
            / "right"
            / view
            / "normal"
            / session_id
            / "images"
            / f"right_{view}_normal_capture_single.png"
        )
        _write_png(image, 20 + index)
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
            "round": "",
            "view": "",
            "camera_serial": "",
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
def complete_manifest(live_config: LiveRunConfig, tmp_path: Path) -> Path:
    return _write_manifest(tmp_path, topology_path=live_config.topology_path)


def _command_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _write_runtime_manifest(command: list[str], *, status: str = "OK", errors: list[str] | None = None) -> Path:
    output_dir = Path(_command_value(command, "--output-dir"))
    output_dir.mkdir(parents=True)
    path = output_dir / "runtime_manifest.json"
    path.write_text(
        json.dumps(
            {
                "part_id": _command_value(command, "--part-id"),
                "capture_session": _command_value(command, "--capture-session"),
                "group_id": _command_value(command, "--group-id"),
                "machine_status": status,
                "inspection_complete": not errors,
                "errors": errors or [],
            },
        ),
        encoding="utf-8",
    )
    return path


def test_build_capture_command_preserves_topology_and_two_round_manual_capture(
    live_config: LiveRunConfig,
) -> None:
    capture_run_root = (live_config.capture_root / live_config.run_id).resolve()

    command = build_capture_command(live_config, capture_run_root)

    assert command[:6] == [
        sys.executable,
        str((live_config.repo_root / "pipeline/zs32_bootstrap_capture.py").resolve()),
        "--topology",
        str(live_config.topology_path.resolve()),
        "--root",
        str(capture_run_root),
    ]
    assert "--manual-load" in command
    assert _command_value(command, "--capture-interval") == "0.2"
    assert _command_value(command, "--short-exposure") == "1500"
    assert _command_value(command, "--long-exposure") == "5500"
    assert _command_value(command, "--hdr-settle-frames") == "1"
    assert _command_value(command, "--timeout-ms") == "2000"
    assert _command_value(command, "--short-dark-threshold") == "80"
    assert _command_value(command, "--long-clip-threshold") == "245"
    assert _command_value(command, "--blend-width") == "50"
    assert _command_value(command, "--blur-size") == "101"
    assert _command_value(command, "--hdr-max-retries") == "0"
    assert _command_value(command, "--hdr-max-clip-pct") == "5"
    assert "--no-align-hdr" in command
    assert command[command.index("--group-count") + 1] == "1"
    assert command[command.index("--images-per-group") + 1] == "1"
    assert "--progress-json" not in command
    assert "--control-json" not in command


def test_build_capture_command_forwards_progress_and_confirmation_files(
    live_config: LiveRunConfig,
    tmp_path: Path,
) -> None:
    config = replace(
        live_config,
        progress_json=tmp_path / "progress.json",
        control_json=tmp_path / "control.json",
    )

    command = build_capture_command(config, config.capture_root / config.run_id)

    assert "--manual-load" in command
    assert _command_value(command, "--progress-json") == str(config.progress_json)
    assert _command_value(command, "--control-json") == str(config.control_json)
    assert command.index("--progress-json") < command.index("--control-json")


def test_build_demo_inference_command_passes_one_config_and_eight_images(
    live_config: LiveRunConfig,
    complete_manifest: Path,
) -> None:
    sample = load_complete_sample(complete_manifest, live_config.topology_path)
    output_dir = (live_config.output_root / sample.capture_session / sample.part_id).resolve()

    command = live_commissioning.build_demo_inference_command(live_config, sample, output_dir)

    assert command[:4] == [
        sys.executable,
        str((live_config.repo_root / "pipeline/zs32_demo_inference.py").resolve()),
        "--demo-config",
        str(live_config.demo_config.resolve()),
    ]
    assert "--runtime-config" not in command
    assert "--runtime-bundle" not in command
    assert "--threshold-artifact" not in command
    assert "--fusion-profile" not in command
    assert "--diagnostic-skip-template" not in command
    assert _command_value(command, "--part-id") == sample.part_id
    assert _command_value(command, "--capture-session") == sample.capture_session
    assert _command_value(command, "--group-id") == sample.group_id
    assert _command_value(command, "--output-dir") == str(output_dir)
    for view in VIEW_ORDER:
        assert _command_value(command, f"--{view.replace('_', '-')}-image") == str(sample.images[view])


def test_build_demo_inference_command_forwards_progress_json(
    live_config: LiveRunConfig,
    complete_manifest: Path,
    tmp_path: Path,
) -> None:
    config = replace(
        live_config,
        progress_json=tmp_path / "progress.json",
        control_json=tmp_path / "control.json",
    )
    sample = load_complete_sample(complete_manifest, config.topology_path)

    command = live_commissioning.build_demo_inference_command(config, sample, tmp_path / "output")

    assert _command_value(command, "--progress-json") == str(config.progress_json)


def test_run_starts_demo_cli_without_socket(live_config: LiveRunConfig) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        commands.append(command)
        if len(commands) == 1:
            _write_manifest(
                Path(_command_value(command, "--root")),
                topology_path=live_config.topology_path,
            )
        else:
            _write_runtime_manifest(command)
        return SimpleNamespace(returncode=0)

    result = run_live_commissioning(live_config, runner)

    assert isinstance(result, LiveRunResult)
    assert result.machine_status == "OK"
    assert result.inspection_complete is True
    assert result.errors == ()
    assert len(commands) == 2
    assert commands[1][1].endswith("pipeline/zs32_demo_inference.py")
    assert result.runtime_manifest_path == result.output_dir / "runtime_manifest.json"


def test_run_leaves_manifest_image_decode_to_demo_worker(
    live_config: LiveRunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        commands.append(command)
        if len(commands) == 1:
            _write_manifest(
                Path(_command_value(command, "--root")),
                topology_path=live_config.topology_path,
            )
            monkeypatch.setattr(
                live_commissioning.cv2,
                "imread",
                lambda *_args, **_kwargs: pytest.fail("Stage35 decoded a manifest image"),
            )
        else:
            _write_runtime_manifest(command)
        return SimpleNamespace(returncode=0)

    result = run_live_commissioning(live_config, runner)

    assert result.machine_status == "OK"
    assert len(commands) == 2


def test_run_submits_demo_argv_to_persistent_worker(
    live_config: LiveRunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(live_config, inference_socket=live_config.capture_root / "worker.sock")
    capture_commands: list[list[str]] = []
    submitted: list[tuple[str, list[str]]] = []

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        capture_commands.append(command)
        _write_manifest(Path(_command_value(command, "--root")), topology_path=config.topology_path)
        return SimpleNamespace(returncode=0)

    class Client:
        def __init__(self, socket_path: Path, *, timeout: float) -> None:
            assert socket_path == config.inference_socket
            assert timeout == 300.0

        def execute(self, job_id: str, argv: list[str]) -> SimpleNamespace:
            submitted.append((job_id, argv))
            _write_runtime_manifest([sys.executable, "demo", *argv])
            return SimpleNamespace(returncode=0, error=None)

    monkeypatch.setattr("capture_data.zs32_live_commissioning.InferenceWorkerClient", Client)

    result = run_live_commissioning(config, runner)

    assert result.machine_status == "OK"
    assert len(capture_commands) == 1
    assert len(submitted) == 1
    assert submitted[0][1][0:2] == ["--demo-config", str(config.demo_config.resolve())]
    assert "zs32_demo_inference.py" not in submitted[0][1]


def test_run_reports_worker_failure_verbatim(
    live_config: LiveRunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(live_config, inference_socket=live_config.capture_root / "worker.sock")

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        _write_manifest(Path(_command_value(command, "--root")), topology_path=config.topology_path)
        return SimpleNamespace(returncode=0)

    class Client:
        def __init__(self, _socket_path: Path, *, timeout: float) -> None:
            assert timeout == 300.0

        def execute(self, _job_id: str, _argv: list[str]) -> SimpleNamespace:
            return SimpleNamespace(returncode=None, error="ValueError: invalid Demo JSON")

    monkeypatch.setattr("capture_data.zs32_live_commissioning.InferenceWorkerClient", Client)

    with pytest.raises(LiveCommissioningError, match="ValueError: invalid Demo JSON"):
        run_live_commissioning(config, runner)


def test_run_failure_progress_keeps_real_error(
    live_config: LiveRunConfig,
    tmp_path: Path,
) -> None:
    config = replace(
        live_config,
        progress_json=tmp_path / "progress.json",
        control_json=tmp_path / "control.json",
    )

    with pytest.raises(LiveCommissioningError, match="capture command failed"):
        run_live_commissioning(
            config,
            lambda _command, **_kwargs: SimpleNamespace(returncode=23),
        )

    payload = json.loads(config.progress_json.read_text(encoding="utf-8"))
    assert payload["state"] == "failed"
    assert payload["message"] == "inspection failed"
    assert "LiveCommissioningError: capture command failed with return code 23" == payload["error"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.pop("machine_status"), "machine_status"),
        (lambda payload: payload.__setitem__("inspection_complete", "yes"), "inspection_complete"),
        (lambda payload: payload.__setitem__("errors", "none"), "errors"),
        (lambda payload: payload.__setitem__("part_id", "wrong"), "part_id"),
    ],
)
def test_run_rejects_invalid_demo_manifest(
    live_config: LiveRunConfig,
    mutate: object,
    match: str,
) -> None:
    invocation = 0

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            _write_manifest(Path(_command_value(command, "--root")), topology_path=live_config.topology_path)
        else:
            path = _write_runtime_manifest(command)
            payload = json.loads(path.read_text(encoding="utf-8"))
            mutate(payload)  # type: ignore[operator]
            path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    with pytest.raises(LiveCommissioningError, match=match):
        run_live_commissioning(live_config, runner)


def test_run_rejects_preexisting_output(live_config: LiveRunConfig) -> None:
    existing = live_config.output_root / "20260714_121500_000001" / "live_part_001_group001_000001"
    existing.mkdir(parents=True)
    commands: list[list[str]] = []

    def runner(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        commands.append(command)
        _write_manifest(Path(_command_value(command, "--root")), topology_path=live_config.topology_path)
        return SimpleNamespace(returncode=0)

    with pytest.raises(LiveCommissioningError, match="output directory already exists"):
        run_live_commissioning(live_config, runner)
    assert len(commands) == 1


@pytest.mark.parametrize("field", ["part_id", "run_id"])
@pytest.mark.parametrize("value", ["", ".", "..", "nested/id", "nested\\id"])
def test_run_rejects_unsafe_identity(
    live_config: LiveRunConfig,
    field: str,
    value: str,
) -> None:
    with pytest.raises(LiveCommissioningError, match=field):
        run_live_commissioning(
            replace(live_config, **{field: value}),
            lambda _command, **_kwargs: SimpleNamespace(returncode=0),
        )


def test_load_complete_sample_validates_exact_eight_decodable_images(
    complete_manifest: Path,
    live_config: LiveRunConfig,
) -> None:
    sample = load_complete_sample(complete_manifest, live_config.topology_path)

    assert tuple(sample.images) == VIEW_ORDER
    assert all(cv2.imread(str(path), cv2.IMREAD_COLOR) is not None for path in sample.images.values())


def test_load_complete_sample_rejects_undecodable_image(
    complete_manifest: Path,
    live_config: LiveRunConfig,
) -> None:
    with complete_manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fields = tuple(rows[0])
    Path(rows[0]["file"]).write_bytes(b"not an image")
    with complete_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="decode"):
        load_complete_sample(complete_manifest, live_config.topology_path)


def test_load_complete_sample_rejects_missing_or_duplicate_view(
    complete_manifest: Path,
    live_config: LiveRunConfig,
) -> None:
    with complete_manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fields = tuple(rows[0])
    rows[1]["view"] = rows[0]["view"]
    with complete_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="canonical views"):
        load_complete_sample(complete_manifest, live_config.topology_path)


def test_load_complete_sample_rejects_identity_or_topology_drift(
    complete_manifest: Path,
    live_config: LiveRunConfig,
) -> None:
    with complete_manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fields = tuple(rows[0])
    rows[0]["camera_serial"] = "wrong"
    with complete_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="camera_serial"):
        load_complete_sample(complete_manifest, live_config.topology_path)


def test_find_single_manifest_rejects_ambiguity(
    complete_manifest: Path,
) -> None:
    complete_manifest.with_name("another.csv").write_bytes(complete_manifest.read_bytes())

    with pytest.raises(ValueError, match="exactly one manifest"):
        find_single_manifest(complete_manifest.parents[1])
