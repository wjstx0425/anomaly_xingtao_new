# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the single ZS32 Demo inference entry point."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pipeline import zs32_demo_inference
from zs32_inspection.domain.views import VIEW_ORDER


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, images: dict[str, Path], **kwargs: object) -> SimpleNamespace:
        self.calls.append({"images": images, **kwargs})
        return SimpleNamespace(
            output_dir=Path(kwargs["output_dir"]),
            machine_status="OK",
            inspection_complete=True,
            errors=(),
        )


def _argv(tmp_path: Path) -> list[str]:
    arguments = [
        "--demo-config",
        str(tmp_path / "demo.json"),
        "--part-id",
        "part-1",
        "--capture-session",
        "session-1",
        "--group-id",
        "group001",
        "--output-dir",
        str(tmp_path / "result"),
    ]
    for view in VIEW_ORDER:
        arguments.extend((f"--{view.replace('_', '-')}-image", str(tmp_path / f"{view}.png")))
    return arguments


def test_run_argv_forwards_exact_eight_view_request(tmp_path: Path) -> None:
    runtime = _Runtime()

    result = zs32_demo_inference.run_argv(
        _argv(tmp_path),
        runtime=runtime,
        startup_config_path=tmp_path / "demo.json",
    )

    assert result.machine_status == "OK"
    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert tuple(call["images"]) == VIEW_ORDER
    assert call["config_path"] == (tmp_path / "demo.json").resolve()
    assert call["part_id"] == "part-1"


def test_worker_rejects_a_different_demo_config(tmp_path: Path) -> None:
    runtime = _Runtime()

    try:
        zs32_demo_inference.run_argv(
            _argv(tmp_path),
            runtime=runtime,
            startup_config_path=tmp_path / "other.json",
        )
    except ValueError as error:
        assert "worker was started" in str(error)
    else:
        raise AssertionError("different Demo config must be rejected")


def test_main_returns_two_and_publishes_failed_progress(tmp_path: Path, monkeypatch: object) -> None:
    progress = tmp_path / "progress.json"
    arguments = _argv(tmp_path) + ["--progress-json", str(progress)]

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("model load failed")

    monkeypatch.setattr(zs32_demo_inference, "prepare_demo_runtime", _raise)

    assert zs32_demo_inference.main(arguments) == 2
    assert progress.is_file()
    assert "model load failed" in progress.read_text(encoding="utf-8")
