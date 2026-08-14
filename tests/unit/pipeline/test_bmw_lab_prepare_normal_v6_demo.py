# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI contracts for the one-command BMW V6 Demo handoff."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "pipeline/bmw_lab_prepare_normal_20260814_v6_demo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bmw_lab_prepare_normal_v6_demo", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raise_incomplete(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise ValueError("八视角训练尚未完成")


def test_cli_reports_incomplete_training_without_partial_output(monkeypatch, capsys) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "publish_v6_demo", _raise_incomplete)

    assert module.main([]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "ValueError: 八视角训练尚未完成",
        "status": "failed",
    }


def test_cli_prints_receipt_and_v6_launch_command(monkeypatch, capsys, tmp_path: Path) -> None:
    module = _load_module()
    output_run = tmp_path / "composite"
    output_config = tmp_path / "v6.json"
    receipt = {"status": "complete", "output_run": str(output_run), "output_config": str(output_config)}
    captured_arguments: dict[str, object] = {}

    def _publish(repo_root: Path, *, output_run: Path, output_config: Path) -> dict[str, object]:
        captured_arguments.update(
            repo_root=repo_root,
            output_run=output_run,
            output_config=output_config,
        )
        return receipt

    monkeypatch.setattr(module, "publish_v6_demo", _publish)

    assert module.main(["--output-run", str(output_run), "--output-config", str(output_config)]) == 0

    assert captured_arguments == {
        "repo_root": module.REPO_ROOT,
        "output_run": output_run,
        "output_config": output_config,
    }
    assert capsys.readouterr().out == (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
        + "uv run --no-sync python pipeline/bmw_lab_eight_view_demo.py "
        + "--config configs/bmw/experiments/bmw_eight_view_demo_v6_right_normal_20260814_v1.json "
        + "--experiment-mode\n"
    )
