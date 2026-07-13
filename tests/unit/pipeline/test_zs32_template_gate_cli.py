# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI contract tests for the ZS32 template gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str) -> ModuleType:
    """Load a numbered/pipeline script as an isolated module."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predict_cli_fails_closed_as_json(tmp_path: Path) -> None:
    """A missing model must produce structured JSON and the invalid exit code."""
    module = _load("predict_zs32_template_gate", "pipeline/predict_zs32_template_gate.py")
    output = tmp_path / "evidence.json"

    code = module.main(
        [
            "--model-dir",
            str(tmp_path / "missing"),
            "--image",
            str(tmp_path / "missing.png"),
            "--hand",
            "left",
            "--view",
            "front",
            "--output-json",
            str(output),
        ],
    )

    assert code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "INVALID_TEMPLATE_GATE"
    assert payload["reason"]
    assert payload["score"] is None
    assert payload["threshold"] is None


def test_train_cli_requires_explicit_deployment_versions() -> None:
    """The trainer must never invent deployment version identities."""
    module = _load("train_zs32_template_gate", "pipeline/train_zs32_template_gate.py")
    parser = module.build_parser()

    required = {
        action.dest
        for action in parser._actions  # noqa: SLF001
        if getattr(action, "required", False)
    }

    expected = {"manifest", "output_dir", "model_version", "threshold_version", "roi_version", "template_version"}
    assert expected <= required
