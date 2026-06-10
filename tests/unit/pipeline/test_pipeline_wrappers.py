# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for numbered pipeline wrapper helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module(name: str, relative_path: str) -> ModuleType:
    """Load a pipeline module from a file path."""
    script_path = Path(__file__).resolve().parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(script_path.parent))
    spec.loader.exec_module(module)
    return module


def test_with_default_command_inserts_all_for_workflow_options() -> None:
    """Workflow wrappers should insert the default subcommand before options."""
    common = _load_module("pipeline_common", "pipeline/_common.py")

    args = common.with_default_command(["--models", "anomaly_dino"], {"all", "train"}, "all")

    assert args == ["all", "--models", "anomaly_dino"]


def test_with_default_command_keeps_explicit_workflow_command() -> None:
    """Workflow wrappers should preserve explicit subcommands."""
    common = _load_module("pipeline_common_explicit", "pipeline/_common.py")

    args = common.with_default_command(["train", "--models", "patchcore"], {"all", "train"}, "all")

    assert args == ["train", "--models", "patchcore"]


def test_process_wrapper_resolves_default_auto_mode() -> None:
    """The processing wrapper should default option-style arguments to auto mode."""
    process = _load_module("pipeline_process", "pipeline/2_process_data.py")

    mode, args = process._resolve_mode(["--data-root", "dataset/c789"])

    assert mode == "auto"
    assert args == ["--data-root", "dataset/c789"]


def test_process_wrapper_resolves_manual_mode() -> None:
    """The processing wrapper should route manual mode explicitly."""
    process = _load_module("pipeline_process_manual", "pipeline/2_process_data.py")

    mode, args = process._resolve_mode(["manual", "--slot-count", "6"])

    assert mode == "manual"
    assert args == ["--slot-count", "6"]


def test_process_wrapper_forwards_manual_slot_arguments() -> None:
    """Manual fixed-slot arguments should be forwarded unchanged."""
    process = _load_module("pipeline_process_manual_slots", "pipeline/2_process_data.py")

    mode, args = process._resolve_mode(["manual", "--slot", "slot01:1,1,832,43,3200,446"])

    assert mode == "manual"
    assert args == ["--slot", "slot01:1,1,832,43,3200,446"]


def test_process_wrapper_resolves_fx11_defect_mode() -> None:
    """The processing wrapper should route the FX11 defect preset."""
    process = _load_module("pipeline_process_fx11_defect", "pipeline/2_process_data.py")

    mode, args = process._resolve_mode(["fx11-defect", "--face", "bottom"])

    assert mode == "fx11-defect"
    assert args == ["--face", "bottom"]


def test_fx11_defect_preset_builds_bottom_manual_args(tmp_path: Path) -> None:
    """The FX11 defect preset should build fixed bottom slot crop arguments."""
    preset = _load_module("pipeline_fx11_defect_crop", "pipeline/fx11_defect_crop.py")
    args = preset.build_parser().parse_args(
        [
            "--face",
            "bottom",
            "--data-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "parts"),
            "--results-root",
            str(tmp_path / "results"),
            "--overwrite",
        ],
    )

    manual_args = preset.build_manual_args("bottom", args)

    assert manual_args[:8] == [
        "--data-root",
        str(tmp_path / "dataset"),
        "--output-root",
        str(tmp_path / "parts"),
        "--hand",
        "no_hand",
        "--position",
        "bottom",
    ]
    assert "--labels" in manual_args
    assert "defect" in manual_args
    assert "--defect-slot-mode" in manual_args
    assert "from-name" in manual_args
    assert ["--slot", "slot01:1,1,830,10,3230,420"] == manual_args[manual_args.index("--slot") : manual_args.index("--slot") + 2]
    assert "--overwrite" in manual_args


def test_run_all_builds_stages_in_order() -> None:
    """The all-in-one wrapper should preserve pipeline stage order."""
    run_all = _load_module("pipeline_run_all", "pipeline/0_run_all.py")
    args = run_all.build_parser().parse_args(
        [
            "--collect",
            "--hand left --position top --label normal",
            "--process",
            "auto --data-root dataset/c789",
            "--train",
            "--models anomaly_dino",
            "--inference",
            "image.png --model anomaly_dino",
        ],
    )

    stages = run_all.build_stages(args)

    assert [stage.name for stage in stages] == [
        "1_collect_data",
        "2_process_data",
        "3_train_model",
        "4_inference",
    ]
    assert stages[0].args == ["--hand", "left", "--position", "top", "--label", "normal"]
    assert stages[1].args == ["auto", "--data-root", "dataset/c789"]


def test_run_all_supports_repeated_collection() -> None:
    """The all-in-one wrapper should allow normal and defect collection runs."""
    run_all = _load_module("pipeline_run_all_collect", "pipeline/0_run_all.py")
    args = run_all.build_parser().parse_args(
        [
            "--collect",
            "--label normal",
            "--collect",
            "--label defect --defect-type scratch",
        ],
    )

    stages = run_all.build_stages(args)

    assert [stage.name for stage in stages] == ["1_collect_data", "1_collect_data"]
    assert stages[0].args == ["--label", "normal"]
    assert stages[1].args == ["--label", "defect", "--defect-type", "scratch"]
