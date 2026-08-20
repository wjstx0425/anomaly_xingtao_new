"""Tests for the minimal BMW four-camera collection wrapper."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import pytest

from pipeline.bmw_lab_collect_data import (
    _ensure_mvs_sdk_import_path,
    build_bootstrap_argv,
    build_parser,
)


REPO_ROOT = Path(__file__).parents[4]
PROFILE = REPO_ROOT / "configs/bmw/capture/bmw_4cam_eight_view_hdr_v1.json"


def _value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_translates_bmw_profile_into_existing_four_camera_hdr_collector(tmp_path: Path) -> None:
    args = Namespace(
        config=PROFILE,
        root=tmp_path,
        label="normal",
        defect_type=None,
        part_id="bmw_normal",
        group_count=3,
        hand="left",
    )

    argv = build_bootstrap_argv(args)

    assert "--legacy-layout" in argv
    assert "--manual-load" in argv
    assert "--hdr" in argv
    assert "--no-align-hdr" in argv
    assert _value(argv, "--short-exposure") == "1500.0"
    assert _value(argv, "--long-exposure") == "6000.0"
    assert _value(argv, "--gain") == "0.0"
    assert _value(argv, "--hdr-settle-frames") == "1"
    assert _value(argv, "--timeout-ms") == "3000"
    assert _value(argv, "--group-count") == "3"
    assert _value(argv, "--part-id") == "bmw_normal"
    assert _value(argv, "--topology").endswith("zs32_4cam_double_side_v1.json")


def test_defect_capture_requires_type() -> None:
    parser = build_parser()
    args = parser.parse_args(["--label", "defect", "--part-id", "bmw_ng"])

    with pytest.raises(ValueError, match="--defect-type"):
        build_bootstrap_argv(args)


def test_adds_vendor_mvs_import_directory_before_hardware_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk_path = tmp_path / "MvImport"
    sdk_path.mkdir()
    (sdk_path / "MvCameraControl_class.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "path", ["existing-path"])

    _ensure_mvs_sdk_import_path(sdk_path)

    assert sys.path[0] == str(sdk_path)
