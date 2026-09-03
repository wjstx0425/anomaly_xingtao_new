"""Tests for the BMW-owned four-camera collection CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from bmw_inspection.cli.collect import _validate, build_parser


def test_requires_explicit_left_or_right_hand() -> None:
    args = build_parser().parse_args(["--label", "normal", "--part-id", "part001"])

    with pytest.raises(ValueError, match="--hand"):
        _validate(args)


def test_accepts_complete_four_camera_capture_request(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--hand", "left", "--label", "normal", "--part-id", "part001", "--root", str(tmp_path)]
    )

    _validate(args)


def test_defect_capture_requires_type() -> None:
    args = build_parser().parse_args(["--hand", "right", "--label", "defect", "--part-id", "part001"])

    with pytest.raises(ValueError, match="--defect-type"):
        _validate(args)
