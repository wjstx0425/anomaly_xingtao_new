# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Headless CLI contracts for the ZS32 inspection dashboard."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from tests.unit.zs32_refactor.dashboard.conftest import eight_view_result_dir as _fixture_result_dir

from zs32_inspection.cli import dashboard


def _result_dir(tmp_path: Path) -> Path:
    return _fixture_result_dir.__wrapped__(tmp_path)


def test_result_dir_and_live_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        dashboard._parser().parse_args(["--result-dir", str(tmp_path), "--live"])


def test_one_offline_or_live_source_is_required() -> None:
    with pytest.raises(SystemExit):
        dashboard._parser().parse_args([])


def test_live_is_an_explicit_not_connected_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert dashboard._run(["--live", "--part-id", "part-1", "--no-gui"]) != 0
    assert "尚未接入" in capsys.readouterr().err


def test_offline_no_gui_saves_decodable_screenshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("HighGUI must not be called")

    monkeypatch.setattr(cv2, "namedWindow", forbidden)
    monkeypatch.setattr(cv2, "imshow", forbidden)
    monkeypatch.setattr(cv2, "waitKey", forbidden)
    eight_view_result_dir = _result_dir(tmp_path)
    screenshot = tmp_path / "screens" / "offline.png"

    assert (
        dashboard._run([
            "--result-dir",
            str(eight_view_result_dir),
            "--no-gui",
            "--save-screenshot",
            str(screenshot),
        ])
        == 0
    )

    decoded = cv2.imread(str(screenshot))
    assert decoded is not None
    assert decoded.shape == (920, 1600, 3)


def test_offline_write_failure_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    eight_view_result_dir = _result_dir(tmp_path)
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)

    result = dashboard._run([
        "--result-dir",
        str(eight_view_result_dir),
        "--no-gui",
        "--save-screenshot",
        str(tmp_path / "failed.png"),
    ])

    assert result != 0
    assert "screenshot" in capsys.readouterr().err
