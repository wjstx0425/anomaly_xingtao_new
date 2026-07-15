"""Owned Stage35 process lifecycle contracts."""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from zs32_inspection.dashboard.contracts import ProgressRecord
from zs32_inspection.dashboard.control import load_progress, write_progress
from zs32_inspection.dashboard.live import Stage35Controller


class _Process:
    def __init__(self) -> None:
        self.pid = 1234
        self.returncode: int | None = None
        self.waits = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        if self.waits == 1 and timeout is not None:
            raise TimeoutError
        self.returncode = -signal.SIGKILL
        return self.returncode


def test_start_confirm_and_close_owned_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = _Process()
    popen_calls: list[tuple[list[str], dict[str, object]]] = []
    signals: list[tuple[int, signal.Signals]] = []

    def popen(command, **kwargs):
        popen_calls.append((list(command), dict(kwargs)))
        return process

    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setattr("os.killpg", lambda pid, sig: signals.append((pid, sig)))
    controller = Stage35Controller(tmp_path)

    controller.start("part-1")
    with pytest.raises(RuntimeError, match="already running"):
        controller.start("part-1")
    write_progress(
        controller.progress_path,
        ProgressRecord("part-1", "session", "waiting_front", "ready", "now", "token-1"),
    )
    assert controller.poll().state == "waiting_front"
    assert controller.confirm() is True
    assert controller.confirm() is False
    assert popen_calls[0][1]["start_new_session"] is True

    controller.close(timeout=0.01)

    assert signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]


def test_poll_returns_failed_progress(tmp_path: Path) -> None:
    controller = Stage35Controller(tmp_path)
    write_progress(
        controller.progress_path,
        ProgressRecord("part-1", None, "failed", "failed", "now", error="boom"),
    )

    assert load_progress(controller.progress_path).error == "boom"
    assert controller.poll().state == "failed"


def test_poll_translates_nonzero_exit_without_progress(tmp_path: Path) -> None:
    controller = Stage35Controller(tmp_path)
    process = _Process()
    process.returncode = 7
    controller.process = process  # type: ignore[assignment]
    controller._part_id = "part-1"

    progress = controller.poll()

    assert progress is not None
    assert progress.state == "failed"
    assert progress.error == "Stage35 exited with return code 7"
