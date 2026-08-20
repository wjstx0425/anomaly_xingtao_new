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


def test_start_passes_demo_config_to_stage35(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process()
    popen_calls: list[list[str]] = []

    def popen(command, **_kwargs):
        popen_calls.append(list(command))
        return process

    monkeypatch.setattr("subprocess.Popen", popen)
    controller = Stage35Controller(
        tmp_path / "work",
        demo_config=Path("configs/zs32/custom_demo.json"),
    )

    controller.start("part-22")

    command = popen_calls[0]
    assert command[command.index("--demo-config") + 1] == "configs/zs32/custom_demo.json"
    assert "--runtime-config" not in command
    process.returncode = 0
    controller.close()


def test_dashboard_worker_starts_once_and_stage35_receives_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _Process()
    worker.pid = 2001
    stage35 = _Process()
    stage35.pid = 2002
    processes = iter((worker, stage35))
    popen_calls: list[list[str]] = []

    def popen(command, **_kwargs):
        popen_calls.append(list(command))
        return next(processes)

    monkeypatch.setattr("subprocess.Popen", popen)
    controller = Stage35Controller(tmp_path, demo_config=Path("configs/zs32/custom_demo.json"))

    controller.start_worker()
    controller.start_worker()
    controller.worker_socket_path.touch()
    controller.start("part-1")

    assert len(popen_calls) == 2
    assert popen_calls[0][1].endswith("pipeline/zs32_inference_worker.py")
    assert popen_calls[0][popen_calls[0].index("--demo-config") + 1] == "configs/zs32/custom_demo.json"
    assert "--runtime-bundle" not in popen_calls[0]
    assert popen_calls[1][popen_calls[1].index("--inference-socket") + 1] == str(controller.worker_socket_path)
    worker.returncode = 0
    stage35.returncode = 0
    controller.close()


def test_dashboard_worker_uses_demo_config_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _Process()
    popen_calls: list[list[str]] = []

    def popen(command, **_kwargs):
        popen_calls.append(list(command))
        return worker

    monkeypatch.setattr("subprocess.Popen", popen)
    controller = Stage35Controller(tmp_path, repo_root=tmp_path / "repo")

    controller.start_worker()

    command = popen_calls[0]
    assert command[command.index("--demo-config") + 1] == str(
        tmp_path / "repo/configs/zs32/zs32_demo.json",
    )
    assert "--runtime-bundle" not in command
    worker.returncode = 0
    controller.close()


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
    controller.log_path.parent.mkdir(parents=True, exist_ok=True)
    controller.log_path.write_text("camera open failed: device busy\n", encoding="utf-8")

    progress = controller.poll()

    assert progress is not None
    assert progress.state == "failed"
    assert progress.error is not None
    assert "return code 7" in progress.error
    assert "camera open failed: device busy" in progress.error
    assert str(controller.log_path) in progress.error


def test_poll_surfaces_worker_exit_log_without_starting_stage35(tmp_path: Path) -> None:
    controller = Stage35Controller(tmp_path)
    worker = _Process()
    worker.returncode = 9
    controller.worker_process = worker  # type: ignore[assignment]
    controller.worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    controller.worker_log_path.write_text("loading model\nCUDA out of memory\n", encoding="utf-8")

    progress = controller.poll()

    assert progress is not None
    assert progress.state == "failed"
    assert progress.error is not None
    assert "CUDA out of memory" in progress.error
    assert str(controller.worker_log_path) in progress.error


def test_worker_loading_is_reported_until_socket_is_ready(tmp_path: Path) -> None:
    controller = Stage35Controller(tmp_path)
    worker = _Process()
    controller.worker_process = worker  # type: ignore[assignment]

    progress = controller.poll()

    assert progress is not None
    assert progress.state == "worker_starting"
    assert controller.running is True

    controller.worker_socket_path.parent.mkdir(parents=True, exist_ok=True)
    controller.worker_socket_path.touch()
    assert controller.poll() is None
    assert controller.running is False
