"""Owned Stage35 subprocess and dashboard control transport."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from .contracts import ConfirmationCommand, ProgressRecord
from .control import load_progress, write_confirmation


class Stage35Controller:
    """Own one Stage35 process group and its progress/control files."""

    def __init__(self, work_dir: Path, *, repo_root: Path | None = None) -> None:
        self.work_dir = work_dir.expanduser().resolve()
        self.repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
        self.progress_path = self.work_dir / "progress.json"
        self.control_path = self.work_dir / "control.json"
        self.log_path = self.work_dir / "stage35.log"
        self.process: subprocess.Popen[bytes] | None = None
        self._log_file = None
        self._confirmed_id: str | None = None
        self._part_id: str | None = None

    def start(self, part_id: str) -> None:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Stage35 is already running")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path.unlink(missing_ok=True)
        self.control_path.unlink(missing_ok=True)
        self._confirmed_id = None
        self._part_id = part_id
        self._log_file = self.log_path.open("ab")
        command = [
            sys.executable,
            str(self.repo_root / "pipeline/35_run_zs32_live_commissioning.py"),
            "--part-id",
            part_id,
            "--progress-json",
            str(self.progress_path),
            "--control-json",
            str(self.control_path),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            start_new_session=True,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

    def poll(self) -> ProgressRecord | None:
        if self.progress_path.is_file():
            return load_progress(self.progress_path)
        if self.process is not None:
            returncode = self.process.poll()
            if returncode not in (None, 0):
                return ProgressRecord(
                    self._part_id or "unknown",
                    None,
                    "failed",
                    "Stage35 exited",
                    "",
                    error=f"Stage35 exited with return code {returncode}",
                )
        return None

    @property
    def running(self) -> bool:
        """Return whether the owned Stage35 child is still active."""
        return self.process is not None and self.process.poll() is None

    def confirm(self) -> bool:
        progress = self.poll()
        if (
            progress is None
            or progress.state not in {"waiting_front", "waiting_back"}
            or not progress.confirmation_id
            or progress.confirmation_id == self._confirmed_id
        ):
            return False
        round_name = "front" if progress.state == "waiting_front" else "back"
        write_confirmation(
            self.control_path,
            ConfirmationCommand(
                "confirm_round", round_name, progress.confirmation_id, progress.part_id
            ),
        )
        self._confirmed_id = progress.confirmation_id
        return True

    def close(self, timeout: float = 5.0) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, TimeoutError):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self.process = None
