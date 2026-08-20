"""Owned Stage35 subprocess and dashboard control transport."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from .contracts import ConfirmationCommand, ProgressRecord
from .control import load_progress, write_confirmation
from .inference_worker import InferenceWorkerClient


class Stage35Controller:
    """Own one Stage35 process group and its progress/control files."""

    def __init__(
        self,
        work_dir: Path,
        *,
        repo_root: Path | None = None,
        demo_config: Path | None = None,
    ) -> None:
        self.work_dir = work_dir.expanduser().resolve()
        self.repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
        self.demo_config = demo_config or (self.repo_root / "configs/zs32/zs32_demo.json")
        self.progress_path = self.work_dir / "progress.json"
        self.control_path = self.work_dir / "control.json"
        self.log_path = self.work_dir / "stage35.log"
        self.worker_socket_path = self.work_dir / "inference_worker.sock"
        self.worker_log_path = self.work_dir / "inference_worker.log"
        self.process: subprocess.Popen[bytes] | None = None
        self.worker_process: subprocess.Popen[bytes] | None = None
        self._log_file = None
        self._worker_log_file = None
        self._confirmed_id: str | None = None
        self._part_id: str | None = None
        self._worker_start_error: str | None = None

    @staticmethod
    def _log_tail(path: Path, *, limit: int = 4000) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-limit:].strip()

    def _failure_progress(self, component: str, returncode: int | None, log_path: Path) -> ProgressRecord:
        detail = self._log_tail(log_path)
        returncode_text = "" if returncode is None else f" with return code {returncode}"
        error = f"{component} exited{returncode_text}; log={log_path}"
        if detail:
            error = f"{error}; tail={detail}"
        return ProgressRecord(
            self._part_id or "unknown",
            None,
            "failed",
            f"{component} failed",
            "",
            error=error,
        )

    def start_worker(self) -> None:
        """Start one independent model worker; repeated calls are harmless."""
        if self.worker_process is not None and self.worker_process.poll() is None:
            return
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.worker_socket_path.unlink(missing_ok=True)
        self._worker_start_error = None
        if self._worker_log_file is not None:
            self._worker_log_file.close()
        self._worker_log_file = self.worker_log_path.open("ab")
        try:
            self.worker_process = subprocess.Popen(
                [
                    sys.executable,
                    str(self.repo_root / "pipeline/zs32_inference_worker.py"),
                    "--socket",
                    str(self.worker_socket_path),
                    "--demo-config",
                    str(self.demo_config),
                ],
                cwd=self.repo_root,
                start_new_session=True,
                stdout=self._worker_log_file,
                stderr=subprocess.STDOUT,
            )
        except OSError as error:
            self.worker_process = None
            self._worker_start_error = f"{type(error).__name__}: {error}; log={self.worker_log_path}"

    def start(self, part_id: str) -> None:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Stage35 is already running")
        if self.worker_process is not None:
            worker_returncode = self.worker_process.poll()
            if worker_returncode is not None:
                progress = self._failure_progress("Inference worker", worker_returncode, self.worker_log_path)
                raise RuntimeError(progress.error or progress.message)
            if not self.worker_socket_path.exists():
                raise RuntimeError(f"Inference worker is not ready; log={self.worker_log_path}")
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
        for option, value in (
            ("--demo-config", self.demo_config),
            ("--inference-socket", self.worker_socket_path if self.worker_process is not None else None),
        ):
            if value is not None:
                command.extend((option, str(value)))
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
        if self._worker_start_error is not None:
            return ProgressRecord(
                self._part_id or "unknown",
                None,
                "failed",
                "Inference worker failed to start",
                "",
                error=self._worker_start_error,
            )
        if self.worker_process is not None:
            worker_returncode = self.worker_process.poll()
            if worker_returncode is not None:
                return self._failure_progress("Inference worker", worker_returncode, self.worker_log_path)
            if not self.worker_socket_path.exists():
                return ProgressRecord(
                    self._part_id or "unknown",
                    None,
                    "worker_starting",
                    f"Loading Demo models; log={self.worker_log_path}",
                    "",
                )
        if self.process is not None:
            returncode = self.process.poll()
            if returncode is not None:
                return self._failure_progress("Stage35", returncode, self.log_path)
        return None

    @property
    def running(self) -> bool:
        """Return whether the owned Stage35 child is still active."""
        stage35_running = self.process is not None and self.process.poll() is None
        worker_loading = (
            self.worker_process is not None
            and self.worker_process.poll() is None
            and not self.worker_socket_path.exists()
        )
        return stage35_running or worker_loading

    @property
    def worker_ready(self) -> bool:
        """Return whether the resident worker is alive and has published its socket."""
        return (
            self.worker_process is not None
            and self.worker_process.poll() is None
            and self.worker_socket_path.exists()
        )

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
        worker = self.worker_process
        if worker is not None and worker.poll() is None:
            try:
                if self.worker_socket_path.exists():
                    InferenceWorkerClient(self.worker_socket_path, timeout=timeout).stop("dashboard-close")
                worker.wait(timeout=timeout)
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=timeout)
                except (subprocess.TimeoutExpired, TimeoutError):
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
        if self._worker_log_file is not None:
            self._worker_log_file.close()
            self._worker_log_file = None
        self.worker_process = None
        self.worker_socket_path.unlink(missing_ok=True)
