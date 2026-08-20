"""Small, serial Unix-socket transport for a resident inference executor."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_MAX_MESSAGE_BYTES = 1024 * 1024
_SOCKET_MODE = 0o600
_STOP_COMMAND = "STOP"


class WorkerTimeoutError(TimeoutError):
    """Raised when a worker request does not finish before its deadline."""


class WorkerProtocolError(ValueError):
    """Raised when the worker returns a malformed response."""


@dataclass(frozen=True)
class WorkerResponse:
    """Result returned for one inference job or control request."""

    job_id: str | None
    returncode: int | None
    error: str | None


class InferenceWorkerServer:
    """Serve one inference request at a time on a private Unix socket."""

    def __init__(self, socket_path: Path, executor: Callable[[list[str]], int]) -> None:
        self.socket_path = socket_path.expanduser().resolve()
        self._executor = executor
        self._ready = threading.Event()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait until the listening socket has been atomically published."""
        return self._ready.wait(timeout)

    def serve_forever(self) -> None:
        """Run serial jobs until a valid STOP request is acknowledged."""
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(self.socket_path):
            raise FileExistsError(f"worker socket already exists: {self.socket_path}")

        temporary_path = self.socket_path.parent / f".inference-worker-{os.getpid()}-{uuid4().hex[:8]}.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        published = False
        try:
            listener.bind(str(temporary_path))
            os.chmod(temporary_path, _SOCKET_MODE)
            listener.listen(1)
            if os.path.lexists(self.socket_path):
                raise FileExistsError(f"worker socket already exists: {self.socket_path}")
            # A same-filesystem hard link publishes the bound inode atomically and,
            # unlike rename(), can never replace a path created by another process.
            os.link(temporary_path, self.socket_path)
            temporary_path.unlink()
            published = True
            self._ready.set()

            stopping = False
            while not stopping:
                connection, _ = listener.accept()
                with connection:
                    response, stopping = self._handle_connection(connection)
                    self._send_response(connection, response)
        finally:
            listener.close()
            if os.path.lexists(temporary_path):
                temporary_path.unlink()
            if published and os.path.lexists(self.socket_path):
                self.socket_path.unlink()
            self._ready.clear()

    def _handle_connection(self, connection: socket.socket) -> tuple[WorkerResponse, bool]:
        try:
            request = _read_json_line(connection)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return WorkerResponse(None, None, "invalid JSON request"), False
        except ValueError as error:
            return WorkerResponse(None, None, str(error)), False

        if not isinstance(request, dict):
            return WorkerResponse(None, None, "request must be a JSON object"), False
        raw_job_id = request.get("job_id")
        job_id = raw_job_id if isinstance(raw_job_id, str) and raw_job_id else None
        if job_id is None:
            return WorkerResponse(None, None, "job_id must be a non-empty string"), False
        argv = request.get("argv")
        if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
            return WorkerResponse(job_id, None, "argv must be a list of strings"), False
        if request.get("command") == _STOP_COMMAND:
            return WorkerResponse(job_id, 0, None), True
        try:
            returncode = self._executor(list(argv))
            if isinstance(returncode, bool) or not isinstance(returncode, int):
                raise TypeError("executor must return an integer return code")
        except Exception as error:  # noqa: BLE001 - one failed job must not terminate the worker
            return WorkerResponse(job_id, None, f"{type(error).__name__}: {error}"), False
        return WorkerResponse(job_id, returncode, None), False

    @staticmethod
    def _send_response(connection: socket.socket, response: WorkerResponse) -> None:
        payload = {
            "job_id": response.job_id,
            "returncode": response.returncode,
            "error": response.error,
        }
        try:
            connection.sendall(_encode_json_line(payload))
        except (BrokenPipeError, ConnectionResetError):
            # A timed-out client may disconnect while its serial job is still finishing.
            return


class InferenceWorkerClient:
    """Submit bounded requests to one resident inference worker."""

    def __init__(self, socket_path: Path, *, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.socket_path = socket_path.expanduser().resolve()
        self.timeout = float(timeout)

    def execute(self, job_id: str, argv: list[str]) -> WorkerResponse:
        """Execute one argv job and wait for its structured response."""
        return self._request({"job_id": job_id, "argv": argv})

    def stop(self, job_id: str = "stop") -> WorkerResponse:
        """Ask the server to acknowledge STOP and exit cleanly."""
        return self._request({"job_id": job_id, "argv": [], "command": _STOP_COMMAND})

    def _request(self, payload: dict[str, Any]) -> WorkerResponse:
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                connection.settimeout(_remaining(deadline))
                try:
                    connection.connect(str(self.socket_path))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    connection.close()
                    time.sleep(min(0.05, _remaining(deadline)))
            with connection:
                connection.settimeout(_remaining(deadline))
                connection.sendall(_encode_json_line(payload))
                connection.settimeout(_remaining(deadline))
                response = _read_json_line(connection)
        except (TimeoutError, socket.timeout) as error:
            raise WorkerTimeoutError(f"worker request timed out after {self.timeout:.3f}s") from error
        return _parse_response(response)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WorkerTimeoutError("worker request timed out")
    return remaining


def _encode_json_line(payload: object) -> bytes:
    return (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _read_json_line(connection: socket.socket) -> object:
    with connection.makefile("rb") as stream:
        raw = stream.readline(_MAX_MESSAGE_BYTES + 1)
    if not raw:
        raise ValueError("empty worker message")
    if len(raw) > _MAX_MESSAGE_BYTES:
        raise ValueError("worker message exceeds size limit")
    if not raw.endswith(b"\n"):
        raise ValueError("worker message must end with a newline")
    return json.loads(raw.decode("utf-8"))


def _parse_response(payload: object) -> WorkerResponse:
    if not isinstance(payload, dict):
        raise WorkerProtocolError("worker response must be a JSON object")
    if set(payload) != {"job_id", "returncode", "error"}:
        raise WorkerProtocolError("worker response fields are invalid")
    job_id = payload["job_id"]
    returncode = payload["returncode"]
    error = payload["error"]
    if job_id is not None and not isinstance(job_id, str):
        raise WorkerProtocolError("worker response job_id is invalid")
    if returncode is not None and (isinstance(returncode, bool) or not isinstance(returncode, int)):
        raise WorkerProtocolError("worker response returncode is invalid")
    if error is not None and not isinstance(error, str):
        raise WorkerProtocolError("worker response error is invalid")
    return WorkerResponse(job_id, returncode, error)
