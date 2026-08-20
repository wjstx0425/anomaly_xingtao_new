"""Deterministic transport tests for the resident inference worker."""

from __future__ import annotations

import json
import importlib.util
import socket
import stat
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from zs32_inspection.dashboard.inference_worker import (
    InferenceWorkerClient,
    InferenceWorkerServer,
    WorkerTimeoutError,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKER_SCRIPT = REPO_ROOT / "pipeline/zs32_inference_worker.py"


def _load_worker(name: str = "pipeline_zs32_demo_worker") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, WORKER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {WORKER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _RunningServer:
    def __init__(self, socket_path: Path, executor: object) -> None:
        self.server = InferenceWorkerServer(socket_path, executor)  # type: ignore[arg-type]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> tuple[InferenceWorkerServer, threading.Thread]:
        self.thread.start()
        assert self.server.wait_until_ready(timeout=1.0)
        return self.server, self.thread

    def __exit__(self, *_args: object) -> None:
        if self.thread.is_alive():
            InferenceWorkerClient(self.server.socket_path, timeout=1.0).stop("test-cleanup")
        self.thread.join(timeout=1.0)


def _raw_request(socket_path: Path, payload: bytes) -> dict[str, object]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1.0)
        connection.connect(str(socket_path))
        connection.sendall(payload + b"\n")
        response = connection.makefile("rb").readline()
    return json.loads(response)


def test_server_executes_jobs_serially_and_publishes_private_socket(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def executor(argv: list[str]) -> int:
        calls.append(argv)
        return 7

    socket_path = tmp_path / "worker.sock"
    with _RunningServer(socket_path, executor):
        response = InferenceWorkerClient(socket_path, timeout=1.0).execute("job-1", ["fuse", "--part-id", "p1"])

        assert response.job_id == "job-1"
        assert response.returncode == 7
        assert response.error is None
        assert calls == [["fuse", "--part-id", "p1"]]
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600


def test_bad_request_and_executor_error_do_not_kill_server(tmp_path: Path) -> None:
    def executor(argv: list[str]) -> int:
        if argv == ["explode"]:
            raise RuntimeError("GPU unavailable")
        return 0

    socket_path = tmp_path / "worker.sock"
    with _RunningServer(socket_path, executor):
        malformed = _raw_request(socket_path, b'{"job_id":"broken","argv":')
        invalid = _raw_request(socket_path, b'{"job_id":"bad","argv":[1]}')
        failed = InferenceWorkerClient(socket_path, timeout=1.0).execute("job-error", ["explode"])
        recovered = InferenceWorkerClient(socket_path, timeout=1.0).execute("job-ok", ["infer"])

        assert malformed == {"job_id": None, "returncode": None, "error": "invalid JSON request"}
        assert invalid == {
            "job_id": "bad",
            "returncode": None,
            "error": "argv must be a list of strings",
        }
        assert failed.job_id == "job-error"
        assert failed.returncode is None
        assert failed.error == "RuntimeError: GPU unavailable"
        assert recovered.returncode == 0
        assert recovered.error is None


def test_client_timeout_is_bounded_and_server_can_finish_later(tmp_path: Path) -> None:
    release = threading.Event()

    def executor(_argv: list[str]) -> int:
        release.wait(timeout=1.0)
        return 0

    socket_path = tmp_path / "worker.sock"
    with _RunningServer(socket_path, executor):
        with pytest.raises(WorkerTimeoutError, match="timed out"):
            InferenceWorkerClient(socket_path, timeout=0.05).execute("slow", ["infer"])
        release.set()


def test_stop_returns_response_exits_cleanly_and_removes_socket(tmp_path: Path) -> None:
    socket_path = tmp_path / "worker.sock"
    server = InferenceWorkerServer(socket_path, lambda _argv: 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    assert server.wait_until_ready(timeout=1.0)

    response = InferenceWorkerClient(socket_path, timeout=1.0).stop("shutdown")
    thread.join(timeout=1.0)

    assert response.job_id == "shutdown"
    assert response.returncode == 0
    assert response.error is None
    assert not thread.is_alive()
    assert not socket_path.exists()


def test_existing_socket_path_is_never_overwritten(tmp_path: Path) -> None:
    socket_path = tmp_path / "worker.sock"
    socket_path.write_text("owned by another process", encoding="utf-8")
    server = InferenceWorkerServer(socket_path, lambda _argv: 0)

    with pytest.raises(FileExistsError, match="already exists"):
        server.serve_forever()

    assert socket_path.read_text(encoding="utf-8") == "owned by another process"


def test_worker_cli_accepts_only_demo_config(tmp_path: Path) -> None:
    worker = _load_worker()
    demo_config = tmp_path / "demo.json"
    socket_path = tmp_path / "worker.sock"

    args = worker.build_parser().parse_args(
        ["--socket", str(socket_path), "--demo-config", str(demo_config)],
    )

    assert args.socket == socket_path
    assert args.demo_config == demo_config
    destinations = {action.dest for action in worker.build_parser()._actions}
    assert "runtime_bundle" not in destinations

    with pytest.raises(SystemExit):
        worker.build_parser().parse_args(
            ["--socket", str(socket_path), "--runtime-bundle", "old.json"],
        )


def test_worker_prepares_runtime_once_and_reuses_it_for_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_worker("pipeline_zs32_demo_worker_reuse")
    demo_config = tmp_path / "demo.json"
    socket_path = tmp_path / "worker.sock"
    prepared: list[Path] = []
    jobs: list[tuple[list[str], object, Path]] = []
    executors: list[object] = []
    closed: list[object] = []
    runtime = SimpleNamespace(close=lambda: closed.append(runtime))

    demo_module = SimpleNamespace(
        prepare_demo_runtime=lambda path: prepared.append(path) or runtime,
        run_argv=lambda argv, *, runtime, startup_config_path: jobs.append(
            (argv, runtime, startup_config_path),
        ),
    )

    class Server:
        def __init__(self, path: Path, executor: object) -> None:
            assert path == socket_path
            executors.append(executor)

        def serve_forever(self) -> None:
            executors[0](["--part-id", "p1"])
            executors[0](["--part-id", "p2"])

    monkeypatch.setattr(worker, "_load_demo_inference", lambda: demo_module)
    monkeypatch.setattr(worker, "InferenceWorkerServer", Server)

    worker.run(
        worker.build_parser().parse_args(
            ["--socket", str(socket_path), "--demo-config", str(demo_config)],
        ),
    )

    assert prepared == [demo_config]
    assert jobs == [
        (["--part-id", "p1"], runtime, demo_config),
        (["--part-id", "p2"], runtime, demo_config),
    ]
    assert closed == [runtime]


def test_worker_closes_runtime_when_server_exits_with_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_worker("pipeline_zs32_demo_worker_cleanup")
    demo_config = tmp_path / "demo.json"
    socket_path = tmp_path / "worker.sock"
    close_calls: list[str] = []
    runtime = SimpleNamespace(close=lambda: close_calls.append("closed"))
    demo_module = SimpleNamespace(prepare_demo_runtime=lambda _path: runtime)

    class Server:
        def __init__(self, path: Path, executor: object) -> None:
            assert path == socket_path
            assert callable(executor)

        def serve_forever(self) -> None:
            raise RuntimeError("server failed")

    monkeypatch.setattr(worker, "_load_demo_inference", lambda: demo_module)
    monkeypatch.setattr(worker, "InferenceWorkerServer", Server)

    with pytest.raises(RuntimeError, match="server failed"):
        worker.run(
            worker.build_parser().parse_args(
                ["--socket", str(socket_path), "--demo-config", str(demo_config)],
            ),
        )

    assert close_calls == ["closed"]
