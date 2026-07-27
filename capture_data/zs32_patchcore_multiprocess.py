# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resident process-isolated PatchCore inference for the eight ZS32 views."""

from __future__ import annotations

import math
import multiprocessing
import tempfile
import threading
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from capture_data.zs32_model_runtime import ModelEvidence, PatchcoreArtifacts, PatchcoreSpec
from zs32_inspection.domain.views import VIEW_ORDER

_PROCESS_COUNTS = frozenset({1, 2, 4, 8})
_SHARD_LAYOUTS: dict[int, tuple[tuple[str, ...], ...]] = {
    1: (VIEW_ORDER,),
    2: (
        ("front", "front_left", "front_right", "front_secondary"),
        ("back", "back_left", "back_right", "back_secondary"),
    ),
    4: (
        ("front_left", "back_right"),
        ("back_left", "front_right"),
        ("front", "front_secondary"),
        ("back", "back_secondary"),
    ),
    8: tuple((view,) for view in VIEW_ORDER),
}
WorkerTarget = Callable[[Connection, dict[str, PatchcoreSpec], tuple[str, ...], dict[str, object]], None]


def view_shards(views: Collection[str], process_count: int) -> tuple[tuple[str, ...], ...]:
    """Return the fixed, latency-balanced shard layout for all eight views."""
    if process_count not in _PROCESS_COUNTS:
        raise ValueError("PatchCore process_count must be 1, 2, 4, or 8")
    supplied = tuple(views)
    if len(supplied) != len(VIEW_ORDER) or set(supplied) != set(VIEW_ORDER):
        raise ValueError(f"PatchCore views must contain exactly the canonical eight views: {VIEW_ORDER}")
    return _SHARD_LAYOUTS[process_count]


@dataclass(slots=True)
class _Worker:
    """Parent-owned process, pipe, and immutable view assignment."""

    views: tuple[str, ...]
    connection: Connection
    process: multiprocessing.Process
    failed_reason: str | None = None


@dataclass(slots=True)
class _RequestState:
    """Per-child response set and explicit batch-boundary state."""

    pending_views: set[str]
    done_count: int = 0
    ack_sent: bool = False
    idle_count: int = 0


def _error_envelope(request_id: str, view: str, exc: Exception) -> dict[str, object]:
    """Build a pickle-safe child error response."""
    return {
        "type": "RESULT",
        "request_id": request_id,
        "view": view,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _evidence_envelope(request_id: str, view: str, evidence: ModelEvidence) -> dict[str, object]:
    """Flatten one PatchCore result into primitive IPC values."""
    artifacts = evidence.patchcore_artifacts
    if artifacts is None:
        raise RuntimeError(f"PatchCore {view} returned no PatchCore artifacts")
    return {
        "type": "RESULT",
        "request_id": request_id,
        "view": view,
        "score": evidence.score,
        "evidence_path": str(evidence.evidence_path.resolve()),
        "raw_anomaly_map_path": str(artifacts.raw_anomaly_map_path.resolve()),
        "mask_path": str(artifacts.mask_path.resolve()),
        "raw_anomaly_map_shape": list(artifacts.raw_anomaly_map_shape),
        "mask_shape": list(artifacts.mask_shape),
        "mask_source": artifacts.mask_source,
        "diagnostic_mask_threshold": artifacts.diagnostic_mask_threshold,
        "error": None,
    }


def _patchcore_worker_main(
    connection: Connection,
    specs: dict[str, PatchcoreSpec],
    views: tuple[str, ...],
    options: dict[str, object],
) -> None:
    """Load one isolated model shard and serve pure-data prediction requests."""
    try:
        # Keep heavy CUDA/Anomalib initialization inside the spawned child.
        from capture_data.zs32_model_runtime import AnomalibPatchcoreBackend

        backend = AnomalibPatchcoreBackend(
            specs,
            views=views,
            accelerator=str(options.get("accelerator", "auto")),
            devices=int(options.get("devices", 1)),
        )
        warmup_value = options.get("warmup_crops")
        if not isinstance(warmup_value, dict) or set(warmup_value) != set(views):
            raise ValueError(f"warmup_crops must contain exactly worker views: {views}")
        with tempfile.TemporaryDirectory(prefix="zs32-patchcore-warmup-") as directory:
            evidence_dir = Path(directory)
            for view in views:
                backend.predict(
                    view,
                    Path(str(warmup_value[view])),
                    evidence_dir / f"{view}.png",
                )
        connection.send({"type": "READY", "views": list(views)})
    except Exception as exc:
        try:
            connection.send(
                {
                    "type": "STARTUP_ERROR",
                    "views": list(views),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        finally:
            connection.close()
        return

    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                return
            if not isinstance(message, dict):
                return
            message_type = message.get("type")
            if message_type == "SHUTDOWN":
                return
            if message_type != "PREDICT":
                return
            request_id = str(message.get("request_id", ""))
            crops = message.get("crops")
            evidence_dir_value = message.get("evidence_dir")
            threshold_value = message.get("diagnostic_mask_threshold")
            if (
                not request_id
                or not isinstance(crops, dict)
                or set(crops) != set(views)
                or not isinstance(evidence_dir_value, str)
                or isinstance(threshold_value, bool)
                or not isinstance(threshold_value, (float, int))
            ):
                for view in views:
                    connection.send(
                        _error_envelope(request_id, view, ValueError("malformed PREDICT request")),
                    )
                continue
            evidence_dir = Path(evidence_dir_value)
            threshold = float(threshold_value)
            for view in views:
                try:
                    evidence = backend.predict(
                        view,
                        Path(str(crops[view])),
                        evidence_dir / f"{view}.png",
                        diagnostic_mask_threshold=threshold,
                    )
                    connection.send(_evidence_envelope(request_id, view, evidence))
                except Exception as exc:
                    connection.send(_error_envelope(request_id, view, exc))
            connection.send({"type": "DONE", "request_id": request_id})
            acknowledgement = connection.recv()
            if acknowledgement != {"type": "DONE_ACK", "request_id": request_id}:
                return
            connection.send({"type": "IDLE", "request_id": request_id})
    finally:
        connection.close()


class ResidentMultiprocessPatchcoreBackend:
    """Own persistent spawned PatchCore workers and aggregate eight-view results."""

    def __init__(
        self,
        specs: Mapping[str, PatchcoreSpec],
        views: Collection[str],
        process_count: int,
        *,
        accelerator: str = "auto",
        devices: int = 1,
        warmup_crops: Mapping[str, Path] | None = None,
        startup_timeout: float = 300.0,
        request_timeout: float = 60.0,
        shutdown_timeout: float = 2.0,
        worker_target: WorkerTarget | None = None,
        worker_options: Mapping[str, object] | None = None,
    ) -> None:
        """Spawn all shards and return only after every child reports ``READY``."""
        self._shards = view_shards(views, process_count)
        self._views = VIEW_ORDER
        if set(specs) != set(self._views) or len(specs) != len(self._views):
            raise ValueError(f"PatchCore specs must contain exactly the canonical eight views: {self._views}")
        self._require_positive_timeout(startup_timeout, "startup_timeout")
        self._require_positive_timeout(request_timeout, "request_timeout")
        self._require_positive_timeout(shutdown_timeout, "shutdown_timeout")
        if warmup_crops is not None and (
            set(warmup_crops) != set(self._views) or len(warmup_crops) != len(self._views)
        ):
            raise ValueError(f"warmup_crops must contain exactly the canonical eight views: {self._views}")
        if worker_target is None and warmup_crops is None:
            raise ValueError("warmup_crops is required for the real PatchCore worker target")

        self._request_timeout = float(request_timeout)
        self._shutdown_timeout = float(shutdown_timeout)
        self._closed = False
        self._lock = threading.Lock()
        self._workers: list[_Worker] = []
        context = multiprocessing.get_context("spawn")
        target = worker_target or _patchcore_worker_main
        common_options = dict(worker_options or {})
        common_options.update({"accelerator": accelerator, "devices": devices})

        try:
            for index, shard in enumerate(self._shards):
                parent_connection, child_connection = context.Pipe(duplex=True)
                shard_options = dict(common_options)
                if warmup_crops is not None:
                    shard_options["warmup_crops"] = {
                        view: str(Path(warmup_crops[view]).expanduser().resolve()) for view in shard
                    }
                process = context.Process(
                    name=f"zs32-patchcore-{index}",
                    target=target,
                    args=(
                        child_connection,
                        {view: specs[view] for view in shard},
                        shard,
                        shard_options,
                    ),
                )
                process.start()
                child_connection.close()
                self._workers.append(_Worker(shard, parent_connection, process))
            self._wait_until_ready(float(startup_timeout))
        except Exception:
            self._abort_startup()
            raise

    @staticmethod
    def _require_positive_timeout(value: float, field: str) -> None:
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{field} must be a positive finite number")

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """Return stable child PIDs in shard order."""
        return tuple(cast("int", worker.process.pid) for worker in self._workers)

    @property
    def worker_processes(self) -> tuple[multiprocessing.Process, ...]:
        """Expose read-only process handles for lifecycle diagnostics."""
        return tuple(worker.process for worker in self._workers)

    @property
    def closed(self) -> bool:
        """Return whether shutdown has begun."""
        return self._closed

    def _wait_until_ready(self, timeout: float) -> None:
        pending = {worker.process.sentinel: worker for worker in self._workers}
        pipe_workers = {worker.connection: worker for worker in self._workers}
        deadline = time.monotonic() + timeout
        while pipe_workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("PatchCore worker READY timeout")
            ready = wait([*pipe_workers, *pending], timeout=remaining)
            if not ready:
                raise RuntimeError("PatchCore worker READY timeout")
            for item in ready:
                if item in pending:
                    worker = pending[item]
                    worker.process.join(timeout=0)
                    raise RuntimeError(
                        f"PatchCore worker exited before READY: pid={worker.process.pid}, "
                        f"exitcode={worker.process.exitcode}",
                    )
            for item in ready:
                if item not in pipe_workers:
                    continue
                worker = pipe_workers[item]
                try:
                    envelope = worker.connection.recv()
                except EOFError as exc:
                    raise RuntimeError(f"PatchCore worker EOF before READY: pid={worker.process.pid}") from exc
                if (
                    not isinstance(envelope, dict)
                    or envelope.get("type") != "READY"
                    or tuple(envelope.get("views", ())) != worker.views
                ):
                    detail = envelope.get("error") if isinstance(envelope, dict) else repr(envelope)
                    raise RuntimeError(f"PatchCore worker sent malformed READY envelope: {detail}")
                del pipe_workers[item]
                del pending[worker.process.sentinel]

    def predict_all(
        self,
        crops: Mapping[str, Path],
        evidence_dir: Path,
        *,
        diagnostic_mask_threshold: float = 0.65,
    ) -> dict[str, ModelEvidence | Exception]:
        """Dispatch one eight-view request and reconstruct fail-closed evidence."""
        with self._lock:
            if self._closed:
                raise RuntimeError("ResidentMultiprocessPatchcoreBackend is closed")
            if set(crops) != set(self._views) or len(crops) != len(self._views):
                raise ValueError(f"crops must contain exactly the canonical eight views: {self._views}")
            if (
                isinstance(diagnostic_mask_threshold, bool)
                or not isinstance(diagnostic_mask_threshold, (float, int))
                or not math.isfinite(float(diagnostic_mask_threshold))
            ):
                raise ValueError("diagnostic_mask_threshold must be finite")
            request_id = uuid4().hex
            resolved_crops = {
                view: str(Path(crops[view]).expanduser().resolve()) for view in self._views
            }
            resolved_evidence_dir = Path(evidence_dir).expanduser().resolve()
            results: dict[str, ModelEvidence | Exception] = {}
            pending: dict[Connection, _RequestState] = {}
            received: dict[Connection, int] = {}

            for worker in self._workers:
                if worker.failed_reason is not None:
                    self._fail_views(
                        results,
                        worker.views,
                        f"PatchCore worker unavailable: {worker.failed_reason}",
                    )
                    continue
                if not worker.process.is_alive():
                    worker.failed_reason = (
                        f"worker exited: pid={worker.process.pid}, exitcode={worker.process.exitcode}"
                    )
                    self._fail_views(
                        results,
                        worker.views,
                        f"PatchCore worker exited: pid={worker.process.pid}, "
                        f"exitcode={worker.process.exitcode}",
                    )
                    continue
                message = {
                    "type": "PREDICT",
                    "request_id": request_id,
                    "crops": {view: resolved_crops[view] for view in worker.views},
                    "evidence_dir": str(resolved_evidence_dir),
                    "diagnostic_mask_threshold": float(diagnostic_mask_threshold),
                }
                try:
                    worker.connection.send(message)
                except (BrokenPipeError, EOFError, OSError) as exc:
                    worker.failed_reason = f"worker send failed: {exc}"
                    self._fail_views(results, worker.views, f"PatchCore worker send failed: {exc}")
                    continue
                pending[worker.connection] = _RequestState(set(worker.views))
                received[worker.connection] = 0

            self._collect_results(
                request_id,
                resolved_evidence_dir,
                pending,
                received,
                results,
            )
            self._retire_failed_workers()
            return {
                view: results.get(view, RuntimeError(f"missing PatchCore response for {view}"))
                for view in self._views
            }

    def _collect_results(
        self,
        request_id: str,
        evidence_dir: Path,
        pending: dict[Connection, _RequestState],
        received: dict[Connection, int],
        results: dict[str, ModelEvidence | Exception],
    ) -> None:
        worker_by_connection = {worker.connection: worker for worker in self._workers}
        worker_by_sentinel = {worker.process.sentinel: worker for worker in self._workers}
        deadline = time.monotonic() + self._request_timeout
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail_pending_after_deadline(pending, received, results)
                return
            active_sentinels = [
                worker.process.sentinel
                for connection, worker in worker_by_connection.items()
                if connection in pending
            ]
            ready = wait([*pending, *active_sentinels], timeout=remaining)
            if not ready:
                self._fail_pending_after_deadline(pending, received, results)
                return
            for item in ready:
                worker = worker_by_sentinel.get(item)
                if worker is None or worker.connection not in pending:
                    continue
                worker.process.join(timeout=0)
                reason = (
                    f"worker exited: pid={worker.process.pid}, "
                    f"exitcode={worker.process.exitcode}"
                )
                worker.failed_reason = reason
                self._fail_connection(
                    pending,
                    results,
                    worker.connection,
                    f"PatchCore {reason}",
                )
            for item in ready:
                if item not in pending:
                    continue
                connection = cast("Connection", item)
                worker = worker_by_connection[connection]
                try:
                    envelope = connection.recv()
                except EOFError:
                    worker.process.join(timeout=0)
                    reason = (
                        f"PatchCore worker exited: pid={worker.process.pid}, "
                        f"exitcode={worker.process.exitcode}"
                        if worker.process.exitcode not in {None, 0}
                        else f"PatchCore worker EOF: pid={worker.process.pid}"
                    )
                    worker.failed_reason = reason
                    self._fail_connection(pending, results, connection, reason)
                    continue
                received[connection] += 1
                self._consume_envelope(
                    request_id,
                    evidence_dir,
                    worker,
                    envelope,
                    pending,
                    results,
                )
                if connection in pending:
                    self._complete_worker_batch(worker, pending)

    def _consume_envelope(
        self,
        request_id: str,
        evidence_dir: Path,
        worker: _Worker,
        envelope: object,
        pending: dict[Connection, _RequestState],
        results: dict[str, ModelEvidence | Exception],
    ) -> None:
        connection = worker.connection
        state = pending[connection]
        expected = state.pending_views
        if not isinstance(envelope, dict):
            worker.failed_reason = "malformed response: not an object"
            self._fail_connection(pending, results, connection, "malformed PatchCore response: not an object")
            return
        envelope_type = envelope.get("type")
        if envelope_type == "IDLE":
            if set(envelope) != {"type", "request_id"}:
                worker.failed_reason = "malformed IDLE envelope"
                self._fail_connection(pending, results, connection, "malformed PatchCore IDLE envelope")
                return
            if envelope.get("request_id") != request_id:
                worker.failed_reason = "stale IDLE request ID"
                self._fail_connection(pending, results, connection, "stale IDLE request ID")
                return
            state.idle_count += 1
            if state.done_count != 1 or not state.ack_sent or expected or state.idle_count != 1:
                worker.failed_reason = "invalid IDLE boundary"
                self._fail_connection(pending, results, connection, "invalid IDLE boundary")
            return
        if envelope_type == "DONE":
            if set(envelope) != {"type", "request_id"}:
                worker.failed_reason = "malformed DONE envelope"
                self._fail_connection(pending, results, connection, "malformed PatchCore DONE envelope")
                return
            if envelope.get("request_id") != request_id:
                worker.failed_reason = "stale DONE request ID"
                self._fail_connection(pending, results, connection, "stale DONE request ID")
                return
            state.done_count += 1
            if state.done_count != 1:
                worker.failed_reason = "duplicate DONE"
                self._fail_connection(pending, results, connection, "duplicate DONE")
                return
            if expected:
                worker.failed_reason = "missing response before DONE"
                self._fail_connection(pending, results, connection, "missing response before DONE")
                return
            try:
                connection.send({"type": "DONE_ACK", "request_id": request_id})
            except (BrokenPipeError, EOFError, OSError) as exc:
                worker.failed_reason = f"DONE acknowledgement failed: {exc}"
                self._fail_connection(pending, results, connection, "DONE acknowledgement failed")
                return
            state.ack_sent = True
            return
        view_value = envelope.get("view")
        view = view_value if isinstance(view_value, str) and view_value in worker.views else None
        if envelope_type != "RESULT" or view is None:
            worker.failed_reason = "malformed response envelope"
            self._fail_connection(pending, results, connection, "malformed PatchCore response envelope")
            return
        if state.done_count:
            worker.failed_reason = f"response after DONE for {view}"
            self._fail_connection(pending, results, connection, f"response after DONE for {view}")
            return
        if view not in expected:
            worker.failed_reason = f"duplicate response for {view}"
            results[view] = RuntimeError(f"duplicate PatchCore response for {view}")
            return
        expected.remove(view)
        if envelope.get("request_id") != request_id:
            worker.failed_reason = f"stale request ID for {view}"
            results[view] = RuntimeError(f"stale request ID in PatchCore response for {view}")
        else:
            error = envelope.get("error")
            if error is not None:
                if not isinstance(error, str) or not error:
                    worker.failed_reason = f"malformed error response for {view}"
                    results[view] = RuntimeError(f"malformed PatchCore error response for {view}")
                else:
                    results[view] = RuntimeError(f"PatchCore {view}: {error}")
            else:
                try:
                    results[view] = self._reconstruct_evidence(envelope, view, evidence_dir)
                except (KeyError, TypeError, ValueError) as exc:
                    worker.failed_reason = f"malformed response for {view}: {exc}"
                    results[view] = RuntimeError(f"malformed PatchCore response for {view}: {exc}")

    def _complete_worker_batch(
        self,
        worker: _Worker,
        pending: dict[Connection, _RequestState],
    ) -> None:
        connection = worker.connection
        state = pending[connection]
        if state.pending_views or state.done_count != 1 or not state.ack_sent or state.idle_count != 1:
            return
        del pending[connection]

    @staticmethod
    def _reconstruct_evidence(
        envelope: Mapping[str, object],
        view: str,
        evidence_dir: Path,
    ) -> ModelEvidence:
        score_value = envelope["score"]
        if isinstance(score_value, bool) or not isinstance(score_value, (float, int)):
            raise TypeError("score must be numeric")
        score = float(score_value)
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        evidence_path = ResidentMultiprocessPatchcoreBackend._expected_path(
            envelope["evidence_path"],
            evidence_dir / f"{view}.png",
            "evidence_path",
        )
        raw_path = ResidentMultiprocessPatchcoreBackend._expected_path(
            envelope["raw_anomaly_map_path"],
            evidence_dir / "raw_maps" / f"{view}.npy",
            "raw_anomaly_map_path",
        )
        mask_path = ResidentMultiprocessPatchcoreBackend._expected_path(
            envelope["mask_path"],
            evidence_dir / "masks" / f"{view}.png",
            "mask_path",
        )
        raw_shape = ResidentMultiprocessPatchcoreBackend._shape(
            envelope["raw_anomaly_map_shape"],
            "raw_anomaly_map_shape",
        )
        mask_shape = ResidentMultiprocessPatchcoreBackend._shape(envelope["mask_shape"], "mask_shape")
        mask_source = envelope["mask_source"]
        if mask_source not in {"pred_mask", "diagnostic_anomaly_map"}:
            raise ValueError("mask_source is invalid")
        threshold_value = envelope["diagnostic_mask_threshold"]
        if threshold_value is None:
            threshold = None
        elif isinstance(threshold_value, bool) or not isinstance(threshold_value, (float, int)):
            raise TypeError("diagnostic_mask_threshold must be numeric or null")
        else:
            threshold = float(threshold_value)
            if not math.isfinite(threshold):
                raise ValueError("diagnostic_mask_threshold must be finite")
        return ModelEvidence(
            score=score,
            evidence_path=evidence_path,
            patchcore_artifacts=PatchcoreArtifacts(
                raw_anomaly_map_path=raw_path,
                mask_path=mask_path,
                mask_source=cast("Literal['pred_mask', 'diagnostic_anomaly_map']", mask_source),
                diagnostic_mask_threshold=threshold,
                raw_anomaly_map_shape=raw_shape,
                mask_shape=mask_shape,
            ),
        )

    @staticmethod
    def _expected_path(value: object, expected: Path, field: str) -> Path:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a path string")
        path = Path(value)
        if not path.is_absolute() or path.resolve() != expected.resolve():
            raise ValueError(f"{field} does not match the assigned artifact path")
        if not path.is_file():
            raise ValueError(f"{field} artifact does not exist")
        return path.resolve()

    @staticmethod
    def _shape(value: object, field: str) -> tuple[int, int]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
        ):
            raise ValueError(f"{field} must contain two positive integers")
        return cast("tuple[int, int]", tuple(value))

    @staticmethod
    def _fail_views(
        results: dict[str, ModelEvidence | Exception],
        views: Collection[str],
        reason: str,
    ) -> None:
        for view in views:
            results[view] = RuntimeError(f"{reason}; affected view={view}")

    def _fail_connection(
        self,
        pending: dict[Connection, _RequestState],
        results: dict[str, ModelEvidence | Exception],
        connection: Connection,
        reason: str,
    ) -> None:
        worker = next(worker for worker in self._workers if worker.connection is connection)
        self._fail_views(results, worker.views, reason)
        pending.pop(connection)

    def _fail_pending_after_deadline(
        self,
        pending: dict[Connection, _RequestState],
        received: Mapping[Connection, int],
        results: dict[str, ModelEvidence | Exception],
    ) -> None:
        worker_by_connection = {worker.connection: worker for worker in self._workers}
        for connection, state in tuple(pending.items()):
            worker = worker_by_connection[connection]
            if not state.pending_views and state.done_count == 0:
                reason = "missing DONE"
                affected_views = worker.views
            elif not state.pending_views and state.done_count == 1 and state.idle_count == 0:
                reason = "missing IDLE after DONE"
                affected_views = worker.views
            else:
                reason = "missing PatchCore response" if received[connection] else "PatchCore request timeout"
                affected_views = state.pending_views
            worker.failed_reason = reason
            self._fail_views(results, affected_views, reason)
            del pending[connection]

    def _retire_failed_workers(self) -> None:
        failed_workers = [worker for worker in self._workers if worker.failed_reason is not None]
        for worker in failed_workers:
            if worker.process.is_alive():
                worker.process.terminate()
        self._join_workers(failed_workers, self._shutdown_timeout)
        for worker in failed_workers:
            if worker.process.is_alive():
                worker.process.kill()
        self._join_workers(failed_workers, self._shutdown_timeout)
        for worker in failed_workers:
            worker.connection.close()

    @staticmethod
    def _join_workers(workers: Collection[_Worker], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        for worker in workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            worker.process.join(timeout=remaining)

    def close(self) -> None:
        """Request normal shutdown, then terminate/kill only overdue children."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for worker in self._workers:
                if worker.process.is_alive():
                    try:
                        worker.connection.send({"type": "SHUTDOWN"})
                    except (BrokenPipeError, EOFError, OSError):
                        pass
            self._join_processes(self._shutdown_timeout)
            for worker in self._workers:
                if worker.process.is_alive():
                    worker.process.terminate()
            self._join_processes(self._shutdown_timeout)
            for worker in self._workers:
                if worker.process.is_alive():
                    worker.process.kill()
            self._join_processes(self._shutdown_timeout)
            for worker in self._workers:
                worker.connection.close()

    def _join_processes(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        for worker in self._workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            worker.process.join(timeout=remaining)

    def _abort_startup(self) -> None:
        self._closed = True
        for worker in self._workers:
            if worker.process.is_alive():
                worker.process.terminate()
        self._join_processes(self._shutdown_timeout)
        for worker in self._workers:
            if worker.process.is_alive():
                worker.process.kill()
        self._join_processes(self._shutdown_timeout)
        for worker in self._workers:
            worker.connection.close()
