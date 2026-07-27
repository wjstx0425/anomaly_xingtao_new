# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the resident, process-isolated ZS32 PatchCore backend."""

from __future__ import annotations

import os
import time
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import pytest

import capture_data.zs32_patchcore_multiprocess as multiprocess_backend
from capture_data.zs32_model_runtime import PatchcoreSpec
from capture_data.zs32_patchcore_multiprocess import (
    ResidentMultiprocessPatchcoreBackend,
    view_shards,
)
from zs32_inspection.domain.views import VIEW_ORDER


def _result(request_id: str, view: str, evidence_dir: str, threshold: float) -> dict[str, object]:
    root = Path(evidence_dir)
    evidence_path = (root / f"{view}.png").resolve()
    raw_path = (root / "raw_maps" / f"{view}.npy").resolve()
    mask_path = (root / "masks" / f"{view}.png").resolve()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(b"overlay")
    np.save(raw_path, np.zeros((2, 3), dtype=np.float32), allow_pickle=False)
    mask_path.write_bytes(b"mask")
    return {
        "type": "RESULT",
        "request_id": request_id,
        "view": view,
        "score": float(VIEW_ORDER.index(view)) + 0.25,
        "evidence_path": str(evidence_path),
        "raw_anomaly_map_path": str(raw_path),
        "mask_path": str(mask_path),
        "raw_anomaly_map_shape": [2, 3],
        "mask_shape": [4, 5],
        "mask_source": "diagnostic_anomaly_map",
        "diagnostic_mask_threshold": threshold,
        "error": None,
    }


def _fake_worker(
    connection: Connection,
    specs: dict[str, PatchcoreSpec],
    views: tuple[str, ...],
    options: dict[str, object],
) -> None:
    assert tuple(specs) == views
    ready_delay = float(options.get("ready_delay", 0.0))
    if ready_delay:
        time.sleep(ready_delay)
    connection.send({"type": "READY", "views": list(views)})
    scenario = str(options.get("scenario", "normal"))
    request_count = 0
    while True:
        message = connection.recv()
        if message["type"] == "SHUTDOWN":
            if scenario == "ignore_shutdown":
                time.sleep(60)
            return
        assert message["type"] == "PREDICT"
        request_count += 1
        if scenario == "timeout":
            time.sleep(60)
        if scenario == "late_once" and request_count == 1:
            time.sleep(float(options["late_delay"]))
        if scenario == "eof":
            connection.close()
            return
        if scenario == "death":
            os._exit(17)
        request_id = str(message["request_id"])
        threshold = float(message["diagnostic_mask_threshold"])
        result_views = list(views)
        if scenario == "one_child_error" and views[0] == "front":
            for view in result_views:
                connection.send(
                    {
                        "type": "RESULT",
                        "request_id": request_id,
                        "view": view,
                        "error": "RuntimeError: child shard failed",
                    },
                )
            connection.send({"type": "DONE", "request_id": request_id})
            assert connection.recv() == {"type": "DONE_ACK", "request_id": request_id}
            connection.send({"type": "IDLE", "request_id": request_id})
            continue
        if scenario == "one_view_error":
            affected = result_views.pop(0)
            connection.send(
                {
                    "type": "RESULT",
                    "request_id": request_id,
                    "view": affected,
                    "error": "RuntimeError: view failed",
                },
            )
        if scenario in {"stale", "duplicate", "missing", "malformed"}:
            affected = result_views.pop(0)
            envelope = _result(request_id, affected, str(message["evidence_dir"]), threshold)
            if scenario == "stale":
                envelope["request_id"] = "old-request"
                connection.send(envelope)
            elif scenario == "duplicate":
                connection.send(envelope)
                connection.send(envelope)
            elif scenario == "malformed":
                del envelope["score"]
                connection.send(envelope)
        for view in result_views:
            connection.send(_result(request_id, view, str(message["evidence_dir"]), threshold))
        if scenario == "tail_duplicate":
            tail_view = result_views[-1]
            connection.send(_result(request_id, tail_view, str(message["evidence_dir"]), threshold))
        if scenario == "missing_done":
            continue
        done_request_id = "old-request" if scenario == "wrong_done" else request_id
        connection.send({"type": "DONE", "request_id": done_request_id})
        if scenario == "duplicate_done":
            connection.send({"type": "DONE", "request_id": request_id})
        if scenario in {"post_done_duplicate", "post_done_wrong"}:
            envelope = _result(request_id, result_views[-1], str(message["evidence_dir"]), threshold)
            if scenario == "post_done_wrong":
                envelope["request_id"] = "old-request"
            connection.send(envelope)
        assert connection.recv() == {"type": "DONE_ACK", "request_id": request_id}
        connection.send({"type": "IDLE", "request_id": request_id})


def _specs(tmp_path: Path) -> dict[str, PatchcoreSpec]:
    specs: dict[str, PatchcoreSpec] = {}
    for view in VIEW_ORDER:
        checkpoint = tmp_path / f"{view}.ckpt"
        checkpoint.write_bytes(view.encode())
        specs[view] = PatchcoreSpec(
            view=view,
            checkpoint=checkpoint.resolve(),
            checkpoint_sha256="0" * 64,
            model_version=f"fake-{view}",
        )
    return specs


def _crops(tmp_path: Path) -> dict[str, Path]:
    crops: dict[str, Path] = {}
    for view in VIEW_ORDER:
        crop = tmp_path / f"{view}.png"
        crop.write_bytes(view.encode())
        crops[view] = crop.resolve()
    return crops


def _backend(
    tmp_path: Path,
    *,
    process_count: int = 1,
    scenario: str = "normal",
    ready_delay: float = 0.0,
    startup_timeout: float = 2.0,
    request_timeout: float = 0.3,
    shutdown_timeout: float = 0.1,
    late_delay: float = 0.2,
) -> ResidentMultiprocessPatchcoreBackend:
    return ResidentMultiprocessPatchcoreBackend(
        _specs(tmp_path),
        VIEW_ORDER,
        process_count,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
        shutdown_timeout=shutdown_timeout,
        worker_target=_fake_worker,
        worker_options={"scenario": scenario, "ready_delay": ready_delay, "late_delay": late_delay},
    )


@pytest.mark.parametrize(
    ("process_count", "expected"),
    [
        (1, (VIEW_ORDER,)),
        (
            2,
            (
                ("front", "front_left", "front_right", "front_secondary"),
                ("back", "back_left", "back_right", "back_secondary"),
            ),
        ),
        (
            4,
            (
                ("front_left", "back_right"),
                ("back_left", "front_right"),
                ("front", "front_secondary"),
                ("back", "back_secondary"),
            ),
        ),
        (8, tuple((view,) for view in VIEW_ORDER)),
    ],
)
def test_view_shards_are_exact_and_deterministic(
    process_count: int,
    expected: tuple[tuple[str, ...], ...],
) -> None:
    assert view_shards(VIEW_ORDER, process_count) == expected
    assert view_shards(reversed(VIEW_ORDER), process_count) == expected
    assert {view for shard in expected for view in shard} == set(VIEW_ORDER)


@pytest.mark.parametrize("process_count", [0, 3, 5, 9])
def test_view_shards_reject_unsupported_process_counts(process_count: int) -> None:
    with pytest.raises(ValueError, match="1, 2, 4, or 8"):
        view_shards(VIEW_ORDER, process_count)


def test_constructor_waits_for_every_worker_ready(tmp_path: Path) -> None:
    started = time.monotonic()
    backend = _backend(tmp_path, process_count=2, ready_delay=0.15)
    elapsed = time.monotonic() - started
    try:
        assert elapsed >= 0.12
        assert len(backend.worker_pids) == 2
    finally:
        backend.close()


def test_constructor_fails_closed_when_ready_deadline_expires(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="READY"):
        _backend(tmp_path, ready_delay=0.2, startup_timeout=0.03)


def test_real_worker_target_requires_complete_warmup_crops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_context(_method: str) -> None:
        raise AssertionError("spawn must not begin before warmup validation")

    monkeypatch.setattr(multiprocess_backend.multiprocessing, "get_context", _unexpected_context)

    with pytest.raises(ValueError, match="warmup_crops.*required"):
        ResidentMultiprocessPatchcoreBackend(_specs(tmp_path), VIEW_ORDER, 1)


def test_predict_all_aggregates_exactly_eight_model_evidence_records(tmp_path: Path) -> None:
    backend = _backend(tmp_path, process_count=4)
    try:
        results = backend.predict_all(_crops(tmp_path), tmp_path / "evidence", diagnostic_mask_threshold=0.71)
    finally:
        backend.close()

    assert tuple(results) == VIEW_ORDER
    assert all(not isinstance(result, Exception) for result in results.values())
    for index, view in enumerate(VIEW_ORDER):
        evidence = results[view]
        assert not isinstance(evidence, Exception)
        assert evidence.score == index + 0.25
        assert evidence.evidence_path == (tmp_path / "evidence" / f"{view}.png").resolve()
        assert evidence.patchcore_artifacts is not None
        assert evidence.patchcore_artifacts.raw_anomaly_map_shape == (2, 3)
        assert evidence.patchcore_artifacts.mask_shape == (4, 5)
        assert evidence.patchcore_artifacts.diagnostic_mask_threshold == 0.71


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("stale", "stale request"),
        ("duplicate", "duplicate"),
        ("missing", "missing"),
        ("malformed", "malformed"),
    ],
)
def test_protocol_errors_fail_closed_only_for_affected_view(
    tmp_path: Path,
    scenario: str,
    message: str,
) -> None:
    backend = _backend(tmp_path, scenario=scenario, request_timeout=0.12)
    try:
        results = backend.predict_all(_crops(tmp_path), tmp_path / "evidence")
        assert isinstance(results["front"], Exception)
        assert message in str(results["front"]).lower()
        if scenario == "missing":
            assert all(isinstance(results[view], Exception) for view in VIEW_ORDER)
        else:
            assert all(not isinstance(results[view], Exception) for view in VIEW_ORDER[1:])
        assert not backend.worker_processes[0].is_alive()

        followup = backend.predict_all(_crops(tmp_path), tmp_path / "followup")
        assert all(isinstance(result, Exception) for result in followup.values())
        assert "unavailable" in str(followup["front"]).lower()
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("scenario", "message"),
    [("timeout", "timeout"), ("eof", "eof"), ("death", "exited")],
)
def test_transport_failures_fail_closed_for_pending_views(
    tmp_path: Path,
    scenario: str,
    message: str,
) -> None:
    backend = _backend(tmp_path, scenario=scenario, request_timeout=0.08)
    try:
        results = backend.predict_all(_crops(tmp_path), tmp_path / "evidence")
    finally:
        backend.close()

    assert tuple(results) == VIEW_ORDER
    assert all(isinstance(result, Exception) for result in results.values())
    assert message in str(results["front"]).lower()


def test_timeout_late_responses_cannot_pollute_the_next_request(tmp_path: Path) -> None:
    backend = _backend(
        tmp_path,
        scenario="late_once",
        request_timeout=0.04,
        shutdown_timeout=0.05,
        late_delay=0.2,
    )
    try:
        first = backend.predict_all(_crops(tmp_path), tmp_path / "first")
        assert all(isinstance(result, Exception) for result in first.values())
        assert "timeout" in str(first["front"]).lower()
        assert not backend.worker_processes[0].is_alive()

        started = time.monotonic()
        second = backend.predict_all(_crops(tmp_path), tmp_path / "second")
        assert time.monotonic() - started < 0.04
        assert all(isinstance(result, Exception) for result in second.values())
        assert "unavailable" in str(second["front"]).lower()
    finally:
        backend.close()


def test_tail_duplicate_after_last_view_fails_current_request_and_discards_worker(tmp_path: Path) -> None:
    backend = _backend(tmp_path, scenario="tail_duplicate")
    try:
        results = backend.predict_all(_crops(tmp_path), tmp_path / "evidence")
        assert isinstance(results["back_secondary"], Exception)
        assert "duplicate" in str(results["back_secondary"]).lower()
        assert not backend.worker_processes[0].is_alive()
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("scenario", "message"),
    [("missing_done", "missing done"), ("duplicate_done", "duplicate done"), ("wrong_done", "stale done")],
)
def test_invalid_done_boundary_fails_current_shard_and_discards_worker(
    tmp_path: Path,
    scenario: str,
    message: str,
) -> None:
    backend = _backend(tmp_path, scenario=scenario, request_timeout=0.08)
    try:
        results = backend.predict_all(_crops(tmp_path), tmp_path / "evidence")
        assert all(isinstance(result, Exception) for result in results.values())
        assert message in str(results["front"]).lower()
        assert not backend.worker_processes[0].is_alive()
    finally:
        backend.close()


@pytest.mark.parametrize("scenario", ["post_done_duplicate", "post_done_wrong"])
def test_result_after_done_fails_current_shard_and_discards_worker(tmp_path: Path, scenario: str) -> None:
    backend = _backend(tmp_path, scenario=scenario)
    try:
        results = backend.predict_all(_crops(tmp_path), tmp_path / "evidence")
        assert all(isinstance(result, Exception) for result in results.values())
        assert "after done" in str(results["front"]).lower()
        assert not backend.worker_processes[0].is_alive()
    finally:
        backend.close()


def test_one_view_worker_error_preserves_other_view_results(tmp_path: Path) -> None:
    backend = _backend(tmp_path, scenario="one_view_error")
    try:
        results = backend.predict_all(_crops(tmp_path), tmp_path / "evidence")
        assert isinstance(results["front"], Exception)
        assert "view failed" in str(results["front"]).lower()
        assert all(not isinstance(results[view], Exception) for view in VIEW_ORDER[1:])
        assert backend.worker_processes[0].is_alive()
    finally:
        backend.close()


def test_one_child_error_preserves_the_other_child_results(tmp_path: Path) -> None:
    backend = _backend(tmp_path, process_count=2, scenario="one_child_error")
    try:
        results = backend.predict_all(_crops(tmp_path), tmp_path / "evidence")
        front_shard = {"front", "front_left", "front_right", "front_secondary"}
        assert all(isinstance(results[view], Exception) for view in front_shard)
        assert all(not isinstance(results[view], Exception) for view in set(VIEW_ORDER) - front_shard)
        assert all(process.is_alive() for process in backend.worker_processes)
    finally:
        backend.close()


@pytest.mark.parametrize("invalid_threshold", [True, False])
def test_predict_all_rejects_boolean_threshold(tmp_path: Path, invalid_threshold: bool) -> None:
    backend = _backend(tmp_path)
    try:
        with pytest.raises(ValueError, match="diagnostic_mask_threshold"):
            backend.predict_all(
                _crops(tmp_path),
                tmp_path / "evidence",
                diagnostic_mask_threshold=invalid_threshold,
            )
    finally:
        backend.close()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_predict_all_rejects_noncanonical_crop_schema(tmp_path: Path, mutation: str) -> None:
    backend = _backend(tmp_path)
    crops = _crops(tmp_path)
    if mutation == "missing":
        del crops["front"]
    else:
        crops["unexpected"] = tmp_path / "unexpected.png"
    try:
        with pytest.raises(ValueError, match="crops must contain exactly"):
            backend.predict_all(crops, tmp_path / "evidence")
    finally:
        backend.close()


def test_two_requests_reuse_the_same_live_child_pids(tmp_path: Path) -> None:
    backend = _backend(tmp_path, process_count=2)
    original_pids = backend.worker_pids
    try:
        first = backend.predict_all(_crops(tmp_path), tmp_path / "first")
        second = backend.predict_all(_crops(tmp_path), tmp_path / "second")
        assert all(not isinstance(value, Exception) for value in first.values())
        assert all(not isinstance(value, Exception) for value in second.values())
        assert backend.worker_pids == original_pids
        assert len(set(original_pids)) == 2
    finally:
        backend.close()


def test_close_is_idempotent_and_bounded_for_uncooperative_child(tmp_path: Path) -> None:
    backend = _backend(tmp_path, scenario="ignore_shutdown", shutdown_timeout=0.05)
    started = time.monotonic()
    backend.close()
    backend.close()

    assert time.monotonic() - started < 1.0
    assert backend.closed is True
    assert not any(process.is_alive() for process in backend.worker_processes)
