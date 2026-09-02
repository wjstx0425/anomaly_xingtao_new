"""Exact live-cycle timing records for the BMW eight-view Demo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bmw_inspection.lab.live_cycle_timing import LiveCycleTiming, persist_cycle_timing


def _timing() -> LiveCycleTiming:
    return LiveCycleTiming(
        capture_id="bmw_demo_20260825_120000",
        started_at="2026-08-25T12:00:00.000+08:00",
        displayed_at="2026-08-25T12:00:12.000+08:00",
        front_capture_ms=1200.0,
        flip_wait_ms=2500.0,
        back_capture_ms=1300.0,
        front_inference_ms=3900.0,
        back_inference_ms=3800.0,
        finalize_ms=200.0,
        persist_ms=5100.0,
        result_display_ms=30.0,
        front_overlap_ms=3800.0,
        total_cycle_ms=12000.0,
    )


def test_live_cycle_timing_exposes_exact_total_and_overlap() -> None:
    timing = _timing()

    assert timing.total_cycle_ms == 12000.0
    assert timing.front_overlap_ms == 3800.0


def test_live_cycle_timing_rejects_negative_or_nonfinite_durations() -> None:
    values = _timing().__dict__ if hasattr(_timing(), "__dict__") else {
        field: getattr(_timing(), field)
        for field in _timing().__dataclass_fields__
    }

    with pytest.raises(ValueError, match="front_capture_ms"):
        LiveCycleTiming(**{**values, "front_capture_ms": -1.0})
    with pytest.raises(ValueError, match="total_cycle_ms"):
        LiveCycleTiming(**{**values, "total_cycle_ms": float("nan")})


def test_persist_cycle_timing_writes_simple_json(tmp_path: Path) -> None:
    timing = _timing()
    (tmp_path / timing.capture_id).mkdir()

    path = persist_cycle_timing(tmp_path, timing)

    assert path == (tmp_path / timing.capture_id / "cycle_timing.json").resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["capture_id"] == timing.capture_id
    assert payload["boundary"] == "first_space_accepted_to_first_result_imshow_returned"
    assert payload["total_cycle_ms"] == 12000.0
    assert "sha256" not in path.read_text(encoding="utf-8")
