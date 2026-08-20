"""Tests for lightweight ZS32 stage timing artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from zs32_inspection.timing import TimingRecorder, merge_timing_payloads


def test_timing_recorder_writes_aggregated_atomic_artifact(tmp_path: Path) -> None:
    recorder = TimingRecorder("stage32")
    recorder.add("runtime_config_load", 0.25)
    recorder.add("runtime_config_load", 0.75)
    recorder.add("patchcore_inference/front", 1.5)

    path = recorder.write(tmp_path / "timing.json", total_seconds=3.0)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["total_seconds"] == 3.0
    assert payload["stages"]["runtime_config_load"] == {"count": 2, "seconds": 1.0}
    assert payload["stages"]["patchcore_inference/front"] == {"count": 1, "seconds": 1.5}
    assert not (tmp_path / ".timing.json.tmp").exists()


def test_merge_timing_payloads_prefixes_child_stage_names() -> None:
    merged = merge_timing_payloads(
        {"schema_version": 1, "stages": {"front_capture": {"count": 1, "seconds": 2.0}}},
        {"schema_version": 1, "stages": {"runtime_config_load": {"count": 1, "seconds": 0.5}}},
        prefixes=("capture", "inference"),
        total_seconds=4.0,
    )

    assert merged["total_seconds"] == 4.0
    assert merged["stages"]["capture/front_capture"]["seconds"] == 2.0
    assert merged["stages"]["inference/runtime_config_load"]["seconds"] == 0.5


def test_merge_timing_payloads_preserves_existing_total_when_not_replaced() -> None:
    merged = merge_timing_payloads(
        {"schema_version": 1, "total_seconds": 7.5, "stages": {}},
        {"schema_version": 1, "stages": {"dashboard_result_parse": {"count": 1, "seconds": 0.1}}},
    )

    assert merged["total_seconds"] == 7.5
