"""Persistence contracts for BMW eight-view Demo evidence."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bmw_inspection.views import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo import (
    BranchStatus,
    DemoBranch,
    DemoBranchResult,
    DemoFinalStatus,
    EightViewInspection,
)
from bmw_inspection.lab.eight_view_demo_persistence import fused_only_sources, persist_inspection
from bmw_inspection.lab.trusted_ok_reference import TrustedOkMatch


def _inspection() -> EightViewInspection:
    images = {view: np.full((12, 16, 3), index * 20, dtype=np.uint8) for index, view in enumerate(VIEW_ORDER)}
    results = tuple(
        DemoBranchResult(
            branch=DemoBranch.TEMPLATE,
            view_id=view,
            status=BranchStatus.PASS,
            score=0.1,
            threshold=0.2,
            elapsed_ms=1.0,
            reason="测试通过",
            overlay=np.full((6, 9, 3), index, dtype=np.uint8),
            details={"deployment_threshold": 0.2, "boxes": ((1, 2, 3, 4),)},
        )
        for index, view in enumerate(VIEW_ORDER)
    )
    return EightViewInspection(
        capture_id="capture-001",
        images=images,
        results=results,
        final_status=DemoFinalStatus.OK,
        elapsed_ms=12.5,
    )


def test_persist_inspection_writes_fused_only_evidence_and_appends_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _inspection()
    capture_config = tmp_path / "capture.json"
    capture_config.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        result_root=tmp_path / "results",
        roi_config=tmp_path / "rois.json",
        capture_config=capture_config,
        demo_id="demo-v3",
    )
    monkeypatch.setattr(
        "bmw_inspection.lab.eight_view_demo_persistence.load_part_rois",
        lambda _path: {view: (1, 2, 10, 8) for view in VIEW_ORDER},
    )
    monkeypatch.setattr(
        "bmw_inspection.lab.eight_view_demo_persistence.load_capture_profile",
        lambda path: SimpleNamespace(
            path=path,
            profile_id="capture-v1",
            slots=tuple(
                SimpleNamespace(
                    slot_id=f"slot-{index}",
                    serial=f"serial-{index}",
                    front_view=VIEW_ORDER[index],
                    back_view=VIEW_ORDER[index + 4],
                )
                for index in range(4)
            ),
            hdr=SimpleNamespace(
                short_exposure_us=1500.0,
                long_exposure_us=6000.0,
                gain=0.0,
                trigger_interval_s=0.2,
                settle_frames=1,
                timeout_ms=3000,
                align=False,
                short_dark_threshold=70.0,
                long_clip_threshold=245.0,
                blend_width=18.0,
                blur_size=31,
                max_retries=0,
                max_clip_pct=12.0,
            ),
        ),
    )

    published = persist_inspection(config, inspection, fused_only_sources(inspection.images))

    assert published == (tmp_path / "results" / "capture-001").resolve()
    assert all((published / "images" / f"{view}_{exposure}.png").is_file() for view in VIEW_ORDER for exposure in ("short", "long", "hdr"))
    assert all((published / "rois" / f"{view}.png").is_file() for view in VIEW_ORDER)
    assert all((published / "evidence" / f"template_{view}.png").is_file() for view in VIEW_ORDER)
    payload = json.loads((published / "inspection.json").read_text(encoding="utf-8"))
    assert payload["source_kind"] == "fused_only"
    assert payload["capture_profile"]["settings_origin"] == "configured_not_camera_readback"
    assert payload["capture_profile"]["hdr"]["short_exposure_us"] == 1500.0
    assert payload["capture_profile"]["hdr"]["long_exposure_us"] == 6000.0
    assert payload["capture_profile"]["camera_slots"][0]["serial"] == "serial-0"
    assert payload["views"]["front"]["source"]["source_kind"] == "fused_only"
    assert payload["results"][0]["overlay"] == "evidence/template_front.png"
    assert payload["results"][0]["details"] == {
        "deployment_threshold": 0.2,
        "boxes": [[1, 2, 3, 4]],
    }
    assert "trusted_ok_references" not in payload
    assert "diagnostic_metadata" not in payload
    with (tmp_path / "results" / "inspection_index.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"capture_id": "capture-001", "demo_id": "demo-v3", "final_status": "OK", "result_path": "capture-001"}]

    persist_inspection(config, inspection, fused_only_sources(inspection.images))
    with (tmp_path / "results" / "inspection_index.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_persist_inspection_writes_trusted_reference_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _inspection()
    current = base.images["front_left"]
    match = TrustedOkMatch(
        view_id="front_left", comparison_mode="roi", physical_part_id="normal-train-001",
        sample_id="normal-train-001_000001", similarity=0.91, shift_x=2, shift_y=-1,
        current_full_image=current, reference_full_image=np.full_like(current, 10),
        current_roi=np.full((6, 9, 3), 20, dtype=np.uint8),
        reference_roi=np.full((6, 9, 3), 30, dtype=np.uint8),
        aligned_reference_roi=np.full((8, 8, 3), 40, dtype=np.uint8),
        difference_overlay=np.full((8, 8, 3), 50, dtype=np.uint8),
    )
    full_match = replace(
        match,
        comparison_mode="full",
        physical_part_id="normal-train-002",
        sample_id="normal-train-002_000001",
        current_roi=current,
        aligned_reference_roi=np.full((8, 8, 3), 60, dtype=np.uint8),
        difference_overlay=np.full((8, 8, 3), 70, dtype=np.uint8),
    )
    inspection = EightViewInspection(
        base.capture_id, base.images, base.results, base.final_status, base.elapsed_ms,
        trusted_ok_by_comparison={
            ("front_left", "roi"): match,
            ("front_left", "full"): full_match,
        },
    )
    capture_config = tmp_path / "capture.json"
    capture_config.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        result_root=tmp_path / "results", roi_config=tmp_path / "rois.json",
        capture_config=capture_config, demo_id="demo-v3",
    )
    monkeypatch.setattr(
        "bmw_inspection.lab.eight_view_demo_persistence.load_part_rois",
        lambda _path: {view: (1, 2, 10, 8) for view in VIEW_ORDER},
    )
    monkeypatch.setattr(
        "bmw_inspection.lab.eight_view_demo_persistence.load_capture_profile",
        lambda path: SimpleNamespace(
            path=path, profile_id="capture-v1", slots=(),
            hdr=SimpleNamespace(
                short_exposure_us=1500.0, long_exposure_us=6000.0, gain=0.0,
                trigger_interval_s=0.2, settle_frames=1, timeout_ms=3000, align=False,
                short_dark_threshold=70.0, long_clip_threshold=245.0, blend_width=18.0,
                blur_size=31, max_retries=0, max_clip_pct=12.0,
            ),
        ),
    )

    published = persist_inspection(config, inspection, fused_only_sources(inspection.images))
    payload = json.loads((published / "inspection.json").read_text(encoding="utf-8"))

    reference = payload["trusted_ok_references"]["front_left/roi"]
    assert payload["reference_is_diagnostic_only"] is True
    assert reference["reference_is_diagnostic_only"] is True
    assert reference["physical_part_id"] == "normal-train-001"
    assert reference["sample_id"] == "normal-train-001_000001"
    assert reference["similarity"] == pytest.approx(0.91)
    assert reference["shift"] == {"x": 2, "y": -1}
    for name in ("full", "roi", "aligned_roi", "difference"):
        path = published / reference["files"][name]
        assert path.is_file()
    assert reference["comparison_mode"] == "roi"
    assert all(path.startswith("references/front_left/roi/") for path in reference["files"].values())
    full_reference = payload["trusted_ok_references"]["front_left/full"]
    assert full_reference["comparison_mode"] == "full"
    assert full_reference["physical_part_id"] == "normal-train-002"
    assert all(path.startswith("references/front_left/full/") for path in full_reference["files"].values())
