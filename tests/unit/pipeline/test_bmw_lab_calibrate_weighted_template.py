from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from pipeline import bmw_lab_calibrate_weighted_template as calibration_module
from pipeline.bmw_lab_calibrate_weighted_template import (
    calibrate,
    summarize_view,
    write_weighted_thresholds,
)


def test_final_test_scores_do_not_change_calibration_threshold() -> None:
    first = summarize_view([0.1, 0.2], [0.9])
    second = summarize_view([0.1, 0.2], [0.01])

    assert first["threshold"] == pytest.approx(0.22)
    assert second["threshold"] == pytest.approx(0.22)
    assert first["final_test_false_reject_count"] == 1
    assert second["final_test_false_reject_count"] == 0


def test_write_weighted_thresholds_preserves_legacy_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "demo.json"
    legacy = {view: 0.01 + index / 1000 for index, view in enumerate(VIEW_ORDER)}
    payload = {
        "template": {
            "thresholds": legacy,
            "weighted_regions": {
                "enabled": True,
                "weight": 3.0,
                "roi_config": "regions.json",
                "thresholds": {view: 0.5 for view in VIEW_ORDER},
            },
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    proposed = {view: 0.02 + index / 1000 for index, view in enumerate(VIEW_ORDER)}

    write_weighted_thresholds(path, proposed)
    updated = json.loads(path.read_text(encoding="utf-8"))

    assert updated["template"]["thresholds"] == legacy
    assert updated["template"]["weighted_regions"]["thresholds"] == proposed


def test_summarize_view_rejects_missing_calibration_scores() -> None:
    with pytest.raises(ValueError, match="calibration"):
        summarize_view([], [0.1])


def test_calibration_report_records_outside_weight_and_effective_ratio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weighted = SimpleNamespace(
        enabled=True,
        weight=3.0,
        outside_weight=0.5,
        roi_config=tmp_path / "regions.json",
    )
    config = SimpleNamespace(
        template_weighted_regions=weighted,
        template_thresholds={view: 0.1 for view in VIEW_ORDER},
    )
    output = SimpleNamespace(score=0.2)
    predictor = SimpleNamespace(predict=lambda _view, _image: output)
    rois = {view: (0, 0, 2, 2) for view in VIEW_ORDER}
    regions = {view: ((object(),) if view == "front" else ()) for view in VIEW_ORDER}
    rows = [
        {
            "view_id": "front",
            "split": "calibration",
            "source_path": str(tmp_path / "front.png"),
        }
    ]
    monkeypatch.setattr(calibration_module, "load_demo_config", lambda _path: config)
    monkeypatch.setattr(
        calibration_module,
        "_build_template_predictor",
        lambda _config: (predictor, rois, regions),
    )
    monkeypatch.setattr(calibration_module, "_normal_rows", lambda _path: rows)
    monkeypatch.setattr(calibration_module.cv2, "imread", lambda *_args: np.zeros((2, 2, 3), dtype=np.uint8))

    report, _thresholds = calibrate(tmp_path / "config.json", tmp_path / "manifest.csv")

    assert report["weight"] == pytest.approx(3.0)
    assert report["outside_weight"] == pytest.approx(0.5)
    assert report["effective_region_ratio"] == pytest.approx(6.0)
