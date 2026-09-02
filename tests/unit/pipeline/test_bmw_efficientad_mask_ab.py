"""Offline BMW EfficientAD mask A/B pipeline contracts."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "pipeline/bmw_lab_evaluate_efficientad_mask_ab.py"


def test_cli_exposes_separate_no_overwrite_build_and_evaluate_commands() -> None:
    namespace = runpy.run_path(SCRIPT)
    parser = namespace["build_parser"]()

    build = parser.parse_args(["build-masks", "--output-dir", "/tmp/masks"])
    evaluate = parser.parse_args(
        ["evaluate", "--mask-index", "/tmp/masks/index.json", "--output-dir", "/tmp/report"]
    )

    assert build.command == "build-masks"
    assert build.fixed_fill_value == 0
    assert evaluate.command == "evaluate"
    assert evaluate.quantile == pytest.approx(0.995)
    assert evaluate.top_k_fraction == pytest.approx(0.001)
    assert evaluate.minimum_component_area == 8


def test_prediction_batches_keep_path_score_label_and_two_dimensional_map() -> None:
    namespace = runpy.run_path(SCRIPT)
    paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    batches = [
        SimpleNamespace(
            image_path=paths,
            pred_score=np.array([0.2, 0.7], dtype=np.float32),
            pred_label=np.array([0, 1]),
            anomaly_map=np.zeros((2, 1, 4, 5), dtype=np.float32),
        )
    ]

    predictions = namespace["_normalize_prediction_batches"](batches)

    assert tuple(predictions) == tuple(paths)
    assert predictions[paths[1]][0] == pytest.approx(0.7)
    assert predictions[paths[1]][1] is True
    assert predictions[paths[1]][2].shape == (4, 5)


def test_prediction_batches_fail_closed_on_duplicate_paths() -> None:
    namespace = runpy.run_path(SCRIPT)
    path = Path("/tmp/a.png")
    batch = SimpleNamespace(
        image_path=[path, path],
        pred_score=np.array([0.2, 0.3]),
        pred_label=np.array([0, 0]),
        anomaly_map=np.zeros((2, 4, 5), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="duplicate"):
        namespace["_normalize_prediction_batches"]([batch])
