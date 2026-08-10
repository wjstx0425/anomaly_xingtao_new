"""Offline EfficientAD score-report helpers."""

import runpy
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.lab.efficientad_analysis import (
    EfficientAdScore,
    fixed_scale_heatmap,
    render_example_sheet,
    render_score_distributions,
    score_metrics,
    select_hard_examples,
    write_score_csv,
)


def test_fixed_scale_heatmap_uses_shared_zero_to_one_scale() -> None:
    anomaly_map = np.array([[0.0, 0.5, 1.5]], dtype=np.float32)

    heatmap = fixed_scale_heatmap(anomaly_map)

    expected = cv2.applyColorMap(np.array([[0, 128, 255]], dtype=np.uint8), cv2.COLORMAP_TURBO)
    assert np.array_equal(heatmap, expected)


def test_select_hard_examples_returns_high_normal_and_low_defect_scores() -> None:
    records = (
        EfficientAdScore("front", "normal", Path("normal-low.png"), 0.1, False),
        EfficientAdScore("front", "normal", Path("normal-high.png"), 0.8, True),
        EfficientAdScore("front", "defect", Path("defect-low.png"), 0.2, False),
        EfficientAdScore("front", "defect", Path("defect-high.png"), 0.9, True),
    )

    selected = select_hard_examples(records, per_label=1)

    assert [item.image_path.name for item in selected["normal"]] == ["normal-high.png"]
    assert [item.image_path.name for item in selected["defect"]] == ["defect-low.png"]


def test_score_metrics_recomputes_runtime_confusion_and_ranking() -> None:
    records = (
        EfficientAdScore("front", "normal", Path("normal-a.png"), 0.1, False),
        EfficientAdScore("front", "normal", Path("normal-b.png"), 0.2, False),
        EfficientAdScore("front", "defect", Path("defect-a.png"), 0.8, True),
        EfficientAdScore("front", "defect", Path("defect-b.png"), 0.9, True),
    )

    metrics = score_metrics(records)

    assert metrics == {
        "auroc": 1.0,
        "f1": 1.0,
        "balanced_accuracy": 1.0,
        "normal_predicted_ng": 0,
        "defect_predicted_ok": 0,
    }


def test_score_report_writes_csv_distribution_and_fixed_scale_examples(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.png"
    defect_path = tmp_path / "defect.png"
    assert cv2.imwrite(str(normal_path), np.full((12, 10, 3), 80, dtype=np.uint8))
    assert cv2.imwrite(str(defect_path), np.full((12, 10, 3), 160, dtype=np.uint8))
    records = (
        EfficientAdScore("front", "normal", normal_path, 0.2, False),
        EfficientAdScore("front", "defect", defect_path, 0.8, True),
    )
    maps = {
        normal_path: np.full((4, 4), 0.2, dtype=np.float32),
        defect_path: np.full((4, 4), 0.8, dtype=np.float32),
    }

    csv_path = write_score_csv(tmp_path / "scores.csv", records, threshold=0.5)
    distribution_path = render_score_distributions(
        tmp_path / "distribution.png",
        records,
        views=("front",),
        threshold=0.5,
    )
    example_path = render_example_sheet(
        tmp_path / "examples.png",
        records,
        maps,
        threshold=0.5,
        per_label=1,
    )

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "view_id,label,prediction,score,threshold,image_path",
        f"front,defect,NG,0.8,0.5,{defect_path}",
        f"front,normal,OK,0.2,0.5,{normal_path}",
    ]
    assert distribution_path.stat().st_size > 0
    assert example_path.stat().st_size > 0


def test_analysis_cli_has_reproducible_report_defaults() -> None:
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(root / "pipeline/bmw_lab_visualize_efficientad_scores.py")

    args = namespace["build_parser"]().parse_args([])

    assert args.threshold == 0.5
    assert args.examples_per_label == 3
    assert args.output_dir.name == "score_analysis"
