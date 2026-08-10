"""Offline EfficientAD score-report helpers."""

import csv
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from matplotlib.axes import Axes

from bmw_inspection.lab.efficientad_analysis import (
    EfficientAdScore,
    fixed_scale_heatmap,
    render_example_sheet,
    render_score_distributions,
    score_metrics,
    select_hard_examples,
    write_score_csv,
)
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


def _placeholder_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test-image-placeholder")
    return path


def _score_image(root: Path, part_id: str, view: str) -> Path:
    return root / f"session__{part_id}_000001__{view}.png"


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


def test_score_distribution_uses_each_views_actual_threshold(tmp_path: Path, monkeypatch) -> None:
    records = (
        EfficientAdScore("front", "normal", Path("front.png"), 0.1, False),
        EfficientAdScore("back", "defect", Path("back.png"), 0.9, True),
    )
    threshold_lines: list[float] = []
    threshold_labels: list[str] = []
    original_axhline = Axes.axhline

    def record_axhline(self, y=0, xmin=0, xmax=1, **kwargs) -> object:
        threshold_lines.append(float(y))
        threshold_labels.append(str(kwargs["label"]))
        return original_axhline(self, y=y, xmin=xmin, xmax=xmax, **kwargs)

    monkeypatch.setattr(Axes, "axhline", record_axhline)

    render_score_distributions(
        tmp_path / "distribution.png",
        records,
        views=("front", "back"),
        threshold={"front": 0.2, "back": math.nextafter(1.0, math.inf)},
    )

    assert threshold_lines == [0.2, math.nextafter(1.0, math.inf)]
    assert threshold_labels == ["阈值 0.20", "阈值 1.0000000000000002 (>1)"]


def test_analysis_cli_has_reproducible_report_defaults() -> None:
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(root / "pipeline/bmw_lab_visualize_efficientad_scores.py")

    args = namespace["build_parser"]().parse_args([])

    assert args.threshold == 0.5
    assert args.examples_per_label == 3
    assert args.output_dir.name == "score_analysis"


def test_analysis_rejects_partial_view_reports_before_data_access(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(root / "pipeline/bmw_lab_visualize_efficientad_scores.py")
    args = namespace["build_parser"]().parse_args([
        "--training-release",
        str(tmp_path / "missing"),
        "--views",
        VIEW_ORDER[0],
    ])

    with pytest.raises(ValueError, match="完整且有序的BMW八视图"):
        namespace["analyze"](args)


def test_analysis_selects_visible_defect_part_union_and_complete_crops(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(root / "pipeline/bmw_lab_visualize_efficientad_scores.py")
    release = tmp_path / "release"
    selected_parts = ("bmw_deform_group006", "bmw_edge_group014")
    unconfirmed_part = "bmw_normal_group999"
    for view in VIEW_ORDER:
        (release / "efficientad" / view / "defect").mkdir(parents=True)
        for part_id in (*selected_parts, unconfirmed_part):
            _placeholder_image(_score_image(release / "crops" / view, part_id, view))
    _placeholder_image(
        _score_image(
            release / "efficientad" / "front" / "defect" / "deform" / selected_parts[0] / "images",
            selected_parts[0],
            "front",
        )
    )
    _placeholder_image(
        _score_image(
            release / "efficientad" / "back_left" / "defect" / "edge" / selected_parts[1] / "images",
            selected_parts[1],
            "back_left",
        )
    )

    selected = namespace["_complete_defect_crop_paths"](release, VIEW_ORDER)

    assert tuple(selected) == VIEW_ORDER
    for view in VIEW_ORDER:
        assert tuple(path.name for path in selected[view]) == tuple(
            f"session__{part_id}_000001__{view}.png" for part_id in selected_parts
        )
        assert all(unconfirmed_part not in path.name for path in selected[view])


def test_analysis_rejects_selected_defect_part_missing_any_crop_view(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(root / "pipeline/bmw_lab_visualize_efficientad_scores.py")
    release = tmp_path / "release"
    part_id = "bmw_edge_group014"
    for view in VIEW_ORDER:
        (release / "efficientad" / view / "defect").mkdir(parents=True)
        (release / "crops" / view).mkdir(parents=True)
        if view != VIEW_ORDER[-1]:
            _placeholder_image(_score_image(release / "crops" / view, part_id, view))
    _placeholder_image(
        _score_image(
            release / "efficientad" / "back_left" / "defect" / "edge" / part_id / "images",
            part_id,
            "back_left",
        )
    )

    with pytest.raises(ValueError, match="exactly one crop for every selected defect part"):
        namespace["_complete_defect_crop_paths"](release, VIEW_ORDER)


def test_analysis_rejects_visible_defect_directory_filename_identity_mismatch(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(root / "pipeline/bmw_lab_visualize_efficientad_scores.py")
    release = tmp_path / "release"
    directory_part = "bmw_edge_group014"
    filename_part = "bmw_edge_group017"
    for view in VIEW_ORDER:
        (release / "efficientad" / view / "defect").mkdir(parents=True)
        _placeholder_image(_score_image(release / "crops" / view, filename_part, view))
    _placeholder_image(
        _score_image(
            release / "efficientad" / "back_left" / "defect" / "edge" / directory_part / "images",
            filename_part,
            "back_left",
        )
    )

    with pytest.raises(ValueError, match="visible defect identity mismatch"):
        namespace["_complete_defect_crop_paths"](release, VIEW_ORDER)


def test_analysis_rejects_png_named_crop_directory(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(root / "pipeline/bmw_lab_visualize_efficientad_scores.py")
    release = tmp_path / "release"
    part_id = "bmw_edge_group014"
    for view in VIEW_ORDER:
        (release / "efficientad" / view / "defect").mkdir(parents=True)
        crop_path = _score_image(release / "crops" / view, part_id, view)
        if view == VIEW_ORDER[-1]:
            crop_path.mkdir(parents=True)
        else:
            _placeholder_image(crop_path)
    _placeholder_image(
        _score_image(
            release / "efficientad" / "back_left" / "defect" / "edge" / part_id / "images",
            part_id,
            "back_left",
        )
    )

    with pytest.raises(ValueError, match="exactly one crop for every selected defect part"):
        namespace["_complete_defect_crop_paths"](release, VIEW_ORDER)


def test_analysis_batches_only_explicit_defect_crop_paths(monkeypatch) -> None:
    from anomalib import data as anomalib_data

    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(root / "pipeline/bmw_lab_visualize_efficientad_scores.py")
    selected_paths = (
        Path("/release/crops/front/session__bmw_deform_group006_000001__front.png"),
        Path("/release/crops/front/session__bmw_edge_group014_000001__front.png"),
    )

    class FakePredictDataset:
        def __init__(self, path: Path) -> None:
            self.image_filenames = [Path(path)]

    class FakeEngine:
        def __init__(self) -> None:
            self.dataset = None

        def predict(self, **kwargs) -> list[SimpleNamespace]:
            self.dataset = kwargs["dataset"]
            return [
                SimpleNamespace(
                    image_path=self.dataset.image_filenames,
                    pred_score=np.array([0.7, 0.8]),
                    pred_label=np.array([1, 1]),
                    anomaly_map=np.zeros((2, 4, 4), dtype=np.float32),
                )
            ]

    monkeypatch.setattr(anomalib_data, "PredictDataset", FakePredictDataset)
    engine = FakeEngine()

    records, maps = namespace["_predictions_for_paths"](
        engine,
        object(),
        selected_paths,
        view="front",
        label="defect",
    )

    assert tuple(engine.dataset.image_filenames) == selected_paths
    assert tuple(record.image_path for record in records) == selected_paths
    assert set(maps) == set(selected_paths)


def test_analysis_writes_strict_complete_part_csv_without_unconfirmed_crops(tmp_path: Path, monkeypatch) -> None:
    from anomalib import data as anomalib_data
    from anomalib import engine as anomalib_engine
    from anomalib import models as anomalib_models

    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(root / "pipeline/bmw_lab_visualize_efficientad_scores.py")
    release = tmp_path / "release"
    model_root = tmp_path / "models"
    output_dir = tmp_path / "analysis"
    normal_parts = ("bmw_normal_group001", "bmw_normal_group002")
    defect_parts = ("bmw_deform_group006", "bmw_edge_group014")
    unconfirmed_part = "bmw_normal_group999"
    for view in VIEW_ORDER:
        _placeholder_image(model_root / view / "model.ckpt")
        (release / "efficientad" / view / "defect").mkdir(parents=True)
        for part_id in normal_parts:
            _placeholder_image(
                _score_image(release / "efficientad" / view / "normal_test" / part_id / "images", part_id, view)
            )
        for part_id in (*defect_parts, unconfirmed_part):
            _placeholder_image(_score_image(release / "crops" / view, part_id, view))
    _placeholder_image(
        _score_image(
            release / "efficientad" / "front" / "defect" / "deform" / defect_parts[0] / "images",
            defect_parts[0],
            "front",
        )
    )
    _placeholder_image(
        _score_image(
            release / "efficientad" / "back_left" / "defect" / "edge" / defect_parts[1] / "images",
            defect_parts[1],
            "back_left",
        )
    )

    class FakePredictDataset:
        def __init__(self, path: Path) -> None:
            self.image_filenames = [Path(path)]

    class FakeEngine:
        def __init__(self, *, logger: bool) -> None:
            assert logger is False

        def predict(self, **kwargs) -> list[SimpleNamespace]:
            if "data_path" in kwargs:
                paths = sorted(Path(kwargs["data_path"]).rglob("*.png"))
                score = 0.1
                predicted = 0
            else:
                paths = list(kwargs["dataset"].image_filenames)
                score = 0.9
                predicted = 1
            count = len(paths)
            return [
                SimpleNamespace(
                    image_path=paths,
                    pred_score=np.full(count, score),
                    pred_label=np.full(count, predicted),
                    anomaly_map=np.zeros((count, 4, 4), dtype=np.float32),
                )
            ]

    class FakeEfficientAd:
        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            return cls()

    def write_placeholder_output(output_path: Path, *args, **kwargs) -> Path:
        return _placeholder_image(Path(output_path))

    monkeypatch.setattr(anomalib_data, "PredictDataset", FakePredictDataset)
    monkeypatch.setattr(anomalib_engine, "Engine", FakeEngine)
    monkeypatch.setattr(anomalib_models, "EfficientAd", FakeEfficientAd)
    monkeypatch.setitem(namespace["analyze"].__globals__, "render_example_sheet", write_placeholder_output)
    monkeypatch.setitem(namespace["analyze"].__globals__, "render_score_distributions", write_placeholder_output)
    monkeypatch.setitem(
        namespace["analyze"].__globals__,
        "score_metrics",
        lambda records: {
            "auroc": 1.0,
            "f1": 1.0,
            "balanced_accuracy": 1.0,
            "normal_predicted_ng": 0,
            "defect_predicted_ok": 0,
        },
    )
    args = namespace["build_parser"]().parse_args([
        "--training-release",
        str(release),
        "--model-root",
        str(model_root),
        "--output-dir",
        str(output_dir),
    ])

    report = namespace["analyze"](args)

    with Path(report["scores_csv"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == (len(normal_parts) + len(defect_parts)) * len(VIEW_ORDER)
    assert report["complete_eight_view"] is True
    assert report["defect_input_policy"] == "visible_defect_part_union_completed_from_crops"
    assert report["defect_part_ids"] == list(defect_parts)
    defect_rows = [row for row in rows if row["label"] == "defect"]
    assert len(defect_rows) == len(defect_parts) * len(VIEW_ORDER)
    for part_id in defect_parts:
        assert {
            row["view_id"]
            for row in defect_rows
            if namespace["part_id_from_image_path"](Path(row["image_path"]), row["view_id"]) == part_id
        } == set(VIEW_ORDER)
    assert all(unconfirmed_part not in row["image_path"] for row in defect_rows)
