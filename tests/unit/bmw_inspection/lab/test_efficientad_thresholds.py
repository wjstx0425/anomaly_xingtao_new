"""Whole-part threshold fitting for the BMW eight-view EfficientAD scores."""

from __future__ import annotations

import csv
import hashlib
import json
import runpy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bmw_inspection.lab.efficientad_thresholds import (
    PartScore,
    PartThresholdFit,
    evaluate_part_thresholds,
    fit_part_thresholds,
    read_part_scores_csv,
    read_part_scores_csv_snapshot,
)
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER


def _part_rows(part_id: str, label: str, scores: dict[str, float] | None = None) -> list[PartScore]:
    values = scores or {}
    return [
        PartScore(
            part_id=part_id,
            view_id=view,
            label=label,
            score=values.get(view, 0.1),
            image_path=Path(f"/{part_id}/images/capture__{part_id}_000001__{view}.png"),
        )
        for view in VIEW_ORDER
    ]


def _eight_view_rows(*, normal_parts: int, defect_parts: int) -> tuple[PartScore, ...]:
    rows: list[PartScore] = []
    for index in range(normal_parts):
        rows.extend(_part_rows(f"normal-{index:03d}", "normal", {view: 0.1 + index / 1000 for view in VIEW_ORDER}))
    for index in range(defect_parts):
        rows.extend(_part_rows(f"defect-{index:03d}", "defect", {VIEW_ORDER[index % len(VIEW_ORDER)]: 0.9}))
    return tuple(rows)


def _tradeoff_rows() -> tuple[PartScore, ...]:
    rows = list(_eight_view_rows(normal_parts=20, defect_parts=0))
    rows = [
        PartScore(
            row.part_id,
            row.view_id,
            row.label,
            0.9
            if row.part_id == "normal-000" and row.view_id == VIEW_ORDER[0]
            else (0.8 if row.part_id == "normal-001" and row.view_id != VIEW_ORDER[0] else row.score),
            row.image_path,
        )
        for row in rows
    ]
    rows.extend(
        _part_rows(
            "defect-a",
            "defect",
            {VIEW_ORDER[0]: 0.5, **dict.fromkeys(VIEW_ORDER[1:], 0.7)},
        )
    )
    rows.extend(_part_rows("defect-b", "defect", {VIEW_ORDER[0]: 0.4, **dict.fromkeys(VIEW_ORDER[1:], 0.05)}))
    return tuple(rows)


def test_joint_fit_limits_union_of_false_positive_parts() -> None:
    rows = _eight_view_rows(normal_parts=20, defect_parts=2)

    fit = fit_part_thresholds(rows, views=VIEW_ORDER, target_part_fpr=0.05)

    assert fit.normal_part_count == 20
    assert fit.allowed_normal_false_positive_count == 1
    assert fit.normal_false_positive_count <= 1
    assert fit.observed_normal_part_fpr <= 0.05
    assert tuple(fit.thresholds) == VIEW_ORDER


def test_fit_optimizes_defect_part_recall_before_image_recall() -> None:
    fit = fit_part_thresholds(_tradeoff_rows(), views=VIEW_ORDER, target_part_fpr=0.05)

    assert fit.defect_detected_count == 2


def test_fit_uses_defect_image_hits_after_defect_part_recall() -> None:
    rows = list(_eight_view_rows(normal_parts=20, defect_parts=0))
    rows = [
        PartScore(
            row.part_id,
            row.view_id,
            row.label,
            0.9
            if row.part_id == "normal-000" and row.view_id in {VIEW_ORDER[0], VIEW_ORDER[2]}
            else (0.8 if row.part_id == "normal-001" and row.view_id == VIEW_ORDER[1] else row.score),
            row.image_path,
        )
        for row in rows
    ]
    rows.extend(
        _part_rows(
            "defect-image-tie",
            "defect",
            {VIEW_ORDER[0]: 0.5, VIEW_ORDER[1]: 0.5, VIEW_ORDER[2]: 0.5},
        )
    )

    fit = fit_part_thresholds(rows, views=VIEW_ORDER, target_part_fpr=0.05)

    assert fit.defect_detected_count == 1
    assert fit.defect_image_hits == 2
    assert fit.thresholds[VIEW_ORDER[0]] < 0.9


def test_general_budget_can_allow_two_false_positive_parts() -> None:
    rows = list(_eight_view_rows(normal_parts=20, defect_parts=0))
    rows = [
        PartScore(
            row.part_id,
            row.view_id,
            row.label,
            0.9
            if row.part_id == "normal-000" and row.view_id == VIEW_ORDER[0]
            else (0.8 if row.part_id == "normal-001" and row.view_id == VIEW_ORDER[1] else row.score),
            row.image_path,
        )
        for row in rows
    ]
    rows.extend(_part_rows("defect-a", "defect", {VIEW_ORDER[0]: 0.5}))
    rows.extend(_part_rows("defect-b", "defect", {VIEW_ORDER[1]: 0.5}))

    fit = fit_part_thresholds(rows, views=VIEW_ORDER, target_part_fpr=0.1)

    assert fit.allowed_normal_false_positive_count == 2
    assert fit.normal_false_positive_count == 2
    assert fit.defect_detected_count == 2


def test_fit_rejects_incomplete_or_inconsistently_grouped_parts() -> None:
    incomplete = _eight_view_rows(normal_parts=1, defect_parts=1)[1:]
    with pytest.raises(ValueError, match="exactly one row for each requested view"):
        fit_part_thresholds(incomplete, views=VIEW_ORDER, target_part_fpr=0.05)

    inconsistent = list(_eight_view_rows(normal_parts=1, defect_parts=1))
    inconsistent[-1] = PartScore(
        part_id=inconsistent[-1].part_id,
        view_id=inconsistent[-1].view_id,
        label="normal",
        score=inconsistent[-1].score,
        image_path=inconsistent[-1].image_path,
    )
    with pytest.raises(ValueError, match="one label"):
        fit_part_thresholds(inconsistent, views=VIEW_ORDER, target_part_fpr=0.05)


def test_fit_rejects_incomplete_defect_parts() -> None:
    rows = list(_eight_view_rows(normal_parts=20, defect_parts=0))
    rows.append(_part_rows("defect-sparse", "defect", {VIEW_ORDER[0]: 0.9})[0])

    with pytest.raises(ValueError, match="exactly one row for each requested view"):
        fit_part_thresholds(rows, views=VIEW_ORDER, target_part_fpr=0.05)


def test_floor_budget_and_tie_breaking_are_deterministic() -> None:
    rows = _eight_view_rows(normal_parts=19, defect_parts=2)

    first = fit_part_thresholds(rows, views=VIEW_ORDER, target_part_fpr=0.05)
    second = fit_part_thresholds(tuple(reversed(rows)), views=VIEW_ORDER, target_part_fpr=0.05)

    assert first.allowed_normal_false_positive_count == 0
    assert first.normal_false_positive_count == 0
    assert first.thresholds == second.thresholds


def test_fit_prefers_fewer_normal_false_positive_parts_after_defect_ties() -> None:
    rows = _eight_view_rows(normal_parts=20, defect_parts=0)

    fit = fit_part_thresholds(rows, views=VIEW_ORDER, target_part_fpr=0.05)

    assert fit.normal_false_positive_count == 0
    for view in VIEW_ORDER:
        assert fit.thresholds[view] > max(row.score for row in rows if row.view_id == view)


def test_final_tie_break_prefers_the_stable_higher_threshold_tuple() -> None:
    rows = list(_eight_view_rows(normal_parts=20, defect_parts=0))
    rows = [
        PartScore(
            row.part_id,
            row.view_id,
            row.label,
            0.9
            if row.part_id == "normal-000" and row.view_id == VIEW_ORDER[0]
            else (0.8 if row.part_id == "normal-001" and row.view_id == VIEW_ORDER[1] else row.score),
            row.image_path,
        )
        for row in rows
    ]
    rows.extend(_part_rows("defect-tie", "defect", {VIEW_ORDER[0]: 0.5, VIEW_ORDER[1]: 0.5}))

    fit = fit_part_thresholds(rows, views=VIEW_ORDER, target_part_fpr=0.05)

    assert fit.thresholds[VIEW_ORDER[0]] > 0.9


def test_evaluation_counts_union_of_view_hits_and_threshold_equality() -> None:
    rows = _part_rows("normal-000", "normal", {VIEW_ORDER[0]: 0.5, VIEW_ORDER[1]: 0.9})
    rows.extend(_part_rows("defect-000", "defect", {VIEW_ORDER[0]: 0.5, VIEW_ORDER[1]: 0.1}))

    evaluation = evaluate_part_thresholds(rows, dict.fromkeys(VIEW_ORDER, 0.5))

    assert evaluation.normal_false_positive_count == 1
    assert evaluation.defect_detected_count == 1
    assert evaluation.defect_image_hits == 1


@pytest.mark.parametrize(
    "thresholds",
    [
        {},
        {**dict.fromkeys(VIEW_ORDER, 0.5), "unexpected": 0.5},
        {**dict.fromkeys(VIEW_ORDER, 0.5), VIEW_ORDER[0]: float("nan")},
    ],
)
def test_evaluation_rejects_invalid_threshold_mappings(thresholds: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        evaluate_part_thresholds(_eight_view_rows(normal_parts=1, defect_parts=1), thresholds)


def test_saturated_noisy_view_can_be_disabled_above_normalized_score_range() -> None:
    rows: list[PartScore] = []
    for index in range(20):
        rows.extend(_part_rows(f"normal-{index:03d}", "normal", {VIEW_ORDER[0]: 1.0}))
    rows.extend(_part_rows("defect-000", "defect", {VIEW_ORDER[1]: 0.9}))

    fit = fit_part_thresholds(rows, views=VIEW_ORDER, target_part_fpr=0.05)

    assert fit.thresholds[VIEW_ORDER[0]] > 1.0
    assert fit.defect_detected_count == 1


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_part_score_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        PartScore("normal-000", VIEW_ORDER[0], "normal", score, Path("normal.png"))


@pytest.mark.parametrize("score", [True, "0.1"])
def test_part_score_rejects_non_numeric_scores(score: object) -> None:
    with pytest.raises(TypeError, match="score must be a real number"):
        PartScore("normal-000", VIEW_ORDER[0], "normal", score, Path("normal.png"))


def test_score_csv_reads_real_six_column_rows_and_supported_identity_paths(tmp_path: Path) -> None:
    score_csv = tmp_path / "efficientad_scores.csv"
    paths = (
        Path(
            "/release/efficientad/front/normal_test/bmw_normal_group001/images/"
            "20260806_100000__bmw_normal_group001_000001__front.png"
        ),
        Path(
            "/release/efficientad/front/defect/edge/bmw_edge_group002/images/"
            "20260806_100000__bmw_edge_group002_000001__front.png"
        ),
        Path("/release/crops/front/20260806_100000__bmw_normal_group003_000001__front.png"),
    )
    with score_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("view_id", "label", "prediction", "score", "threshold", "image_path"))
        writer.writerow(("front", "normal", "OK", "0.1", "0.5", paths[0]))
        writer.writerow(("front", "defect", "NG", "0.9", "0.5", paths[1]))
        writer.writerow(("front", "normal", "OK", "0.2", "0.5", paths[2]))

    rows = read_part_scores_csv(score_csv)

    assert [row.part_id for row in rows] == [
        "bmw_normal_group001",
        "bmw_edge_group002",
        "bmw_normal_group003",
    ]
    assert [row.score for row in rows] == [0.1, 0.9, 0.2]


def test_score_csv_rejects_ambiguous_images_parent_without_sample_identity(tmp_path: Path) -> None:
    score_csv = tmp_path / "efficientad_scores.csv"
    score_csv.write_text(
        "view_id,label,score,image_path\nfront,normal,0.1,/release/not-a-part/images/arbitrary.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot derive physical part identity"):
        read_part_scores_csv(score_csv)


def test_score_csv_snapshot_hashes_the_same_bytes_it_parses(tmp_path: Path, monkeypatch) -> None:
    score_csv = tmp_path / "efficientad_scores.csv"
    rows = _eight_view_rows(normal_parts=1, defect_parts=1)
    with score_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("view_id", "label", "score", "image_path"))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "view_id": row.view_id,
                "label": row.label,
                "score": row.score,
                "image_path": row.image_path,
            })
    original_bytes = score_csv.read_bytes()
    original_read_bytes = Path.read_bytes
    read_count = 0

    def mutate_after_read(path: Path) -> bytes:
        nonlocal read_count
        data = original_read_bytes(path)
        if path == score_csv:
            read_count += 1
            path.write_text("tampered-after-snapshot\n", encoding="utf-8")
        return data

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)

    parsed, digest = read_part_scores_csv_snapshot(score_csv)

    assert read_count == 1
    assert len(parsed) == 16
    assert digest == hashlib.sha256(original_bytes).hexdigest()
    assert score_csv.read_text(encoding="utf-8") == "tampered-after-snapshot\n"


def test_cli_writes_hash_bound_demo_only_assets(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[4]
    score_csv = tmp_path / "efficientad_scores.csv"
    rows = _eight_view_rows(normal_parts=20, defect_parts=2)
    with score_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("view_id", "label", "score", "image_path"))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "view_id": row.view_id,
                "label": row.label,
                "score": row.score,
                "image_path": row.image_path,
            })
    score_bytes = score_csv.read_bytes()
    expected_sha256 = hashlib.sha256(score_bytes).hexdigest()
    namespace = runpy.run_path(root / "pipeline/bmw_lab_calibrate_efficientad_thresholds.py")
    output_dir = tmp_path / "thresholds"
    original_snapshot = namespace["read_part_scores_csv_snapshot"]
    snapshot_calls = 0

    def snapshot_then_mutate(path: Path) -> tuple[tuple[PartScore, ...], str]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        snapshot = original_snapshot(path)
        Path(path).write_text("tampered-after-cli-snapshot\n", encoding="utf-8")
        return snapshot

    monkeypatch.setitem(namespace["calibrate"].__globals__, "read_part_scores_csv_snapshot", snapshot_then_mutate)

    defaults = namespace["build_parser"]().parse_args([])
    assert defaults.scores_csv.name == "efficientad_scores.csv"
    assert defaults.output_dir == defaults.scores_csv.parent
    assert defaults.target_part_fpr == 0.05
    assert tuple(defaults.views) == VIEW_ORDER

    assert namespace["main"](["--scores-csv", str(score_csv), "--output-dir", str(output_dir)]) == 0
    assert snapshot_calls == 1

    thresholds = json.loads((output_dir / "part_thresholds.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "part_threshold_report.json").read_text(encoding="utf-8"))
    required_fields = {
        "thresholds",
        "normal_part_count",
        "normal_false_positive_count",
        "observed_normal_part_fpr",
        "defect_part_count",
        "defect_detected_count",
        "demo_only",
        "test_used_for_selection",
        "calibrated_at_utc",
        "source_csv_sha256",
    }
    for payload in (thresholds, report):
        assert required_fields <= payload.keys()
        assert payload["source_csv_sha256"] == expected_sha256
        assert payload["demo_only"] is True
        assert payload["test_used_for_selection"] is True
        assert payload["normal_part_count"] == 20
        assert payload["normal_false_positive_count"] <= 1
        assert payload["defect_part_count"] == 2
        assert isinstance(payload["observed_normal_part_fpr"], float)
        assert all(isinstance(value, float) for value in payload["thresholds"].values())
    assert tuple(thresholds["thresholds"]) == VIEW_ORDER
    assert thresholds["calibrated_at_utc"] == report["calibrated_at_utc"]
    calibrated_at = datetime.strptime(thresholds["calibrated_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    assert calibrated_at.utcoffset().total_seconds() == 0
    score_csv.write_bytes(score_bytes)
    assert (
        namespace["main"]([
            "--scores-csv",
            str(score_csv),
            "--output-dir",
            str(tmp_path / "partial"),
            "--views",
            VIEW_ORDER[0],
        ])
        == 2
    )


def test_calibration_cli_renders_final_per_view_thresholds_without_inference(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[4]
    score_csv = tmp_path / "efficientad_scores.csv"
    rows = _eight_view_rows(normal_parts=1, defect_parts=1)
    with score_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("view_id", "label", "score", "image_path"))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "view_id": row.view_id,
                "label": row.label,
                "score": row.score,
                "image_path": row.image_path,
            })
    thresholds = {view: 0.2 + index / 100 for index, view in enumerate(VIEW_ORDER)}
    thresholds["back"] = 1.0000000000000002
    fit = PartThresholdFit(
        thresholds=thresholds,
        target_part_fpr=0.05,
        allowed_normal_false_positive_count=0,
        normal_part_count=1,
        normal_false_positive_count=0,
        observed_normal_part_fpr=0.0,
        defect_part_count=1,
        defect_detected_count=1,
        defect_image_hits=1,
    )
    namespace = runpy.run_path(root / "pipeline/bmw_lab_calibrate_efficientad_thresholds.py")
    captured: dict[str, object] = {}

    def render(
        output_path: Path,
        records: tuple[object, ...],
        *,
        views: tuple[str, ...],
        threshold: object,
    ) -> Path:
        captured.update(output_path=output_path, records=records, views=views, threshold=threshold)
        output_path.write_bytes(b"calibrated-plot")
        return output_path

    monkeypatch.setitem(namespace["calibrate"].__globals__, "fit_part_thresholds", lambda *_args, **_kwargs: fit)
    monkeypatch.setitem(namespace["calibrate"].__globals__, "render_score_distributions", render)
    output_dir = tmp_path / "thresholds"
    args = namespace["build_parser"]().parse_args([
        "--scores-csv",
        str(score_csv),
        "--output-dir",
        str(output_dir),
    ])

    report = namespace["calibrate"](args)

    plot_path = output_dir.resolve() / "efficientad_calibrated_score_distributions.png"
    assert captured["output_path"] == plot_path
    assert captured["views"] == VIEW_ORDER
    assert captured["threshold"] == thresholds
    plotted = captured["records"]
    assert [item.view_id for item in plotted] == [row.view_id for row in rows]
    assert [item.label for item in plotted] == [row.label for row in rows]
    assert [item.score for item in plotted] == [row.score for row in rows]
    assert [item.image_path for item in plotted] == [row.image_path for row in rows]
    assert all(item.predicted_anomalous is (item.score >= thresholds[item.view_id]) for item in plotted)
    assert report["calibrated_score_distributions"] == str(plot_path)
    persisted = json.loads((output_dir / "part_threshold_report.json").read_text(encoding="utf-8"))
    assert persisted["calibrated_score_distributions"] == str(plot_path)
