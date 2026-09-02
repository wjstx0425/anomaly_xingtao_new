"""Tests for the BMW Template / EfficientAD NG reviewer."""

from __future__ import annotations

import csv
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bmw_inspection.lab.ng_review import ReviewDataset


FIELDS = (
    "case_id",
    "decision",
    "review_note",
    "capture_id",
    "branch",
    "view_id",
    "panel_path",
    "custom_field",
)


def _write_review_csv(path: Path) -> None:
    rows = (
        {
            "case_id": "T001",
            "decision": "",
            "review_note": "",
            "capture_id": "part-002",
            "branch": "template",
            "view_id": "front",
            "panel_path": "/tmp/T001.jpg",
            "custom_field": "keep-a",
        },
        {
            "case_id": "E001",
            "decision": "真实缺陷",
            "review_note": "小凹坑",
            "capture_id": "part-002",
            "branch": "efficientad",
            "view_id": "front_left",
            "panel_path": "/tmp/E001.jpg",
            "custom_field": "keep-b",
        },
        {
            "case_id": "E002",
            "decision": "",
            "review_note": "",
            "capture_id": "part-001",
            "branch": "efficientad",
            "view_id": "back",
            "panel_path": "/tmp/E002.jpg",
            "custom_field": "keep-c",
        },
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_load_groups_in_csv_order_and_reports_progress(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    _write_review_csv(path)

    dataset = ReviewDataset.load(path)

    assert [group.capture_id for group in dataset.groups] == ["part-002", "part-001"]
    assert [case.case_id for case in dataset.groups[0].cases] == ["T001", "E001"]
    assert dataset.progress.case_count == 3
    assert dataset.progress.reviewed_case_count == 1
    assert dataset.progress.capture_count == 2
    assert dataset.progress.completed_capture_count == 0


def test_decision_validation_and_batch_update_preserve_existing_choice(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    _write_review_csv(path)
    dataset = ReviewDataset.load(path)

    with pytest.raises(ValueError, match="decision"):
        dataset.set_decision("T001", "可能误判")

    changed = dataset.set_remaining_for_capture("part-002", "误判")

    assert changed == 1
    assert dataset.find_case("T001").decision == "误判"
    assert dataset.find_case("E001").decision == "真实缺陷"
    assert dataset.progress.reviewed_case_count == 2
    assert dataset.progress.completed_capture_count == 1


def test_save_round_trip_preserves_rows_columns_and_notes(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    _write_review_csv(path)
    dataset = ReviewDataset.load(path)
    dataset.set_decision("E002", "不确定")
    dataset.set_note("E002", "请放大后复核")

    dataset.save()
    reloaded = ReviewDataset.load(path)

    assert reloaded.fieldnames == FIELDS
    assert [case.case_id for case in reloaded.cases] == ["T001", "E001", "E002"]
    assert reloaded.find_case("E002").decision == "不确定"
    assert reloaded.find_case("E002").review_note == "请放大后复核"
    assert [case.row["custom_field"] for case in reloaded.cases] == ["keep-a", "keep-b", "keep-c"]
    assert not list(tmp_path.glob(".review.csv.*.tmp"))


def test_load_rejects_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("case_id,capture_id\nT001,part-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        ReviewDataset.load(path)


def test_cli_parser_uses_shared_review_package_and_accepts_override(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[4] / "pipeline/bmw_lab_review_ng_cases.py"
    namespace = runpy.run_path(script)

    default_args = namespace["build_parser"]().parse_args([])
    custom_args = namespace["build_parser"]().parse_args(["--review-csv", str(tmp_path / "custom.csv")])

    assert default_args.review_csv.name == "review_cases.csv"
    assert default_args.review_csv.parent.name == "bmw_template_efficientad_ng_review_left_20260816_v1"
    assert custom_args.review_csv == tmp_path / "custom.csv"


def test_template_review_uses_original_roi_instead_of_misleading_difference_composite() -> None:
    script = Path(__file__).resolve().parents[4] / "pipeline/bmw_lab_review_ng_cases.py"
    namespace = runpy.run_path(script)
    current_array = np.zeros((80, 120, 3), dtype=np.uint8)
    current_array[10:70, 25:95] = (30, 140, 220)
    current = Image.fromarray(current_array)
    misleading_composite = Image.new("RGB", (512, 512), (255, 0, 255))

    display, title, explanation = namespace["_review_evidence_image"](
        current,
        misleading_composite,
        branch="template",
    )

    assert np.array_equal(np.asarray(display), current_array)
    assert title == "Template整体匹配异常（原始ROI）"
    assert "不提供像素级缺陷定位" in explanation


def test_configure_fonts_applies_cjk_family_to_tk_and_ttk() -> None:
    script = Path(__file__).resolve().parents[4] / "pipeline/bmw_lab_review_ng_cases.py"
    namespace = runpy.run_path(script)
    configured_fonts: dict[str, dict[str, object]] = {}
    style_settings: dict[str, dict[str, object]] = {}
    option_settings: list[tuple[str, str]] = []

    class FakeFont:
        def __init__(self, name: str) -> None:
            self.name = name

        def configure(self, **kwargs: object) -> None:
            configured_fonts[self.name] = kwargs

    font_module = SimpleNamespace(
        nametofont=lambda name, root=None: FakeFont(name),
        Font=lambda root=None, name=None, **kwargs: FakeFont(str(name)),
    )

    class FakeStyle:
        def configure(self, name: str, **kwargs: object) -> None:
            style_settings[name] = kwargs

    root = SimpleNamespace(option_add=lambda pattern, value: option_settings.append((pattern, value)))

    fonts = namespace["_configure_fonts"](
        root,
        font_module=font_module,
        style_factory=lambda _root: FakeStyle(),
    )

    assert configured_fonts["TkDefaultFont"]["family"] == "song ti"
    assert configured_fonts["TkDefaultFont"]["size"] == 12
    assert style_settings["."]["font"] == "TkDefaultFont"
    assert style_settings["TButton"]["font"] == "TkDefaultFont"
    assert style_settings["TRadiobutton"]["font"] == "TkDefaultFont"
    assert option_settings == [("*Font", "TkDefaultFont")]
    assert set(fonts) == {"title", "section", "case"}
