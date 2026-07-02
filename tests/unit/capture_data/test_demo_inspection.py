# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the C789 two-sided inspection demo."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np


def load_demo_module() -> ModuleType:
    """Load the demo script from its file path."""
    script_path = Path(__file__).resolve().parents[3] / "capture_data" / "demo_inspection.py"
    spec = importlib.util.spec_from_file_location("capture_data_demo_inspection", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load demo script from {script_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_full_fixture(path: Path, value: int = 120) -> None:
    """Write an image large enough for the C789 and FX11 presets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((3200, 4300, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_report_fixture(root: Path, view: str, normal_test: list[float], defect: list[float], summary: float) -> None:
    """Write minimal report CSVs for threshold profile tests."""
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "summary.csv").write_text(
        "\n".join(
            [
                "model,view,deploy_threshold",
                f"anomaly_dino,{view},{summary}",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    rows = ["model,view,label,pred_score"]
    rows.extend(f"anomaly_dino,{view},normal_test,{score}" for score in normal_test)
    rows.extend(f"anomaly_dino,{view},defect,{score}" for score in defect)
    (reports / "predictions.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_face_order_is_top_then_bottom() -> None:
    """The demo must inspect the top side before the bottom side."""
    demo = load_demo_module()

    assert demo.FACE_ORDER == ("top", "bottom")


def test_resolve_face_configs_uses_c789_defaults(tmp_path: Path) -> None:
    """Default face configs should point at the trained C789 top/bottom models."""
    demo = load_demo_module()
    args = demo.build_parser().parse_args(["--part-profile", "c789", "--output-dir", str(tmp_path)])

    configs = demo.resolve_face_configs(args)

    assert configs["top"].title == "正面"
    assert configs["top"].view == "left_top"
    assert configs["top"].preset_name == "c789_left_top_3x2"
    assert configs["top"].gain == 0.0
    assert configs["bottom"].title == "底面"
    assert configs["bottom"].view == "left_bottom"
    assert configs["bottom"].preset_name == "c789_left_bottom_3x2"
    assert configs["bottom"].gain == 0.0


def test_threshold_profile_demo_uses_report_score_distribution(tmp_path: Path) -> None:
    """Demo thresholds should use fixed top and normal-safe bottom handling."""
    demo = load_demo_module()
    top_root = tmp_path / "top"
    bottom_root = tmp_path / "bottom"
    _write_report_fixture(top_root, "left_top", normal_test=[0.2, 0.3], defect=[0.7], summary=0.3)
    _write_report_fixture(bottom_root, "left_bottom", normal_test=[0.2, 0.5], defect=[0.4], summary=0.5)
    args = demo.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--part-profile",
            "c789",
            "--top-output-root",
            str(top_root),
            "--bottom-output-root",
            str(bottom_root),
        ],
    )

    configs = demo.resolve_face_configs(args)

    assert configs["top"].threshold == demo.PROFILE_DEMO_THRESHOLDS["c789"]["top"]
    assert "固定" in configs["top"].threshold_source
    assert configs["bottom"].threshold == 0.5
    assert "重叠" in configs["bottom"].threshold_source


def test_threshold_profile_report_uses_summary_csv(tmp_path: Path) -> None:
    """Report thresholds should keep the workflow summary.csv value."""
    demo = load_demo_module()
    top_root = tmp_path / "top"
    bottom_root = tmp_path / "bottom"
    _write_report_fixture(top_root, "left_top", normal_test=[0.2, 0.3], defect=[0.7], summary=0.31)
    _write_report_fixture(bottom_root, "left_bottom", normal_test=[0.2, 0.5], defect=[0.4], summary=0.51)
    args = demo.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--part-profile",
            "c789",
            "--top-output-root",
            str(top_root),
            "--bottom-output-root",
            str(bottom_root),
            "--threshold-profile",
            "report",
        ],
    )

    configs = demo.resolve_face_configs(args)

    assert configs["top"].threshold == 0.31
    assert configs["bottom"].threshold == 0.51
    assert "summary.csv" in configs["top"].threshold_source


def test_profile_specific_checkpoint_overrides_work_for_switchable_parts(tmp_path: Path) -> None:
    """Profile-specific checkpoint flags should work even when the profile is not initially selected."""
    demo = load_demo_module()
    c789_top = tmp_path / "c789_top.ckpt"
    fx11_bottom = tmp_path / "fx11_bottom.ckpt"
    args = demo.build_parser().parse_args(
        [
            "--part-profile",
            "fx11",
            "--c789-top-ckpt-path",
            str(c789_top),
            "--c789-bottom-threshold",
            "0.42",
            "--fx11-bottom-ckpt-path",
            str(fx11_bottom),
        ],
    )

    c789_configs = demo.resolve_face_configs(args, "c789")
    fx11_configs = demo.resolve_face_configs(args, "fx11")

    assert c789_configs["top"].ckpt_path == c789_top
    assert c789_configs["bottom"].threshold == 0.42
    assert fx11_configs["bottom"].ckpt_path == fx11_bottom


def test_capture_defaults_match_training_hdr_recipe(tmp_path: Path) -> None:
    """Default HDR capture settings should match the C789 training capture recipe."""
    demo = load_demo_module()

    args = demo.build_parser().parse_args(["--output-dir", str(tmp_path)])

    assert args.hdr_settle_frames == 8
    assert args.short_dark_threshold == 80.0
    assert args.blend_width == 50.0
    assert args.blur_size == 101
    assert args.predict_batch_size == 1


def test_parse_mock_defects() -> None:
    """Mock defect parser should keep defects separated by face."""
    demo = load_demo_module()

    defects = demo.parse_mock_defects(["top:slot02,slot05", "bottom:none"])

    assert defects == {"top": {"slot02", "slot05"}, "bottom": set()}


def test_face_and_part_statuses(tmp_path: Path) -> None:
    """Face status and whole-part status should be reported independently."""
    demo = load_demo_module()
    top = demo.FaceResult(
        "top",
        "正面",
        tmp_path / "top.png",
        tmp_path / "top_crops",
        tmp_path / "top.csv",
        [
            demo.SlotResult("slot01", 0.1, 0.5, 0, tmp_path / "slot01.png"),
            demo.SlotResult("slot02", 0.7, 0.5, 1, tmp_path / "slot02.png"),
        ],
    )
    bottom = demo.FaceResult(
        "bottom",
        "底面",
        tmp_path / "bottom.png",
        tmp_path / "bottom_crops",
        tmp_path / "bottom.csv",
        [demo.SlotResult("slot01", 0.2, 0.5, 0, tmp_path / "slot01.png")],
    )

    assert demo.format_face_result(top) == "正面: NG, 缺陷位置: slot02"
    assert demo.format_face_result(bottom) == "底面: OK, 缺陷位置: 无"
    assert demo.part_status({"top": top}) == "NG"
    assert demo.part_status({"top": top, "bottom": bottom}) == "NG"
    assert demo.all_defect_positions({"top": top, "bottom": bottom}) == ["正面 slot02"]


def test_defect_regions_from_prediction_mask_are_mapped_to_full_image(tmp_path: Path) -> None:
    """Prediction masks should become full-image defect contours."""
    demo = load_demo_module()
    args = demo.build_parser().parse_args(["--part-profile", "c789", "--output-dir", str(tmp_path)])
    face = demo.resolve_face_configs(args)["top"]
    slot = face.preset.slots[0]
    mask = np.zeros((slot.box.height, slot.box.width), dtype=np.uint8)
    mask[20:60, 30:90] = 1

    regions = demo.defect_regions_from_artifact({"pred_mask": mask}, face, slot, min_area=10)

    assert len(regions) == 1
    assert regions[0].bbox == (30, 20, 90, 60)
    assert regions[0].full_bbox == (
        face.preset.roi.x1 + slot.box.x1 + 30,
        face.preset.roi.y1 + slot.box.y1 + 20,
        face.preset.roi.x1 + slot.box.x1 + 90,
        face.preset.roi.y1 + slot.box.y1 + 60,
    )


def test_stamp_line_parser_accepts_fixed_two_line_format() -> None:
    """Stamp parsing should return part number plus YYYYMMDD date."""
    demo = load_demo_module()

    assert demo.parse_stamp_lines("5027875", "20260521") == ("5027875", "20260521", "OK")
    assert demo.parse_stamp_lines("20260521", "5027875") == ("5027875", "20260521", "OK")
    assert demo.parse_stamp_lines("5027875", "20269999")[2] == "FAIL"


def test_stamp_roi_crop_applies_mirror_horizontal() -> None:
    """Stamp ROI orientation should handle mirrored camera text."""
    demo = load_demo_module()
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    crop[:, 0] = (10, 0, 0)
    crop[:, 3] = (200, 0, 0)

    stamp = demo._crop_stamp_roi(
        crop,
        {"x1": 0, "y1": 0, "x2": 4, "y2": 4, "rotate_180": False, "mirror_horizontal": True},
    )

    assert stamp is not None
    assert int(stamp[0, 0, 0]) == 200
    assert int(stamp[0, 3, 0]) == 10


def test_slot_ocr_text_shows_raw_digits_when_confidence_is_low(tmp_path: Path) -> None:
    """The GUI should show partial OCR digits instead of hiding them completely."""
    demo = load_demo_module()
    slot = demo.SlotResult(
        "slot01",
        0.1,
        0.5,
        0,
        tmp_path / "slot01.png",
        stamp_ocr=demo.StampOcrResult(status="FAIL", raw_text="5027875\n20260521", confidence=0.1),
    )
    result = demo.FaceResult("top", "正面", tmp_path / "top.png", tmp_path / "crops", tmp_path / "pred.csv", [slot])

    assert demo._slot_ocr_text(result, "slot01") == "钢印? 5027875 / 20260521"


def test_archive_csv_appends_without_duplicate_header(tmp_path: Path) -> None:
    """Archive CSV appends should keep a single header row."""
    demo = load_demo_module()
    path = tmp_path / "archive.csv"
    rows = [{"created_at": "t1", "inspection_id": "run1"}]

    demo._write_archive_rows(path, ("created_at", "inspection_id"), rows)
    demo._write_archive_rows(path, ("created_at", "inspection_id"), rows)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines.count("created_at,inspection_id") == 1
    assert len(lines) == 3


def test_render_dashboard_returns_nonblank_canvas(tmp_path: Path) -> None:
    """The OpenCV dashboard renderer should be testable without opening a window."""
    demo = load_demo_module()
    args = demo.build_parser().parse_args(["--output-dir", str(tmp_path)])
    configs = demo.resolve_face_configs(args)
    state = demo.DemoState(current_face="top", current_image=np.full((3200, 3700, 3), 180, dtype=np.uint8))

    canvas = demo.render_dashboard(state, configs)

    assert canvas.shape == (920, 1600, 3)
    assert canvas.std() > 0


def test_auto_mock_demo_writes_two_independent_face_results(tmp_path: Path, capsys) -> None:
    """Auto mock mode should run top then bottom and save final artifacts."""
    demo = load_demo_module()
    top_image = tmp_path / "top.png"
    bottom_image = tmp_path / "bottom.png"
    screenshot = tmp_path / "ui.png"
    _write_full_fixture(top_image, 100)
    _write_full_fixture(bottom_image, 140)
    args = demo.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--demo-top-image",
            str(top_image),
            "--demo-bottom-image",
            str(bottom_image),
            "--mock-predictions",
            "--mock-defects",
            "top:slot02",
            "--mock-defects",
            "bottom:none",
            "--auto-run",
            "--no-gui",
            "--save-ui-screenshot",
            str(screenshot),
        ],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = demo.resolve_face_configs(args)
    predictors = demo.build_predictors(configs, args)
    state = demo.DemoState()

    demo.run_auto_demo(state, configs, predictors, args)

    captured = capsys.readouterr()
    assert "正面: NG, 缺陷位置: slot02" in captured.out
    assert "底面: OK, 缺陷位置: 无" in captured.out
    assert "整件: NG, 缺陷位置: 正面 slot02" in captured.out
    assert "[quality] 正面:" in captured.out
    assert "[timing] 正面:" in captured.out
    assert state.finished
    assert state.results["top"].status == "NG"
    assert state.results["bottom"].status == "OK"
    top_dir = tmp_path / "run" / "fx11" / "top"
    bottom_dir = tmp_path / "run" / "fx11" / "bottom"
    assert (top_dir / "top_predictions.csv").is_file()
    assert (bottom_dir / "bottom_predictions.csv").is_file()
    assert (top_dir / "top_annotated.png").is_file()
    assert (bottom_dir / "bottom_annotated.png").is_file()
    trace_path = top_dir / "top_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["quality"]["status"] in {"OK", "WARN"}
    assert trace["model"]["predict_batch_size"] == 1
    assert trace["model"]["effective_predict_batch_size"] == 1
    assert trace["timings_ms"]["model_inference"] >= 0
    assert trace["defect_regions"][0]["slot"] == "slot02"
    assert "stamp_ocr" in trace
    assert trace["artifacts"]["annotated_image"].endswith("top_annotated.png")
    slots_csv = tmp_path / "run" / "archive" / "inspection_slots.csv"
    parts_csv = tmp_path / "run" / "archive" / "inspection_parts.csv"
    assert slots_csv.is_file()
    assert parts_csv.is_file()
    with slots_csv.open(newline="", encoding="utf-8") as file:
        slot_rows = list(csv.DictReader(file))
    with parts_csv.open(newline="", encoding="utf-8") as file:
        part_rows = list(csv.DictReader(file))
    assert len(slot_rows) == 12
    assert len(part_rows) == 6
    assert any(row["slot"] == "slot02" and row["status"] == "NG" for row in slot_rows)
    assert any(row["slot"] == "slot02" and row["status"] == "NG" for row in part_rows)
    assert screenshot.is_file()
    assert not (top_dir / "raw_exposures").exists()
