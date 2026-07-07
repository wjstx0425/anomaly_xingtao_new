# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage 24: visualize traditional-operator CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (344, 214)
TILE_WIDTH = 368
TILE_HEIGHT = 365
HEADER_HEIGHT = 64
REASON_PATTERN = re.compile(r"raw_delta=(\S+).*?region=(\S+).*?score=([0-9.]+).*?threshold=([0-9.]+)")
REGION_PATTERN = re.compile(r"region=(r(?P<row>[0-9]+)_c(?P<col>[0-9]+))")
REGION_GRID = (4, 8)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Pipeline stage 24: visualize traditional-operator results.")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="traditional_predictions.csv path.")
    parser.add_argument("--cases-csv", type=Path, help="Optional traditional_cases.csv path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for rendered report images.")
    parser.add_argument("--report-name", default="traditional", help="Output filename prefix.")
    parser.add_argument(
        "--mode",
        choices=("defect-all", "false-positives", "false-negatives", "positives", "all"),
        default="all",
        help="Which cases to render.",
    )
    parser.add_argument("--max-cases", type=int, help="Maximum cases to render.")
    parser.add_argument("--columns", type=int, default=4, help="Contact sheet columns.")
    parser.add_argument(
        "--prefer-evidence-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show evidence overlay when available; otherwise show source image.",
    )
    return parser


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    """Read a CSV file, returning an empty list for omitted paths."""
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _case_key(row: dict[str, str]) -> tuple[str, str]:
    """Return the source/slot key used to match case and branch rows."""
    return row.get("source_path", ""), row.get("slot_id", "")


def _gt_defect_type(row: dict[str, str]) -> str:
    """Return the human label type, preferring explicit CSV columns."""
    explicit = (row.get("gt_defect_type") or "").strip()
    if explicit:
        return explicit
    source_path = Path(row.get("source_path", ""))
    return source_path.stem.split("_", maxsplit=1)[0] if source_path.stem else ""


def _branch_rows_by_case(predictions: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Group branch predictions by source path and slot id."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in predictions:
        grouped.setdefault(_case_key(row), []).append(row)
    return grouped


def _cases_from_predictions(predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build simple case rows when a cases CSV is not provided."""
    cases = []
    grouped = _branch_rows_by_case(predictions)
    for (source_path, slot_id), rows in sorted(grouped.items()):
        positive_branches = sorted({row.get("branch", "") for row in rows if row.get("pred_label") == "1"})
        first = rows[0]
        cases.append(
            {
                "case_id": f"{Path(source_path).stem}_{slot_id}",
                "label": "defect" if "defect" in source_path.lower() else "unknown",
                "part_id": first.get("part_id", ""),
                "side": first.get("side", ""),
                "view": first.get("view", ""),
                "slot_id": slot_id,
                "pred_label": "1" if positive_branches else "0",
                "positive_branches": ";".join(positive_branches),
                "gt_defect_type": _gt_defect_type(first),
                "source_path": source_path,
            },
        )
    return cases


def _filter_cases(cases: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    """Filter cases according to the requested report mode."""
    if mode == "all":
        return cases
    if mode == "defect-all":
        return [case for case in cases if case.get("label") == "defect"]
    if mode == "false-positives":
        return [case for case in cases if case.get("label", "").startswith("normal") and case.get("pred_label") == "1"]
    if mode == "false-negatives":
        return [case for case in cases if case.get("label") == "defect" and case.get("pred_label") != "1"]
    if mode == "positives":
        return [case for case in cases if case.get("pred_label") == "1"]
    return cases


def _status_for_case(case: dict[str, str]) -> tuple[str, tuple[int, int, int]]:
    """Return a human status and BGR color for a case row."""
    label = case.get("label", "")
    pred = case.get("pred_label") == "1"
    if label == "defect" and pred:
        return "HIT", (0, 145, 0)
    if label == "defect" and not pred:
        return "MISS", (0, 0, 220)
    if label.startswith("normal") and pred:
        return "FP", (0, 0, 220)
    if label.startswith("normal") and not pred:
        return "TN", (0, 145, 0)
    return ("POS" if pred else "NEG"), ((0, 145, 0) if pred else (90, 90, 90))


def _load_image(path_text: str, size: tuple[int, int]) -> np.ndarray:
    """Load an image into a fixed-size canvas."""
    image = cv2.imread(path_text)
    canvas = np.full((size[1], size[0], 3), 24, np.uint8)
    if image is None:
        cv2.putText(canvas, "missing image", (16, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return canvas
    height, width = image.shape[:2]
    scale = min(size[0] / width, size[1] / height)
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    y = (size[1] - resized.shape[0]) // 2
    x = (size[0] - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _load_source_image(path_text: str) -> np.ndarray:
    """Load a source image without resizing."""
    image = cv2.imread(path_text)
    if image is not None:
        return image
    return np.full((320, 480, 3), 24, np.uint8)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (35, 35, 35),
    max_chars: int = 43,
) -> None:
    """Draw up to three wrapped text lines."""
    x, y = origin
    words = str(text).split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if len(candidate) > max_chars and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, text_line in enumerate(lines[:3]):
        cv2.putText(
            image,
            text_line,
            (x, y + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _safe_name(value: str) -> str:
    """Return a filesystem-safe name fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def _selected_evidence_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return rows worth visualizing for a case."""
    positives = [row for row in rows if row.get("pred_label") == "1"]
    if positives:
        return positives
    geometry_rows = [row for row in rows if row.get("branch") == "geometry"]
    return geometry_rows[:1]


def _localization_path(case: dict[str, str], output_dir: Path, report_name: str) -> Path:
    """Return the per-case localization image path."""
    case_id = case.get("case_id") or Path(case.get("source_path", "")).stem
    slot_id = case.get("slot_id", "")
    return output_dir / "localization" / f"{_safe_name(report_name)}_{_safe_name(case_id)}_{_safe_name(slot_id)}_localization.jpg"


def _region_box(region: str, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    """Convert a geometry region name such as r02_c06 into image coordinates."""
    match = re.fullmatch(r"r(?P<row>[0-9]+)_c(?P<col>[0-9]+)", region)
    if match is None:
        return None
    row = int(match.group("row"))
    col = int(match.group("col"))
    rows, cols = REGION_GRID
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    height, width = image_shape[:2]
    x1 = int(round(col * width / cols))
    x2 = int(round((col + 1) * width / cols))
    y1 = int(round(row * height / rows))
    y2 = int(round((row + 1) * height / rows))
    return x1, y1, x2, y2


def _evidence_color(row: dict[str, str]) -> tuple[int, int, int]:
    """Return BGR overlay color for one evidence row."""
    evidence_type = row.get("evidence_type", "")
    if evidence_type == "missing_mask":
        return 0, 0, 255
    if evidence_type == "extra_mask":
        return 255, 0, 0
    if evidence_type == "dark_line":
        return 0, 255, 255
    if evidence_type == "texture_residual":
        return 0, 165, 255
    return 255, 0, 255


def _draw_region_localization(image: np.ndarray, row: dict[str, str]) -> bool:
    """Draw a coarse geometry region heatmap for one branch row."""
    reason = row.get("reason", "")
    match = REGION_PATTERN.search(reason)
    if match is None:
        return False
    region = match.group(1)
    box = _region_box(region, image.shape)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    color = np.array(_evidence_color(row), dtype=np.uint8)
    overlay = image.copy()
    overlay[y1:y2, x1:x2] = (0.35 * overlay[y1:y2, x1:x2] + 0.65 * color).astype(np.uint8)
    image[y1:y2, x1:x2] = overlay[y1:y2, x1:x2]
    cv2.rectangle(image, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), tuple(int(v) for v in color), 2)
    rows, cols = REGION_GRID
    height, width = image.shape[:2]
    grid_color = (70, 70, 70)
    for index in range(1, cols):
        x = int(round(index * width / cols))
        cv2.line(image, (x, 0), (x, height - 1), grid_color, 1)
    for index in range(1, rows):
        y = int(round(index * height / rows))
        cv2.line(image, (0, y), (width - 1, y), grid_color, 1)
    return True


def _draw_localization_label(image: np.ndarray, case: dict[str, str], rows: list[dict[str, str]]) -> None:
    """Draw a compact localization legend on the image."""
    label = f"GT:{_gt_defect_type(case) or case.get('label', '')}  {_summarize_positive_evidence(rows)}"
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 30), (0, 0, 0), -1)
    cv2.putText(image, label[:110], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def _render_localization_image(
    case: dict[str, str],
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    prefer_evidence_image: bool,
) -> None:
    """Render one per-case localization image."""
    selected = _selected_evidence_rows(rows)
    evidence_image = ""
    if prefer_evidence_image:
        evidence_image = next((row.get("evidence_path", "") for row in selected if row.get("evidence_path")), "")
    image = _load_source_image(evidence_image or case.get("source_path", ""))
    drew_region = False
    for row in selected:
        drew_region = _draw_region_localization(image, row) or drew_region
    if not drew_region and not selected:
        cv2.putText(image, "No traditional evidence", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif not drew_region and selected:
        cv2.putText(image, "Evidence has no local region", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    _draw_localization_label(image, case, selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def _render_localization_images(
    cases: list[dict[str, str]],
    predictions_by_case: dict[tuple[str, str], list[dict[str, str]]],
    output_dir: Path,
    *,
    report_name: str,
    prefer_evidence_image: bool,
) -> dict[tuple[str, str], str]:
    """Render per-case localization images and return source paths by case key."""
    outputs: dict[tuple[str, str], str] = {}
    for case in cases:
        rows = predictions_by_case.get(_case_key(case), [])
        output_path = _localization_path(case, output_dir, report_name)
        _render_localization_image(case, rows, output_path, prefer_evidence_image=prefer_evidence_image)
        outputs[_case_key(case)] = str(output_path)
    return outputs


def _summarize_positive_evidence(rows: list[dict[str, str]]) -> str:
    """Return a compact evidence summary that separates evidence from GT class."""
    positives = [row for row in rows if row.get("pred_label") == "1"]
    if not positives:
        geometry_rows = [row for row in rows if row.get("branch") == "geometry"]
        if not geometry_rows:
            return "Evidence: none"
        row = geometry_rows[0]
        return f"Evidence: geometry negative {row.get('score', '')}/{row.get('threshold', '')}"
    row = positives[0]
    branch = row.get("branch", "")
    evidence_type = row.get("evidence_type") or row.get("defect_type") or "positive"
    reason = _short_reason(row.get("reason", ""))
    return f"Evidence: {branch}:{evidence_type} {reason}".strip()


def _short_reason(reason: str) -> str:
    """Extract a readable reason fragment."""
    match = REASON_PATTERN.search(reason)
    if match is None:
        return reason[:48]
    raw_delta, region, score, threshold = match.groups()
    return f"raw_delta={raw_delta} region={region} {float(score):.0f}/{float(threshold):.0f}"


def _image_path_for_case(
    case: dict[str, str],
    rows: list[dict[str, str]],
    *,
    prefer_evidence_image: bool,
    localization_paths: dict[tuple[str, str], str] | None = None,
) -> str:
    """Return the image path to show for a case."""
    if localization_paths:
        localization_path = localization_paths.get(_case_key(case))
        if localization_path:
            return localization_path
    if prefer_evidence_image:
        positives = [row for row in rows if row.get("pred_label") == "1" and row.get("evidence_path")]
        if positives:
            return positives[0]["evidence_path"]
    return case.get("source_path", "")


def _render_page(
    cases: list[dict[str, str]],
    predictions_by_case: dict[tuple[str, str], list[dict[str, str]]],
    output_path: Path,
    *,
    title: str,
    columns: int,
    prefer_evidence_image: bool,
    localization_paths: dict[tuple[str, str], str] | None = None,
) -> None:
    """Render one contact-sheet page."""
    rows = max(1, math.ceil(len(cases) / columns))
    sheet = np.full((HEADER_HEIGHT + rows * TILE_HEIGHT, columns * TILE_WIDTH, 3), 244, np.uint8)
    cv2.putText(sheet, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (22, 22, 22), 2, cv2.LINE_AA)
    for index, case in enumerate(cases):
        row_index, column_index = divmod(index, columns)
        x0 = column_index * TILE_WIDTH
        y0 = HEADER_HEIGHT + row_index * TILE_HEIGHT
        status, color = _status_for_case(case)
        prediction_rows = predictions_by_case.get(_case_key(case), [])
        image_path = _image_path_for_case(
            case,
            prediction_rows,
            prefer_evidence_image=prefer_evidence_image,
            localization_paths=localization_paths,
        )
        thumb = _load_image(image_path, IMAGE_SIZE)
        sheet[y0 + 8 : y0 + 8 + IMAGE_SIZE[1], x0 + 12 : x0 + 12 + IMAGE_SIZE[0]] = thumb
        cv2.rectangle(sheet, (x0 + 5, y0 + 5), (x0 + TILE_WIDTH - 6, y0 + TILE_HEIGHT - 6), color, 4)
        cv2.putText(
            sheet,
            f"{status} {case.get('slot_id', '')}",
            (x0 + 12, y0 + 246),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        _draw_text(sheet, f"GT: {_gt_defect_type(case) or case.get('label', '')}", (x0 + 12, y0 + 270))
        _draw_text(sheet, f"File: {Path(case.get('source_path', '')).name}", (x0 + 12, y0 + 290))
        _draw_text(sheet, _summarize_positive_evidence(prediction_rows), (x0 + 12, y0 + 312))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def run_visualization(args: argparse.Namespace) -> list[Path]:
    """Render traditional-operator result sheets and return output paths."""
    predictions = _read_csv(args.predictions_csv)
    cases = _read_csv(args.cases_csv) or _cases_from_predictions(predictions)
    selected_cases = _filter_cases(cases, args.mode)
    if args.max_cases is not None:
        selected_cases = selected_cases[: args.max_cases]
    predictions_by_case = _branch_rows_by_case(predictions)
    localization_paths = _render_localization_images(
        selected_cases,
        predictions_by_case,
        args.output_dir,
        report_name=args.report_name,
        prefer_evidence_image=bool(args.prefer_evidence_image),
    )
    outputs: list[Path] = []
    page_size = max(1, int(args.columns) * 10)
    for page_index, start in enumerate(range(0, len(selected_cases), page_size), start=1):
        page_cases = selected_cases[start : start + page_size]
        output_path = args.output_dir / f"{args.report_name}_{args.mode}_page{page_index:02d}.jpg"
        _render_page(
            page_cases,
            predictions_by_case,
            output_path,
            title=f"{args.report_name} {args.mode} page {page_index}",
            columns=max(1, int(args.columns)),
            prefer_evidence_image=bool(args.prefer_evidence_image),
            localization_paths=localization_paths,
        )
        outputs.append(output_path)
    return outputs


def main() -> None:
    """Run the visualization CLI."""
    args = build_parser().parse_args()
    outputs = run_visualization(args)
    for output_path in outputs:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
