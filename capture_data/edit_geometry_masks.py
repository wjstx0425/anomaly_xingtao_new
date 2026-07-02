# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Interactively edit manual geometry mask PNGs for C789 slot templates."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_TYPES = ("expected", "allowed", "ignore", "watch_edge")
KEY_SAMPLE_IDS = ("less_1_2_slot02", "less_2_1_slot03", "more_2_2_slot04")
WINDOW_NAME = "C789 Geometry Mask Editor"

MASK_COLORS = {
    "allowed": (0, 180, 0),
    "expected": (0, 255, 255),
    "ignore": (180, 80, 220),
    "watch_edge": (0, 220, 255),
}


@dataclass(frozen=True)
class SlotReviewItem:
    """Review-pack paths and counts for one slot."""

    slot: str
    reference_path: Path
    review_sheet: Path
    masks: dict[str, Path]
    normal_count: int
    stress_count: int
    defect_count: int


@dataclass(frozen=True)
class EditorConfig:
    """Configuration for geometry mask editing and validation."""

    review_pack: Path
    template_dir: Path
    stress_root: Path
    defect_root: Path
    anomaly_predictions: Path
    stress_output_dir: Path
    defect_output_dir: Path
    threshold_margin: float = 0.05
    threshold_mode: str = "region"
    tolerance_px: int = 3
    min_component_area: int = 64
    hole_dilation: int = 32
    max_window_width: int = 1900
    max_window_height: int = 1050


@dataclass(frozen=True)
class KeySampleStatus:
    """Fused prediction labels for one key sample."""

    anomaly: int | None = None
    geometry: int | None = None
    final: int | None = None


@dataclass(frozen=True)
class EvalSummary:
    """Human-readable validation metrics from geometry output CSVs."""

    stress_fp: int = 0
    stress_total: int = 0
    geometry_positives: int = 0
    geometry_total: int = 0
    fused_positives: int = 0
    fused_total: int = 0
    key_samples: dict[str, KeySampleStatus] = field(default_factory=dict)

    def lines(self) -> list[str]:
        """Return compact display lines."""
        lines = [
            f"stress FP: {self.stress_fp}/{self.stress_total}",
            f"geometry positives: {self.geometry_positives}/{self.geometry_total}",
            f"fused recall: {self.fused_positives}/{self.fused_total}",
        ]
        for sample_id in KEY_SAMPLE_IDS:
            status = self.key_samples.get(sample_id, KeySampleStatus())
            lines.append(
                f"{sample_id}: A={_label_text(status.anomaly)} G={_label_text(status.geometry)} F={_label_text(status.final)}",
            )
        return lines


def _label_text(value: int | None) -> str:
    """Return a display value for a binary label."""
    return "?" if value is None else str(value)


def _int_value(value: str | int | float | bool | None) -> int:
    """Parse a CSV integer-ish label."""
    if isinstance(value, bool):
        return int(value)
    if value in {None, ""}:
        return 0
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no"}:
        return 0
    return int(float(text))


def _is_defect_row(row: dict[str, str]) -> bool:
    """Return whether a CSV row represents a defect sample."""
    return _int_value(row.get("gt_label")) == 1 or row.get("label") == "defect" or row.get("dataset_label") == "defect"


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _resolve_path(value: str, review_pack: Path) -> Path:
    """Resolve a review-pack manifest path."""
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = review_pack / path
    if candidate.exists():
        return candidate
    return path


def load_review_manifest(review_pack: Path) -> list[SlotReviewItem]:
    """Load slot review items from a manual review pack."""
    manifest_path = review_pack / "review_pack_manifest.csv"
    if not manifest_path.is_file():
        msg = f"Missing review pack manifest: {manifest_path}"
        raise FileNotFoundError(msg)

    items: list[SlotReviewItem] = []
    for row in _read_csv(manifest_path):
        masks = {
            "expected": _resolve_path(row["expected_mask"], review_pack),
            "allowed": _resolve_path(row["allowed_mask"], review_pack),
            "ignore": _resolve_path(row["ignore_mask"], review_pack),
            "watch_edge": _resolve_path(row["watch_edge_mask"], review_pack),
        }
        items.append(
            SlotReviewItem(
                slot=row["slot"],
                reference_path=_resolve_path(row.get("reference_image") or row["reference_path"], review_pack),
                review_sheet=_resolve_path(row["review_sheet"], review_pack),
                masks=masks,
                normal_count=_int_value(row.get("normal_count")),
                stress_count=_int_value(row.get("stress_count")),
                defect_count=_int_value(row.get("defect_count")),
            ),
        )
    if not items:
        msg = f"No rows found in {manifest_path}"
        raise RuntimeError(msg)
    return items


def mask_dir_from_review_manifest(review_pack: Path) -> Path:
    """Return the shared manual mask directory from a review manifest."""
    items = load_review_manifest(review_pack)
    parents = {path.parent for item in items for path in item.masks.values()}
    if len(parents) != 1:
        msg = f"Manual masks must share one directory for template compilation: {sorted(str(path) for path in parents)}"
        raise ValueError(msg)
    return next(iter(parents))


def read_binary_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a mask where any non-zero pixel is active."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        msg = f"Could not read mask: {path}"
        raise FileNotFoundError(msg)
    binary = mask > 0
    if shape is not None and binary.shape != shape:
        height, width = shape
        binary = cv2.resize(binary.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return binary


def write_binary_mask(path: Path, mask: np.ndarray) -> None:
    """Write a binary mask as a pure black/white PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (mask.astype(bool).astype(np.uint8)) * 255
    if not cv2.imwrite(str(path), image):
        msg = f"Failed to write mask: {path}"
        raise RuntimeError(msg)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a mask to image shape if needed."""
    if mask.shape == shape:
        return mask.astype(bool)
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def compose_overlay(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    active_mask: str,
    alpha: float = 0.45,
) -> np.ndarray:
    """Compose a visual overlay without mutating mask arrays."""
    if image.ndim != 3:
        msg = "Overlay image must be BGR"
        raise ValueError(msg)
    overlay = image.copy()
    shape = image.shape[:2]
    for mask_name in ("allowed", "expected", "ignore", "watch_edge"):
        mask = _resize_mask(masks[mask_name], shape)
        if not np.any(mask):
            continue
        color = np.array(MASK_COLORS[mask_name], dtype=np.float32)
        weight = min(0.85, max(0.05, alpha + (0.18 if mask_name == active_mask else 0.0)))
        overlay[mask] = ((1.0 - weight) * overlay[mask].astype(np.float32) + weight * color).astype(np.uint8)

    for mask_name in MASK_TYPES:
        mask = _resize_mask(masks[mask_name], shape)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = 2 if mask_name == active_mask else 1
        cv2.drawContours(overlay, contours, -1, MASK_COLORS[mask_name], thickness, cv2.LINE_AA)
    return overlay


def summarize_eval_outputs(stress_output_dir: Path, defect_output_dir: Path) -> EvalSummary:
    """Summarize stress and fused defect prediction CSVs."""
    stress_rows = _read_csv(stress_output_dir / "geometry_predictions.csv")
    defect_rows = _read_csv(defect_output_dir / "geometry_predictions.csv")
    fused_rows = _read_csv(defect_output_dir / "fused_predictions.csv")

    defect_geometry_rows = [row for row in defect_rows if _is_defect_row(row)]
    defect_fused_rows = [row for row in fused_rows if _is_defect_row(row)]
    key_samples: dict[str, KeySampleStatus] = {}
    for sample_id in KEY_SAMPLE_IDS:
        hit = next(
            (
                row
                for row in fused_rows
                if sample_id in " ".join([row.get("source_path", ""), row.get("image_path", ""), row.get("frame_id", "")])
            ),
            None,
        )
        if hit is None:
            key_samples[sample_id] = KeySampleStatus()
        else:
            key_samples[sample_id] = KeySampleStatus(
                anomaly=_int_value(hit.get("anomaly_deploy_pred_label")),
                geometry=_int_value(hit.get("geometry_pred_label")),
                final=_int_value(hit.get("final_pred_label")),
            )

    return EvalSummary(
        stress_fp=sum(_int_value(row.get("geometry_pred_label")) for row in stress_rows),
        stress_total=len(stress_rows),
        geometry_positives=sum(_int_value(row.get("geometry_pred_label")) for row in defect_geometry_rows),
        geometry_total=len(defect_geometry_rows),
        fused_positives=sum(_int_value(row.get("final_pred_label")) for row in defect_fused_rows),
        fused_total=len(defect_fused_rows),
        key_samples=key_samples,
    )


def run_validation(config: EditorConfig) -> EvalSummary:
    """Run manual-template compilation and geometry/fused evaluation."""
    mask_dir = mask_dir_from_review_manifest(config.review_pack)
    threshold_path = config.stress_output_dir / "geometry_thresholds.csv"
    commands = [
        [
            sys.executable,
            str(REPO_ROOT / "capture_data" / "build_manual_geometry_templates.py"),
            "--mask-dir",
            str(mask_dir),
            "--output-dir",
            str(config.template_dir),
        ],
        [
            sys.executable,
            str(REPO_ROOT / "capture_data" / "evaluate_geometry_shape.py"),
            "--data-root",
            str(config.stress_root),
            "--template-dir",
            str(config.template_dir),
            "--output-dir",
            str(config.stress_output_dir),
            "--calibrate-thresholds",
            "--threshold-margin",
            str(config.threshold_margin),
            "--threshold-mode",
            config.threshold_mode,
            "--tolerance-px",
            str(config.tolerance_px),
            "--min-component-area",
            str(config.min_component_area),
            "--hole-dilation",
            str(config.hole_dilation),
        ],
        [
            sys.executable,
            str(REPO_ROOT / "capture_data" / "evaluate_geometry_shape.py"),
            "--data-root",
            str(config.defect_root),
            "--template-dir",
            str(config.template_dir),
            "--thresholds",
            str(threshold_path),
            "--anomaly-predictions",
            str(config.anomaly_predictions),
            "--output-dir",
            str(config.defect_output_dir),
            "--tolerance-px",
            str(config.tolerance_px),
            "--min-component-area",
            str(config.min_component_area),
            "--hole-dilation",
            str(config.hole_dilation),
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return summarize_eval_outputs(config.stress_output_dir, config.defect_output_dir)


def _read_reference(path: Path) -> np.ndarray:
    """Read a reference image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Could not read reference image: {path}"
        raise FileNotFoundError(msg)
    return image


def _mask_snapshot(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a deep copy of all masks."""
    return {name: mask.copy() for name, mask in masks.items()}


class GeometryMaskEditor:
    """OpenCV-based editor for manual geometry masks."""

    def __init__(self, config: EditorConfig) -> None:
        self.config = config
        self.items = load_review_manifest(config.review_pack)
        self.index = 0
        self.active_mask = "watch_edge"
        self.brush_radius = 15
        self.alpha = 0.42
        self.dirty_slots: set[str] = set()
        self.status = "Ready"
        self.validation_summary: EvalSummary | None = None
        self.references = {item.slot: _read_reference(item.reference_path) for item in self.items}
        self.masks_by_slot = {
            item.slot: {
                mask_name: read_binary_mask(mask_path, self.references[item.slot].shape[:2])
                for mask_name, mask_path in item.masks.items()
            }
            for item in self.items
        }
        self.undo_stacks: dict[str, list[dict[str, np.ndarray]]] = {item.slot: [] for item in self.items}
        self.redo_stacks: dict[str, list[dict[str, np.ndarray]]] = {item.slot: [] for item in self.items}
        self.painting = False
        self.paint_value = True
        self.display_scale = 1.0
        self.image_origin = (0, 48)
        self.display_size = (0, 0)

    @property
    def item(self) -> SlotReviewItem:
        """Return the current slot review item."""
        return self.items[self.index]

    @property
    def slot(self) -> str:
        """Return the current slot name."""
        return self.item.slot

    def run(self) -> None:
        """Run the interactive OpenCV editor."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if self.handle_key(key):
                break
        cv2.destroyWindow(WINDOW_NAME)

    def render(self) -> np.ndarray:
        """Render the editor canvas."""
        image = self.references[self.slot]
        masks = self.masks_by_slot[self.slot]
        overlay = compose_overlay(image, masks, self.active_mask, self.alpha)
        height, width = overlay.shape[:2]
        info_width = 470
        status_height = 48
        self.display_scale = min(
            max(0.1, (self.config.max_window_width - info_width) / max(1, width)),
            max(0.1, (self.config.max_window_height - status_height) / max(1, height)),
            1.0,
        )
        display_w = max(1, round(width * self.display_scale))
        display_h = max(1, round(height * self.display_scale))
        self.display_size = (display_w, display_h)
        if self.display_scale != 1.0:
            display = cv2.resize(overlay, (display_w, display_h), interpolation=cv2.INTER_AREA)
        else:
            display = overlay
        canvas_h = max(status_height + display_h, 520)
        canvas_w = display_w + info_width
        canvas = np.full((canvas_h, canvas_w, 3), (24, 24, 24), dtype=np.uint8)
        canvas[status_height : status_height + display_h, 0:display_w] = display
        self._draw_status(canvas, display_w)
        self._draw_info_panel(canvas, display_w, status_height)
        return canvas

    def _draw_status(self, canvas: np.ndarray, display_w: int) -> None:
        """Draw the top status row."""
        dirty = "*" if self.slot in self.dirty_slots else ""
        text = (
            f"{self.slot}{dirty} | mask={self.active_mask} | brush={self.brush_radius}px | "
            f"scale={self.display_scale:.2f} | {self.status}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 47), (38, 38, 38), -1)
        cv2.putText(canvas, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.line(canvas, (0, 47), (display_w, 47), MASK_COLORS[self.active_mask], 2, cv2.LINE_AA)

    def _draw_info_panel(self, canvas: np.ndarray, x: int, y: int) -> None:
        """Draw shortcuts and validation summary."""
        item = self.item
        lines = [
            "Shortcuts",
            "1-6: slot",
            "e/a/i/w: expected/allowed/ignore/watch",
            "left drag: paint white",
            "right drag: erase black",
            "[ / ]: brush size",
            "u / r: undo / redo",
            "s: save current slot masks",
            "v: validate saved masks",
            "q / Esc: quit",
            "",
            f"normal/stress/defect: {item.normal_count}/{item.stress_count}/{item.defect_count}",
            "",
            "Validation",
        ]
        if self.validation_summary is None:
            lines.append("No validation in this session")
        else:
            lines.extend(self.validation_summary.lines())
        y_cursor = y + 24
        for line in lines:
            color = (255, 255, 255) if line in {"Shortcuts", "Validation"} else (210, 210, 210)
            scale = 0.62 if line not in {"Shortcuts", "Validation"} else 0.72
            thickness = 2 if line in {"Shortcuts", "Validation"} else 1
            cv2.putText(canvas, line, (x + 16, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            y_cursor += 28

    def handle_key(self, key: int) -> bool:
        """Handle one OpenCV key code. Return True to quit."""
        if key in {27, ord("q")}:
            return True
        if ord("1") <= key <= ord("9"):
            requested = key - ord("1")
            if requested < len(self.items):
                self.index = requested
                self.status = f"Switched to {self.slot}"
            return False
        mask_keys = {ord("e"): "expected", ord("a"): "allowed", ord("i"): "ignore", ord("w"): "watch_edge"}
        if key in mask_keys:
            self.active_mask = mask_keys[key]
            self.status = f"Active mask: {self.active_mask}"
            return False
        if key == ord("["):
            self.brush_radius = max(1, self.brush_radius - 2)
            self.status = f"Brush: {self.brush_radius}px"
            return False
        if key == ord("]"):
            self.brush_radius = min(120, self.brush_radius + 2)
            self.status = f"Brush: {self.brush_radius}px"
            return False
        if key == ord("u"):
            self.undo()
            return False
        if key == ord("r"):
            self.redo()
            return False
        if key == ord("s"):
            self.save_current_slot()
            return False
        if key == ord("v"):
            self.validate_saved_masks()
            return False
        return False

    def _push_undo(self) -> None:
        """Record an undo snapshot for the current slot."""
        stack = self.undo_stacks[self.slot]
        stack.append(_mask_snapshot(self.masks_by_slot[self.slot]))
        del stack[:-40]
        self.redo_stacks[self.slot].clear()

    def undo(self) -> None:
        """Undo the latest edit for the current slot."""
        stack = self.undo_stacks[self.slot]
        if not stack:
            self.status = "Nothing to undo"
            return
        self.redo_stacks[self.slot].append(_mask_snapshot(self.masks_by_slot[self.slot]))
        self.masks_by_slot[self.slot] = stack.pop()
        self.dirty_slots.add(self.slot)
        self.status = "Undo"

    def redo(self) -> None:
        """Redo the latest undone edit for the current slot."""
        stack = self.redo_stacks[self.slot]
        if not stack:
            self.status = "Nothing to redo"
            return
        self.undo_stacks[self.slot].append(_mask_snapshot(self.masks_by_slot[self.slot]))
        self.masks_by_slot[self.slot] = stack.pop()
        self.dirty_slots.add(self.slot)
        self.status = "Redo"

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        """Handle OpenCV mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._push_undo()
            self.painting = True
            self.paint_value = True
            self._paint_at(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._push_undo()
            self.painting = True
            self.paint_value = False
            self._paint_at(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.painting:
            self._paint_at(x, y)
        elif event in {cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP}:
            self.painting = False

    def _paint_at(self, x: int, y: int) -> None:
        """Apply the active brush at display coordinates."""
        origin_x, origin_y = self.image_origin
        display_w, display_h = self.display_size
        local_x = x - origin_x
        local_y = y - origin_y
        if local_x < 0 or local_y < 0 or local_x >= display_w or local_y >= display_h:
            return
        image = self.references[self.slot]
        height, width = image.shape[:2]
        px = int(np.clip(round(local_x / self.display_scale), 0, width - 1))
        py = int(np.clip(round(local_y / self.display_scale), 0, height - 1))
        radius = max(1, round(self.brush_radius / max(0.1, self.display_scale)))
        mask = self.masks_by_slot[self.slot][self.active_mask].astype(np.uint8)
        cv2.circle(mask, (px, py), radius, 1 if self.paint_value else 0, -1, cv2.LINE_8)
        self.masks_by_slot[self.slot][self.active_mask] = mask > 0
        self.dirty_slots.add(self.slot)
        action = "paint" if self.paint_value else "erase"
        self.status = f"{action} {self.active_mask} at {px},{py}"

    def save_current_slot(self) -> None:
        """Save current slot masks after backing up originals."""
        item = self.item
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config.review_pack / "manual_masks_backup" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        for mask_name in MASK_TYPES:
            source = item.masks[mask_name]
            if source.is_file():
                shutil.copy2(source, backup_dir / source.name)
            write_binary_mask(source, self.masks_by_slot[self.slot][mask_name])
        self.dirty_slots.discard(self.slot)
        self.status = f"Saved {self.slot}; backup: {backup_dir}"

    def validate_saved_masks(self) -> None:
        """Run validation for saved masks."""
        if self.dirty_slots:
            self.status = "Save dirty masks with s before validation"
            return
        self.status = "Running validation..."
        try:
            self.validation_summary = run_validation(self.config)
        except subprocess.CalledProcessError as error:
            self.status = f"Validation failed: exit {error.returncode}"
            return
        except Exception as error:  # noqa: BLE001 - surface GUI status instead of crashing.
            self.status = f"Validation failed: {error}"
            return
        self.status = "Validation complete"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-pack",
        type=Path,
        default=Path("results/c789_100_hardened/left_top_geometry/manual_review_pack"),
        help="Manual geometry review pack directory.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("results/c789_100_hardened/left_top_geometry/manual_templates"),
        help="Output directory for compiled manual templates.",
    )
    parser.add_argument(
        "--stress-root",
        type=Path,
        default=Path("dataset/c789_stress_normal_group_split/locked/left/top"),
        help="Locked stress-normal crop root.",
    )
    parser.add_argument(
        "--defect-root",
        type=Path,
        default=Path("dataset/c789_100_left_top_parts/left/top/defect"),
        help="Defect crop root.",
    )
    parser.add_argument(
        "--anomaly-predictions",
        type=Path,
        default=Path("results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv"),
        help="Existing AnomalyDINO predictions CSV.",
    )
    parser.add_argument(
        "--stress-output-dir",
        type=Path,
        default=Path("results/c789_100_hardened/left_top_geometry/manual_stress_locked"),
        help="Output directory for stress geometry validation.",
    )
    parser.add_argument(
        "--defect-output-dir",
        type=Path,
        default=Path("results/c789_100_hardened/left_top_geometry/manual_defect_fused"),
        help="Output directory for defect geometry and fused validation.",
    )
    parser.add_argument("--threshold-margin", type=float, default=0.05, help="Stress calibration margin ratio.")
    parser.add_argument(
        "--threshold-mode",
        choices=("region", "slot"),
        default="region",
        help="Stress calibration threshold mode.",
    )
    parser.add_argument("--tolerance-px", type=int, default=3, help="Geometry tolerance in pixels.")
    parser.add_argument("--min-component-area", type=int, default=64, help="Minimum component area.")
    parser.add_argument("--hole-dilation", type=int, default=32, help="Preset hole mask dilation.")
    parser.add_argument("--max-window-width", type=int, default=1900, help="Maximum OpenCV canvas width.")
    parser.add_argument("--max-window-height", type=int, default=1050, help="Maximum OpenCV canvas height.")
    return parser


def main() -> None:
    """Run the interactive editor."""
    args = build_parser().parse_args()
    config = EditorConfig(
        review_pack=args.review_pack,
        template_dir=args.template_dir,
        stress_root=args.stress_root,
        defect_root=args.defect_root,
        anomaly_predictions=args.anomaly_predictions,
        stress_output_dir=args.stress_output_dir,
        defect_output_dir=args.defect_output_dir,
        threshold_margin=args.threshold_margin,
        threshold_mode=args.threshold_mode,
        tolerance_px=args.tolerance_px,
        min_component_area=args.min_component_area,
        hole_dilation=args.hole_dilation,
        max_window_width=args.max_window_width,
        max_window_height=args.max_window_height,
    )
    GeometryMaskEditor(config).run()


if __name__ == "__main__":
    main()
