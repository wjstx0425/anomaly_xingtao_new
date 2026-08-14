#!/usr/bin/env python3
"""Interactively draw per-view polygons that EfficientAD should ignore."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.efficientad_ignore_mask import (  # noqa: E402
    load_ignore_mask_asset,
    polygons_to_ignore_mask,
    save_ignore_mask_asset,
)
from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402
from bmw_inspection.lab.eight_view_roi import fit_image_for_display  # noqa: E402


DEFAULT_CAPTURE_ID = "bmw_right_normal_group002_000001"
DEFAULT_INSPECTION_ROOT = REPO_ROOT / "results/bmw_eight_view_demo_v3_ng_evidence_v1"
DEFAULT_ROI_CONFIG = REPO_ROOT / "configs/bmw/rois/bmw_right_hdr_eight_view_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/bmw_efficientad_manual_ignore_masks/bmw_right_manual_ignore_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_images(source_paths: dict[str, Path]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for view in VIEW_ORDER:
        path = source_paths[view]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法读取 {view} ROI：{path}")
        images[view] = image
    return images


def load_training_release_roi_images(
    training_release: Path,
    *,
    representative_sample: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Path], str]:
    """Load one complete normal/train ROI crop set from an immutable training release."""
    root = Path(training_release).expanduser().resolve()
    manifest = root / "manifests/crop_manifest.csv"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError(f"training release crop manifest does not exist: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "session_id", "view_id", "source_class", "split", "crop_path", "crop_sha256"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("training release crop manifest lacks selector fields")
        rows = [
            row
            for row in reader
            if row["source_class"] == "normal"
            and row["split"] == "train"
            and (representative_sample is None or row["sample_id"] == representative_sample)
        ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["sample_id"], row["session_id"]), []).append(row)
    candidates = [
        (identity, group)
        for identity, group in sorted(grouped.items())
        if len(group) == len(VIEW_ORDER) and {row["view_id"] for row in group} == set(VIEW_ORDER)
    ]
    if len(candidates) != 1 and representative_sample is not None:
        raise ValueError(f"representative sample does not identify one complete normal/train sample: {representative_sample}")
    if not candidates:
        raise ValueError("training release has no complete normal/train ROI representative sample")
    (sample_id, _session_id), selected = candidates[0]
    by_view = {row["view_id"]: row for row in selected}
    source_paths: dict[str, Path] = {}
    for view in VIEW_ORDER:
        relative = Path(by_view[view]["crop_path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or root not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f"training release ROI crop is invalid: {view}")
        if _sha256(path) != by_view[view]["crop_sha256"]:
            raise ValueError(f"training release ROI crop SHA mismatch: {view}")
        source_paths[view] = path
    return _load_images(source_paths), source_paths, f"{root.name}:{sample_id}"


def load_selector_roi_images(
    *,
    inspection_root: Path,
    capture_id: str,
    training_release: Path | None,
    representative_sample: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Path], str]:
    """Load either existing Demo ROIs or one complete training-release representative."""
    if training_release is not None:
        return load_training_release_roi_images(training_release, representative_sample=representative_sample)
    roi_dir = Path(inspection_root).expanduser().resolve() / capture_id / "rois"
    source_paths = {view: roi_dir / f"{view}.png" for view in VIEW_ORDER}
    return _load_images(source_paths), source_paths, capture_id


def load_seed_polygons(
    index_path: Path,
    *,
    images: dict[str, np.ndarray],
    source_paths: dict[str, Path],
    public_roi_config: Path,
) -> dict[str, list[list[tuple[int, int]]]]:
    """Load polygons only when they exactly reconstruct a SHA-verified mask asset."""
    path = Path(index_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    shapes = {view: images[view].shape[:2] for view in VIEW_ORDER}
    asset = load_ignore_mask_asset(
        path,
        expected_views=VIEW_ORDER,
        expected_roi_config_sha256=_sha256(Path(public_roi_config).expanduser().resolve()),
        expected_shapes=shapes,
    )
    records = payload.get("views")
    if not isinstance(records, dict):
        raise ValueError("续画 mask 索引缺少八视角记录")
    loaded: dict[str, list[list[tuple[int, int]]]] = {}
    for view in VIEW_ORDER:
        record = records[view]
        source_path = Path(source_paths[view]).expanduser().resolve()
        if record.get("source_roi_sha256") != _sha256(source_path):
            raise ValueError(f"续画参考 ROI SHA 不匹配：{view}")
        raw_polygons = record.get("polygons")
        if not isinstance(raw_polygons, list):
            raise ValueError(f"续画多边形记录不正确：{view}")
        polygons = [[tuple(point) for point in polygon] for polygon in raw_polygons]
        reconstructed = polygons_to_ignore_mask(shapes[view], polygons)
        if not np.array_equal(reconstructed, asset.masks[view]):
            raise ValueError(f"续画多边形无法重建已校验 mask：{view}")
        loaded[view] = [[(int(x), int(y)) for x, y in polygon] for polygon in polygons]
    return loaded


def display_to_source_point(
    x: int,
    y: int,
    *,
    display_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> tuple[int, int]:
    """Convert a display click to a clamped source-image coordinate."""
    display_height, display_width = display_shape
    source_height, source_width = source_shape
    source_x = round(x * source_width / display_width)
    source_y = round(y * source_height / display_height)
    return min(max(source_x, 0), source_width - 1), min(max(source_y, 0), source_height - 1)


def _source_to_display_point(
    point: tuple[int, int],
    *,
    display_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> tuple[int, int]:
    display_height, display_width = display_shape
    source_height, source_width = source_shape
    return round(point[0] * display_width / source_width), round(point[1] * display_height / source_height)


def _render(
    displayed: np.ndarray,
    *,
    source_shape: tuple[int, int],
    polygons: list[list[tuple[int, int]]],
    current: list[tuple[int, int]],
    view: str,
    ordinal: int,
) -> np.ndarray:
    canvas = displayed.copy()
    display_shape = canvas.shape[:2]
    for polygon in polygons:
        points = np.asarray(
            [_source_to_display_point(point, display_shape=display_shape, source_shape=source_shape) for point in polygon],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        tint = canvas.copy()
        cv2.fillPoly(tint, [points], (0, 0, 255))
        canvas = cv2.addWeighted(canvas, 0.7, tint, 0.3, 0.0)
        cv2.polylines(canvas, [points], True, (0, 255, 255), 2, cv2.LINE_AA)
    if current:
        points = np.asarray(
            [_source_to_display_point(point, display_shape=display_shape, source_shape=source_shape) for point in current],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(canvas, [points], False, (0, 255, 0), 2, cv2.LINE_AA)
        for x, y in points[:, 0, :]:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        f"{ordinal}/8 {view} | closed={len(polygons)} current_points={len(current)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def select_polygons(
    view: str,
    image: np.ndarray,
    *,
    ordinal: int,
    max_display_width: int,
    max_display_height: int,
    initial_polygons: list[list[tuple[int, int]]] | None = None,
) -> list[list[tuple[int, int]]] | None:
    """Collect zero or more ignore polygons for one view; return ``None`` on cancel."""
    displayed = fit_image_for_display(
        image,
        max_display_width=max_display_width,
        max_display_height=max_display_height,
    )
    polygons = [] if initial_polygons is None else [[tuple(point) for point in polygon] for polygon in initial_polygons]
    current: list[tuple[int, int]] = []
    window = f"BMW EfficientAD ignore mask - {view}"

    def close_current() -> bool:
        if not current:
            return True
        if len(current) < 3:
            print(f"{view}: 当前多边形只有 {len(current)} 个点，至少需要 3 个点。")
            return False
        polygons.append(current.copy())
        current.clear()
        return True

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONUP:
            current.append(
                display_to_source_point(
                    x,
                    y,
                    display_shape=displayed.shape[:2],
                    source_shape=image.shape[:2],
                )
            )
        elif event == cv2.EVENT_RBUTTONUP:
            close_current()

    print(
        f"[{ordinal}/8] {view}: 已预载 {len(polygons)} 个多边形。左键逐点；右键或 Enter 闭合；可继续添加。"
        " U 撤销，R 清空后重画，N 清空并进入下一视角，S 保留/保存本视角，Esc 取消整次发布。"
    )
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, displayed.shape[1], displayed.shape[0])
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            cv2.imshow(
                window,
                _render(
                    displayed,
                    source_shape=image.shape[:2],
                    polygons=polygons,
                    current=current,
                    view=view,
                    ordinal=ordinal,
                ),
            )
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in (27, ord("q")):
                return None
            if key in (10, 13):
                close_current()
            elif key == ord("u"):
                if current:
                    current.pop()
                elif polygons:
                    polygons.pop()
            elif key == ord("r"):
                current.clear()
                polygons.clear()
            elif key == ord("n"):
                return []
            elif key == ord("s") and close_current():
                return polygons
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def build_parser() -> argparse.ArgumentParser:
    """Build the manual EfficientAD ignore-mask selector CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection-root", type=Path, default=DEFAULT_INSPECTION_ROOT)
    parser.add_argument("--capture-id", default=DEFAULT_CAPTURE_ID)
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    parser.add_argument("--training-release", type=Path)
    parser.add_argument("--representative-sample")
    parser.add_argument("--from-index", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--max-display-height", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Select all eight views, then atomically publish their ignore masks."""
    args = build_parser().parse_args(argv)
    try:
        images, source_paths, source_capture_id = load_selector_roi_images(
            inspection_root=args.inspection_root,
            capture_id=args.capture_id,
            training_release=args.training_release,
            representative_sample=args.representative_sample,
        )
        if args.from_index is None:
            seed_polygons = {view: [] for view in VIEW_ORDER}
            print("未提供旧 mask；以八个空多边形开始。")
        else:
            seed_polygons = load_seed_polygons(
                args.from_index,
                images=images,
                source_paths=source_paths,
                public_roi_config=args.roi_config,
            )
            print(f"已校验并载入旧 mask：{Path(args.from_index).expanduser().resolve()}")
        polygons: dict[str, list[list[tuple[int, int]]]] = {}
        for ordinal, view in enumerate(VIEW_ORDER, start=1):
            selected = select_polygons(
                view,
                images[view],
                ordinal=ordinal,
                max_display_width=args.max_display_width,
                max_display_height=args.max_display_height,
                initial_polygons=seed_polygons[view],
            )
            if selected is None:
                print("已取消；没有发布任何 mask。", file=sys.stderr)
                return 1
            polygons[view] = selected
        index_path = save_ignore_mask_asset(
            args.output,
            images=images,
            polygons_by_view=polygons,
            source_paths=source_paths,
            source_capture_id=source_capture_id,
            public_roi_config=args.roi_config,
        )
    except (FileExistsError, OSError, TypeError, ValueError, cv2.error) as error:
        print(f"BMW EfficientAD 手动 mask 选择失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "index": str(index_path.resolve()),
                "output": str(index_path.parent.resolve()),
                "polygon_count_by_view": {view: len(polygons[view]) for view in VIEW_ORDER},
                "semantics": {"0": "inspect", "255": "ignore"},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
