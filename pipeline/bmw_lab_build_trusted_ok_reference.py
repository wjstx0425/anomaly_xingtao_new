#!/usr/bin/env python3
"""从八个左手0820正常件直接生成实验室可信OK参考库。"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER  # noqa: E402


DEFAULT_MANIFEST = Path(
    "/home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared/"
    "bmw_left_0820_v1/manifests/dataset_manifest.csv"
)
DEFAULT_ROI_CONFIG = REPO_ROOT / "configs/bmw/rois/bmw_left_0820_v1.json"
DEFAULT_OUTPUT = Path(
    "/home/yunjing/anomaly_xingtao_new/dataset/bmw_trusted_ok_reference/"
    "bmw_left_0820_train_normal_v1"
)
DEFAULT_PART_IDS = tuple(
    f"bmw_left_normal_group{group}"
    for group in ("002", "004", "005", "006", "008", "011", "012", "013")
)
_REQUIRED_MANIFEST_FIELDS = {
    "sample_id",
    "physical_part_id",
    "session_id",
    "view_id",
    "source_path",
    "source_class",
    "business_label",
    "split",
}


def _load_rois(path: Path) -> tuple[int, int, dict[str, tuple[int, int, int, int]]]:
    """Read only the current image dimensions and eight ROI rectangles."""
    roi_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(roi_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取ROI配置：{roi_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("ROI配置必须是JSON对象")
    width = payload.get("image_width")
    height = payload.get("image_height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("ROI配置缺少有效的图像尺寸")
    raw_rois = payload.get("part_rois")
    if not isinstance(raw_rois, dict):
        raise ValueError("ROI配置缺少part_rois")
    rois: dict[str, tuple[int, int, int, int]] = {}
    for view in VIEW_ORDER:
        raw = raw_rois.get(view)
        if (
            not isinstance(raw, list)
            or len(raw) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        ):
            raise ValueError(f"{view} ROI必须包含四个整数")
        x1, y1, x2, y2 = raw
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{view} ROI越界")
        rois[view] = (x1, y1, x2, y2)
    return width, height, rois


def _selected_rows(
    manifest_path: Path,
    physical_part_ids: Sequence[str],
) -> dict[str, dict[str, dict[str, str]]]:
    """Select the requested complete train/normal/OK parts in canonical view order."""
    requested = tuple(physical_part_ids)
    if not requested or len(set(requested)) != len(requested) or any(not value for value in requested):
        raise ValueError("physical_part_ids必须是非空且不重复的零件列表")
    path = Path(manifest_path).expanduser().resolve()
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing_fields = _REQUIRED_MANIFEST_FIELDS - set(reader.fieldnames or ())
            if missing_fields:
                raise ValueError(f"manifest缺少字段：{','.join(sorted(missing_fields))}")
            rows = [row for row in reader if row["physical_part_id"] in requested]
    except OSError as error:
        raise ValueError(f"无法读取manifest：{path}: {error}") from error

    selected: dict[str, dict[str, dict[str, str]]] = {}
    for part_id in requested:
        part_rows = [row for row in rows if row["physical_part_id"] == part_id]
        views = [row["view_id"] for row in part_rows]
        if len(part_rows) != len(VIEW_ORDER) or set(views) != set(VIEW_ORDER):
            raise ValueError(f"{part_id}八视图不完整")
        if len(views) != len(set(views)):
            raise ValueError(f"{part_id}包含重复视角")
        if any(
            row["source_class"] != "normal"
            or row["business_label"] != "OK"
            or row["split"] != "train"
            for row in part_rows
        ):
            raise ValueError(f"{part_id}不是train normal OK零件")
        if len({row["sample_id"] for row in part_rows}) != 1:
            raise ValueError(f"{part_id}包含多个sample_id")
        if len({row["session_id"] for row in part_rows}) != 1:
            raise ValueError(f"{part_id}包含多个session_id")
        selected[part_id] = {row["view_id"]: row for row in part_rows}
    return selected


def build_trusted_ok_reference(
    *,
    manifest_path: Path,
    roi_config_path: Path,
    output_dir: Path,
    physical_part_ids: Sequence[str] = DEFAULT_PART_IDS,
) -> Path:
    """Write ROI crops and a minimal reference index without any SHA binding."""
    width, height, rois = _load_rois(roi_config_path)
    selected = _selected_rows(manifest_path, physical_part_ids)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    references: list[dict[str, str]] = []

    for part_id in physical_part_ids:
        for view in VIEW_ORDER:
            row = selected[part_id][view]
            source = Path(row["source_path"]).expanduser()
            if not source.is_absolute():
                source = Path(manifest_path).expanduser().resolve().parent / source
            source = source.resolve()
            if not source.is_file():
                raise ValueError(f"可信OK原图不存在：{source}")
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                raise ValueError(f"可信OK原图无法读取：{source}")
            if image.dtype != np.uint8:
                raise ValueError(f"可信OK原图必须是uint8：{source}")
            if image.shape[:2] != (height, width):
                raise ValueError(f"可信OK原图尺寸与ROI配置不一致：{source}")
            x1, y1, x2, y2 = rois[view]
            crop = image[y1:y2, x1:x2]
            roi_path = destination / "roi" / view / f"{row['sample_id']}.png"
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(roi_path), crop):
                raise OSError(f"无法写入可信OK ROI：{roi_path}")
            references.append(
                {
                    "physical_part_id": part_id,
                    "sample_id": row["sample_id"],
                    "view_id": view,
                    "full_image_path": str(source),
                    "roi_image_path": roi_path.relative_to(destination).as_posix(),
                }
            )

    index_path = destination / "reference_index.json"
    index_path.write_text(
        json.dumps({"references": references}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index_path


def build_parser() -> argparse.ArgumentParser:
    """Build the direct laboratory trusted-OK generator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--part-id",
        dest="part_ids",
        action="append",
        help="覆盖默认候选时重复传入；每个零件必须是完整train normal OK八视图。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate one directly editable trusted-OK index and its ROI crops."""
    args = build_parser().parse_args(argv)
    try:
        index_path = build_trusted_ok_reference(
            manifest_path=args.manifest,
            roi_config_path=args.roi_config,
            output_dir=args.output,
            physical_part_ids=tuple(args.part_ids or DEFAULT_PART_IDS),
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"BMW可信OK图库生成失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "reference_index": str(index_path),
                "part_count": len(args.part_ids or DEFAULT_PART_IDS),
                "reference_count": len(args.part_ids or DEFAULT_PART_IDS) * len(VIEW_ORDER),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
