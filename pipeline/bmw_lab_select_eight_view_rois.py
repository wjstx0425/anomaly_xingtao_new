#!/usr/bin/env python3
"""从BMW prepared或raw八视图样本逐视图选择固定零件ROI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.lab.eight_view_dataset import SOURCE_CLASSES, VIEW_ORDER  # noqa: E402
from bmw_inspection.lab.eight_view_roi import (  # noqa: E402
    EightViewRoiConfig,
    fit_image_for_display,
    save_roi_config,
    select_raw_representative_images,
    select_representative_images,
    source_roi,
)


DEFAULT_PREPARED_ROOT = REPO_ROOT / "dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1"
DEFAULT_PREPARED_OUTPUT = REPO_ROOT / "configs/bmw/rois/bmw_hdr_eight_view_v1.json"
DEFAULT_RAW_OUTPUT = REPO_ROOT / "configs/bmw/rois/bmw_right_hdr_eight_view_v1.json"


def _window_name(view: str) -> str:
    """Return an ASCII-only title compatible with OpenCV's Qt backend."""
    return f"BMW 8-view ROI - {view}"


def _select_one(view: str, image_path: Path, *, max_width: int, max_height: int) -> tuple[int, int, int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取ROI参考图：{image_path}")
    displayed = fit_image_for_display(image, max_display_width=max_width, max_display_height=max_height)
    window = _window_name(view)
    try:
        print(f"正在选择 {view}：鼠标框选零件区域后按 Enter；按 C 取消。")
        selection = tuple(int(value) for value in cv2.selectROI(window, displayed, True, False))
        return source_roi(selection, displayed.shape[:2], image.shape[:2])
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def build_parser() -> argparse.ArgumentParser:
    """Build the eight-view ROI selector CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--prepared-root",
        type=Path,
        help="已有prepared release；不传数据源参数时使用旧默认release。",
    )
    source.add_argument("--raw-root", type=Path, help="原始采集根目录，用于生成可复用fixed-setup ROI。")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-part-id", help="可选：指定一个normal/train物理件作为ROI参考。")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--source-class", choices=SOURCE_CLASSES)
    parser.add_argument("--sample-id", help="raw模式下可选：指定一个完整八视图sample。")
    parser.add_argument("--session-id", help="raw模式下可选：限定采集会话，避免同名sample选到旧数据。")
    parser.add_argument("--profile-id", default="bmw-right-hdr-eight-view-v1")
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--max-display-height", type=int, default=720)
    parser.add_argument("--force", action="store_true", help="显式覆盖已有ROI配置。")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Select and save all eight ROIs."""
    args = build_parser().parse_args(argv)
    try:
        if args.raw_root is None:
            selection = select_representative_images(
                args.prepared_root or DEFAULT_PREPARED_ROOT,
                physical_part_id=args.physical_part_id,
            )
            output = args.output or DEFAULT_PREPARED_OUTPUT
            binding_mode = "prepared_manifest"
            capture_scope = None
        else:
            if args.physical_part_id is not None:
                raise ValueError("--physical-part-id cannot be used with --raw-root")
            selection = select_raw_representative_images(
                args.raw_root,
                profile_id=args.profile_id,
                capture_scope=args.hand,
                source_class=args.source_class,
                sample_id=args.sample_id,
                session_id=args.session_id,
            )
            output = args.output or DEFAULT_RAW_OUTPUT
            binding_mode = "fixed_setup"
            capture_scope = args.hand
        print(f"ROI参考件：{selection.physical_part_id} / {selection.sample_id}")
        rois = {
            view: _select_one(
                view,
                selection.images[view],
                max_width=args.max_display_width,
                max_height=args.max_display_height,
            )
            for view in VIEW_ORDER
        }
        config = EightViewRoiConfig(
            dataset_id=selection.dataset_id,
            source_manifest=selection.manifest_path,
            source_manifest_sha256=selection.manifest_sha256,
            representative_sample_id=selection.sample_id,
            image_width=selection.image_width,
            image_height=selection.image_height,
            part_rois=rois,
            binding_mode=binding_mode,
            capture_scope=capture_scope,
        )
        save_roi_config(output, config, force=args.force)
    except (FileExistsError, OSError, TypeError, ValueError, cv2.error) as error:
        print(f"BMW八视图ROI选择失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(Path(output).resolve()), "part_rois": rois}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
