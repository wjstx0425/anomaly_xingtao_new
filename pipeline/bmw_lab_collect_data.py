#!/usr/bin/env python3
"""快速采集BMW四相机、正反面八视图HDR训练图像。"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.capture.config import load_capture_profile  # noqa: E402
from capture_data.collect_multicamera_dataset import HikvisionAdapter, SDK_PATH  # noqa: E402
from zs32_inspection.cli.bootstrap_capture import main as bootstrap_main  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/bmw/capture/bmw_4cam_eight_view_hdr_v1.json"


def _ensure_mvs_sdk_import_path(sdk_path: Path | str = SDK_PATH) -> None:
    """Expose the installed Hikvision Python SDK to the topology-driven backend."""
    resolved = Path(sdk_path).expanduser().resolve()
    if not (resolved / "MvCameraControl_class.py").is_file():
        raise RuntimeError(f"海康MVS Python SDK不存在：{resolved}")
    text = str(resolved)
    if text not in sys.path:
        sys.path.insert(0, text)


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally small BMW data-collection command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="BMW四机HDR采集配置。")
    parser.add_argument("--list-devices", action="store_true", help="只列出当前连接的相机。")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "dataset/bmw_lab_raw")
    parser.add_argument("--label", choices=("normal", "defect"))
    parser.add_argument("--defect-type", help="缺陷采集时必填，例如 no_streak 或 scratch。")
    parser.add_argument("--part-id", help="本批物理零件前缀，例如 bmw_normal。")
    parser.add_argument("--group-count", type=int, default=1, help="本批采集的物理零件数量。")
    parser.add_argument(
        "--hand",
        choices=("left", "right"),
        default="left",
        help="兼容旧落盘目录的字段；BMW固定使用默认left即可。",
    )
    return parser


def build_bootstrap_argv(args: Namespace) -> list[str]:
    """Translate the BMW profile into the already-tested four-camera collector."""
    if args.label is None:
        raise ValueError("--label is required")
    if not args.part_id:
        raise ValueError("--part-id is required")
    if args.group_count <= 0:
        raise ValueError("--group-count must be positive")
    defect_type = args.defect_type.strip() if isinstance(args.defect_type, str) else ""
    if args.label == "defect" and not defect_type:
        raise ValueError("--defect-type is required when --label defect")
    if args.label == "normal" and defect_type:
        raise ValueError("--defect-type is only valid when --label defect")

    profile = load_capture_profile(args.config)
    hdr = profile.hdr
    argv = [
        "--topology",
        str(profile.bootstrap_topology_path),
        "--output-root",
        str(Path(args.root).expanduser().resolve()),
        "--legacy-layout",
        "--hand",
        args.hand,
        "--label",
        args.label,
        "--part-id",
        args.part_id,
        "--group-count",
        str(args.group_count),
        "--images-per-group",
        "1",
        "--manual-load",
        "--hdr",
        "--short-exposure",
        str(hdr.short_exposure_us),
        "--long-exposure",
        str(hdr.long_exposure_us),
        "--gain",
        str(hdr.gain),
        "--capture-interval",
        str(hdr.trigger_interval_s),
        "--hdr-settle-frames",
        str(hdr.settle_frames),
        "--timeout-ms",
        str(hdr.timeout_ms),
        "--no-align-hdr" if not hdr.align else "--align-hdr",
        "--short-dark-threshold",
        str(hdr.short_dark_threshold),
        "--long-clip-threshold",
        str(hdr.long_clip_threshold),
        "--blend-width",
        str(hdr.blend_width),
        "--blur-size",
        str(hdr.blur_size),
        "--hdr-max-retries",
        str(hdr.max_retries),
        "--hdr-max-clip-pct",
        str(hdr.max_clip_pct),
    ]
    if defect_type:
        argv.extend(("--defect-type", defect_type))
    return argv


def _list_devices() -> int:
    _ensure_mvs_sdk_import_path()
    adapter = HikvisionAdapter.load()
    for device in adapter.list_devices():
        print(f"{device.index}\t{device.model}\t{device.serial}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Collect one or more complete BMW eight-view HDR samples."""
    args = build_parser().parse_args(argv)
    try:
        if args.list_devices:
            return _list_devices()
        _ensure_mvs_sdk_import_path()
        print("BMW四相机八视图HDR采集：正面4张，翻面后反面4张。")
        return bootstrap_main(build_bootstrap_argv(args))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW采集启动失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
