#!/usr/bin/env python3
"""Collect BMW left/right parts with four cameras and two HDR rounds."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bmw_inspection.capture.config import load_capture_profile  # noqa: E402
from bmw_inspection.capture.hardware import HikvisionAdapter  # noqa: E402
from bmw_inspection.lab.eight_view_demo_capture import FourCameraHdrSession  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs/bmw/capture/bmw_4cam_eight_view_hdr_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("dataset/bmw_lab_raw"))
    parser.add_argument("--hand", choices=("left", "right"))
    parser.add_argument("--label", choices=("normal", "defect"))
    parser.add_argument("--defect-type")
    parser.add_argument("--part-id")
    parser.add_argument("--group-count", type=int, default=1)
    parser.add_argument("--no-prompt", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.hand is None or args.label is None or not args.part_id:
        raise ValueError("--hand, --label and --part-id are required")
    if args.group_count <= 0:
        raise ValueError("--group-count must be positive")
    defect_type = (args.defect_type or "").strip()
    if args.label == "defect" and not defect_type:
        raise ValueError("--defect-type is required when --label defect")
    if args.label == "normal" and defect_type:
        raise ValueError("--defect-type is only valid when --label defect")
    load_capture_profile(args.config)


def _wait(message: str, disabled: bool) -> None:
    if not disabled:
        input(message)


def _output_dir(root: Path, hand: str, view: str, label: str, defect_type: str, session_id: str) -> Path:
    middle = Path("normal") if label == "normal" else Path("defect") / defect_type
    path = root / hand / view / middle / session_id / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_image(path: Path, image: object) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _collect(args: argparse.Namespace) -> int:
    _validate(args)
    root = Path(args.root).expanduser().resolve()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    manifest_path = root / "manifests" / f"{session_id}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with FourCameraHdrSession(args.config) as camera:
        for group_index in range(1, args.group_count + 1):
            sample_id = f"{args.part_id}_group{group_index:03d}_000001"
            _wait(f"[{group_index}/{args.group_count}] 放置零件正面，回车拍摄：", args.no_prompt)
            front = camera.capture_round("front")
            _wait("翻转同一零件到反面，回车拍摄：", args.no_prompt)
            back = camera.capture_round("back")
            images = {**front, **back}
            for view, image in images.items():
                label_text = args.label if args.label == "normal" else f"defect_{args.defect_type}"
                name = f"{args.hand}_{view}_{label_text}_{sample_id}_fused.png"
                path = _output_dir(root, args.hand, view, args.label, args.defect_type or "", session_id) / name
                _write_image(path, image)
                source = camera.last_sources[view]
                prefix = path.stem.removesuffix("_fused")
                _write_image(path.with_name(prefix + "_short.png"), source.short_image)
                _write_image(path.with_name(prefix + "_long.png"), source.long_image)
                serial = next(slot.serial for slot in camera.profile.slots if view in {slot.front_view, slot.back_view})
                rows.append({
                    "session_id": session_id,
                    "sample_id": sample_id,
                    "hand": args.hand,
                    "view_id": view,
                    "label": args.label,
                    "defect_type": args.defect_type or "",
                    "camera_serial": serial,
                    "source_path": str(path.relative_to(root)),
                })
            print(f"已保存完整八视图：{sample_id}")
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"采集清单：{manifest_path}")
    return 0


def _list_devices() -> int:
    for device in HikvisionAdapter.load().list_devices():
        print(f"{device.index}\t{device.model}\t{device.serial}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _list_devices() if args.list_devices else _collect(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"BMW采集启动失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
