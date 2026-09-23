"""Scan camera exposures and compare saved brackets with two exposure-fusion methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from bmw_inspection.capture.config import CameraSlot, load_capture_profile
from bmw_inspection.capture.exposure_fusion import fuse_exposures
from bmw_inspection.capture.hardware import (
    GroupedTriggerPacer,
    HikvisionAdapter,
    capture_single_round,
    open_cameras,
)
from bmw_inspection.cli.collect import DEFAULT_CONFIG


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone live-capture and offline-replay interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--output", type=Path, help="New directory; existing directories are never overwritten")
    parser.add_argument("--exposures-us", nargs="+", type=float, help="2–12 increasing values; default: profile pair")
    parser.add_argument("--pairs", nargs="+", help="Requested exposure pairs, e.g. 750:3000 1500:6000; default: all pairs")
    parser.add_argument("--serial", nargs="+", help="Camera serials, including new devices; default: configured four")
    parser.add_argument("--round", choices=("front", "back"), default="front")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--gain", type=float, help="Fixed gain; default: capture profile")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--align", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--replay", type=Path, help="Replay this program's completed session.json without cameras")
    return parser


def _exposures(values: list[float]) -> list[float]:
    if not 2 <= len(values) <= 12 or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("exposures must contain 2–12 finite positive microsecond values")
    if values != sorted(set(values)):
        raise ValueError("exposures must be unique and strictly increasing")
    return values


def _pairs(texts: list[str] | None, exposures: list[float]) -> list[tuple[float, float]]:
    if texts is None:
        return list(combinations(exposures, 2))
    result = []
    for text in texts:
        parts = text.split(":")
        if len(parts) != 2:
            raise ValueError("each pair must be SHORT:LONG in microseconds")
        short, long = map(float, parts)
        if short not in exposures or long not in exposures or short >= long:
            raise ValueError("pair values must be in --exposures-us and SHORT < LONG")
        if (short, long) in result:
            raise ValueError("duplicate exposure pair")
        result.append((short, long))
    return result


def _json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _save(path: Path, image: np.ndarray) -> str:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("images must be nonempty uint8 BGR")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"cannot encode {path}")
    data = encoded.tobytes()
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def _read(root: Path, record: dict) -> np.ndarray:
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("source image must stay inside its session directory")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"source checksum mismatch: {path}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected uint8 BGR source: {path}")
    return image


def _capture(args: argparse.Namespace, output: Path, session: dict) -> None:
    profile = load_capture_profile(args.config)
    slots = list(profile.slots)
    if args.serial is not None:
        if len(set(args.serial)) != len(args.serial) or any(not serial.strip() for serial in args.serial):
            raise ValueError("--serial must list distinct nonempty serials")
        configured = {slot.serial: slot for slot in profile.slots}
        slots = [
            configured.get(serial) or CameraSlot(
                slot_id=f"test_{index + 1:02d}", serial=serial,
                front_view=f"camera_{index + 1:02d}_front", back_view=f"camera_{index + 1:02d}_back",
            )
            for index, serial in enumerate(args.serial)
        ]
    adapter = HikvisionAdapter.load()
    available = {device.serial: device for device in adapter.list_devices()}
    missing = [slot.serial for slot in slots if slot.serial not in available]
    if missing:
        raise RuntimeError(f"missing cameras: {', '.join(missing)}")
    hdr = session["metadata"]["hdr"]
    pacer = GroupedTriggerPacer(hdr["trigger_interval_s"])
    devices = [available[slot.serial] for slot in slots]
    session["metadata"]["devices"] = [asdict(device) for device in devices]
    session["metadata"]["selected_slots"] = [asdict(slot) for slot in slots]
    _json(output / "session.json", session)
    with open_cameras(devices, adapter, hdr["gain"]) as handles:
        for repeat in range(1, args.repeat + 1):
            if not args.no_prompt:
                input(f"第 {repeat}/{args.repeat} 轮：固定零件与光源，保持 {args.round} 面静止，回车开始曝光扫描：")
            for index, exposure in enumerate(session["metadata"]["exposures_us"]):
                # Same grouped software trigger, settling and pacing as the existing collector.
                frames = capture_single_round(
                    handles, adapter, exposure, hdr["timeout_ms"], pacer, settle_frames=hdr["settle_frames"],
                )
                received_at = datetime.now(timezone.utc).isoformat()
                for slot, handle, frame in zip(slots, handles, frames, strict=True):
                    image = frame.final_image
                    roi = args.roi
                    if roi and not (0 <= roi[0] < roi[2] <= image.shape[1] and 0 <= roi[1] < roi[3] <= image.shape[0]):
                        raise ValueError("ROI is outside the captured image")
                    view = slot.front_view if args.round == "front" else slot.back_view
                    relative = f"raw/{view}/r{repeat:03d}_e{index:02d}.png"
                    checksum = _save(output / relative, image)
                    actual = adapter.get_exposure(handle)
                    if not math.isfinite(actual) or actual <= 0:
                        raise RuntimeError("camera returned invalid exposure readback")
                    session["records"].append({
                        "path": relative, "sha256": checksum, "camera_serial": slot.serial,
                        "view": view, "repeat": repeat, "kind": "single", "exposures_us": [exposure],
                        "exposure_readback_us": actual, "received_at": received_at,
                    })
                    _json(output / "session.json", session)
                print(f"已保存第 {repeat} 轮曝光 {exposure:g} μs，共 {len(frames)} 台相机", flush=True)


def _fuse(output: Path, session: dict, pairs: list[tuple[float, float]]) -> list[dict]:
    records = list(session["records"])
    hdr = session["metadata"]["hdr"]
    groups: dict[tuple[str, str, int], dict[float, dict]] = {}
    for record in records:
        if record["kind"] != "single" or len(record["exposures_us"]) != 1:
            raise ValueError("session must contain only single-exposure source records")
        key = record["camera_serial"], record["view"], record["repeat"]
        exposure = record["exposures_us"][0]
        group = groups.setdefault(key, {})
        if exposure in group:
            raise ValueError("duplicate source exposure within one camera/repeat")
        group[exposure] = record
    if not groups:
        raise ValueError("session has no source images")
    for group_index, group in enumerate(groups.values()):
        if set(group) != set(session["metadata"]["exposures_us"]):
            raise ValueError("incomplete exposure bracket")
        for pair_index, (short, long) in enumerate(pairs):
            sources = [group[short], group[long]]
            images = [_read(output, source) for source in sources]
            if images[0].shape != images[1].shape:
                raise ValueError("source images in one bracket must have the same shape")
            if hdr["align"]:
                # Align once so both methods receive exactly the same pixels.
                cv2.createAlignMTB().process(images, images)
            selective = fuse_exposures(
                images, align=False, short_dark_threshold=hdr["short_dark_threshold"],
                long_clip_threshold=hdr["long_clip_threshold"], blend_width=hdr["blend_width"],
                blur_size=hdr["blur_size"],
            )
            mertens_float = cv2.createMergeMertens(1.0, 1.0, 1.0).process(images)
            if not np.isfinite(mertens_float).all():
                raise RuntimeError("Mertens produced non-finite pixels")
            mertens = np.rint(np.clip(mertens_float, 0, 1) * 255).astype(np.uint8)
            for kind, image in (("selective", selective), ("mertens", mertens)):
                relative = f"fusion/g{group_index:03d}_p{pair_index:03d}_{kind}.png"
                checksum = _save(output / relative, image)
                records.append({
                    "path": relative, "sha256": checksum, "camera_serial": sources[0]["camera_serial"],
                    "view": sources[0]["view"], "repeat": sources[0]["repeat"], "kind": kind,
                    "exposures_us": [short, long], "source_paths": [source["path"] for source in sources],
                })
            print(f"已融合 {sources[0]['view']} 第 {sources[0]['repeat']} 轮：{short:g}/{long:g} μs", flush=True)
    return records


def run(args: argparse.Namespace) -> Path:
    """Capture or replay one experiment into a fresh evidence directory."""
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.gain is not None:
        if not math.isfinite(args.gain):
            raise ValueError("--gain must be finite")
    if args.roi and not (0 <= args.roi[0] < args.roi[2] and 0 <= args.roi[1] < args.roi[3]):
        raise ValueError("ROI requires 0 <= X1 < X2 and 0 <= Y1 < Y2")
    source_root = None
    if args.replay:
        if args.exposures_us is not None or args.serial is not None or args.gain is not None or args.repeat != 1:
            raise ValueError("replay does not accept capture overrides: exposures, serial, gain, repeat")
        source_root = args.replay.resolve().parent
        session = json.loads(args.replay.read_text(encoding="utf-8"))
        if session.get("schema_version") != 1 or session.get("status") != "complete":
            raise ValueError("replay requires a completed schema_version=1 session.json")
        session["metadata"]["replayed_from"] = str(args.replay.resolve())
        # Keep the original capture settings; only explicitly selected alignment/ROI may change.
        if args.roi is not None:
            session["metadata"]["roi"] = args.roi
        _exposures(session["metadata"]["exposures_us"])
    else:
        profile = load_capture_profile(args.config)
        hdr = asdict(profile.hdr)
        if args.gain is not None:
            hdr["gain"] = args.gain
        exposures = _exposures(args.exposures_us or [hdr["short_exposure_us"], hdr["long_exposure_us"]])
        if exposures[-1] / 1000 + 100 >= hdr["timeout_ms"]:
            raise ValueError("timeout_ms must exceed the longest exposure by more than 100 ms")
        session = {
            "schema_version": 1, "status": "capturing", "records": [],
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(), "profile_path": str(profile.path),
                "profile_id": profile.profile_id, "hdr": hdr, "exposures_us": exposures,
                "roi": args.roi, "round": args.round, "repeat_count": args.repeat,
                "source_kind": "camera_exposure_bracket", "exposure_unit": "microseconds",
            },
        }
    if args.align is not None:
        session["metadata"]["hdr"]["align"] = args.align
    pairs = _pairs(args.pairs, session["metadata"]["exposures_us"])
    session["metadata"]["pairs_us"] = pairs
    session["metadata"]["opencv_version"] = cv2.__version__
    session["metadata"]["mertens"] = {"contrast": 1.0, "saturation": 1.0, "exposure": 1.0, "gamma": 1.0}
    output = (args.output or Path("results/bmw_exposure") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    session["status"] = "capturing" if source_root is None else "replaying"
    _json(output / "session.json", session)
    try:
        if source_root is None:
            _capture(args, output, session)
        else:
            for index, record in enumerate(session["records"]):
                image = _read(source_root, record)
                record["original_path"] = record["path"]
                record["path"] = f"raw/replay_{index:05d}.png"
                record["sha256"] = _save(output / record["path"], image)
        session["status"] = "fusing"
        _json(output / "session.json", session)
        records = _fuse(output, session, pairs)
        from bmw_inspection.capture.exposure_report import write_report

        write_report(output, records, session["metadata"])
        session["status"] = "complete"
        _json(output / "session.json", session)
    except BaseException as error:
        session["status"] = "failed"
        session["error"] = f"{type(error).__name__}: {error}"
        _json(output / "session.json", session)
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    """Run the exposure test without loading inspection models."""
    args = build_parser().parse_args(argv)
    try:
        if args.list_devices:
            for device in HikvisionAdapter.load().list_devices():
                print(f"{device.index}\t{device.model}\t{device.serial}")
            return 0
        output = run(args)
        print(f"测试完成，对比页面：{output / 'index.html'}")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, cv2.error, EOFError) as error:
        print(f"曝光测试失败：{error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("曝光测试已中断；已保存图像和失败状态保留在输出目录。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
