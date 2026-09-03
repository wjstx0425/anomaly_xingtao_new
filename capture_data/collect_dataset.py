import sys
import csv
import ctypes
import argparse
import termios
import tty
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from exposure_fusion import fuse_exposures

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *


'''
conda activate capture-data


python capture_data/collect_dataset.py   --hand no_hand   --position top   --label normal   --part-id part001   --group-count 60   --images-per-group 5   --manual-load   --hdr   --save-hdr-sources   --align-hdr   --short-exposure 4000   --long-exposure 35000   --short-dark-threshold 80   --long-clip-threshold 245   --blend-width 50   --blur-size 101   --hdr-settle-frames 8   --gain 0   --root ./dataset/fx11_2

python capture_data/collect_dataset.py   --hand no_hand   --position top   --label defect   --part-id part001   --group-count 20   --images-per-group 1   --manual-load   --hdr   --save-hdr-sources   --align-hdr   --short-exposure 4000   --long-exposure 35000   --short-dark-threshold 80   --long-clip-threshold 245   --blend-width 50   --blur-size 101   --hdr-settle-frames 8   --gain 0   --root ./dataset/fx11_2

'''


POSITION_CONFIG = {
    "bottom": {"exposure": 8000, "gain": 16, "fps": 10},
    "side": {"exposure": 8000, "gain": 9, "fps": 10},
    "top":{"exposure": 4000, "gain": 10, "fps":10},
}


def check_ret(ret, msg):
    if ret != 0:
        raise RuntimeError(f"{msg} failed, ret=0x{ret:x}")


def set_enum(cam, name, value):
    check_ret(cam.MV_CC_SetEnumValue(name, value), name)


def set_float(cam, name, value):
    check_ret(cam.MV_CC_SetFloatValue(name, float(value)), name)


def set_bool(cam, name, value):
    check_ret(cam.MV_CC_SetBoolValue(name, bool(value)), name)


def convert_frame_to_bgr(data_buf, frame_info):
    w, h = frame_info.nWidth, frame_info.nHeight
    data = np.frombuffer(data_buf, dtype=np.uint8, count=frame_info.nFrameLen)

    if frame_info.enPixelType == PixelType_Gvsp_Mono8:
        return cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)

    if frame_info.enPixelType == PixelType_Gvsp_BayerRG8:
        return cv2.cvtColor(data.reshape(h, w), cv2.COLOR_BAYER_RG2BGR)

    if frame_info.enPixelType == PixelType_Gvsp_BGR8_Packed:
        return data.reshape(h, w, 3)

    raise RuntimeError("Unsupported pixel type")


def open_camera(index):
    dev_list = MV_CC_DEVICE_INFO_LIST()
    check_ret(MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dev_list), "Enum")

    if dev_list.nDeviceNum == 0:
        raise RuntimeError("No camera")
    if index >= dev_list.nDeviceNum:
        raise RuntimeError(f"Device index {index} out of range")

    cam = MvCamera()
    info = ctypes.cast(dev_list.pDeviceInfo[index], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents

    check_ret(cam.MV_CC_CreateHandle(info), "CreateHandle")
    check_ret(cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "OpenDevice")

    return cam


def setup_camera(cam, exposure, gain, fps):
    set_enum(cam, "ExposureAuto", 0)
    set_float(cam, "ExposureTime", exposure)

    set_enum(cam, "GainAuto", 0)
    try:
        set_float(cam, "Gain", gain)
    except:
        pass

    try:
        set_bool(cam, "AcquisitionFrameRateEnable", True)
        set_float(cam, "AcquisitionFrameRate", fps)
    except:
        pass

    set_enum(cam, "TriggerMode", 0)


def make_dir(root, hand, pos, label, defect, part):
    t = datetime.now().strftime("%Y%m%d_%H%M%S")

    if label == "normal":
        path = Path(root)/hand/pos/"normal"/f"{part}_{t}"
    else:
        path = Path(root)/hand/pos/"defect"/defect/f"{part}_{t}"

    img_dir = path/"images"
    img_dir.mkdir(parents=True, exist_ok=True)

    return path, img_dir


def write_csv_header(p):
    if p.exists(): return
    with open(p, "w", newline="") as f:
        csv.writer(f).writerow([
            "file",
            "time",
            "hand",
            "pos",
            "label",
            "defect",
            "part",
            "group",
            "capture_mode",
            "exposure",
            "gain",
            "fps",
            "short_exposure",
            "long_exposure",
            "short_dark_threshold",
            "long_clip_threshold",
            "blend_width",
            "blur_size",
            "source_short",
            "source_long",
            "hdr_attempt",
            "fused_clip_pct",
        ])


def exposure_name(exposure):
    if float(exposure).is_integer():
        return str(int(exposure))
    return str(exposure).replace(".", "p")


def grab_frame(cam, buf, frame, timeout_ms):
    ret = cam.MV_CC_GetOneFrameTimeout(buf, len(buf), frame, timeout_ms)
    if ret != 0:
        print("frame fail", hex(ret))
        return None
    return convert_frame_to_bgr(buf, frame)


def discard_frames(cam, buf, frame, count, timeout_ms):
    for _ in range(count):
        grab_frame(cam, buf, frame, timeout_ms)


def capture_at_exposure(cam, exposure, buf, frame, args):
    set_float(cam, "ExposureTime", exposure)
    discard_frames(cam, buf, frame, args.hdr_settle_frames, args.timeout_ms)
    image = grab_frame(cam, buf, frame, args.timeout_ms)
    if image is None:
        raise RuntimeError(f"Failed to capture image at exposure={exposure}")
    return image


def image_clip_pct(image, threshold=250):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray >= threshold) * 100.0)


def write_meta_row(
    args,
    csv_path,
    path,
    source_short="",
    source_long="",
    hdr_attempt="",
    fused_clip_pct="",
):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            str(path),
            datetime.now().isoformat(),
            args.hand,
            args.position,
            args.label,
            args.defect_type,
            args.part_id,
            args.group_id,
            "hdr_fused" if args.hdr else "single",
            "" if args.hdr else args.exposure,
            args.gain,
            args.fps,
            args.short_exposure if args.hdr else "",
            args.long_exposure if args.hdr else "",
            args.short_dark_threshold if args.hdr else "",
            args.long_clip_threshold if args.hdr else "",
            args.blend_width if args.hdr else "",
            args.blur_size if args.hdr else "",
            str(source_short),
            str(source_long),
            hdr_attempt,
            fused_clip_pct,
        ])


def capture_single(cam, args, img_dir, csv_path, buf, frame):
    discard_frames(cam, buf, frame, 5, 1000)

    for i in range(args.images_per_group):
        img = grab_frame(cam, buf, frame, args.timeout_ms)
        if img is None:
            continue

        name = f"{args.hand}_{args.position}_{args.label}_{args.part_id}_g{args.group_id:03d}_{i:06d}.png"
        path = img_dir/name

        cv2.imwrite(str(path), img)
        write_meta_row(args, csv_path, path)
        print("saved", path)


def capture_hdr(cam, args, img_dir, csv_path, source_dir, buf, frame):
    short_label = exposure_name(args.short_exposure)
    long_label = exposure_name(args.long_exposure)

    for i in range(args.images_per_group):
        for attempt in range(args.hdr_max_retries + 1):
            short_img = capture_at_exposure(cam, args.short_exposure, buf, frame, args)
            long_img = capture_at_exposure(cam, args.long_exposure, buf, frame, args)
            fused = fuse_exposures(
                [short_img, long_img],
                method="selective",
                align=args.align_hdr,
                short_dark_threshold=args.short_dark_threshold,
                long_clip_threshold=args.long_clip_threshold,
                blend_width=args.blend_width,
                blur_size=args.blur_size,
            )
            fused_clip_pct = image_clip_pct(fused)
            if fused_clip_pct <= args.hdr_max_clip_pct:
                break
            if attempt < args.hdr_max_retries:
                print(
                    "HDR fused image looks overexposed "
                    f"(clip={fused_clip_pct:.2f}%), retrying..."
                )
            else:
                print(
                    "Warning: HDR fused image still looks overexposed "
                    f"(clip={fused_clip_pct:.2f}%)"
                )

        base = f"{args.hand}_{args.position}_{args.label}_{args.part_id}_g{args.group_id:03d}_{i:06d}"
        path = img_dir/f"{base}.png"
        source_short = ""
        source_long = ""

        if args.save_hdr_sources:
            source_short = source_dir/f"{base}_exp{short_label}.png"
            source_long = source_dir/f"{base}_exp{long_label}.png"
            cv2.imwrite(str(source_short), short_img)
            cv2.imwrite(str(source_long), long_img)

        cv2.imwrite(str(path), fused)
        write_meta_row(
            args,
            csv_path,
            path,
            source_short,
            source_long,
            attempt + 1,
            f"{fused_clip_pct:.4f}",
        )
        print("saved", path)


def capture(cam, args, img_dir, csv_path, source_dir=None):
    frame = MV_FRAME_OUT_INFO_EX()
    buf = (ctypes.c_ubyte * (50*1024*1024))()

    if args.hdr:
        capture_hdr(cam, args, img_dir, csv_path, source_dir, buf, frame)
    else:
        capture_single(cam, args, img_dir, csv_path, buf, frame)


def wait_for_manual_capture():
    prompt = "放好零件后按 Enter 或 s..."

    if not sys.stdin.isatty():
        input(prompt)
        return

    print(prompt, end="", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        while True:
            key = sys.stdin.read(1)
            if key in {"\n", "\r"} or key.lower() == "s":
                print()
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--device", type=int, default=0)
    p.add_argument("--hand", required=True, choices=["left","right", "no_hand"])
    p.add_argument("--position", required=True, choices=["bottom", "side", "top"])
    p.add_argument("--label", required=True, choices=["normal","defect"])
    p.add_argument("--defect-type", default="")
    p.add_argument("--part-id", default="part001")

    p.add_argument("--group-count", type=int, default=1)
    p.add_argument("--images-per-group", type=int, default=5)
    p.add_argument("--manual-load", action="store_true")
    p.add_argument("--exposure", type=float, default=None)
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--fps", type=float, default=None)

    p.add_argument("--hdr", action="store_true")
    p.add_argument("--short-exposure", type=float, default=7000.0)
    p.add_argument("--long-exposure", type=float, default=40000.0)
    p.add_argument("--short-dark-threshold", type=float, default=70.0)
    p.add_argument("--long-clip-threshold", type=float, default=245.0)
    p.add_argument("--blend-width", type=float, default=18.0)
    p.add_argument("--blur-size", type=int, default=31)
    p.add_argument("--hdr-settle-frames", type=int, default=5)
    p.add_argument("--align-hdr", action="store_true")
    p.add_argument("--save-hdr-sources", action="store_true")
    p.add_argument("--hdr-max-retries", type=int, default=2)
    p.add_argument("--hdr-max-clip-pct", type=float, default=12.0)
    p.add_argument("--timeout-ms", type=int, default=3000)

    p.add_argument("--root", default="./dataset")

    args = p.parse_args()

    cfg = POSITION_CONFIG[args.position]
    args.exposure = cfg["exposure"] if args.exposure is None else args.exposure
    args.gain = cfg["gain"] if args.gain is None else args.gain
    args.fps = cfg["fps"] if args.fps is None else args.fps

    cam = open_camera(args.device)

    try:
        initial_exposure = args.short_exposure if args.hdr else args.exposure
        setup_camera(cam, initial_exposure, args.gain, args.fps)

        # ✅ 关键修复：只启动一次
        check_ret(cam.MV_CC_StartGrabbing(), "StartGrabbing")

        session, img_dir = make_dir(
            args.root, args.hand, args.position,
            args.label, args.defect_type, args.part_id
        )
        source_dir = session/"raw_exposures"
        if args.hdr and args.save_hdr_sources:
            source_dir.mkdir(parents=True, exist_ok=True)

        csv_path = session/"meta.csv"
        write_csv_header(csv_path)

        print("session", session)
        print("images", img_dir)

        for g in range(1, args.group_count+1):
            args.group_id = g

            print(f"\n=== Group {g}/{args.group_count} ===")

            if args.manual_load:
                wait_for_manual_capture()

            capture(cam, args, img_dir, csv_path, source_dir)

    finally:
        try:
            cam.MV_CC_StopGrabbing()
        except:
            pass
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()


if __name__ == "__main__":
    main()
