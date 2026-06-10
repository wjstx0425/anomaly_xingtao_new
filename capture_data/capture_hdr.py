"""Capture short/long exposure pairs and save a fused image."""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import cv2

import capture as camera_utils
from exposure_fusion import fuse_exposures


def open_camera(device_index: int):
    """Open a Hikvision camera by device index."""
    device_list = camera_utils.MV_CC_DEVICE_INFO_LIST()
    ret = camera_utils.MvCamera.MV_CC_EnumDevices(
        camera_utils.MV_GIGE_DEVICE | camera_utils.MV_USB_DEVICE,
        device_list,
    )
    camera_utils.check_ret(ret, "Enum devices")

    if device_list.nDeviceNum == 0:
        raise RuntimeError("No camera found")

    print(f"Found {device_list.nDeviceNum} device(s)")
    if device_index >= device_list.nDeviceNum:
        raise RuntimeError(f"Device index {device_index} out of range")

    cam = camera_utils.MvCamera()
    device_info = ctypes.cast(
        device_list.pDeviceInfo[device_index],
        ctypes.POINTER(camera_utils.MV_CC_DEVICE_INFO),
    ).contents

    camera_utils.check_ret(cam.MV_CC_CreateHandle(device_info), "Create handle")
    camera_utils.check_ret(cam.MV_CC_OpenDevice(camera_utils.MV_ACCESS_Exclusive, 0), "Open device")

    if device_info.nTLayerType == camera_utils.MV_GIGE_DEVICE:
        packet_size = cam.MV_CC_GetOptimalPacketSize()
        if packet_size > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

    return cam


def setup_camera(cam, gain: float, fps: float, trigger: bool) -> None:
    """Configure fixed gain, frame rate, and trigger mode."""
    camera_utils.set_enum(cam, "ExposureAuto", 0)
    camera_utils.set_enum(cam, "GainAuto", 0)
    try:
        camera_utils.set_float(cam, "Gain", gain)
    except Exception as exc:
        print(f"Warning: set Gain failed ({exc}), continue with camera default gain")

    try:
        camera_utils.set_bool(cam, "AcquisitionFrameRateEnable", True)
        camera_utils.set_float(cam, "AcquisitionFrameRate", fps)
    except Exception as exc:
        print(f"Warning: set frame rate failed: {exc}")

    if trigger:
        camera_utils.set_enum(cam, "TriggerMode", 1)
        camera_utils.set_enum(cam, "TriggerSource", 7)
    else:
        camera_utils.set_enum(cam, "TriggerMode", 0)


def grab_one_frame(
    cam,
    data_buf,
    buffer_size: int,
    frame_info,
    timeout_ms: int,
    trigger: bool,
) -> cv2.typing.MatLike | None:
    """Grab one frame from the camera."""
    if trigger:
        ret = cam.MV_CC_SetCommandValue("TriggerSoftware")
        camera_utils.check_ret(ret, "Software trigger")

    ret = cam.MV_CC_GetOneFrameTimeout(data_buf, buffer_size, frame_info, timeout_ms)
    if ret != 0:
        print(f"Frame failed, ret=0x{ret:x}")
        return None

    return camera_utils.convert_frame_to_bgr(data_buf, frame_info)


def capture_at_exposure(
    cam,
    exposure: float,
    data_buf,
    buffer_size: int,
    frame_info,
    timeout_ms: int,
    trigger: bool,
    settle_frames: int,
) -> cv2.typing.MatLike:
    """Set exposure, discard settling frames, and return one image."""
    camera_utils.set_float(cam, "ExposureTime", exposure)

    for _ in range(settle_frames):
        grab_one_frame(cam, data_buf, buffer_size, frame_info, timeout_ms, trigger)

    image = grab_one_frame(cam, data_buf, buffer_size, frame_info, timeout_ms, trigger)
    if image is None:
        raise RuntimeError(f"Failed to capture image at exposure={exposure}")
    return image


def exposure_name(exposure: float) -> str:
    """Format an exposure value for filenames."""
    if exposure.is_integer():
        return str(int(exposure))
    return str(exposure).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out", type=str, default="./hik_hdr_images")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg", "bmp"])
    parser.add_argument("--short-exposure", type=float, default=7000.0)
    parser.add_argument("--long-exposure", type=float, default=40000.0)
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--trigger", action="store_true")
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--no-save-sources", action="store_true")
    parser.add_argument(
        "--fusion-method",
        choices=["selective", "mertens"],
        default="selective",
    )
    parser.add_argument("--short-dark-threshold", type=float, default=70.0)
    parser.add_argument("--long-clip-threshold", type=float, default=245.0)
    parser.add_argument("--blend-width", type=float, default=18.0)
    parser.add_argument("--blur-size", type=int, default=31)
    args = parser.parse_args()

    out_dir = Path(args.out)
    fused_dir = out_dir / "fused"
    source_dir = out_dir / "sources"
    fused_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_sources:
        source_dir.mkdir(parents=True, exist_ok=True)

    cam = open_camera(args.device)
    try:
        setup_camera(cam, args.gain, args.fps, args.trigger)
        camera_utils.check_ret(cam.MV_CC_StartGrabbing(), "Start grabbing")

        frame_info = camera_utils.MV_FRAME_OUT_INFO_EX()
        buffer_size = 50 * 1024 * 1024
        data_buf = (ctypes.c_ubyte * buffer_size)()

        exposures = [args.short_exposure, args.long_exposure]
        exposure_labels = [exposure_name(value) for value in exposures]

        for index in range(args.count):
            images = []
            for exposure in exposures:
                image = capture_at_exposure(
                    cam,
                    exposure,
                    data_buf,
                    buffer_size,
                    frame_info,
                    args.timeout_ms,
                    args.trigger,
                    args.settle_frames,
                )
                images.append(image)

            if not args.no_save_sources:
                for image, label in zip(images, exposure_labels, strict=True):
                    source_path = source_dir / f"{index:06d}_exp{label}.{args.format}"
                    cv2.imwrite(str(source_path), image)

            fused = fuse_exposures(
                images,
                method=args.fusion_method,
                align=args.align,
                short_dark_threshold=args.short_dark_threshold,
                long_clip_threshold=args.long_clip_threshold,
                blend_width=args.blend_width,
                blur_size=args.blur_size,
            )
            fused_path = fused_dir / f"{index:06d}.{args.format}"
            cv2.imwrite(str(fused_path), fused)
            print(f"Saved {fused_path}")

            time.sleep(0.01)

        cam.MV_CC_StopGrabbing()

    finally:
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()


if __name__ == "__main__":
    main()
