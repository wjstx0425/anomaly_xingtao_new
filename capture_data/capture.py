import os
import sys
import time
import ctypes
import argparse
from pathlib import Path
#侧面4000 9
#底部 6000 15

#ZS32
#


import cv2
import numpy as np

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *


def check_ret(ret, msg):
    if ret != 0:
        raise RuntimeError(f"{msg} failed, ret=0x{ret:x}")


def set_enum(cam, name, value):
    ret = cam.MV_CC_SetEnumValue(name, value)
    check_ret(ret, f"Set {name}")


def set_float(cam, name, value):
    target = float(value)
    st_value = MVCC_FLOATVALUE()
    ret = cam.MV_CC_GetFloatValue(name, st_value)
    if ret == 0:
        clamped = min(max(target, st_value.fMin), st_value.fMax)
        if clamped != target:
            print(
                f"Warning: {name}={target} out of range "
                f"[{st_value.fMin}, {st_value.fMax}], clamped to {clamped}"
            )
        target = clamped

    ret = cam.MV_CC_SetFloatValue(name, target)
    check_ret(ret, f"Set {name}")


def set_bool(cam, name, value):
    ret = cam.MV_CC_SetBoolValue(name, bool(value))
    check_ret(ret, f"Set {name}")


def get_float(cam, name):
    st_value = MVCC_FLOATVALUE()
    ret = cam.MV_CC_GetFloatValue(name, st_value)
    check_ret(ret, f"Get {name}")
    return st_value.fCurValue


def convert_frame_to_bgr(data_buf, frame_info):
    width = frame_info.nWidth
    height = frame_info.nHeight
    pixel_type = frame_info.enPixelType
    frame_len = frame_info.nFrameLen

    data = np.frombuffer(data_buf, dtype=np.uint8, count=frame_len)

    if pixel_type == PixelType_Gvsp_Mono8:
        img = data.reshape(height, width)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if pixel_type == PixelType_Gvsp_BayerRG8:
        img = data.reshape(height, width)
        return cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)

    if pixel_type == PixelType_Gvsp_BayerGB8:
        img = data.reshape(height, width)
        return cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)

    if pixel_type == PixelType_Gvsp_BayerGR8:
        img = data.reshape(height, width)
        return cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR)

    if pixel_type == PixelType_Gvsp_BayerBG8:
        img = data.reshape(height, width)
        return cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)

    if pixel_type == PixelType_Gvsp_RGB8_Packed:
        img = data.reshape(height, width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if pixel_type == PixelType_Gvsp_BGR8_Packed:
        return data.reshape(height, width, 3)

    raise RuntimeError(f"Unsupported pixel type: {pixel_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out", type=str, default="./hik_images")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg", "bmp"])

    parser.add_argument("--exposure", type=float, default=10000.0, help="Exposure time in us")
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=10.0)

    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument("--auto-gain", action="store_true")

    parser.add_argument("--trigger", action="store_true", help="Enable software trigger")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    check_ret(ret, "Enum devices")

    if device_list.nDeviceNum == 0:
        raise RuntimeError("No camera found")

    print(f"Found {device_list.nDeviceNum} device(s)")

    if args.device >= device_list.nDeviceNum:
        raise RuntimeError(f"Device index {args.device} out of range")

    cam = MvCamera()
    device_info = ctypes.cast(
        device_list.pDeviceInfo[args.device],
        ctypes.POINTER(MV_CC_DEVICE_INFO)
    ).contents

    ret = cam.MV_CC_CreateHandle(device_info)
    check_ret(ret, "Create handle")

    try:
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        check_ret(ret, "Open device")

        if device_info.nTLayerType == MV_GIGE_DEVICE:
            packet_size = cam.MV_CC_GetOptimalPacketSize()
            if packet_size > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

        # 关闭自动曝光/自动增益后，才能设置手动值
        if args.auto_exposure:
            set_enum(cam, "ExposureAuto", 2)  # Continuous
        else:
            set_enum(cam, "ExposureAuto", 0)  # Off
            set_float(cam, "ExposureTime", args.exposure)

        if args.auto_gain:
            set_enum(cam, "GainAuto", 2)  # Continuous
        else:
            set_enum(cam, "GainAuto", 0)  # Off
            try:
                set_float(cam, "Gain", args.gain)
            except Exception as e:
                print(
                    f"Warning: set Gain failed ({e}), "
                    "continue with camera default gain"
                )

        # 设置帧率
        try:
            set_bool(cam, "AcquisitionFrameRateEnable", True)
            set_float(cam, "AcquisitionFrameRate", args.fps)
        except Exception as e:
            print(f"Warning: set frame rate failed: {e}")

        if args.trigger:
            set_enum(cam, "TriggerMode", 1)          # On
            set_enum(cam, "TriggerSource", 7)        # Software
        else:
            set_enum(cam, "TriggerMode", 0)          # Off

        print("Current parameters:")
        for name in ["ExposureTime", "Gain", "AcquisitionFrameRate"]:
            try:
                print(f"  {name}: {get_float(cam, name)}")
            except Exception:
                pass

        ret = cam.MV_CC_StartGrabbing()
        check_ret(ret, "Start grabbing")

        frame_info = MV_FRAME_OUT_INFO_EX()
        buffer_size = 50 * 1024 * 1024
        data_buf = (ctypes.c_ubyte * buffer_size)()

        for i in range(args.count):
            if args.trigger:
                ret = cam.MV_CC_SetCommandValue("TriggerSoftware")
                check_ret(ret, "Software trigger")

            ret = cam.MV_CC_GetOneFrameTimeout(
                data_buf,
                buffer_size,
                frame_info,
                3000
            )

            if ret != 0:
                print(f"Frame {i} failed, ret=0x{ret:x}")
                continue

            img = convert_frame_to_bgr(data_buf, frame_info)
            filename = out_dir / f"{i:06d}.{args.format}"
            cv2.imwrite(str(filename), img)
            print(f"Saved {filename}")

            time.sleep(0.01)

        cam.MV_CC_StopGrabbing()

    finally:
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()


if __name__ == "__main__":
    main()