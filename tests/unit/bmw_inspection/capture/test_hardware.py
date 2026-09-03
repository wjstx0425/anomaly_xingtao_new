"""Hardware-free checks for BMW-owned four-camera HDR primitives."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bmw_inspection.capture.hardware import CameraHandle, DeviceDescription, GroupedTriggerPacer, capture_hdr_round


class _Adapter:
    def __init__(self) -> None:
        self.exposures: dict[int, float] = {}

    def set_exposure(self, handle: CameraHandle, exposure: float) -> None:
        self.exposures[handle.device.index] = exposure

    def trigger(self, handle: CameraHandle) -> None:
        del handle

    def read(self, handle: CameraHandle, timeout_ms: int) -> np.ndarray:
        del timeout_ms
        value = int(self.exposures[handle.device.index] / 1000) + handle.device.index
        return np.full((8, 8, 3), value, dtype=np.uint8)


def test_hdr_round_returns_one_result_for_each_of_four_cameras() -> None:
    handles = [
        CameraHandle(DeviceDescription(index, "camera", f"serial-{index}"), object(), object(), object(), True)
        for index in range(4)
    ]
    config = SimpleNamespace(
        short_exposure=1500.0,
        long_exposure=6000.0,
        hdr_settle_frames=0,
        timeout_ms=3000,
        align_hdr=False,
        short_dark_threshold=70.0,
        long_clip_threshold=245.0,
        blend_width=18.0,
        blur_size=3,
        hdr_max_retries=0,
        hdr_max_clip_pct=100.0,
    )

    results = capture_hdr_round(
        handles,
        _Adapter(),
        config,
        pacer=GroupedTriggerPacer(0.0),
    )

    assert [result.camera_slot for result in results] == [0, 1, 2, 3]
    assert all(result.short_image.shape == (8, 8, 3) for result in results)
    assert all(result.long_image.shape == (8, 8, 3) for result in results)
    assert all(result.fused_image.shape == (8, 8, 3) for result in results)
