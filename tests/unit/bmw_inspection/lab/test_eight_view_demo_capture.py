"""Source-image retention contracts for BMW four-camera HDR acquisition."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np

from bmw_inspection.lab import eight_view_demo_capture as capture_module
from bmw_inspection.views import VIEW_ORDER
from bmw_inspection.lab.eight_view_demo_capture import FourCameraHdrSession


def test_capture_round_returns_fused_images_and_caches_all_hdr_sources(monkeypatch) -> None:
    slots = tuple(
        SimpleNamespace(front_view=VIEW_ORDER[index], back_view=VIEW_ORDER[index + 4])
        for index in range(4)
    )
    session = FourCameraHdrSession.__new__(FourCameraHdrSession)
    session.profile = SimpleNamespace(slots=slots, front_views=VIEW_ORDER[:4], back_views=VIEW_ORDER[4:])
    session._handles = object()
    session._adapter = object()
    session._pacer = object()
    session.last_sources = MappingProxyType({})
    monkeypatch.setattr(capture_module, "_runtime_config", lambda _profile: object())

    def fake_capture(_handles, _adapter, _config, *, pacer):
        return [
            SimpleNamespace(
                camera_slot=index,
                short_image=np.full((4, 5, 3), index + 1, dtype=np.uint8),
                long_image=np.full((4, 5, 3), index + 11, dtype=np.uint8),
                fused_image=np.full((4, 5, 3), index + 21, dtype=np.uint8),
                fused_clip_pct=float(index),
                attempt=2,
            )
            for index in range(4)
        ]

    import bmw_inspection.capture.hardware as multicamera

    monkeypatch.setattr(multicamera, "capture_hdr_round", fake_capture)
    front = session.capture_round("front")
    back = session.capture_round("back")

    assert tuple(front) == VIEW_ORDER[:4]
    assert tuple(back) == VIEW_ORDER[4:]
    assert tuple(session.last_sources) == VIEW_ORDER
    assert int(front["front"][0, 0, 0]) == 21
    assert int(session.last_sources["front"].short_image[0, 0, 0]) == 1
    assert int(session.last_sources["back_secondary"].long_image[0, 0, 0]) == 14
    assert session.last_sources["front"].fused_clip_pct == 0.0
    assert session.last_sources["front"].attempt == 2
