"""Entry-point integration for BMW v3 evidence persistence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bmw_inspection.lab.eight_view_dataset import VIEW_ORDER
from pipeline import bmw_lab_eight_view_demo as entrypoint
from bmw_inspection.lab.eight_view_demo_ui import EightViewUiState


def test_persist_result_adapts_offline_images_and_preserves_live_sources(monkeypatch) -> None:
    images = {view: np.zeros((4, 5, 3), dtype=np.uint8) for view in VIEW_ORDER}
    inspection = SimpleNamespace(images=images)
    config = object()
    live_sources = {view: object() for view in VIEW_ORDER}
    calls = []
    monkeypatch.setattr(entrypoint, "fused_only_sources", lambda value: ("offline", value))
    monkeypatch.setattr(
        entrypoint,
        "persist_inspection",
        lambda used_config, used_inspection, used_sources: calls.append(
            (used_config, used_inspection, used_sources)
        ),
    )

    entrypoint._persist_result(config, inspection, None)
    entrypoint._persist_result(config, inspection, live_sources)

    assert calls == [
        (config, inspection, ("offline", images)),
        (config, inspection, live_sources),
    ]


def test_o_shortcut_handler_toggles_only_when_inspection_exists() -> None:
    idle = EightViewUiState()
    assert entrypoint._handle_trusted_ok_shortcut(idle, ord("o")) is idle
    legacy = SimpleNamespace(results=("unchanged",), trusted_ok_by_view={}, diagnostic_metadata={})
    legacy_state = EightViewUiState(inspection=legacy)  # type: ignore[arg-type]
    assert entrypoint._handle_trusted_ok_shortcut(legacy_state, ord("o")) is legacy_state
    inspection = SimpleNamespace(
        results=("unchanged",), trusted_ok_by_view={"front": object()}, diagnostic_metadata={}
    )
    state = EightViewUiState(inspection=inspection)  # type: ignore[arg-type]

    toggled = entrypoint._handle_trusted_ok_shortcut(state, ord("O"))

    assert toggled.trusted_ok_mode is True
    assert toggled.inspection is inspection
    assert toggled.inspection.results == ("unchanged",)
    assert entrypoint._handle_trusted_ok_shortcut(toggled, ord("x")) is toggled
