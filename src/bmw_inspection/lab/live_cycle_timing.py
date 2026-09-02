"""Exact live GUI cycle timing for the BMW eight-view laboratory Demo."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


_BOUNDARY = "first_space_accepted_to_first_result_imshow_returned"


@dataclass(frozen=True, slots=True)
class LiveCycleTiming:
    """One measured live cycle from accepted Space to submitted RESULT frame."""

    capture_id: str
    started_at: str
    displayed_at: str
    front_capture_ms: float
    flip_wait_ms: float
    back_capture_ms: float
    front_inference_ms: float
    back_inference_ms: float
    finalize_ms: float
    persist_ms: float
    result_display_ms: float
    front_overlap_ms: float
    total_cycle_ms: float

    def __post_init__(self) -> None:
        if not self.capture_id.strip() or not self.started_at.strip() or not self.displayed_at.strip():
            raise ValueError("完整周期计时身份和墙钟时间不能为空")
        for field in fields(self)[3:]:
            value = float(getattr(self, field.name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"完整周期计时字段无效：{field.name}")

    def to_payload(self) -> dict[str, str | float]:
        """Return the simple JSON-compatible timing record."""
        return {
            "capture_id": self.capture_id,
            "boundary": _BOUNDARY,
            "started_at": self.started_at,
            "displayed_at": self.displayed_at,
            **{field.name: float(getattr(self, field.name)) for field in fields(self)[3:]},
        }


def persist_cycle_timing(result_root: Path, timing: LiveCycleTiming) -> Path:
    """Write the timing record beside the already-persisted inspection."""
    path = Path(result_root).expanduser().resolve() / timing.capture_id / "cycle_timing.json"
    path.write_text(
        json.dumps(timing.to_payload(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


__all__ = ["LiveCycleTiming", "persist_cycle_timing"]
