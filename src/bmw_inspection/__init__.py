"""BMW single-camera bright-streak Demo."""

from .contracts import (
    BrightStreakConfig,
    BrightStreakMetrics,
    BrightStreakResult,
    DemoStatus,
    load_config,
)
from .roi_selector import save_roi, select_roi

__all__ = [
    "BrightStreakConfig",
    "BrightStreakMetrics",
    "BrightStreakResult",
    "DemoStatus",
    "load_config",
    "save_roi",
    "select_roi",
]
