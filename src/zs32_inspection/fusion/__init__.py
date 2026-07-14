"""Strict, fail-closed fusion for ZS32 multi-view evidence."""

from .engine import StrictFusionEngine
from .policy import FusionPolicy

__all__ = ["FusionPolicy", "StrictFusionEngine"]
