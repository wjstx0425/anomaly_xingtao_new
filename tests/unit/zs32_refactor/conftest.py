"""Prevent ZS32 refactor tests from executing outside Linux + NVIDIA hosts."""

from __future__ import annotations

import sys

import pytest


def _has_authoritative_linux_cuda() -> bool:
    """Require a working driver and CUDA-visible PyTorch, not a stale binary."""
    if sys.platform != "linux":
        return False
    try:
        import torch

        from zs32_inspection.cli._environment import require_linux_nvidia

        require_linux_nvidia()
    except Exception:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip every ZS32 test when the authoritative platform is unavailable."""
    if _has_authoritative_linux_cuda():
        return
    marker = pytest.mark.skip(
        reason="ZS32 tests require Linux, a working NVIDIA driver, and CUDA-visible PyTorch"
    )
    for item in items:
        item.add_marker(marker)
