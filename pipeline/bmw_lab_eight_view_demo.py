#!/usr/bin/env python3
"""Compatibility wrapper for the packaged BMW eight-view CLI."""

import sys

from bmw_inspection.cli import eight_view_demo as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
