#!/usr/bin/env python3
"""Compatibility wrapper for the packaged BMW four-camera collector."""

import sys

from bmw_inspection.cli import collect as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
