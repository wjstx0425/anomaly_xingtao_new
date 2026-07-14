#!/usr/bin/env python3
"""Thin wrapper for the Linux-only ZS32 candidate validation command."""

from zs32_inspection.cli.validate_candidate import main


if __name__ == "__main__":
    raise SystemExit(main())
