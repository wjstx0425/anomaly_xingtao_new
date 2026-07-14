#!/usr/bin/env python3
"""Thin wrapper for the Linux-only ZS32 candidate registration command."""

from zs32_inspection.cli.register_candidate import main


if __name__ == "__main__":
    raise SystemExit(main())
