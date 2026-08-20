# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare or validate the immutable ZS32 0723 retraining-data release."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture_data.prepare_zs32_0723_retraining import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
