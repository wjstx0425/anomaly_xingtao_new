"""Linux-only CLI boundary for versioned acquisition assets."""

from __future__ import annotations

import sys

import pytest

from zs32_inspection.cli import capture


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ZS32 capture is Linux-only")


def test_capture_cli_has_no_loose_camera_config_escape_hatch() -> None:
    parser = capture._parser()
    assert "--camera-config" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--gate-publication", "/immutable/gate-publication",
                "--raw-root", "/data/raw",
                "--capture-session", "session-1",
                "--capture-set-id", "capture-1",
                "--part-instance-id", "part-1",
                "--hand", "right",
                "--camera-config", "/tmp/loose.json",
            ]
        )
