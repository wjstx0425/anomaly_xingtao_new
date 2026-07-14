from __future__ import annotations

import json
from pathlib import Path

from zs32_inspection.dashboard.control import (
    consume_confirmation,
    load_progress,
    write_confirmation,
    write_progress,
)
from zs32_inspection.dashboard.contracts import ConfirmationCommand, ProgressRecord


def test_progress_round_trip_uses_atomic_json_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "progress.json"
    record = ProgressRecord(
        part_id="part-1",
        capture_session="session-1",
        state="waiting_front",
        message="Confirm front and capture",
        timestamp="2026-07-14T12:34:56Z",
        confirmation_id="token-1",
    )

    write_progress(path, record)

    assert load_progress(path) == record
    assert json.loads(path.read_text(encoding="utf-8"))["error"] is None
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_confirmation_accepts_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    command = ConfirmationCommand("confirm_round", "front", "token-1", "part-1")

    write_confirmation(path, command)

    assert consume_confirmation(path, command) is True
    assert consume_confirmation(path, command) is False
    assert not path.exists()


def test_confirmation_rejects_stale_wrong_round_and_wrong_part(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    expected = ConfirmationCommand("confirm_round", "front", "token-1", "part-1")
    for command in (
        ConfirmationCommand("confirm_round", "front", "old-token", "part-1"),
        ConfirmationCommand("confirm_round", "back", "token-1", "part-1"),
        ConfirmationCommand("confirm_round", "front", "token-1", "part-2"),
    ):
        write_confirmation(path, command)
        assert consume_confirmation(path, expected) is False
        assert not path.exists()
