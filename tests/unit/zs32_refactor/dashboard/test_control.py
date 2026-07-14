from __future__ import annotations

import json
from pathlib import Path

import pytest

from zs32_inspection.dashboard.control import (
    consume_confirmation,
    load_progress,
    write_confirmation,
    write_progress,
)
from zs32_inspection.dashboard.contracts import ConfirmationCommand, ProgressRecord


def test_progress_round_trip_uses_atomic_json_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "nested" / "progress.json"
    replace_calls: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def spy_replace(source: Path, target: Path) -> Path:
        replace_calls.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", spy_replace)
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
    assert replace_calls == [(path.with_name(f".{path.name}.tmp"), path)]


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


@pytest.mark.parametrize(
    ("action", "round_name", "confirmation_id", "part_id"),
    [
        ("approve", "front", "token-1", "part-1"),
        ("confirm_round", "middle", "token-1", "part-1"),
        ("confirm_round", "front", "", "part-1"),
        ("confirm_round", "front", "token-1", ""),
    ],
)
def test_confirmation_command_rejects_invalid_runtime_values(
    action: str,
    round_name: str,
    confirmation_id: str,
    part_id: str,
) -> None:
    with pytest.raises(ValueError):
        ConfirmationCommand(action, round_name, confirmation_id, part_id)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    [
        "{broken",
        "[]",
        '{"action":"confirm_round","round":"front","confirmation_id":"token-1"}',
        (
            '{"action":"confirm_round","round":"front","confirmation_id":"token-1",'
            '"part_id":"part-1","extra":true}'
        ),
        '{"action":"approve","round":"front","confirmation_id":"token-1","part_id":"part-1"}',
        '{"action":"confirm_round","round":"middle","confirmation_id":"token-1","part_id":"part-1"}',
    ],
    ids=["corrupt-json", "array", "missing-field", "extra-field", "wrong-action", "wrong-round"],
)
def test_consume_confirmation_discards_invalid_payload(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "control.json"
    path.write_text(payload, encoding="utf-8")
    expected = ConfirmationCommand("confirm_round", "front", "token-1", "part-1")

    assert consume_confirmation(path, expected) is False
    assert not path.exists()
