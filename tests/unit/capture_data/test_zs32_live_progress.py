"""Live progress/control transport at capture boundaries."""

from __future__ import annotations

import io
from pathlib import Path

from zs32_inspection.capture import bootstrap
from zs32_inspection.capture.bootstrap import (
    BootstrapRoundCoordinator,
    DashboardRoundCoordinator,
)
from zs32_inspection.capture.contracts import CaptureRequest, CaptureRoundPlan
from zs32_inspection.dashboard.contracts import ConfirmationCommand
from zs32_inspection.dashboard.control import load_progress, write_confirmation
from zs32_inspection.domain.identity import Hand, PartIdentity


class _InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class _InteractiveOutput(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_terminal_confirmation_waits_without_a_default_deadline(monkeypatch) -> None:
    stdin = _InteractiveInput("\n")
    stdout = _InteractiveOutput()
    observed_timeouts: list[float | None] = []

    def _select(readers, writers, errors, timeout):
        del writers, errors
        observed_timeouts.append(timeout)
        return readers, [], []

    monkeypatch.setattr(bootstrap.sys, "stdin", stdin)
    monkeypatch.setattr(bootstrap.sys, "stdout", stdout)
    monkeypatch.setattr(bootstrap.select, "select", _select)
    coordinator = BootstrapRoundCoordinator("operator")
    request = CaptureRequest("session", "set", PartIdentity("part-1", Hand.RIGHT))
    round_plan = CaptureRoundPlan("front", "place front")

    confirmation = coordinator.confirm_round(
        request, round_plan, round_index=1, round_count=2
    )

    assert confirmation.round_id == "front"
    assert observed_timeouts == [None]


def test_dashboard_confirmation_is_consumed_once_without_tty(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    control = tmp_path / "control.json"
    coordinator = DashboardRoundCoordinator("operator", progress, control, poll_interval=0.0)
    assert coordinator.timeout_seconds is None
    request = CaptureRequest("session", "set", PartIdentity("part-1", Hand.RIGHT))
    round_plan = CaptureRoundPlan("front", "place front")

    coordinator.publish_waiting(request, round_plan)
    waiting = load_progress(progress)
    write_confirmation(
        control,
        ConfirmationCommand("confirm_round", "front", str(waiting.confirmation_id), "part-1"),
    )
    confirmation = coordinator.confirm_round(request, round_plan, round_index=1, round_count=2)

    assert confirmation.round_id == "front"
    assert load_progress(progress).state == "capturing_front"
    assert coordinator.consume_current_confirmation(request, round_plan) is False
