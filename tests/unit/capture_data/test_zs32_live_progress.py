"""Live progress/control transport at capture boundaries."""

from __future__ import annotations

from pathlib import Path

from zs32_inspection.capture.bootstrap import DashboardRoundCoordinator
from zs32_inspection.capture.contracts import CaptureRequest, CaptureRoundPlan
from zs32_inspection.dashboard.contracts import ConfirmationCommand
from zs32_inspection.dashboard.control import load_progress, write_confirmation
from zs32_inspection.domain.identity import Hand, PartIdentity


def test_dashboard_confirmation_is_consumed_once_without_tty(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    control = tmp_path / "control.json"
    coordinator = DashboardRoundCoordinator("operator", progress, control, poll_interval=0.0)
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
