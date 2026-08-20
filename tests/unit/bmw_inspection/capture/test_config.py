"""Contract tests for the BMW four-camera eight-view HDR profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bmw_inspection.capture.config import load_capture_profile


REPO_ROOT = Path(__file__).parents[4]
PROFILE_PATH = REPO_ROOT / "configs/bmw/capture/bmw_4cam_eight_view_hdr_v1.json"


def test_shipped_profile_binds_the_fixed_four_camera_eight_view_hdr_contract() -> None:
    profile = load_capture_profile(PROFILE_PATH)

    assert profile.profile_id == "bmw-4cam-eight-view-hdr-v1"
    assert tuple((slot.slot_id, slot.serial) for slot in profile.slots) == (
        ("center", "DA9805574"),
        ("left", "DA9625347"),
        ("right", "DB0998274"),
        ("secondary", "DB0968108"),
    )
    assert profile.front_views == ("front", "front_left", "front_right", "front_secondary")
    assert profile.back_views == ("back", "back_left", "back_right", "back_secondary")
    assert profile.all_views == profile.front_views + profile.back_views
    assert len(set(profile.all_views)) == 8
    assert profile.hdr.short_exposure_us == 1500
    assert profile.hdr.long_exposure_us == 6000
    assert profile.hdr.short_exposure_us < profile.hdr.long_exposure_us
    assert profile.hdr.gain == 0
    assert profile.hdr.trigger_interval_s == 0.2
    assert profile.hdr.settle_frames == 1
    assert profile.hdr.timeout_ms == 3000


@pytest.mark.parametrize("profile_id", ["../escape", "bmw profile", "bmw/profile", ""])
def test_rejects_path_unsafe_profile_id(tmp_path: Path, profile_id: str) -> None:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    payload["profile_id"] = profile_id
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="profile_id.*path-safe"):
        load_capture_profile(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("short_exposure_us", 0, "short_exposure_us must be positive"),
        ("long_exposure_us", 1500, "short_exposure_us must be below long_exposure_us"),
        ("trigger_interval_s", 0, "trigger_interval_s must be positive"),
        ("settle_frames", 0, "settle_frames must be positive"),
        ("timeout_ms", 0, "timeout_ms must be positive"),
    ],
)
def test_rejects_non_positive_or_non_ordered_hdr_settings(
    tmp_path: Path, field: str, value: int | float, message: str
) -> None:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    payload["hdr"][field] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_capture_profile(path)
