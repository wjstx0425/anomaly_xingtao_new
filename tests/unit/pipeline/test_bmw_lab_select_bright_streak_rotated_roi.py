"""CLI contracts for the manual rotated bright-streak ROI selector."""

from __future__ import annotations

from pipeline.bmw_lab_select_bright_streak_rotated_roi import _source_point, build_parser


def test_source_point_maps_scaled_display_to_original() -> None:
    assert _source_point((320, 180), (720, 1280), (3036, 4024)) == (1006, 759)


def test_parser_defaults_to_the_confirmed_broken_capture() -> None:
    args = build_parser().parse_args([])

    assert args.image.name == "front_left_hdr.png"
    assert "bmw_demo_20260813_164043" in str(args.image)
