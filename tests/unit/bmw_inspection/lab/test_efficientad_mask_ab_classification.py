"""Conservative business-category assignment for the BMW mask A/B report."""

from bmw_inspection.lab.efficientad_mask_ab import classify_diagnostic_hotspot


def test_only_mask_excluded_hotspot_gets_fixture_background_category() -> None:
    assert classify_diagnostic_hotspot("background") == ("fixture_or_background", "candidate_mask_spatial_rule")
    assert classify_diagnostic_hotspot("foreground") == ("uncertain", "requires_human_business_label")
