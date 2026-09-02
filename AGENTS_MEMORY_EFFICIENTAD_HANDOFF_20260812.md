# BMW EfficientAD Fixture/Stain Handoff Memory

- Primary handoff: `docs/BMW_EFFICIENTAD_FIXTURE_STAIN_FALSE_POSITIVE_HANDOFF_20260812.md`.
- Baseline: branch `agent/bmw-21only-diagnostics`, handoff commit
  `268a9b67eb5ecf0040bedf1a971793924c3635b9`, V3 config
  `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`.
- Current EfficientAD decides from Anomalib `pred_score` over the complete rectangular ROI. Its anomaly map is only
  diagnostic, has no foreground mask, and raw maps are not persisted in existing inspection records.
- The V3 threshold asset is normal-only and test-selected, adds `0.05`, and has no evaluated defect parts. Do not fix
  fixture/stain false positives by another blind threshold increase.
- Recommended sequence: same-image masked-map A/B first, then an EfficientAD-only eight-view foreground-mask contract,
  user-approved acceptable-stain normals, physical-part-isolated splits, V4 retraining, calibration-only score/threshold
  selection, and an independent no-overwrite V4 Demo config.
- Do not alter the shared ROI or V1/V2/V3. Missing parts must remain fail-closed through Template or a dedicated
  presence check. Trusted-OK v2 can support mask construction/diagnostics but is not independent acceptance data.
