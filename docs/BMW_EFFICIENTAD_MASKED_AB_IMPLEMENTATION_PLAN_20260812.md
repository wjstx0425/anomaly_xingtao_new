# BMW EfficientAD Masked A/B Implementation Plan

## Goal

Build the smallest no-overwrite offline experiment that re-infers saved V3 ROI images, preserves the current
EfficientAD `pred_score`, exports the returned raw anomaly map, and compares foreground/background map aggregations.
Do not modify V1/V2/V3 runtime assets or start full retraining.

## Boundaries

- Use the existing public ROI unchanged.
- Treat EfficientAD heatmaps as diagnostic hot regions, never defect segmentation.
- Keep current `pred_score` and map aggregations as different score domains; do not reuse the V3 threshold for map scores.
- Every foreground mask is binary, original-ROI-sized, SHA-bound, and reviewed through an overlay.
- Existing现场 records without human labels remain `uncertain`.
- Publish into a new atomic no-overwrite result directory.

## Tasks

1. Add focused tests for binary mask validation, fixed outside fill, map resizing, foreground/background contribution,
   quantile/top-k aggregation, and minimum connected-area aggregation. Run them RED.
2. Implement the pure mask/A-B helpers and run the focused tests GREEN.
3. Add focused CLI tests for immutable output and evidence/report paths. Run them RED, then implement the offline CLI.
4. Build eight candidate masks from the trusted-OK median references, save SHA-bound masks and visual overlays, and
   review the overlays before inference.
5. Re-infer the saved V3 ROI set with each view checkpoint loaded once. Save ROI, mask, masked ROI, raw `.npy` map,
   masked heatmap, per-row CSV, per-capture CSV, per-view CSV, and JSON report.
6. Compare fixture/background contribution objectively. Keep acceptable-stain and visible-defect metrics unverified
   until the user supplies capture/view labels.
7. If and only if the A/B shows a useful fixture/background reduction without obviously masking the part, publish an
   independent mask candidate asset. Defer V4 training/runtime activation until physical-part calibration/final-test
   data and stain/defect labels are available.
8. Run focused verification and update `AGENTS_MEMORY.md` with exact artifacts, commands, results, and boundaries.
