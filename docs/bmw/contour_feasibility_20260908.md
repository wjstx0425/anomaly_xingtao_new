# BMW left contour image review — 2026-09-08

Source: `dataset/bmw_lab_raw_clean_0820/left`, views `front_left` and `front_secondary`.

## Inventory and review scope

Each view contains 211 fused PNG images: 90 normal from `20260820_162040_682498`, 47 normal from `20260823_161606_428291`, 42 normal from `20260823_164432_522052`, and 32 generic defects from `20260820_155511_803522`. These are image/capture counts, not independently established physical-part counts. The defect directory is `defect/defect`, not a deformation-specific annotation.

Visually reviewed every defect in two 16-image contact sheets per view and 12 normal samples per view (four evenly spaced sorted samples from each session). Six selected images were reviewed with larger crops. Local source paths, crop coordinates and contact sheets are under `artifacts/bmw_contour_review_20260908/` (excluded from Git). Original images were not modified. Normal and defect group001 are examples, not a before/after pair.

## Observations

- `front_left`: dark background separates much of the narrow part silhouette. Defect group020 has a downward-bent bottom tab and group027 an upward-bent top tab. Group023 shows a visibly displaced lower section; group024 a changed upper section. These are useful candidates for local silhouette/endpoint comparison.
- `front_secondary`: right and lower outer edges are useful candidates. Defect group014 has an irregular lower edge; group019 has an altered lower-right outline. Surface markings, ribs, reflected highlights and some surface defects also occur inside the outline, so outline inspection alone does not cover all generic defects.
- Existing public `front_left` ROI `[1575,91,2517,2866]` clips portions of the extreme tabs in group020/group027. Use a separate expanded contour search region; do not change all existing branches' shared ROI merely to conduct this experiment.
- Normal images across the three sessions have appearance/pose variation that should be included in false-positive checks. A single normal reference is insufficient evidence for a production tolerance.
- Two 0823 normal sessions reuse retake sample names; use session plus sample plus view for identity. Cross-view capture correspondence is not proof of distinct physical parts or before/after correspondence.

## Proposed next experiment

Use stable body regions for translation/small-angle rigid alignment. Inspect top/bottom tabs and side edges separately in `front_left`; inspect the right outer arc and bottom edge separately in `front_secondary`. Prototype threshold/contour extraction with OpenCV, then narrow-band normal-direction edge measurements where the silhouette is unstable. Keep both missing and excess silhouette regions, with displacement and contiguous abnormal-length metrics. Avoid nonrigid alignment, free scaling and aggressive morphology that can conceal deformation. Mark clipped/unobservable edges as invalid rather than PASS.

Build normal envelopes from confirmed normals and evaluate held-out sessions plus manually confirmed visible contour defects. Do not label all 32 generic defects as expected contour positives. This review did not run a contour detector, choose thresholds, measure accuracy or change runtime behavior.

OpenCV contour extraction reference: https://docs.opencv.org/4.13.0/d4/d73/tutorial_py_contours_begin.html
