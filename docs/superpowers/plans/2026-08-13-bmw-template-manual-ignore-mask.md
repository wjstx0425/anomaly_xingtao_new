# BMW Template Manual Ignore Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the SHA-bound eight-view manual ignore mask for Template alignment, similarity, risk, difference statistics, and diagnostic heatmaps, then prove the behavior on saved ROI images before publishing an independent candidate config.

**Architecture:** Keep Template inputs and models unchanged. Transform each original-ROI mask through the same aspect-fit and reflected padding as Template preprocessing, then evaluate masked CCOEFF_NORMED across the existing translation window using correlation/integral identities. Preserve current unmasked evidence beside masked evidence. Offline A/B runs saved ROIs only and publishes no-overwrite JSON/CSV evidence.

**Tech Stack:** Python, NumPy, OpenCV, pytest, uv.

## Global Constraints

- Reuse `bmw_right_manual_ignore_v3`; do not change the public ROI or EfficientAD mask behavior.
- `255=ignore`, `0=inspect`; ignored pixels do not select templates, shifts, scores, differences, or hotspots.
- Template input pixels are never filled or modified.
- V3/V4 configs and embedded Template thresholds remain unchanged.
- A masked-score threshold must be a separate SHA-bound candidate asset; current Template thresholds may be shown only as an A/B reference.
- Template heatmaps remain “诊断热区”, not defect segmentation.

---

### Task 1: Masked Template scoring kernel

**Files:**
- Create: `src/bmw_inspection/lab/template_ignore_mask.py`
- Create: `tests/unit/bmw_inspection/lab/test_template_ignore_mask.py`

**Interfaces:**
- Produces: `prepare_template_inspect_mask(mask, target_size) -> np.ndarray`
- Produces: `masked_ccoeff_normed_map(padded_query, template, padded_inspect_mask) -> np.ndarray`
- Produces: `select_masked_template_match(...) -> MaskedTemplateMatch`

- [ ] Write failing tests proving all-inspect parity with OpenCV, ignored corruption invariance, shifted-mask alignment, and fail-closed insufficient/constant valid pixels.
- [ ] Run the focused test and confirm failure because the module does not exist.
- [ ] Implement the correlation/integral masked CCOEFF formula and exact nearest-neighbor/reflected mask geometry.
- [ ] Run the focused test and confirm all tests pass.

### Task 2: Template predictor evidence integration

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

**Interfaces:**
- Consumes: Task 1 scoring kernel and original-ROI masks.
- Produces: optional Template masked scoring with raw unmasked and masked evidence fields.

- [ ] Write failing tests showing a masked corruption changes NG to PASS while a defect outside the mask remains NG, and an empty mask preserves the current path.
- [ ] Run the tests and confirm behavior is absent.
- [ ] Add optional Template masks/index SHA/thresholds without changing default construction.
- [ ] Render ignored pixels neutral in the difference heatmap and restrict statistics/hotspots to inspect pixels.
- [ ] Run Template and model-suite focused regression tests.

### Task 3: Saved-ROI offline A/B and candidate gate

**Files:**
- Create: `pipeline/bmw_lab_report_template_manual_ignore_ab.py`
- Create: `tests/unit/pipeline/test_bmw_lab_report_template_manual_ignore_ab.py`
- Create only after evidence passes: `configs/bmw/experiments/bmw_eight_view_demo_v5_template_manual_ignore_mask_v1.json`

**Interfaces:**
- Consumes: saved `rois/<view>.png`, current Template models, and v3 mask asset.
- Produces: per-image CSV and report JSON with raw/masked similarity, risk, shifts, threshold-reference decisions, mask contribution, and per-capture fail-close evidence.

- [ ] Write failing tests for deterministic row schema, no-overwrite publication, and per-capture summary.
- [ ] Implement the CLI and run it on the current saved inspection root.
- [ ] Compare trusted OK and known pose/fail-close samples; do not infer defect labels from heatmaps.
- [ ] If evidence is favorable, fit a separate candidate threshold from calibration-only data and publish a SHA-bound V5 config; otherwise stop with the A/B report and explicit blocker.
- [ ] Run only relevant tests, `py_compile`, asset SHA checks, and `git diff --check`.
