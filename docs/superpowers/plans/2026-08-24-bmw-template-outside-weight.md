# BMW Template Outside Weight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven development and execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep critical ROI weight at `3.0`, reduce non-ROI Template weight to `0.5`, recalibrate thresholds, and preserve the current Template40 configs for rollback.

**Architecture:** Extend the existing weighted Template configuration with one scalar `outside_weight`. Build the same aligned 512x512 weight map with a configurable background value, then reuse the existing weighted CCOEFF scorer and calibration pipeline. New candidate configs and result roots remain separate from the unchanged rollback configs.

**Tech Stack:** Python 3.13, NumPy, OpenCV, pytest, JSON, uv.

## Global Constraints

- ROI weight remains exactly `3.0`; outside weight is exactly `0.5`.
- Existing `bmw_eight_view_demo_{left,right}_0823_template40_v1.json` files remain byte-for-byte unchanged.
- Back views, Template models, ROI coordinates, masks, YOLO, EfficientAD, bright-streak, HDR capture, and fusion rules do not change.
- Calibration uses calibration normals only; final-test data is reporting-only.
- No SHA, publisher, receipt, or provenance flow is introduced.

---

### Task 1: Weight-map and config contract

**Files:**
- Modify: `src/bmw_inspection/lab/template_region_weighting.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Test: `tests/unit/bmw_inspection/lab/test_template_region_weighting.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_lab_config.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

**Interfaces:**
- `TemplateWeightedRegionsConfig.outside_weight: float`
- `build_aligned_template_weights(..., region_weight: float, outside_weight: float, ...) -> np.ndarray`
- `EightViewTemplatePredictor(..., weighted_outside_weight: float = 1.0)`

- [ ] Add failing tests for `3.0/0.5/0.0` weight-map values, config parsing, legacy default `1.0`, and result details.
- [ ] Run the focused tests and verify failures are caused by the missing `outside_weight` behavior.
- [ ] Implement the minimum parameter validation and propagation.
- [ ] Run the focused tests and verify they pass.

### Task 2: Calibration propagation and versioned configs

**Files:**
- Modify: `pipeline/bmw_lab_calibrate_weighted_template.py`
- Test: `tests/unit/pipeline/test_bmw_lab_calibrate_weighted_template.py`
- Create: `configs/bmw/experiments/bmw_eight_view_demo_left_0823_template40_outside05_v1.json`
- Create: `configs/bmw/experiments/bmw_eight_view_demo_right_0823_template40_outside05_v1.json`

**Interfaces:**
- Calibration predictor receives `weighted.outside_weight`.
- Calibration report records `outside_weight` and `effective_region_ratio`.

- [ ] Add a failing calibration propagation/report test.
- [ ] Implement propagation and report metadata.
- [ ] Create candidate configs from the existing rollback configs, changing only demo/result identity and `outside_weight` before calibration.
- [ ] Run left and right calibration with `--write-config`, storing reports under the new result roots.
- [ ] Confirm only front four weighted thresholds differ and back thresholds remain unchanged.

### Task 3: Focused verification and handoff

**Files:**
- Modify: `AGENTS_MEMORY.md`

- [ ] Run focused pytest for weighting, config, predictor, calibration, and active-config contracts.
- [ ] Run Python syntax checks and `git diff --check`.
- [ ] Run one no-GUI offline eight-view replay with one new candidate config.
- [ ] Record new/rollback launch commands, calibration reports, replay result, and remaining live-camera boundary in `AGENTS_MEMORY.md`.

