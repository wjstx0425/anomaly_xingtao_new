# BMW Reviewed 40-Template Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train both hands' eight Template models from exactly the surviving review copies, recalibrate ordinary and weight-3.0 thresholds, and create new rollback-safe Demo configs.

**Architecture:** A focused trainer reads `candidate_manifest.csv`, treats missing candidate copies as rejected, writes every surviving preprocessed crop as a resident template, and never performs a second selection pass. The same runtime Template scorer evaluates 0823 calibration/final-test normal rows once, exposing both legacy and weighted risks; new configs are copied from the current mixed profiles while the current profiles remain unchanged.

**Tech Stack:** Python 3.13, OpenCV, NumPy, existing BMW Template runtime, pytest, uv.

## Global Constraints

- Use exactly the surviving review PNGs; do not refill or reselect.
- Require 3–40 templates per hand/view.
- Keep current model directories and current mixed configs unchanged for rollback.
- Fit ordinary and weighted thresholds as calibration-normal maximum plus 10%; final_test is reporting only.
- Keep fixed 25 checks and do not alter EfficientAD, YOLO, bright-streak, ROI, masks, HDR, or fusion.
- Do not add SHA, receipt, provenance, publisher, rebind, or immutable release logic.

---

### Task 1: Reviewed model writer and calibration

**Files:**
- Create: `src/bmw_inspection/lab/template_review_training.py`
- Test: `tests/unit/bmw_inspection/lab/test_template_review_training.py`

**Interfaces:**
- Produces `write_reviewed_template_models(...)`, `calibrate_reviewed_template_models(...)`, and `write_reviewed_demo_config(...)`.

- [ ] Write failing tests proving deleted copies are omitted without refill, all survivors become templates, fewer than three fails, model JSON contains no SHA fields, calibration uses only calibration normals, final_test cannot change thresholds, and the base config remains byte-identical.
- [ ] Run the test file and confirm RED because the module is absent.
- [ ] Implement the minimal model writer, runtime-score calibration, report, and copied-config writer.
- [ ] Run the test file and confirm GREEN.

### Task 2: Bilateral CLI, training, replay, and memory

**Files:**
- Create: `pipeline/bmw_lab_train_reviewed_templates.py`
- Test: `tests/unit/pipeline/test_bmw_lab_train_reviewed_templates.py`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Default output models: `results/bmw_lab_one_click/bmw_{right,left}_template_40_reviewed_0823_v1`.
- Default candidate configs: `configs/bmw/experiments/bmw_eight_view_demo_{right,left}_0823_template40_v1.json`.

- [ ] Write a failing CLI test for bilateral defaults and exact argument forwarding.
- [ ] Implement the thin CLI and run focused GREEN tests, syntax checks, and `git diff --check`.
- [ ] Run the CLI once, verify per-view template counts and report thresholds, and verify the two old configs are byte-identical.
- [ ] Run one offline 0823 normal replay per hand with the new configs; require 25 rows and zero ERROR, while reporting actual OK/NG instead of forcing acceptance.
- [ ] Update `AGENTS_MEMORY.md` with exact assets, thresholds, replay outcomes, and remaining live-camera/GPU boundary.

