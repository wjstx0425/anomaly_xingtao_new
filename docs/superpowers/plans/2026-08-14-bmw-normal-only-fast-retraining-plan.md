# BMW Normal-Only Fast Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven development and verify every focused test before completion.

**Goal:** Add one minimal CLI that retrains Template, EfficientAD, and the traditional bright-streak rule from a new all-normal eight-view release while skipping YOLO.

**Architecture:** A small orchestration module validates inputs, derives a reusable fixed-setup ROI, materializes ROI crops, merges only historical `no_streak` rows into a bright-streak manifest, and invokes existing stage implementations. EfficientAD score calibration is generalized from exactly 21 held-out parts to the actual complete part count.

**Tech Stack:** Python 3.11+, argparse, OpenCV, existing BMW lab modules, pytest, uv.

## Global Constraints

- YOLO must never be selected or trained.
- Existing releases, runs, ROI assets, and Demo configs must not be overwritten.
- New normal data is the only Template/EfficientAD training source.
- Historical data contributes only `no_streak` rows to bright-streak recalibration.

---

### Task 1: Dynamic EfficientAD normal calibration

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_train_all.py`
- Test: `tests/unit/bmw_inspection/lab/test_efficientad_only.py`

- [ ] Add a regression test proving score calibration accepts an arbitrary complete normal part count instead of exactly 21.
- [ ] Run the focused test and confirm the current hard-coded count fails it.
- [ ] Replace the fixed record/part count with the count derived from the scored identities while preserving the existing `1/21` target part FPR.
- [ ] Run the focused test to green.

### Task 2: Normal-only selective orchestration

**Files:**
- Create: `src/bmw_inspection/lab/normal_only_retraining.py`
- Create: `pipeline/bmw_lab_retrain_normal_only.py`
- Create: `tests/unit/bmw_inspection/lab/test_normal_only_retraining.py`

- [ ] Add failing tests for all-normal validation, fixed-setup ROI derivation, historical `no_streak` filtering, stage order, and absence of YOLO.
- [ ] Run the focused tests and confirm failure because the module/entrypoint is absent.
- [ ] Implement the smallest orchestration API with injectable stage handlers so unit tests do not start GPU training.
- [ ] Implement the argparse wrapper and dry-run report.
- [ ] Run the focused tests to green.

### Task 3: Handoff and verification

**Files:**
- Modify: `AGENTS_MEMORY.md`

- [ ] Document the exact new entrypoint, dataset contract, outputs, and the fact that it does not activate Demo models.
- [ ] Run focused BMW tests, `compileall`, CLI `--help`, and `git diff --check`.
- [ ] Report the exact prepared-data and training commands for session `20260814_091058_791487`.

