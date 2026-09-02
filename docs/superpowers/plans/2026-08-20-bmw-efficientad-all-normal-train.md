# BMW EfficientAD All-Normal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow an explicit laboratory mode where every current `normal` part trains EfficientAD while independent validation and threshold fitting remain pending.

**Architecture:** Preserve the prepared release's stratified splits for Template and YOLO. Add an opt-in flag to the ROI materializer and trainers that routes every `normal` row into each view's EfficientAD `normal` directory, disables Anomalib validation/test splitting during checkpoint training, and records that deployment thresholds require a future independent release.

**Tech Stack:** Python 3.13, pytest, OpenCV, Anomalib EfficientAD, argparse.

## Global Constraints

- Default behavior remains the existing stratified EfficientAD train/calibration/final-test layout.
- Only `source_class=normal` is promoted; defect classes are never used for EfficientAD weight training.
- Template and YOLO retain their existing split assignments.
- All-normal training produces checkpoints only and must not claim validation metrics or deployable thresholds.

---

### Task 1: Materialize the opt-in EfficientAD layout

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_training_data.py`
- Modify: `pipeline/bmw_lab_materialize_training_data.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_training_data.py`

**Interfaces:**
- Consumes: `materialize_training_data(..., efficientad_all_normal_train: bool = False)`.
- Produces: report fields `efficientad_all_normal_train` and `efficientad_normal_train_count_by_view`.

- [x] Add a failing test proving every normal split is linked under `efficientad/<view>/normal` only, while Template and YOLO splits remain unchanged.
- [x] Run the focused test and confirm the current calibration/held-out layout fails the new assertions.
- [x] Add the boolean argument, routing, report metadata, and CLI flag.
- [x] Run the complete materializer unit-test file.

### Task 2: Train checkpoints without internal validation leakage

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_train_all.py`
- Modify: `pipeline/bmw_lab_train_all.py`
- Modify: `src/bmw_inspection/lab/left_normal_training.py`
- Modify: `pipeline/bmw_lab_train_left_normal.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_train_all.py`
- Test: `tests/unit/bmw_inspection/lab/test_left_normal_training.py`

**Interfaces:**
- Consumes: `LabTrainingConfig.efficientad_all_normal_train` and `LeftNormalTrainingConfig.efficientad_all_normal_train`.
- Produces: Anomalib folder settings with no validation/test split, no `engine.test()` call, and metrics marked `pending_external_validation`.

- [x] Add failing tests for flag propagation, no-test data options, report semantics, and rejection of `--stage all` in all-normal mode.
- [x] Run the focused tests and confirm failure for the missing option.
- [x] Implement the minimal config, CLI, materialization propagation, and checkpoint-only training behavior.
- [x] Run focused training-orchestration tests, compile changed Python files, and run `git diff --check`.
