# BMW Template 40-Candidate Review Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a deterministic, manually reviewable 40-candidate Template package for every left/right BMW view without training or changing active Demo configuration.

**Architecture:** Add one small library that loads the two prepared manifests and ROI files, ranks normal train crops with deterministic diversity plus physical-part coverage, and writes candidate crops, contact sheets, a CSV, and instructions. Add one thin CLI with concrete 0823 defaults. Keep the existing Template training and runtime untouched until manual review is complete.

**Tech Stack:** Python 3.13, OpenCV, NumPy, csv/json, pytest, uv.

## Global Constraints

- Work directly in `/home/yunjing/anomaly_xingtao_new/.worktrees/bmw-eight-view-handoff`.
- Use both left and right 0823 prepared datasets.
- Select exactly 40 candidates per hand and canonical view from train/normal/OK rows.
- Deleted review copies are rejected and are never automatically refilled.
- Do not train models or modify active configs in this phase.
- Do not add SHA, schema-version, receipt, provenance, publisher, rebind, or immutable-release logic.
- Preserve all unrelated dirty-worktree changes.

---

### Task 1: Deterministic candidate selection and package writer

**Files:**
- Create: `src/bmw_inspection/lab/template_review.py`
- Test: `tests/unit/bmw_inspection/lab/test_template_review.py`

**Interfaces:**
- Consumes: prepared manifest CSV, public ROI JSON, hand name, and candidate count.
- Produces: `build_template_review_package(specs, output_root, candidate_count=40) -> Path` and a complete review directory.

- [ ] **Step 1: Write failing tests**

Create fixtures with two hands, eight views, more than 40 train/normal/OK rows, duplicate physical parts across sessions, and synthetic readable images. Assert exactly 40 candidates per hand/view, deterministic ordering, new-part coverage before repeated parts, correct ROI dimensions, and exclusion of calibration/final_test/defect rows.

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw-template-review-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/bmw_inspection/lab/test_template_review.py
```

Expected: fail because `bmw_inspection.lab.template_review` does not exist.

- [ ] **Step 3: Implement minimal library**

Implement `ReviewSource`, `ReviewCandidate`, manifest/ROI parsing, image reads, ROI crops, deterministic center-plus-farthest selection, physical-part-first eligibility, 5×8 contact sheets, CSV output, and Chinese review instructions. Refuse an existing output directory and insufficient eligible rows; do not hash files or mutate sources.

- [ ] **Step 4: Run GREEN**

Run the Task 1 command again. Expected: all tests pass.

### Task 2: 0823 CLI and real package generation

**Files:**
- Create: `pipeline/bmw_lab_prepare_template_review.py`
- Create: `tests/unit/pipeline/test_bmw_lab_prepare_template_review.py`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: optional CLI overrides with concrete left/right 0823 defaults.
- Produces: `dataset/bmw_lab_labeling/bmw_template_40_review_0823_v1` and a JSON terminal summary.

- [ ] **Step 1: Write failing CLI test**

Load the script with `runpy`, point it at temporary manifests/ROIs/output, and assert a zero exit code plus the expected summary; also assert the parser default count is 40.

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw-template-review-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/pipeline/test_bmw_lab_prepare_template_review.py
```

Expected: fail because the CLI does not exist.

- [ ] **Step 3: Implement minimal CLI**

Add arguments for both manifests, both ROI configs, output root, and candidate count. Use the fixed 0823 paths as defaults, print the generated root and counts, and return code 2 with a concise error message on invalid data.

- [ ] **Step 4: Run focused tests and syntax checks**

Run both new test files, compile both new Python files, and run `git diff --check`. Expected: tests pass and all commands return zero.

- [ ] **Step 5: Generate and verify the real review package**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw-template-review-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python pipeline/bmw_lab_prepare_template_review.py
```

Then verify 640 readable candidate PNGs, 16 readable contact sheets, per-view ROI dimensions, manifest membership, and unchanged active Demo configs. Record the generated path and review procedure in `AGENTS_MEMORY.md`.

## Self-review

- Spec coverage: both hands, eight views, 40 candidates, two-session identity, delete-to-reject, no refill, and no active integration are covered.
- Placeholder scan: no TBD/TODO/“similar to” steps remain.
- Type consistency: the CLI calls the exact `build_template_review_package` interface defined in Task 1.

