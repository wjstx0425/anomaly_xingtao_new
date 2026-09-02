# BMW Bilateral Template and EfficientAD Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one command that sequentially trains right- and left-hand Template models plus eight-view EfficientAD checkpoints, with every normal part used for EfficientAD training.

**Architecture:** Generalize the existing left normal-only training configuration with an explicit `capture_scope`, preserving `left` as its default. Add a small bilateral orchestrator that constructs one right and one left configuration with `efficientad_all_normal_train=True`, runs both with `stage="train"`, and stops on the first error. A thin CLI supplies the current 0820 prepared releases and ROI files as defaults.

**Tech Stack:** Python 3.13, argparse, dataclasses, pytest, Anomalib EfficientAD.

## Global Constraints

- Train only Template and EfficientAD; do not invoke YOLO or bright-streak stages.
- EfficientAD must use every `source_class=normal` row and remain checkpoint-only pending external validation.
- Run right then left sequentially on one GPU and stop immediately on failure.
- Preserve immutable no-overwrite training and result directories.
- Keep the existing left-only command compatible by defaulting `capture_scope` to `left`.

---

### Task 1: Generalize hand identity and add the bilateral runner

**Files:**
- Modify: `src/bmw_inspection/lab/left_normal_training.py`
- Create: `src/bmw_inspection/lab/bilateral_normal_training.py`
- Create: `pipeline/bmw_lab_train_bilateral_normal.py`
- Modify: `pipeline/bmw_lab_train_left_normal.py`
- Create: `tests/unit/bmw_inspection/lab/test_bilateral_normal_training.py`
- Modify: `tests/unit/bmw_inspection/lab/test_left_normal_training.py`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: `run_left_normal_training(config, dry_run=..., stage="train")` and the existing all-normal materializer/trainer behavior.
- Produces: `BilateralNormalTrainingConfig`, `build_hand_configs(config)`, `run_bilateral_normal_training(config, dry_run=False, runner=...)`, and CLI `pipeline/bmw_lab_train_bilateral_normal.py`.

- [x] **Step 1: Write failing hand-identity and bilateral orchestration tests**

Add tests that assert a right-scoped normal configuration passes right report/ROI identities, both generated configs force `efficientad_all_normal_train=True`, execution order is `right,left`, and a right-side exception prevents the left runner call.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q \
  tests/unit/bmw_inspection/lab/test_bilateral_normal_training.py \
  tests/unit/bmw_inspection/lab/test_left_normal_training.py -k 'bilateral or right_capture_scope'
```

Expected: collection/import failure for the missing bilateral module and failure for the missing `capture_scope` field.

- [x] **Step 3: Implement the minimal runner and CLI**

Add `capture_scope: str = "left"` to `LeftNormalTrainingConfig`; validate it as `left|right` and compare both prepared-report and ROI `capture_scope` against it. Implement bilateral defaults for `bmw_right_0820_v1` and `bmw_left_0820_v1`, construct both hand configs with the all-normal flag forced true, and call the existing training runner sequentially with `stage="train"`.

- [x] **Step 4: Run focused and compatibility verification**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync pytest -q \
  tests/unit/bmw_inspection/lab/test_bilateral_normal_training.py \
  tests/unit/bmw_inspection/lab/test_left_normal_training.py \
  tests/unit/bmw_inspection/lab/test_eight_view_training_data.py \
  tests/unit/bmw_inspection/lab/test_eight_view_train_all.py
```

Expected: all tests pass.

- [x] **Step 5: Verify the real 0820 dry-run and code health**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --no-sync python \
  pipeline/bmw_lab_train_bilateral_normal.py --dry-run
```

Expected: right and left plans both report `dry_run`, `stage=train`, and `all_normal_train=true`; no output directories are created. Then run `py_compile` on the changed Python files and `git diff --check` on this task's paths.

- [x] **Step 6: Record the operational contract**

Append the exact command, default release identities, sequential/fail-fast behavior, and external-validation requirement to `AGENTS_MEMORY.md`.
