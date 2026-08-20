# BMW New Dataset ROI Redefinition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale BMW ROI with one selected on the new OK dataset and validate all 6 OK plus 7 NG images through the production detector.

**Architecture:** Keep the existing manual ROI selector and detector. Change only the selector's default reference and configuration state first; after the operator saves a new ROI, lock all 13 labeled images into the detector regression test and tune only generic JSON thresholds.

**Tech Stack:** Python, OpenCV, NumPy, pytest, uv.

**Implementation status (2026-08-05):** Complete. Operator ROI `[1872, 1180, 1953, 1793]` is saved; all 6 OK and
7 no-streak NG images pass their exact labels, and the focused BMW suite passes 34 tests.

## Global Constraints

- The one ROI must serve every 4024x3036 image under `dataset/bmw/OK` and `dataset/bmw/NG`.
- OK means a continuous bright streak; NG means the bright streak is completely absent.
- Do not branch on filenames or paths.
- Keep camera serial `DA9625347` and preserve `ERROR` as separate from part-level NG.
- Do not change ZS32 code or introduce deep learning.

---

### Task 1: Point ROI selection at the new reference and invalidate stale coordinates

**Files:**
- Modify: `pipeline/bmw_select_bright_streak_roi.py`
- Modify: `configs/bmw/bright_streak_demo.json`
- Test: `tests/unit/bmw_inspection/test_roi_selector.py`

**Interfaces:**
- Consumes: `select_roi(image_path, config_path, max_display_width=1280, max_display_height=720)`.
- Produces: a default selector command that opens `dataset/bmw/OK/Image_20260805172921398.bmp` and a config with `roi_xyxy=null` until Enter confirmation.

- [ ] **Step 1: Add a failing default-reference regression assertion**

Import the pipeline module by path and assert that `DEFAULT_IMAGE` resolves to the first new OK BMP and exists.

- [ ] **Step 2: Run the focused test and confirm the old default fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/test_roi_selector.py`

Expected: the default-reference assertion fails because the old root-level BMP no longer exists.

- [ ] **Step 3: Change the default image and clear the stale ROI**

Set `DEFAULT_IMAGE = REPO_ROOT / "dataset/bmw/OK/Image_20260805172921398.bmp"` and set `roi_xyxy` to JSON `null`. Preserve all camera and threshold fields.

- [ ] **Step 4: Run the focused selector test**

Expected: all selector tests pass.

- [ ] **Step 5: Operator selects the ROI**

Run:

```bash
uv run --no-sync python pipeline/bmw_select_bright_streak_roi.py
```

Draw one rectangle around the complete bright-streak inspection corridor and press Enter. The selector prints and atomically persists half-open source coordinates.

### Task 2: Lock and validate the complete labeled dataset

**Files:**
- Modify: `tests/unit/bmw_inspection/test_detector.py`
- Modify if required: `configs/bmw/bright_streak_demo.json`
- Update: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: the operator-selected `roi_xyxy` and `detect_bright_streak(image, config)`.
- Produces: one regression that enumerates every BMP below the OK and NG roots and enforces exact business labels.

- [ ] **Step 1: Replace the two obsolete image cases with dataset enumeration**

Assert there are exactly 6 OK and 7 NG BMPs, then require every OK result to equal `DemoStatus.OK` and every NG result to equal `DemoStatus.NG_NO_STREAK`.

- [ ] **Step 2: Run the detector test and record failures by filename, status, reason, and metrics**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/test_detector.py`

Expected: the test either passes or reports the generic threshold mismatches that require tuning.

- [ ] **Step 3: Tune only generic JSON thresholds if required**

Retain one shared configuration for all images. Do not add filename conditions, reference-image hashes, or per-image coordinates.

- [ ] **Step 4: Run the BMW acceptance suite**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q \
  tests/unit/bmw_inspection/test_contracts.py \
  tests/unit/bmw_inspection/test_roi_selector.py \
  tests/unit/bmw_inspection/test_detector.py \
  tests/unit/bmw_inspection/test_camera.py \
  tests/unit/bmw_inspection/test_demo_app.py
```

Expected: all BMW tests pass, including all 13 labeled BMPs and the synthetic `NG_BROKEN` case.

- [ ] **Step 5: Update project memory**

Record the final ROI coordinates, dataset counts, exact acceptance results, and the fact that real USB/MVS capture remains a site-only check.
