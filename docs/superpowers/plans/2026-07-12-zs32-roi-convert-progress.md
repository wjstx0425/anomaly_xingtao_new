# ZS32 ROI Conversion Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add visible `Preflight` and `Cropping` progress bars to stage-29 ROI dataset conversion without changing its command or output semantics.

**Architecture:** Wrap the two existing manifest-sized loops with `rich.progress.track`. Keep progress presentation inside `capture_data/zs32_view_roi_dataset.py`; the numbered wrapper remains unchanged because the convert function already owns both loops.

**Tech Stack:** Python 3.13, Rich, pytest, Ruff, uv

## Global Constraints

- Keep `uv run --no-sync python pipeline/29_zs32_fixed_roi.py convert --overwrite` unchanged.
- Show separate `Preflight` and `Cropping` progress bars with totals equal to manifest row count.
- Preserve ROI clipping, dropped-box reporting, split manifest, and dataset output behavior.
- Do not rewrite the 11GB production output during automated verification.

---

### Task 1: Add Two-Phase Conversion Progress

**Files:**
- Modify: `capture_data/zs32_view_roi_dataset.py`
- Modify: `tests/unit/capture_data/test_zs32_view_roi_dataset.py`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: `_preflight(rows, repo_root, image_width, image_height, rois)` and `crop_zs32_yolo_dataset(...)`.
- Produces: calls `rich.progress.track(sequence, description=..., total=len(sequence))` once for each phase.

- [ ] **Step 1: Write the failing test**

Add a test that replaces module-level `track`, runs the existing two-image synthetic conversion, and asserts:

```python
assert progress_calls == [("Preflight", 2), ("Cropping", 2)]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-sync pytest \
  tests/unit/capture_data/test_zs32_view_roi_dataset.py::test_crop_dataset_reports_two_progress_phases -q
```

Expected: fail because the module does not expose or call `track`.

- [ ] **Step 3: Implement the minimal progress wrappers**

Import `track` from `rich.progress`, then change the loops to:

```python
for row in track(rows, description="Preflight", total=len(rows)):
    ...

for row, image_path, transformed, row_clipped, row_dropped in track(
    prepared,
    description="Cropping",
    total=len(prepared),
):
    ...
```

- [ ] **Step 4: Run GREEN and repository checks**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache \
  uv run --no-sync pytest \
  tests/unit/capture_data/test_zs32_view_roi_dataset.py \
  tests/unit/capture_data/test_prepare_zs32_yolo_dataset.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync ruff check \
  capture_data/zs32_view_roi_dataset.py \
  tests/unit/capture_data/test_zs32_view_roi_dataset.py
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m compileall -q \
  capture_data/zs32_view_roi_dataset.py pipeline/29_zs32_fixed_roi.py
```

Expected: all tests pass, Ruff exits 0, compileall exits 0.

- [ ] **Step 5: Record the behavior and commit**

Update `AGENTS_MEMORY.md` with the two progress phase names and verification result, then commit only the progress-related source, tests, docs, and memory changes:

```bash
git add capture_data/zs32_view_roi_dataset.py \
  tests/unit/capture_data/test_zs32_view_roi_dataset.py \
  docs/superpowers/plans/2026-07-12-zs32-roi-convert-progress.md \
  AGENTS_MEMORY.md
git commit -m "feat: show ZS32 ROI conversion progress"
```
