# BMW Eight-View ROI and Training Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select eight dataset-bound ROIs and materialize immutable EfficientAD, Template, and truth-preserving YOLO datasets.

**Architecture:** Add one typed ROI/config module and one training-data materializer beside the standalone eight-view preparer, with thin pipeline CLIs. Canonical crops are written once and branch layouts reference them through relative symlinks.

**Tech Stack:** Python, OpenCV, CSV/JSON/hashlib, pathlib symlinks, pytest, uv.

## Global Constraints

- Preserve the legacy six-view BMW runtime and enums.
- Bind ROI bytes to the prepared dataset manifest SHA-256.
- Never infer visible defects or YOLO boxes from a part-level folder.
- Refuse overwrite except explicit atomic ROI config replacement.

---

### Task 1: ROI contract and selector

**Files:**
- Create: `src/bmw_inspection/lab/eight_view_roi.py`
- Create: `pipeline/bmw_lab_select_eight_view_rois.py`
- Create: `tests/unit/bmw_inspection/lab/test_eight_view_roi.py`

- [ ] Write failing tests for representative-sample selection, display-to-source coordinate mapping, exact eight-view ROI validation, manifest hash binding, and overwrite behavior.
- [ ] Run the focused test and confirm RED.
- [ ] Implement the typed ROI config, atomic writer, pure mapping helpers, and GUI wrapper.
- [ ] Re-run the focused test and confirm GREEN.

### Task 2: Canonical crops and branch layouts

**Files:**
- Create: `src/bmw_inspection/lab/eight_view_training_data.py`
- Create: `pipeline/bmw_lab_materialize_training_data.py`
- Create: `tests/unit/bmw_inspection/lab/test_eight_view_training_data.py`

- [ ] Write failing tests for exact crop pixels, source hash checks, branch-good `no_streak`, Template reference isolation, pending YOLO publication, complete YOLO labels, and no-overwrite publication.
- [ ] Run the focused test and confirm RED.
- [ ] Implement single-copy crops, relative branch symlinks, YOLO validation, reports, and atomic publication.
- [ ] Re-run the focused and adjacent BMW tests.

### Task 3: Documentation and real-data handoff

**Files:**
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`

- [ ] Document exact ROI selection and materialization commands.
- [ ] Run CLI help, compileall, focused tests, diff check, and a synthetic end-to-end materialization smoke.
- [ ] Hand off the GUI ROI command; after the user selects ROI, run the real materializer without reimplementing code.
