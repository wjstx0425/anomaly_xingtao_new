# BMW YOLO Fixed Ignore Region Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Filter the fixed `front_secondary` fixture detection from YOLO business decisions while retaining diagnostic evidence.

**Architecture:** Extend the existing YOLO config with an optional per-view rectangle mapping. Pass the validated immutable mapping into `EightViewYoloPredictor`, which classifies boxes whose centers fall inside configured rectangles as ignored before computing final count and decision score.

**Tech Stack:** Python 3.13, dataclasses, OpenCV, pytest, uv.

## Global Constraints

- Ignore rectangle: `front_secondary: [1580, 450, 1756, 800]`.
- Do not change the shared ROI, model checkpoint, or non-YOLO branches.
- Preserve ignored boxes in diagnostic details and overlays.
- Keep all existing configurations backward compatible.

---

### Task 1: Configuration and runtime filtering

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_left_normal_20260814_v1.json`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

**Interfaces:**
- Consumes: optional JSON `yolo.ignore_regions`, mapping view names to lists of `[x1, y1, x2, y2]` integer rectangles.
- Produces: `EightViewDemoConfig.yolo_ignore_regions` and `EightViewYoloPredictor(..., ignore_regions=...)`.

- [ ] **Step 1: Write failing tests** for loader validation and predictor filtering, including the real fixture coordinates.
- [ ] **Step 2: Run focused tests** and verify failure is caused by the missing `ignore_regions` interface.
- [ ] **Step 3: Implement minimal parsing and center-in-rectangle filtering**, retaining ignored boxes in `details` and overlay.
- [ ] **Step 4: Add the approved rectangle to the left Demo config.**
- [ ] **Step 5: Run focused tests and saved-image replay** and confirm the fixed fixture becomes YOLO PASS.
- [ ] **Step 6: Update `AGENTS_MEMORY.md`** with the exact config, rectangle and verified behavior.
