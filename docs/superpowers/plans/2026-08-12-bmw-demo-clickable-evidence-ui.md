# BMW Demo Clickable Evidence UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add clickable algorithm/view navigation and a same-coordinate trusted-OK versus current-evidence detail page to the BMW eight-view OpenCV Demo.

**Architecture:** Keep the immutable inspection result unchanged. Add pure UI selection/hit-test/detail-image helpers in the renderer module, and let the OpenCV loop translate queued mouse releases and keyboard input into those state transitions.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, Pillow, pytest, uv.

## Global Constraints

- Do not modify detector/model/threshold/config/fusion behavior or files.
- Do not stage unrelated EfficientAD or bright-streak files already present in the worktree.
- Keep the 1600×900 all-Chinese Swiss-grid canvas and existing keyboard shortcuts.
- Trusted OK is diagnostic only and is shown for NG/ERROR; PASS does not trigger new matching.
- Detail pairs must use the same view, algorithm coordinate domain, ROI, orientation, and fit rule.

---

### Task 1: Pure UI state, hit testing, and detail rendering

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo_ui.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py`

**Interfaces:**
- Produces `DemoUiPage`, `DashboardHit`, `dashboard_hit_test(x, y)`, `select_branch(state, branch)`, `apply_dashboard_click(state, hit)`, `evidence_detail_images(state)`, and `render_eight_view_screen(state)`.
- `render_eight_view_screen` dispatches to the existing dashboard or the new detail renderer without changing `EightViewInspection`.

- [ ] Write failing tests for algorithm-card ERROR/NG-first selection, all-PASS fallback, view-card selection, empty panel no-op, and selected borders.
- [ ] Run `uv run --no-sync pytest tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py -q` and confirm failures are caused by missing interfaces.
- [ ] Implement minimal frozen UI page state, stable canvas hit regions, and shared mouse/keyboard selection helpers.
- [ ] Write failing tests for Template, YOLO, EfficientAD, and clockwise-rotated bright-streak detail pairs, plus PASS and missing-reference messages.
- [ ] Implement the 1600×900 two-column detail renderer using original arrays and the existing `_fit_image` helper.
- [ ] Re-run the UI test file and require all tests pass.

### Task 2: Mouse queue and keyboard page behavior

**Files:**
- Modify: `pipeline/bmw_lab_eight_view_demo.py`
- Test: `tests/unit/pipeline/test_bmw_lab_eight_view_demo.py`

**Interfaces:**
- Consumes `dashboard_hit_test`, `apply_dashboard_click`, `DemoUiPage`, and `render_eight_view_screen` from Task 1.
- Produces a left-button-release queue, logical-canvas coordinate mapping, and deterministic key handling for detail Esc/main Esc/Q/R.

- [ ] Write failing tests for 1440×810 and resized-window coordinate conversion, queued left-click consumption, detail Esc returning to main, main Esc no-op, and Q exit.
- [ ] Run `uv run --no-sync pytest tests/unit/pipeline/test_bmw_lab_eight_view_demo.py -q` and confirm expected failures.
- [ ] Register `cv2.setMouseCallback`, consume mouse releases in the main loop, render through `render_eight_view_screen`, and enforce the page-specific keyboard rules.
- [ ] Re-run both UI test files and require all tests pass.
- [ ] Run `uv run --no-sync python -m compileall -q src/bmw_inspection/lab/eight_view_demo_ui.py pipeline/bmw_lab_eight_view_demo.py`, `git diff --check`, and verify the UI diff contains no algorithm/config files.
- [ ] Stage exactly the two implementation files and two test files, then commit with message `feat: add clickable BMW evidence UI`.
