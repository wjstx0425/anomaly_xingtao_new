# BMW EfficientAD Expand Manual Ignore Mask Implementation Plan

> **For agentic workers:** Execute inline in the existing isolated worktree; do not dispatch subagents for this small task.

**Goal:** Preload the SHA-verified v1 manual ignore polygons, let the operator add or redraw regions, and publish a no-overwrite v2 asset.

**Architecture:** Extend only the existing OpenCV selector. Reconstruct every seeded mask from the saved polygons and require exact equality with the SHA-verified v1 mask before opening the GUI. Keep v1 and the current V4 config unchanged until v2 selection finishes.

**Tech Stack:** Python, OpenCV, NumPy, pytest, uv.

## Global Constraints

- `0=inspect`, `255=ignore`.
- Existing polygons are preloaded; `S` keeps them, new polygons extend their union, `R` clears the current view.
- Publish to `bmw_right_manual_ignore_v2`; never overwrite v1.

### Task 1: Seeded selector

**Files:**
- Modify: `pipeline/bmw_lab_select_efficientad_ignore_masks.py`
- Test: `tests/unit/pipeline/test_bmw_lab_select_efficientad_ignore_masks.py`

- [ ] Add a failing test for v1 defaults and exact polygon-to-mask reconstruction.
- [ ] Run the focused test and confirm failure because seed loading is absent.
- [ ] Add `--from-index`, v2 output default, seed validation, and initial polygon display.
- [ ] Run the focused selector and mask-asset tests.
- [ ] Start the GUI on `DISPLAY=:1`; publish only after all eight views are confirmed.
