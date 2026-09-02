# BMW Left Front Threshold Tightening Implementation Plan

> **For agentic workers:** Execute inline with TDD; preserve unrelated worktree changes and do not commit or push.

**Goal:** Make the confirmed left-hand small-foot deformation fail Template and EfficientAD in `front`, `front_left`, and `front_right`.

**Architecture:** Change only the six deployment thresholds in the active left 0820 mixed JSON config. Lock the values through the existing config contract test, then replay the saved HDR capture `bmw_demo_20260821_192534` through the complete 25-check pipeline.

**Tech Stack:** JSON configuration, Python, pytest, uv.

## Global Constraints

- Preserve all model weights, ROI coordinates, masks, YOLO settings, bright-streak settings, the remaining five views, and fusion rules.
- Template thresholds: `front=0.015`, `front_left=0.011`, `front_right=0.008`.
- EfficientAD thresholds: `front=0.33`, `front_left=0.36`, `front_right=0.37`.
- Do not commit, push, reset, clean, or overwrite the original saved capture.

---

### Task 1: Lock and apply the six thresholds

**Files:**
- Modify: `tests/unit/bmw_inspection/lab/test_bmw_0820_mixed_configs.py`
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_left_0820_mixed_v1.json`

- [x] Add exact left-hand Template and EfficientAD threshold assertions to the config contract test.
- [x] Run the focused test and confirm it fails on the previous values.
- [x] Change only the six deployment threshold values and update their existing source/status labels to identify this single field-deformation anchor.
- [x] Run the focused config tests and active model/config tests; expect PASS.

### Task 2: Replay the confirmed deformation capture

**Files:**
- Read: `results/bmw_lab_one_click/bmw_eight_view_demo_left_0820_mixed_v1/bmw_demo_20260821_192534/images/*_hdr.png`
- Create: a temporary `/tmp` capture-set made only of symlinks
- Modify: `AGENTS_MEMORY.md`

- [x] Replay the eight saved HDR images through the active left mixed config under a new capture ID.
- [x] Confirm 25 checks, 0 ERROR, and NG for all six targeted Template/EfficientAD rows.
- [x] Confirm the other 19 branch/view statuses and scores remain unchanged.
- [x] Update `AGENTS_MEMORY.md`, run `git diff --check`, and report CPU/GPU/camera verification boundaries.
