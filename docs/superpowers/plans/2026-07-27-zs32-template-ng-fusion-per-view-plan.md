# ZS32 Template-NG Per-View Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the global Template gate stops a part, publish and display each view's own Template result as its Fusion result while PatchCore and YOLO remain skipped.

**Architecture:** The Demo runtime remains the authority for branch state. On a successful Template pass with at least one NG view, it emits per-view available Fusion evidence whose status mirrors that view's Template status. The Dashboard compositor keeps showing the real skipped state on PatchCore/YOLO layers but prioritizes the available Fusion status on the Fusion layer.

**Tech Stack:** Python, OpenCV, pytest, JSON runtime manifest.

## Global Constraints

- Any Template `NG_TEMPLATE` still immediately stops the whole part.
- PatchCore and YOLO must not run after the global Template gate triggers.
- The overall machine status remains `NG_TEMPLATE`.
- Template errors remain errors; they must not be converted into a PASS or NG Fusion decision.
- Preserve all unrelated changes in the dirty worktree and do not create a commit.

---

### Task 1: Publish per-view Fusion results after Template short-circuit

**Files:**
- Modify: `tests/unit/capture_data/test_zs32_demo_runtime.py`
- Modify: `capture_data/zs32_demo_runtime.py`

**Interfaces:**
- Consumes: each view's existing `branch_records[view]["template"]["status"]`.
- Produces: Fusion records with `state="available"`, per-view `status`, `score=None`, `threshold=None`, a reason, and a valid `evidence_path`.

- [x] **Step 1: Change the Template-NG regression test first**

Assert that the NG Template view has Fusion `NG_TEMPLATE`, other Template-PASS views have Fusion `PASS`, all Fusion branches are available with real evidence, and PatchCore/YOLO remain skipped and uncalled.

- [x] **Step 2: Run the focused test and confirm it fails on Fusion `SKIPPED`**

Run:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q \
  --confcutdir=tests/unit/capture_data \
  tests/unit/capture_data/test_zs32_demo_runtime.py::test_template_ng_globally_skips_patchcore_and_yolo
```

Expected: failure because the current runtime publishes Fusion as skipped.

- [x] **Step 3: Implement the minimum runtime change**

For a Template-NG run without Template errors, create `evidence/fusion/<view>.png` with `_fusion_evidence()` and publish Fusion as available with the corresponding Template status. Preserve the existing error path and branch counts.

- [x] **Step 4: Run the focused test and confirm it passes**

Run the command from Step 2. Expected: `1 passed`.

### Task 2: Make the Fusion page lead with the Fusion decision

**Files:**
- Modify: `tests/unit/zs32_refactor/dashboard/test_compositor.py`
- Modify: `src/zs32_inspection/dashboard/compositor.py`
- Modify: `AGENTS_MEMORY.md`
- Modify: `capture_data/AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: an available Fusion branch with `PASS` or `NG_TEMPLATE`.
- Produces: a Fusion-layer notice beginning with `Fusion: <status>`, while preserving truthful PatchCore/YOLO skipped details.

- [x] **Step 1: Add a failing compositor regression**

Build a view with Fusion available and PatchCore/YOLO skipped. Assert that the composed Fusion notice starts with the per-view Fusion result and still says the downstream branches were not executed.

- [x] **Step 2: Run the compositor test and confirm it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q \
  --confcutdir=tests/unit/zs32_refactor/dashboard \
  tests/unit/zs32_refactor/dashboard/test_compositor.py
```

- [x] **Step 3: Implement the notice ordering**

In the Fusion compositor, emit `Fusion: <status>` first for an available Fusion branch, followed by any PatchCore/YOLO unavailable notices.

- [x] **Step 4: Update folder memories and verify focused suites**

Run the runtime, parser, compositor, and render tests with their local `--confcutdir` settings. Run `git diff --check` on the touched files and inspect the focused diff.
