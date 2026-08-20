# ZS32 v14 Demo Model Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the selected v14 Template and eight-view PatchCore artifacts in the active ZS32 Demo configuration.

**Architecture:** Change only the model paths and their paired deployment thresholds in the existing Demo JSON. Preserve topology, ROI, YOLO, and eight-process execution.

**Tech Stack:** JSON, Python, pytest, uv

## Global Constraints

- Template thresholds must equal the v14 `model.json` `high_threshold` values.
- PatchCore thresholds must equal the v14 `eight_view_summary.csv` `deploy_threshold` values.
- All eight views and all referenced files must exist.
- YOLO, ROI, topology, and `patchcore_process_count=8` must not change.

---

### Task 1: Replace the active artifacts and verify the production contract

**Files:**
- Modify: `configs/zs32/zs32_demo.json`
- Modify: `tests/unit/capture_data/test_zs32_demo_config.py`
- Modify: `pipeline/AGENTS_MEMORY.md`
- Modify: `capture_data/AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: v14 Template `model.json` and PatchCore `eight_view_summary.csv`
- Produces: one production Demo configuration with eight v14 checkpoint paths and paired thresholds

- [x] **Step 1: Extend the repository-config regression assertion**

Assert that the loaded Template root is the selected v14 directory, all
PatchCore parents belong to the selected v14 root, and the loaded thresholds
equal the published v14 values.

- [x] **Step 2: Run the focused test and confirm it fails**

Run:
`UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/capture_data/test_zs32_demo_config.py`

Expected: the new repository-config assertions fail against the old v13/v11
configuration.

- [x] **Step 3: Update the production JSON**

Replace `models.template_dir`, all eight `models.patchcore` paths, all eight
Template thresholds, and all eight PatchCore thresholds from the authoritative
v14 artifacts. Leave every other field byte-for-byte equivalent.

- [x] **Step 4: Verify artifact identity and configuration loading**

Check Template `model.sha256`, all 40 declared Template PNG hashes, eight
checkpoint paths, JSON parsing, and the production Demo configuration loader.

- [x] **Step 5: Run focused regression tests**

Run:
`UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/capture_data/test_zs32_demo_config.py tests/unit/capture_data/test_zs32_demo_runtime.py`

Expected: all selected tests pass.

- [x] **Step 6: Update folder memory and close the plan**

Record the deployed v14 roots, threshold sources, unchanged fields, and rollback
roots in both relevant folder memory files, then mark all plan steps complete.
