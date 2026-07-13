# ZS32 PatchCore ROI Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stage 30 to select 12 hand/view-specific ROIs and safely generate cropped `dataset/right` and `dataset/left` trees for PatchCore retraining.

**Architecture:** Put discovery, identity validation, ROI config, preflight, cropping, and reporting in `capture_data/zs32_patchcore_roi_dataset.py`. Keep `pipeline/30_crop_zs32_patchcore_dataset.py` as a thin `select`/`convert` wrapper. Reuse `capture_data.select_roi` while leaving stage 29 and YOLO transformation unchanged.

**Tech Stack:** Python 3.13, OpenCV, CSV/JSON, pathlib, pytest, Ruff, uv.

## Global Constraints

- Inputs default to `dataset/right` and `dataset/left`; never modify either source tree.
- Output defaults to `dataset/zs32_patchcore_roi` and preserves `hand/view/{normal,normal_test,defect}/...`.
- Configure exactly 12 independent pixel half-open `xyxy` ROIs: two hands times six canonical views.
- Correct filename-view versus parent-view mismatches into the filename-derived view and record them; reject filename hand mismatches.
- Default to no overwrite; `--overwrite` may replace output only after complete preflight succeeds.
- Do not modify stage 29 behavior or YOLO images/labels.
- Use `uv`; offline tests must not require GUI interaction, network, cameras, or GPU.

---

### Task 1: Strict ROI configuration and PatchCore image discovery

**Files:**
- Create: `capture_data/zs32_patchcore_roi_dataset.py`
- Create: `tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py`

**Interfaces:**
- Produces: `HANDS`, `VIEWS`, `IMAGE_EXTENSIONS`, `ROI`, `SourceImage`, `load_patchcore_roi_config(path: Path)`, and `discover_patchcore_images(dataset_root: Path)`.
- Consumed by Tasks 2-3: validated `{hand: {view: roi}}` mapping and stable `SourceImage` records.

- [x] **Step 1: Write failing strict-config tests**

Create a complete 12-ROI JSON fixture and assert:

```python
width, height, rois, payload = load_patchcore_roi_config(config_path)
assert (width, height) == (40, 30)
assert rois["right"]["front"] == (1, 2, 20, 25)
assert set(rois) == {"right", "left"}
```

Parametrize missing/extra hands and views, bad schema/coordinate system, non-integer values including booleans, and out-of-bounds ROIs.

- [x] **Step 2: Run config tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --no-sync pytest -q \
  tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py -k config
```

Expected: import failure because the new module is absent.

- [x] **Step 3: Implement strict config loading**

Use these exact constants and reject `bool` as an integer value:

```python
HANDS = ("right", "left")
VIEWS = ("front", "front_left", "front_right", "back", "back_left", "back_right")
IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff")
ROI = tuple[int, int, int, int]
```

Return `(width, height, rois, payload)` without mutating the JSON payload.

- [x] **Step 4: Run config tests and verify GREEN**

Run Step 2 again. Expected: all config tests pass.

- [x] **Step 5: Write failing discovery/identity tests**

Require a frozen `SourceImage` with:

```python
source_path: Path
hand: str
source_view: str
resolved_view: str
label: str
defect_type: str
session_id: str
relative_tail: Path
view_corrected: bool
```

Test stable ordering, case-insensitive extensions, longest prefix matching, front/back correction, unknown prefixes, hand mismatch, missing hand/view/normal data, and target collisions after correction.

- [x] **Step 6: Run discovery tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --no-sync pytest -q \
  tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py -k discovery
```

- [x] **Step 7: Implement discovery and destination mapping**

Match filename prefixes using views sorted by descending length. For corrected views, replace only the first hand-relative path component, retain label/type/session/images/basename, and reject duplicate targets before conversion.

- [x] **Step 8: Run all Task 1 tests**

Run the full new test file. Expected: strict config and discovery tests pass offline.

### Task 2: Interactive 12-ROI selection

**Files:**
- Modify: `capture_data/zs32_patchcore_roi_dataset.py`
- Modify: `tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py`

**Interfaces:**
- Consumes: `discover_patchcore_images` and optional existing config.
- Produces: `select_patchcore_rois(repo_root: Path, dataset_root: Path, config_path: Path, preview_dir: Path, max_window_width: int = 1600, max_window_height: int = 1000) -> dict[str, object]`.

- [x] **Step 1: Write a failing selection test**

Monkeypatch `select_roi` and `save_overlay`; create one normal image per hand/view. Assert fixed right-then-left order, existing initial ROI reuse, preview names `<hand>_<view>_roi.png`, repo-relative references, and a complete 12-entry JSON payload.

- [x] **Step 2: Run the selection test and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --no-sync pytest -q \
  tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py -k selection
```

- [x] **Step 3: Implement reference selection and atomic config writing**

Choose the first stable normal image for each hand/view, require a common source size, reuse `select_roi`/`save_overlay`, and write `<config>.tmp` before atomic replacement only after all 12 selections succeed.

- [x] **Step 4: Run selection and full module tests GREEN**

Run the selection tests, then the complete new test file.

### Task 3: Fail-closed preflight and transactional conversion

**Files:**
- Modify: `capture_data/zs32_patchcore_roi_dataset.py`
- Modify: `tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py`

**Interfaces:**
- Consumes: validated config and `SourceImage` records.
- Produces: `crop_patchcore_dataset(repo_root: Path, dataset_root: Path, output_root: Path, roi_config: Path, overwrite: bool = False) -> dict[str, object]`, `crop_manifest.csv`, `roi_config.json`, and `summary.json`.

- [x] **Step 1: Write failing protection tests**

Cover unreadable images, size mismatch, unsafe output paths, existing output without overwrite, collision, and invalid source with `overwrite=True` preserving an existing sentinel.

- [x] **Step 2: Run protection tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --no-sync pytest -q \
  tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py -k 'preflight or protection'
```

- [x] **Step 3: Implement complete read-only preflight**

Resolve safe boundaries, read every source, validate dimensions/ROIs, calculate crop shapes, and build prepared rows without creating, deleting, or changing output.

- [x] **Step 4: Write failing conversion/manifest tests**

Use distinct pixel patterns and require exact crop pixels, different hand/view sizes, preserved hierarchy/extensions, corrected path metadata, untouched sources, one complete manifest row per source, config snapshot, and exact summary counts.

- [x] **Step 5: Run conversion tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --no-sync pytest -q \
  tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py -k 'conversion or manifest or summary'
```

- [x] **Step 6: Implement transactional conversion**

Write to a sibling temporary directory after preflight. Use PNG compression `1` only for PNG. Write metadata last. For overwrite, preserve the old output as a backup until atomic replacement succeeds, and restore it if replacement fails.

- [x] **Step 7: Run all Task 3 tests GREEN**

Run the entire new test file and confirm source/sentinel assertions remain intact.

### Task 4: Stage 30 CLI, documentation, and verification

**Files:**
- Create: `pipeline/30_crop_zs32_patchcore_dataset.py`
- Modify: `tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py`
- Modify: `pipeline/README.md`
- Modify: `CHANGELOG.md`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: core selection/conversion functions.
- Produces: documented `select` and `convert` commands with repository-local defaults.

- [x] **Step 1: Write failing CLI parser tests**

Assert `select` defaults to `dataset`, `dataset/zs32_patchcore_roi_config.json`, and `dataset/zs32_patchcore_roi_previews`; assert `convert` defaults to the same dataset/config, `dataset/zs32_patchcore_roi`, and `overwrite=False`.

- [x] **Step 2: Run parser tests and verify RED**

Expected: stage 30 wrapper is absent.

- [x] **Step 3: Implement the thin stage 30 wrapper**

Follow stage 29 bootstrap style. `select` prints all 12 coordinates. `convert` prints output/manifest paths, total images, corrected-view count, and per-hand/view totals.

- [x] **Step 4: Run parser tests and CLI help GREEN**

```bash
uv run --offline --no-sync python pipeline/30_crop_zs32_patchcore_dataset.py --help
uv run --offline --no-sync python pipeline/30_crop_zs32_patchcore_dataset.py select --help
uv run --offline --no-sync python pipeline/30_crop_zs32_patchcore_dataset.py convert --help
```

- [x] **Step 5: Update operational documentation**

Add stage 30 and Chinese copy-paste commands to `pipeline/README.md`, including 12 manual ROIs, preservation, output layout, overwrite, manifest, and PatchCore training roots. Add an Unreleased changelog entry and update `AGENTS_MEMORY.md` with actual verification status.

- [x] **Step 6: Run synthetic conversion coverage**

Create a complete tiny 12-view source/config fixture under `/tmp`, invoke `convert`, and assert cropped images plus manifest/config/summary. Do not invoke GUI during automation.

- [x] **Step 7: Run final repository verification**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --no-sync pytest -q \
  tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py \
  tests/unit/capture_data/test_zs32_view_roi_dataset.py \
  tests/unit/pipeline/test_pipeline_wrappers.py
uv run --offline --no-sync python -m py_compile \
  capture_data/zs32_patchcore_roi_dataset.py \
  pipeline/30_crop_zs32_patchcore_dataset.py \
  tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py
uv run --offline --no-sync ruff check --select F,I \
  capture_data/zs32_patchcore_roi_dataset.py \
  pipeline/30_crop_zs32_patchcore_dataset.py \
  tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py
git diff --check
```

- [x] **Step 8: Run real-data read-only discovery**

Report actual right/left image totals, source sizes, missing required views, and corrected-view count without creating output. Do not guess left-hand ROI coordinates or run conversion before the user completes `select`.

### Task 5: Minimal visible crop progress

**Files:**
- Modify: `capture_data/zs32_patchcore_roi_dataset.py`
- Modify: `tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py`
- Modify: `pipeline/README.md`

**Interface:** Keep `crop_patchcore_dataset(...)` unchanged; display one Rich progress bar while writing prepared images.

- [x] **Step 1: Write a failing test**

Monkeypatch module-level `track`, run a 12-image conversion fixture, and assert it receives all prepared images with `description="Cropping images"`.

- [x] **Step 2: Verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --no-sync pytest -q \
  tests/unit/capture_data/test_zs32_patchcore_roi_dataset.py -k visible_progress
```

Expected: failure because the PatchCore module does not import or call `track`.

- [x] **Step 3: Implement the minimal progress bar**

Import `track` from `rich.progress` and replace only the conversion loop with `for item in track(prepared, description="Cropping images")`.

- [x] **Step 4: Verify GREEN and update docs**

Run the focused test, full stage-30 test file, Ruff F/I, `py_compile`, and `git diff --check`. Add one README sentence stating that `convert` displays progress.
