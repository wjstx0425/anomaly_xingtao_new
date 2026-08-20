# BMW Eight-View Data Preparer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an immutable BMW eight-view HDR raw-capture index with physical-part splits and truthful per-branch labels.

**Architecture:** Add a standalone eight-view dataset module beside the legacy six-view BMW laboratory package, plus a thin pipeline CLI. Parse the append-only capture CSVs into typed immutable rows, validate the fixed topology, split physical parts deterministically within source classes, then atomically publish manifests and a receipt without copying images.

**Tech Stack:** Python 3.10+, standard-library CSV/dataclasses/hashlib/tempfile, OpenCV image validation, pytest, uv.

## Global Constraints

- Do not modify the legacy six-view `ViewId` or runtime contract.
- Treat `no_streak` as business NG, EfficientAD/Template branch-normal, YOLO negative, and front-left bright-streak NG.
- Never infer per-view defects or YOLO boxes from a part-level defect directory.
- Keep raw images read-only and refuse release overwrite.

---

### Task 1: Strict raw-capture parser and split contract

**Files:**
- Create: `src/bmw_inspection/lab/eight_view_dataset.py`
- Create: `tests/unit/bmw_inspection/lab/test_eight_view_dataset.py`

**Interfaces:**
- Produces: `VIEW_ORDER`, `PreparedImage`, `read_complete_capture_rows(raw_root)`, `assign_stratified_part_splits(rows, seed)`.

- [ ] Write failing tests for complete-sample filtering, exact eight-view validation, capture-native part-instance identity, deterministic stratified splits, and incomplete-sample exclusion.
- [ ] Run `UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest tests/unit/bmw_inspection/lab/test_eight_view_dataset.py -q` and confirm failures are due to the missing module.
- [ ] Implement the minimal typed parser and split functions.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Branch manifests and immutable publication

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_dataset.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_dataset.py`

**Interfaces:**
- Produces: `prepare_eight_view_dataset(raw_root, output_root, dataset_id, seed=42, verify_image_hash=True, dry_run=False)`.

- [ ] Add failing tests for `no_streak` branch semantics, review-only defect visibility, dry-run no-write, receipt hashes, and overwrite refusal.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Implement CSV/report serialization through a staging directory followed by an atomic rename.
- [ ] Re-run the focused tests and the existing BMW dataset tests.

### Task 3: Thin CLI and operational documentation

**Files:**
- Create: `pipeline/bmw_lab_prepare_eight_view_data.py`
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_dataset.py`

**Interfaces:**
- CLI: `uv run --no-sync python pipeline/bmw_lab_prepare_eight_view_data.py --raw-root ... --output-root ... --dataset-id ...`.

- [ ] Add a failing CLI test for dry-run JSON output and Chinese error exit code 2.
- [ ] Implement the parser/wrapper and document the exact dry-run and publication commands.
- [ ] Run focused tests, CLI help, compileall, real-data dry-run, real-data publication, and `git diff --check`.
- [ ] Update the folder memory with the verified artifact path, counts, label semantics, and remaining ROI/annotation boundary.
