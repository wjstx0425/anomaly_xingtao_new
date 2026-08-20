# BMW Right Reusable Eight-View ROI Implementation Plan

> **For agentic workers:** Implement inline in the current dirty workspace. Do not dispatch subagents or modify
> unrelated ZS32 files. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select one right-part eight-view ROI profile now and reuse it for future normal and no-streak captures.

**Architecture:** Extend the existing BMW raw reader with an optional capture-scope filter, add a raw-sample source to
the existing ROI selector, and introduce a backward-compatible fixed-setup ROI binding mode. Keep schema-v1 strict and
allow schema-v2 materialization only when source dimensions match exactly.

**Tech Stack:** Python 3.13, OpenCV, CSV/JSON/hashlib, pytest, uv.

## Global Constraints

- Preserve existing schema-v1 ROI behavior.
- Canonical views remain `front`, `front_left`, `front_right`, `front_secondary`, `back`, `back_left`, `back_right`,
  `back_secondary`.
- Never scale ROI coordinates between different image dimensions.
- Never overwrite an existing ROI or dataset release without an explicit existing opt-in.
- Do not train models or open cameras in this change.

---

### Task 1: Right capture filtering and raw representative selection

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_dataset.py`
- Modify: `src/bmw_inspection/lab/eight_view_roi.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_dataset.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_roi.py`

**Interfaces:**
- `read_complete_capture_rows(raw_root, *, verify_image_hash=True, capture_scope=None)`
- `select_raw_representative_images(raw_root, *, capture_scope, source_class=None, sample_id=None)`

- [ ] Add failing tests proving `capture_scope="right"` excludes left samples and raw selection returns one complete
  canonical sample plus its source manifest digest.
- [ ] Run the focused tests and confirm they fail because the arguments/API do not exist.
- [ ] Implement exact top-level capture-scope filtering and deterministic raw sample selection.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Fixed-setup ROI schema-v2 and materialization reuse

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_roi.py`
- Modify: `src/bmw_inspection/lab/eight_view_training_data.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_roi.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_training_data.py`

**Interfaces:**
- `EightViewRoiConfig.binding_mode: str`
- `EightViewRoiConfig.capture_scope: str | None`
- `load_roi_config(path) -> EightViewRoiConfig`

- [ ] Add failing tests for schema-v2 JSON round-trip and cross-release materialization with equal dimensions.
- [ ] Confirm RED while the current loader only accepts schema-v1 and materializer requires exact manifest identity.
- [ ] Implement schema-v2 validation/save/load and dimension-locked reusable materialization.
- [ ] Re-run focused tests and preserve schema-v1 strict-binding coverage.

### Task 3: CLI wiring and operator handoff

**Files:**
- Modify: `pipeline/bmw_lab_select_eight_view_rois.py`
- Modify: `pipeline/bmw_lab_prepare_eight_view_data.py`
- Modify: `pipeline/README.md`

**Interfaces:**
- ROI selector raw mode: `--raw-root`, `--hand`, `--source-class`, `--sample-id`, `--profile-id`.
- Data preparer scope: `--hand left|right`.

- [ ] Add CLI assertions to the focused tests before implementation and confirm RED.
- [ ] Wire raw selection to schema-v2 output and pass `--hand` into prepared-dataset creation.
- [ ] Run a real-data dry-run with `--hand right --skip-image-hash` and confirm 30 complete samples / 240 images.
- [ ] Run CLI help, focused tests, compile checks, and document the exact GUI command.
