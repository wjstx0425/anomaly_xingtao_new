# BMW Right One-Click Training Implementation Plan

> **For agentic workers:** Implement inline in the current dirty workspace. Do not modify unrelated ZS32 files.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one command that prepares the reviewed right-hand YOLO labels and trains all four BMW laboratory branches.

**Architecture:** A right-hand CLI prepares a deterministic label cache from the Label Studio ZIP/JSON, constructs the
existing `LabTrainingConfig` with right-hand paths, and delegates all training stages to the existing orchestrator. The
orchestrator derives expected label/crop counts from the selected dataset instead of old left-hand constants.

**Tech Stack:** Python 3.13, zipfile/JSON/CSV, OpenCV, Anomalib EfficientAD, Ultralytics YOLO, pytest, uv.

## Global Constraints

- Keep YOLO26n batch at 32 by default.
- Keep EfficientAD train batch fixed at 1.
- Treat `normal` and `no_streak` as EfficientAD normal data.
- Fail fast and never overwrite an inconsistent label cache, training release, or run directory.
- Do not start GPU training during implementation verification.

---

### Task 1: Dataset-derived training contracts

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_train_all.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_train_all.py`

**Interfaces:**
- `required_yolo_label_names(prepared_root: Path) -> tuple[str, ...]`
- `training_release_report_is_complete(report, views) -> bool`

- [ ] Add failing tests for a non-248 reviewed-label set and non-132 per-view crop count.
- [ ] Run the focused tests and confirm RED.
- [ ] Derive expected labels from the prepared manifest and validate reusable releases structurally.
- [ ] Re-run focused tests and confirm GREEN.

### Task 2: Right-hand label preparation and CLI

**Files:**
- Create: `src/bmw_inspection/lab/right_train_all.py`
- Create: `pipeline/bmw_lab_train_right.py`
- Create: `tests/unit/bmw_inspection/lab/test_right_train_all.py`

**Interfaces:**
- `prepare_reviewed_yolo_labels(zip_path, json_path, output_root) -> dict[str, object]`
- CLI options preserve right-hand defaults and expose epochs, YOLO batch/image size, GPU, workers, seed, and dry-run.

- [ ] Add failing tests for ZIP/JSON filename matching, cache creation/reuse, and CLI defaults.
- [ ] Run focused tests and confirm RED.
- [ ] Implement safe label preparation and the delegating right-hand CLI.
- [ ] Re-run focused tests and confirm GREEN.

### Task 3: Lightweight verification and handoff

**Files:**
- Modify: `pipeline/README.md`

- [ ] Document the right-hand dry-run and real training commands.
- [ ] Run focused tests, CLI help, compile checks, and real-data `--dry-run`.
- [ ] Report generated paths and the exact command without launching GPU training.
