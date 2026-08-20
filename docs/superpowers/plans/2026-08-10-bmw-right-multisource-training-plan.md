# BMW Right Multisource Training Implementation Plan

> **For agentic workers:** Implement inline in the current dirty workspace. Do not modify unrelated ZS32 files.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train right-hand EfficientAD/Template/bright-streak on two right releases and YOLO26n on paired left/right data.

**Architecture:** Build one immutable symlink-based composite release, then delegate model training to the existing BMW
eight-view orchestrator. Namespace repeated right-hand sample identities while preserving source splits and labels.

**Tech Stack:** Python 3.13, CSV/JSON/pathlib, Anomalib EfficientAD, Ultralytics YOLO, pytest, uv.

## Global Constraints

- New right batch is excluded from YOLO.
- Left YOLO labels remain paired only with their left images.
- YOLO batch defaults to 32; EfficientAD batch remains 1.
- Existing releases and results are never overwritten.
- Verification does not start GPU training.

---

### Task 1: Composite release contract

**Files:**
- Create: `src/bmw_inspection/lab/multisource_training_data.py`
- Create: `tests/unit/bmw_inspection/lab/test_multisource_training_data.py`

**Interfaces:**
- `build_multisource_training_data(..., dry_run: bool) -> dict[str, object]`

- [ ] Add a failing test covering right branch merge, namespaced Template rows, paired left/right YOLO, and dry-run.
- [ ] Run the test and confirm RED because the module is absent.
- [ ] Implement deterministic validation, symlink assembly, reports, and atomic publication.
- [ ] Re-run the test and confirm GREEN.

### Task 2: Reusable model-asset preflight and CLI

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_train_all.py`
- Create: `pipeline/bmw_lab_train_right_multisource.py`
- Modify: `tests/unit/bmw_inspection/lab/test_right_train_all.py`

**Interfaces:**
- `preflight_model_assets(config: LabTrainingConfig) -> dict[str, object]`
- CLI defaults point to the two right releases, left YOLO release, and reviewed right labels.

- [ ] Add failing assertions for the reusable preflight and CLI defaults.
- [ ] Confirm RED before implementation.
- [ ] Implement the shared preflight and fail-fast multisource CLI.
- [ ] Re-run focused tests and confirm GREEN.

### Task 3: Handoff verification

**Files:**
- Modify: `pipeline/README.md`

- [ ] Document dry-run and real training commands.
- [ ] Run focused tests, core Ruff checks, compilation, and real-data dry-run.
- [ ] Confirm dry-run creates no composite release or model result directory.
