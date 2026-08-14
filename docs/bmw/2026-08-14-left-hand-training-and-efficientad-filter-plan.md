# BMW Left-Hand Training and EfficientAD Filter Implementation Plan

> **For agentic workers:** Execute each task test-first. Preserve unrelated dirty work and stage only task-owned files.

**Goal:** Prepare a runnable left-hand normal-data pipeline and replace EfficientAD single-pixel max decisions with a calibrated component-aware score while reusing the exact shared YOLO checkpoint.

**Architecture:** Existing prepare, ROI, materialization, Template, and EfficientAD primitives remain the foundation. New code adds left-hand fail-closed orchestration, blank mask selection from materialized crops, and one pure component scorer shared by calibration and runtime. All new outputs are candidate-only, versioned, SHA-bound, and no-overwrite.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, anomalib/EfficientAD, pytest, uv.

## Global Constraints

- Use `uv`; do not train YOLO.
- Prepared data and ROI must both identify `capture_scope=left`.
- Template and EfficientAD thresholds must be fitted from left-hand normal data and bound to their model SHA values.
- EfficientAD calibration and runtime must use the same mask, component policy, and score function.
- Shared YOLO SHA remains `0e9591f2fa2487ad12000847f1d80137901ed69989e0e95cb5c8907b95ba3913`.
- Existing right-hand configurations and immutable releases remain unchanged.

---

### Task 1: Left ROI and blank mask contracts

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_training_data.py`
- Modify: `pipeline/bmw_lab_select_efficientad_ignore_masks.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_training_data.py`
- Test: `tests/unit/pipeline/test_bmw_lab_select_efficientad_ignore_masks.py`

**Interfaces:**
- Materialization rejects a fixed-setup ROI whose `capture_scope` differs from the prepared report.
- Mask selector accepts either existing Demo ROI images or `--training-release` plus an optional representative sample.
- `--from-index` becomes optional; omission starts eight empty polygon lists.

- [ ] Add failing tests for cross-hand ROI rejection and blank mask source loading.
- [ ] Run the two focused test files and confirm the new assertions fail for the missing behavior.
- [ ] Add the minimum capture-scope check and selector source/seed helpers.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Component-aware EfficientAD score shared by runtime and calibration

**Files:**
- Create: `src/bmw_inspection/lab/efficientad_component_filter.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Test: `tests/unit/bmw_inspection/lab/test_efficientad_component_filter.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`

**Interfaces:**
- `ComponentFilterPolicy` contains low/seed/P95/hard-peak/area/line gates.
- `score_anomaly_components(anomaly_map, ignore_mask, policy)` returns the decision score, accepted/rejected component statistics, accepted mask, and hotspot.
- Runtime uses the component score only when a SHA-verified component-filter artifact is configured; legacy configs keep their existing behavior.
- Details and overlay expose component area, peak, mean, P95, bounding box, and acceptance reason.

- [ ] Add failing pure-function tests: isolated shallow PASS, tiny hard point NG, thin long line NG, broad anomaly NG, ignored area excluded.
- [ ] Verify RED.
- [ ] Implement immutable policy/result types and deterministic OpenCV connected-component scoring.
- [ ] Verify pure-function GREEN.
- [ ] Add failing config/runtime tests for artifact SHA, eight-view policy coverage, score source, and overlay details.
- [ ] Implement optional config loading and runtime integration without changing legacy behavior.
- [ ] Run focused tests and confirm GREEN.

### Task 3: Left normal-only training and robust EfficientAD calibration

**Files:**
- Create: `src/bmw_inspection/lab/left_normal_training.py`
- Create: `pipeline/bmw_lab_train_left_normal.py`
- Test: `tests/unit/bmw_inspection/lab/test_left_normal_training.py`

**Interfaces:**
- The stage order is `materialize`, `template`, `efficientad`, `score_component_maps`, `calibrate_component_thresholds`; it contains no YOLO or bright-streak stage.
- Template trains from train rows and fits normal-only per-view candidate thresholds from calibration rows; its artifact binds all eight model JSON hashes.
- EfficientAD map scoring loads the left mask and component policy, writes per-image component score CSV, then calls the existing whole-part threshold fitter with `target_part_fpr=0.05`.
- EfficientAD threshold artifact binds all eight checkpoint hashes, mask-index SHA, policy SHA, score source, sample count, and FPR resolution.
- Dry-run validates layout and identities without GPU work.

- [ ] Add failing tests for exact stage order, no YOLO, cross-hand refusal, Template threshold provenance, and EfficientAD threshold provenance.
- [ ] Verify RED.
- [ ] Implement the minimal orchestration and CLI with fresh output IDs and no-overwrite behavior.
- [ ] Verify GREEN with injected lightweight stage handlers.

### Task 4: Operator commands, verification, and project memory

**Files:**
- Create or modify: `docs/bmw/LEFT_HAND_QUICKSTART.md`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Quickstart gives copy-paste commands from completed left normal capture through ROI, materialization/training, blank mask, component calibration, later bright ROI, and V3 retraining.
- It states that YOLO is reused and records the exact checkpoint SHA.

- [ ] Document the post-capture command sequence and expected artifact paths.
- [ ] Run focused pytest files, `compileall`, CLI `--help`, a fixture dry-run, and `git diff --check`.
- [ ] Inspect the final diff for unrelated files and update `AGENTS_MEMORY.md` with verified facts and remaining live-data blockers.
