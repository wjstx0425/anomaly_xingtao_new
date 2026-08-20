# ZS32 0727 Template v12 Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an eight-view right-hand Template model from only the 19 confirmed-normal `zs32_0727` physical parts and publish it as the platform-default versioned v12 bundle without changing v11 PatchCore, YOLO, or ROI assets.

**Architecture:** Add a focused Template-release builder that converts the Stage30 crop manifest into an explicit physical-part-level four-role manifest. Train and score an immutable v12 Template artifact, then republish the existing hash-bound Stage37/Stage34 runtime chain with only Template assets changed and switch defaults after fresh-process validation.

**Tech Stack:** Python 3.10+, `uv`, pytest, OpenCV/Pillow, CSV/JSON manifests, existing ZS32 Stage30/34/35/37 CLIs.

## Global Constraints

- Use only `dataset/zs32_0727` groups `group001` through `group019`; never use `zs32_0723` images for the new Template.
- Treat all 19 groups as distinct confirmed-normal physical parts and keep all eight views of a part in one role.
- Use fixed seed 42 and exact role counts `train=11`, `model_val=3`, `calibration=3`, `final_test=2`.
- Preserve the existing ROI bytes and version `zs32-eight-view-roi-9412b2838cdb`.
- Require all eight Template views, including `front_secondary` and `back_secondary`.
- Do not modify or delete v11/v10 artifacts; every v12 output is a new path.
- Keep PatchCore, YOLO, YOLO labels, and ROI identical to v11.
- Keep `commissioning_only=true` and `production_release_allowed=false`; no defect metrics may be claimed.
- Manage Python execution with `uv`.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Explicit 19-Part Template Release Builder

**Files:**
- Create: `capture_data/prepare_zs32_template_release.py`
- Create: `pipeline/prepare_zs32_template_release.py`
- Create: `tests/unit/capture_data/test_prepare_zs32_template_release.py`

**Interfaces:**
- Consumes: Stage30 `crop_manifest.csv`, copied `roi_config.json`, source name, seed, and four role counts.
- Produces: `prepare_template_release(crop_manifest: Path, output_root: Path, *, source_name: str, expected_parts: int, train_count: int, model_val_count: int, calibration_count: int, final_test_count: int, seed: int = 42, dry_run: bool = False) -> dict[str, Any]` and CLI subcommands `prepare`/`validate`.

- [ ] **Step 1: Write failing builder tests**

Create fixtures with 19 physical groups and eight views each. Assert deterministic `11/3/3/2` part counts, 152 manifest rows, one role per physical part, all eight views per part, copied ROI bytes, SHA receipts, exclusion/failure for incomplete parts, duplicate cross-part hashes, wrong labels/hands, overwrite, and validation leakage.

- [ ] **Step 2: Verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/capture_data/test_prepare_zs32_template_release.py
```

Expected: failure because `capture_data/prepare_zs32_template_release.py` does not exist.

- [ ] **Step 3: Implement the minimal builder**

Implement strict CSV loading, `groupNNN` extraction, `physical_part_id = session_id:group_id`,
eight-view completeness, encoded-file SHA uniqueness, deterministic seeded role assignment,
symlink materialization, `template_manifest.csv`, `physical_part_splits.csv`, `summary.json`,
`sha256_receipts.csv`, copied ROI, atomic publish, dry-run, and fail-closed validation.

- [ ] **Step 4: Add the thin pipeline wrapper**

The wrapper must only add the repository root to `sys.path`, import `main`, and
`raise SystemExit(main())`.

- [ ] **Step 5: Verify GREEN**

Run the Task 1 pytest command and expect all tests to pass.

### Task 2: Build and Validate the 0727 Release

**Files:**
- Create: `dataset/zs32_0727_template_roi_v1/`
- Create: `dataset/zs32_0727_template_release_v1/`

**Interfaces:**
- Consumes: raw `dataset/zs32_0727`, v11 ROI config, and Task 1 CLI.
- Produces: immutable 152-image ROI crop release and explicit four-role Template manifest.

- [ ] **Step 1: Run Stage30 ROI conversion**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python pipeline/30_crop_zs32_patchcore_dataset.py convert \
  --repo-root /home/yunjing/anomaly_xingtao_new \
  --dataset-root /home/yunjing/anomaly_xingtao_new/dataset/zs32_0727 \
  --config /home/yunjing/anomaly_xingtao_new/dataset/zs32_all_plus_0723_retraining_release_v2/roi_config.json \
  --output-root /home/yunjing/anomaly_xingtao_new/dataset/zs32_0727_template_roi_v1 \
  --hand right
```

Expected: 152 outputs, eight views × 19, zero corrected views.

- [ ] **Step 2: Run release preparation**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python pipeline/prepare_zs32_template_release.py prepare \
  --crop-manifest dataset/zs32_0727_template_roi_v1/crop_manifest.csv \
  --output-root dataset/zs32_0727_template_release_v1 \
  --source-name zs32_0727 --expected-parts 19 \
  --train-count 11 --model-val-count 3 --calibration-count 3 --final-test-count 2 \
  --seed 42
```

- [ ] **Step 3: Validate release evidence**

Run the CLI `validate` command and an independent CSV assertion for exact counts, 152 unique
paths/hashes, eight views, 19 parts, and zero part-role leakage. Render/inspect one ROI crop per
view before training.

### Task 3: Train and Score Template v12

**Files:**
- Create: `results/zs32_template_gate_right_0727_eight_view_v12/`
- Create: `results/zs32_template_gate_right_0727_eight_view_v12/evaluation.json`
- Create: `results/zs32_template_gate_right_0727_eight_view_v12/evaluation.csv`

**Interfaces:**
- Consumes: Task 2 explicit `template_manifest.csv`.
- Produces: one eight-group Template `model.json`, `model.sha256`, 40 templates, calibration evidence, and role/view score report.

- [ ] **Step 1: Train with explicit roles**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python pipeline/train_zs32_template_gate.py \
  --manifest dataset/zs32_0727_template_release_v1/template_manifest.csv \
  --output-dir results/zs32_template_gate_right_0727_eight_view_v12 \
  --required-hand right --width 512 --max-shift 12 \
  --templates-per-group 5 --max-train-per-group 120 \
  --normal-quantile 1.0 --normal-only --evaluation-fraction 0 \
  --model-version zs32-right-0727-eight-view-template-v12 \
  --threshold-version zs32-right-0727-temporary-threshold-v12 \
  --roi-version zs32-eight-view-roi-9412b2838cdb \
  --template-version zs32-right-0727-eight-view-template-v12
```

- [ ] **Step 2: Verify artifact integrity**

Compare `sha256sum model.json` with `model.sha256`, call `load_model()` in a fresh process,
verify eight exact groups and five templates per group, and verify every embedded template SHA.

- [ ] **Step 3: Score every explicit role**

Use `PreparedTemplateGate` to emit per-image CSV and per-view/role JSON aggregates containing
count, risk min/median/max, PASS/REVIEW/NG counts, and thresholds. Do not mutate thresholds from
model-val or final-test outcomes.

### Task 4: Publish the Versioned v12 Runtime Chain

**Files:**
- Create: `config/fusion/zs32_right_eight_view_24_group_commissioning_template_0727_v12.json`
- Create: `config/fusion/zs32_eight_view_24group_template_0727_v12_bundle_source.json`
- Create: `results/zs32_runtime_assets_eight_view_template_0727_v12/`
- Create: `results/zs32_24group_template_0727_v12_commissioning/`
- Create: `results/zs32_runtime_bundle_eight_view_template_0727_v12/`

**Interfaces:**
- Consumes: Template v12 plus unchanged v11 PatchCore/YOLO/ROI and existing Stage33 sources.
- Produces: hash-bound 24-group commissioning bundle.

- [ ] **Step 1: Write failing config/default contract tests**

Extend existing Stage34/Stage37 tests to require the new source/profile shapes, strict 24 groups,
new Template SHA binding, and unchanged PatchCore/YOLO/ROI SHA values.

- [ ] **Step 2: Verify RED**

Run the targeted runtime-bundle and commissioning tests; expect failure because v12 configs and
artifacts do not exist.

- [ ] **Step 3: Add v12 profile and source config**

Copy the established v11 24-group structure, change only version/bundle identifiers and Template
model path, and keep `expected_versions=[]` until Stage37 binds actual hashes.

- [ ] **Step 4: Publish assets, thresholds, and final bundle**

Run Stage37 `publish-assets`, Stage34 publication with the v11 PatchCore overrides and YOLO
threshold `0.07`, then Stage37 `finalize`. Use new v12 output directories only.

- [ ] **Step 5: Verify GREEN and immutable bindings**

Fresh-load v12 and assert 24 records, eight required Template groups, both secondary views,
new Template SHA, unchanged PatchCore/YOLO/ROI SHA, and commissioning/non-production flags.
Fresh-load v11 and v10 rollback bundles in separate processes.

### Task 5: Switch Defaults, Document Operations, and Final Verification

**Files:**
- Modify: `pipeline/35_run_zs32_live_commissioning.py`
- Modify: `src/zs32_inspection/dashboard/live.py`
- Modify: relevant CLI/Dashboard default-path tests
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`
- Create: `docs/ZS32_0727_TEMPLATE_V12_REPLACEMENT_20260727.md`

**Interfaces:**
- Consumes: validated Task 4 v12 runtime bundle.
- Produces: default CLI/Dashboard selection of v12, explicit v11 rollback command, and durable project memory.

- [ ] **Step 1: Write failing default-path tests**

Assert both default entry points resolve to
`results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json`.

- [ ] **Step 2: Verify RED**

Run only the affected CLI and Dashboard tests; expect current v11-default assertions to fail.

- [ ] **Step 3: Switch the two defaults**

Change only the default constants/arguments. Do not alter explicit `--runtime-config` behavior.

- [ ] **Step 4: Add operational documentation and memory**

Record exact preparation/training/publication/smoke commands, threshold location, data limits,
artifact hashes, and explicit v11 rollback command. Update both scoped memory files.

- [ ] **Step 5: Run final verification**

Run targeted unit tests, fresh-load v12/v11/v10, independent SHA comparisons, exact 24-group
checks, default-path assertions, and a Stage35 non-GUI preflight. If GPU/hardware is unavailable,
report the exact unrun live smoke command without claiming it passed.

