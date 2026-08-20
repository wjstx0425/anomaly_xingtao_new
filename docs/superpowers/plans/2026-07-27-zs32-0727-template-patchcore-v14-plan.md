# ZS32 0727 Template and PatchCore v14 Preparation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned, leakage-controlled eight-view training release from 104 distinct `zs32_0727` normal parts and 23 unique `zs32_all/right` defect parts, then provide runnable Template and PatchCore v14 commands.

**Architecture:** Crop only the new normal images with the established eight-view ROI. Reuse the existing defect crops through file symlinks, filter the proven duplicate defect session, assign deterministic physical-part roles, and publish separate Template manifest and PatchCore folder surfaces without modifying any source dataset.

**Tech Stack:** Python 3.13, OpenCV, CSV/JSON manifests, pytest, anomalib PatchCore, OpenCV Template gate

## Global Constraints

- Preserve both source datasets and all v11/v12/v13 model artifacts.
- Physical identity is `<session_id>:<groupNNN>`; every part keeps all eight views in one role.
- Normal roles are `62 train / 16 model_val / 13 calibration / 13 final_test`.
- Unique defect roles are `0 train / 6 model_val / 11 calibration / 6 final_test`, stratified by defect type where possible.
- Exclude `zs32_right_deform_20260714_201404_438451409`, whose five parts are byte-identical duplicates.
- Template selection uses only normal `train`; defect rows may affect calibration and scoring but never become templates.
- PatchCore memory-bank fitting uses only normal `train`; calibration normal maps to `normal_test`, calibration defects map to `defect`; model-val/final-test remain holdouts.
- Both secondary views are required.
- No pixel-level metric claims because no masks exist.

---

### Task 1: Add the deterministic release builder

**Files:**
- Create: `capture_data/prepare_zs32_0727_template_patchcore.py`
- Create: `pipeline/prepare_zs32_0727_template_patchcore.py`
- Test: `tests/unit/capture_data/test_prepare_zs32_0727_template_patchcore.py`

**Interfaces:**
- Consumes: new-normal crop manifest, old defect crop manifest, output root, seed.
- Produces: `physical_part_splits.csv`, `template_manifest.csv`, `patchcore_manifest.csv`, `patchcore/crop_manifest.csv`, `roi_config.json`, `release_summary.json`, and file-level symlinks.

- [x] **Step 1: Write failing fixture tests**

Test exact normal/defect role counts, eight-view identity isolation, duplicate-session rejection, Template label/split semantics, PatchCore train/calibration routing, holdout preservation, and symlink targets.

- [x] **Step 2: Verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync pytest \
  tests/unit/capture_data/test_prepare_zs32_0727_template_patchcore.py -q
```

Expected: failure because the new builder module does not exist.

- [x] **Step 3: Implement prepare and validate subcommands**

The prepare command must refuse an existing output, verify exact counts and eight-view completeness before writing, stage atomically, and use deterministic seed-42 role assignment. The validate command must independently re-read all manifests, resolve every symlink, and reject count, role, identity, path, or duplicate-content violations.

- [x] **Step 4: Verify GREEN**

Run the new unit test plus `py_compile` and `git diff --check`.

### Task 2: Materialize and validate the real training release

**Files:**
- Create: `dataset/zs32_0727_right_normal_roi_104part_v1/`
- Create: `dataset/zs32_0727_plus_zs32_all_unique_defect_release_v1/`

**Interfaces:**
- Consumes: merged 0727 manifest, ROI SHA `9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65`, and existing defect crops.
- Produces: 832 new normal crops and a versioned release containing 104 normal and 23 unique defect eight-view identities.

- [x] **Step 1: Crop the new normal images**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/30_crop_zs32_patchcore_dataset.py convert \
  --repo-root /home/yunjing/anomaly_xingtao_new \
  --dataset-root /home/yunjing/anomaly_xingtao_new/dataset/zs32_0727 \
  --config /home/yunjing/anomaly_xingtao_new/dataset/zs32_all_right_patchcore_roi/roi_config.json \
  --output-root /home/yunjing/anomaly_xingtao_new/dataset/zs32_0727_right_normal_roi_104part_v1 \
  --hand right
```

- [x] **Step 2: Build the release**

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run --no-sync python \
  pipeline/prepare_zs32_0727_template_patchcore.py prepare \
  --normal-crop-manifest dataset/zs32_0727_right_normal_roi_104part_v1/crop_manifest.csv \
  --defect-crop-manifest dataset/zs32_all_right_patchcore_roi/crop_manifest.csv \
  --output-root dataset/zs32_0727_plus_zs32_all_unique_defect_release_v1 \
  --seed 42
```

- [x] **Step 3: Validate the release**

Run the matching `validate` subcommand and independently confirm 104 normal parts, 23 defect parts, eight views per part, no duplicate image hashes, valid symlinks, and correct PatchCore directory buckets.

### Task 3: Publish the training handoff

**Files:**
- Create: `docs/ZS32_0727_TEMPLATE_PATCHCORE_V14_TRAINING.md`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: validated release.
- Produces: exact Template and PatchCore commands with new result roots.

- [x] **Step 1: Document the Template command**

Use `pipeline/train_zs32_template_gate.py` without `--normal-only`, with the explicit four-role manifest and v14 identities.

- [x] **Step 2: Document the PatchCore command**

Use `pipeline/run_patchcore_roi_eight_views.sh` against the release `patchcore/` root, strong WRN50 layer2+3 parameters, seed 42, and new v14 result root.

- [x] **Step 3: Final verification**

Verify both CLI `--help` surfaces, all referenced inputs, GPU preflight command, offline backbone-cache assumptions, and `git diff --check`. Do not start training.
