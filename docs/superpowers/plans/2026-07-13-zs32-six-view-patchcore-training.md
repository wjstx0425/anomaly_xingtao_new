# ZS32 Six-View PatchCore Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train six independent fixed-configuration PatchCore models, resume safely after interruption, and produce one cross-view metrics CSV.

**Architecture:** Keep the repository's `pipeline/8_train_custom_models.py` entrypoint as the only trainer. A serial shell orchestrator gives each view its own output root, selects full/resume/evaluate-only mode from durable artifacts, records failures without stopping later views, and merges completed per-view reports without sharing checkpoints or thresholds.

**Tech Stack:** Bash, Python 3.13, Lightning, anomalib PatchCore, pytest/static CLI checks.

## Global Constraints

- Use `/home/yunjing/anomalib` because `/home/ljl/anomaly_xingtao` is absent and this checkout's `mygithub` remote is `anomaly_xingtao`.
- Use `/home/yunjing/anomalib/dataset/right` because `/DATA/ljl/right` is absent.
- Fixed model configuration: `wide_resnet50_2`, `layer2`, `256,256`, ROI `full`, coreset `0.01`, neighbors `9`, precision `float16`, train/eval batch `4`, workers `2`, deploy FPR `0.05`, seed `42`.
- Run suffix is `wrn_l2_s256_r001_k9_fp16_fpr005_seed42`.
- Train `right_front`, `right_front_left`, `right_front_right`, `right_back`, `right_back_left`, and `right_back_right` serially on GPU `0`.
- Never delete or overwrite completed experiment artifacts, and never copy a threshold between views.

---

### Task 1: Seed each PatchCore experiment deterministically

**Files:**
- Modify: `examples/api/03_models/zs32_defect_workflow.py`
- Test: `tests/unit/pipeline/test_pipeline_wrappers.py`

**Interfaces:**
- Consumes: CLI `--seed` integer.
- Produces: one `seed_everything(args.seed, workers=True)` call before each `(view, model)` model/datamodule construction.

- [x] Add a source-level regression test that requires the Lightning import and per-experiment seed call.
- [x] Run the targeted test and confirm it fails because the import/call is absent.
- [x] Add the import and call at the start of the inner training loop.
- [x] Run the targeted test and confirm it passes.

### Task 2: Add serial, resumable six-view runner and report merger

**Files:**
- Create: `pipeline/run_wrn50_fixed_six_views.sh`
- Create: `tests/unit/pipeline/test_run_wrn50_fixed_six_views.py`

**Interfaces:**
- Consumes: `DATA_ROOT [RUN_BASE] [GPU]`, with repository-local defaults and optional `PYTHON_BIN` override for testability.
- Produces: per-view output roots, per-view runner logs, `failed_views.txt` when needed, and `six_view_summary.csv` when all six summaries exist.

- [x] Add tests for fixed arguments, serial order, completed-view skip, checkpoint evaluate-only resume, manifest skip-preprocess resume, failure recording, and summary merge fields.
- [x] Run the targeted tests and confirm they fail because the runner is absent.
- [x] Implement the minimal Bash orchestrator and CSV merge.
- [x] Run Bash syntax, targeted tests, and dry-run checks.

### Task 3: Execute and verify fixed training

**Files:**
- Update: `AGENTS_MEMORY.md`
- Generate: `results/six_view_fixed_seed42/**`

**Interfaces:**
- Consumes: cached `wide_resnet50_2` weights, six-view dataset, GPU `0`.
- Produces: six independent checkpoints, six independent deployment thresholds, per-view logs, and `results/six_view_fixed_seed42/six_view_summary.csv`.

- [x] Verify cached weights with `HF_HUB_OFFLINE=1`.
- [x] Run the shell orchestrator with actual adjusted paths and GPU `0`.
- [x] Resume the same command after any recoverable interruption.
- [x] Validate all six summary rows, checkpoint paths, metrics, and failure log state.
- [x] Update `AGENTS_MEMORY.md` with exact commands, paths, status, and results.
