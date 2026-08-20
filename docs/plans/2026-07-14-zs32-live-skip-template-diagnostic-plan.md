# ZS32 Live Skip-Template Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one explicit Stage35 diagnostic flag that skips template matching and Stage18 fusion while still capturing six views and running all six PatchCore models plus YOLO.

**Architecture:** Reuse Stage32 `infer` without `--template-model-dir` or `--threshold-artifact`; add an explicit diagnostic contract to Stage32 rather than fabricating template evidence. Stage35 conditionally builds that command, validates the diagnostic summary and both six-view CSVs, then prints artifact paths. The existing default `fuse --fusion-profile zs32-right-18-commissioning` path remains byte-for-byte unchanged in behavior.

**Tech Stack:** Python 3.13, argparse, csv/json, subprocess, pathlib, pytest, Ruff, existing ZS32ModelRuntime, anomalib/Ultralytics GPU runtime.

## Global Constraints

- The new switch is exactly `--diagnostic-skip-template`.
- Diagnostic mode is right-hand, one-part, manual-load HDR capture with the existing serials and defaults.
- Diagnostic mode runs Stage32 `infer`, never `fuse`, and omits both `--template-model-dir` and `--threshold-artifact`.
- It must run PatchCore and YOLO for all six canonical views and preserve continuous scores/evidence.
- It must not create `template_match.csv`, Stage18 fusion output, or an audit.
- Every diagnostic result must state `diagnostic_skip_template=true`, `inspection_complete=false`, `strict_fusion=false`, `commissioning_only=true`, and `production_release_allowed=false`.
- No existing template model, threshold artifact, production profile, or default Stage35 behavior may be modified.

---

### Task 1: Publish an explicit Stage32 diagnostic infer contract

**Files:**
- Modify: `pipeline/32_run_zs32_multimodel_inference.py`
- Modify: `tests/unit/pipeline/test_zs32_multimodel_inference.py`

**Interfaces:**
- Consumes: existing `infer` mode with no template model and `ZS32ModelRuntime.run(...)`.
- Produces: parser flag `--diagnostic-skip-template`; diagnostic fields in `runtime_summary.json` and `runtime_manifest.json`.

- [x] **Step 1: Write failing parser and validation tests**

Add tests proving the flag parses, is rejected with `fuse`, and is rejected when either `--template-model-dir` or `--threshold-artifact` is supplied. The error must name `--diagnostic-skip-template`.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run --frozen pytest tests/unit/pipeline/test_zs32_multimodel_inference.py -k diagnostic_skip_template -q
```

Expected: failure because the parser/contract does not exist.

- [x] **Step 3: Implement the minimal Stage32 flag and validation**

Add the boolean parser option. In `_validate_mode`, accept it only when `mode == "infer"`, `template_model_dir is None`, and `threshold_artifact is None`; otherwise raise `ValueError` before runtime/model construction.

- [x] **Step 4: Write a failing diagnostic publication test**

Use the existing fake runtime pattern to run `main()` in diagnostic mode. Assert that both JSON surfaces contain the exact identity and policy fields:

```python
{
    "part_id": "diagnostic-part",
    "capture_session": "session-001",
    "group_id": "group001",
    "hand": "right",
    "diagnostic_skip_template": True,
    "inspection_complete": False,
    "strict_fusion": False,
    "commissioning_only": True,
    "production_release_allowed": False,
}
```

Also assert `patchcore_csv` and `yolo_csv` in the summary are absolute paths and no template/fusion/audit artifact is created.

- [x] **Step 5: Implement diagnostic summary/manifest publication**

After `runtime.run`, extend the existing non-fusion summary with identity, diagnostic policy, and absolute CSV paths only when the flag is set. Atomically update the existing runtime manifest with the same diagnostic/policy fields and a note that template/Stage18 were intentionally skipped. Do not change ordinary `infer` or `fuse` summaries.

- [x] **Step 6: Verify Task 1 GREEN**

Run the whole Stage32 unit file plus focused Ruff/format checks for the touched files.

### Task 2: Add Stage35 diagnostic orchestration and fail-closed CSV validation

**Files:**
- Modify: `capture_data/zs32_live_commissioning.py`
- Modify: `tests/unit/capture_data/test_zs32_live_commissioning.py`

**Interfaces:**
- Consumes: Task 1 diagnostic summary and the existing PatchCore/YOLO CSV schemas.
- Produces: `LiveRunConfig.diagnostic_skip_template: bool`; conditional `build_stage32_command`; diagnostic fields/CSV paths on `LiveRunResult`.

- [x] **Step 1: Write failing exact-command tests**

For `diagnostic_skip_template=True`, assert the child command contains `infer --diagnostic-skip-template`, all six image flags, runtime config and GPU flags; assert it contains none of `fuse`, `--fusion-profile`, `--template-model-dir`, or `--threshold-artifact`. Keep the existing default command assertion unchanged.

- [x] **Step 2: Run the command tests and verify RED**

Run the two Stage32-command test cases. Expected: the diagnostic config field/command behavior is missing.

- [x] **Step 3: Implement the conditional command builder**

Add the frozen config boolean defaulting to `False`. Build the shared identity/image/runtime/GPU arguments once; choose `infer` plus the diagnostic flag for diagnostic mode, otherwise retain the current locked `fuse` arguments.

- [x] **Step 4: Write failing diagnostic-result contract tests**

Use the injected child runner to create a complete manifest, diagnostic summary and two CSV files. Each CSV must contain exactly six canonical views, shared sample identity, the expected branch (`anomaly_<view>` or `yolo`), a finite numeric `score`, existing source/evidence paths, and matching `manifest_identity`. Assert rejection for missing/extra/duplicate view, wrong identity/branch, non-finite score, missing evidence file, missing CSV, forbidden template/fusion/audit artifact, wrong policy flag, or non-diagnostic summary.

- [x] **Step 5: Implement diagnostic validation and result publication**

Add a dedicated validator rather than weakening the full-fusion and template-short-circuit validators. Return `audit_path=None`, `diagnostic_skip_template=True`, and absolute `patchcore_csv`/`yolo_csv` paths. Diagnostic success remains a valid process result even though its machine status is `REVIEW` and inspection is incomplete.

- [x] **Step 6: Verify Task 2 GREEN**

Run the complete live-core unit file and focused Ruff/format checks.

### Task 3: Expose the Stage35 operator switch and document the diagnostic boundary

**Files:**
- Modify: `pipeline/35_run_zs32_live_commissioning.py`
- Modify: `tests/unit/pipeline/test_zs32_live_commissioning_cli.py`
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: Task 2 `LiveRunConfig` and `LiveRunResult` fields.
- Produces: operator command `pipeline/35_run_zs32_live_commissioning.py --part-id ID --diagnostic-skip-template`.

- [x] **Step 1: Write failing CLI tests**

Assert the parser exposes the boolean switch, `_config_from_args` propagates it, and diagnostic output contains:

```text
template: skipped (diagnostic)
stage18_audit: none (diagnostic infer; fusion was not run)
patchcore_csv: /absolute/path/patchcore.csv
yolo_csv: /absolute/path/yolo.csv
production_release_allowed: false
```

- [x] **Step 2: Run CLI tests and verify RED**

Run the focused Stage35 CLI tests. Expected: the flag and diagnostic output are missing.

- [x] **Step 3: Implement parser/config/output changes**

Add the flag, pass it into `LiveRunConfig`, and branch terminal formatting on `result.diagnostic_skip_template`. Do not relabel the diagnostic REVIEW as OK or NG.

- [x] **Step 4: Update operator docs and project memories**

Document the exact command, output locations, absence of template/Stage18, and non-production boundary in `pipeline/README.md`. Record the reusable command/contract in both memory files without changing earlier default-flow history.

- [x] **Step 5: Run final verification**

```bash
uv run --frozen pytest \
  tests/unit/pipeline/test_zs32_multimodel_inference.py \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py \
  tests/unit/capture_data/test_collect_multicamera_dataset.py -q
uv run --with ruff ruff check \
  pipeline/32_run_zs32_multimodel_inference.py \
  capture_data/zs32_live_commissioning.py \
  pipeline/35_run_zs32_live_commissioning.py \
  tests/unit/pipeline/test_zs32_multimodel_inference.py \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py
uv run --with ruff ruff format --check \
  capture_data/zs32_live_commissioning.py \
  pipeline/35_run_zs32_live_commissioning.py \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py
/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py --help
git diff --check
```

Expected: all focused tests pass; help exposes the diagnostic switch; existing default Stage35 tests remain green.
