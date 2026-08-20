# ZS32 Live 18-Group Commissioning Implementation Plan

> **For agentic workers:** Execute each task with test-first development and independent review. Do not modify the production 36-group profile.

**Goal:** Provide one terminal command that captures one right-hand ZS32 part from three serial-bound cameras, validates the six-view manifest, runs the existing 18-group Stage32/18 commissioning fusion, and prints the audit result.

**Architecture:** Keep the numbered Stage35 entrypoint thin. Put manifest parsing, command construction, orchestration, and audit formatting in a testable `capture_data` module. Reuse the current collector and Stage32 as child processes under the same `/home/yunjing/anomalib/.venv/bin/python` interpreter.

**Tech Stack:** Python 3.13, argparse, csv/json, subprocess, pathlib, pytest, Ruff, Hikvision MVS through the existing collector, anomalib/Ultralytics through Stage32.

## Global Constraints

- Right hand only; profile is always `zs32-right-18-commissioning`.
- Do not add quality gate, registration, Geometry, GUI, service mode, or model persistence.
- Camera routing is serial-bound: `DA9805574`, `DA9625347`, `DB0998274`.
- Default capture is one HDR sample with `1500/6000 us`, gain `0`, settle frames `1`.
- Never start Stage32 unless the manifest proves one complete, unique six-view sample.
- Every terminal result must state `production_release_allowed=false`.

---

### Task 1: Lock the manifest and audit contracts

**Files:**
- Create: `capture_data/zs32_live_commissioning.py`
- Create: `tests/unit/capture_data/test_zs32_live_commissioning.py`

**Interfaces:**
- Produces: `CapturedSample`, `find_single_manifest(root: Path) -> Path`, `load_complete_sample(path: Path) -> CapturedSample`, `format_audit_report(audit: Mapping[str, object]) -> str`.

- [x] **Step 1: Write failing manifest tests**

Create fixtures containing six image rows plus one sample row. Assert exact identity mapping and rejection of missing/duplicate views, incomplete rows, inconsistent identities, absent image files, and multiple manifests.

```python
sample = load_complete_sample(manifest)
assert sample.part_id == "live_part_001_group001_000001"
assert sample.capture_session == "20260714_120000_000001"
assert tuple(sample.images) == CANONICAL_VIEWS
```

- [x] **Step 2: Verify red**

Run:

```bash
uv run --frozen pytest tests/unit/capture_data/test_zs32_live_commissioning.py -q
```

Expected: collection failure because the module does not exist.

- [x] **Step 3: Implement immutable sample parsing**

Use a frozen dataclass and reject every ambiguous manifest before returning:

```python
@dataclass(frozen=True)
class CapturedSample:
    part_id: str
    capture_session: str
    group_id: str
    images: dict[str, Path]
    manifest_path: Path
```

`part_id` is the manifest `sample_id`; all file paths are resolved and verified with `is_file()`.

- [x] **Step 4: Add failing audit formatting tests, then implement**

Require canonical view order and branch order `template_match`, `anomaly_<view>`, `yolo`. Read `score`, `low_threshold`, `high_threshold`, and `computed_evidence_level` from Stage18 audit `views`.

- [x] **Step 5: Verify green**

Run the Task 1 test file and Ruff for the new core and tests.

### Task 2: Build and orchestrate the two child commands

**Files:**
- Modify: `capture_data/zs32_live_commissioning.py`
- Modify: `tests/unit/capture_data/test_zs32_live_commissioning.py`

**Interfaces:**
- Produces: `build_capture_command(config, capture_run_root) -> list[str]`, `build_stage32_command(config, sample, output_dir) -> list[str]`, `run_live_commissioning(config, command_runner=subprocess.run) -> LiveRunResult`.

- [x] **Step 1: Write failing exact-command tests**

Assert capture includes:

```text
--hand right --label normal --group-count 1 --images-per-group 1
--manual-load --hdr --short-exposure 1500 --long-exposure 6000
--gain 0 --hdr-settle-frames 1
```

Assert Stage32 includes all six explicit image arguments, GPU flags, locked asset paths, and `--fusion-profile zs32-right-18-commissioning` without quality/registration/geometry flags.

- [x] **Step 2: Verify red**

Run only the two command tests and confirm the builders are missing.

- [x] **Step 3: Implement command builders**

Every child command begins with `sys.executable`. Paths are absolute. The capture root is a new per-run directory under the configured capture root; runtime output is `<output_root>/<capture_session>/<part_id>` and must not exist.

- [x] **Step 4: Write failing orchestration tests**

Inject a fake command runner. On capture invocation it writes a complete manifest; on Stage32 invocation it writes `runtime_summary.json` and the audit. Assert capture failure, invalid manifest, Stage32 failure, missing/invalid result JSON, and successful business statuses are handled separately.

- [x] **Step 5: Implement orchestration**

Return nonzero only for execution/contract errors. Preserve valid `OK`, `NG_*`, and `REVIEW` outputs. Require summary/audit identity agreement for full fusion and require `commissioning_only=true`, `production_release_allowed=false`. A validated template short circuit preserves `NG_TEMPLATE` or exception-shaped `REVIEW` with `audit=None` instead of requiring a Stage18 artifact that was never produced. Image decode/dimension failures are Stage32 execution errors, not fabricated `INVALID_CAPTURE` summaries.

- [x] **Step 6: Verify green**

Run the core test file and Ruff.

### Task 3: Add the Stage35 CLI and correct right-hand prompts

**Files:**
- Create: `pipeline/35_run_zs32_live_commissioning.py`
- Create: `tests/unit/pipeline/test_zs32_live_commissioning_cli.py`
- Modify: `capture_data/collect_multicamera_dataset.py`
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`

**Interfaces:**
- Stage35 CLI consumes `--part-id` or `--list-devices`; optional overrides expose asset roots, capture parameters, timeout, and serials without exposing the fusion profile.

- [x] **Step 1: Write failing parser/default tests**

Assert the minimal command requires `--part-id`, defaults to the approved asset paths and capture values, and `--list-devices` does not require `--part-id`.

- [x] **Step 2: Write failing prompt test**

Call `capture_group` with `hand="right"` and a recording prompt. Assert it contains `右手件` for both front and back rounds and never contains `左手件`.

- [x] **Step 3: Verify red**

Run the two focused test nodes and confirm the entrypoint/prompt behavior is absent.

- [x] **Step 4: Implement the CLI and dynamic prompt**

The CLI calls the core runner, prints its formatted report and absolute artifact paths, and exits with the core error code. Replace the hard-coded hand text in both collector capture functions with `"右手件" if args.hand == "right" else "左手件"`.

- [x] **Step 5: Verify green**

Run Stage35 parser tests plus the complete multicamera collector test file.

### Task 4: Document and exercise the operator path

**Files:**
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

- [x] **Step 1: Document preflight and one-part commands**

Add exactly these operator shapes:

```bash
/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py --list-devices
/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py --part-id live_part_001
```

State the two Enter prompts, output locations, business status behavior, and non-production boundary.

- [x] **Step 2: Add a no-camera historical-manifest harness**

Use a small Python harness to extract one complete sample from `/home/yunjing/anomalib/dataset/manifests/20260711_165347_078120.csv` into a temporary one-sample CSV, call `load_complete_sample` on that file, and build the Stage32 command without executing it. Confirm six unique right-hand files and the expected identity mapping.

- [x] **Step 3: Run final verification**

```bash
uv run --frozen pytest \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py \
  tests/unit/capture_data/test_collect_multicamera_dataset.py \
  tests/unit/pipeline/test_zs32_multimodel_inference.py -q
uv run --with ruff ruff check capture_data/zs32_live_commissioning.py pipeline/35_run_zs32_live_commissioning.py tests/unit/capture_data/test_zs32_live_commissioning.py tests/unit/pipeline/test_zs32_live_commissioning_cli.py
uv run --with ruff ruff format --check capture_data/zs32_live_commissioning.py pipeline/35_run_zs32_live_commissioning.py tests/unit/capture_data/test_zs32_live_commissioning.py tests/unit/pipeline/test_zs32_live_commissioning_cli.py
python -m compileall -q capture_data/zs32_live_commissioning.py pipeline/35_run_zs32_live_commissioning.py
/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py --help
git diff --check
```

Expected: all tests and static checks pass; CLI help lists the minimal command and approved defaults.

Observed on 2026-07-14: the four-file regression set passed `183` tests; focused Ruff and format checks,
Python compilation, both locked JSON parses, CLI help, and `git diff --check` all exited zero. Camera discovery
also returned the three configured serials with exit code zero; a real capture remains an operator-present step.
