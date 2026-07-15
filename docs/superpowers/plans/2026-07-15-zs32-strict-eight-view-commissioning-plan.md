# ZS32 Strict Eight-View Commissioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every ZS32 right-hand view a fully modeled Template/PatchCore/YOLO/Fusion view, publish replaceable hash-bound runtime bundles, and complete one four-camera commissioning run through the OpenCV dashboard.

**Architecture:** Keep legacy six-view artifacts immutable while introducing a strict versioned eight-view path. A two-phase bundle publisher first emits immutable Stage32 runtime assets, then finalizes one Stage35 bundle after Stage33/34 publish the 24-group threshold artifact; Stage35 resolves that single bundle into capture, inference, fusion, progress/control, and dashboard inputs.

**Tech Stack:** Python 3.10+, dataclasses, JSON/CSV, OpenCV, NumPy, anomalib PatchCore, Ultralytics YOLO, pytest, uv, MVS four-camera SDK.

This plan supersedes Tasks 6-9 of `2026-07-14-zs32-eight-view-inspection-dashboard.md`. Tasks 1-5 of that plan remain the accepted offline-dashboard baseline.

## Global Constraints

- Product is exactly `ZS32`, hand is exactly `right`, and view order is exactly `front/front_left/front_right/front_secondary/back/back_left/back_right/back_secondary`.
- `MODELED_VIEWS == VIEW_ORDER`; all eight views require Template, PatchCore, YOLO, and Fusion dashboard records. New manifests may not publish secondary as unsupported or with empty branches.
- Stage18 commissioning evidence is exactly `8 × 3 = 24` groups: `template_match`, `anomaly_{view}`, and `yolo` for every view.
- New artifacts are always `commissioning_only=true` and `production_release_allowed=false`; no result may be described as production-ready.
- Keep legacy six-view configs and artifacts unchanged for historical replay.
- Template, eight PatchCore checkpoints, shared YOLO, ROI, profiles, and thresholds are selected through one versioned runtime bundle, not hard-coded in Stage32, Stage35, or dashboard code.
- Runtime assets, source images, masks, scores, and paths fail closed on missing files, invalid geometry, non-finite values, identity conflict, or SHA-256 drift.
- Template runs all eight views before aggregate gating. A template stop produces a structurally complete eight-view manifest and marks all downstream branches `SKIPPED`; it does not fabricate scores or later progress states.
- Stage35 uses `configs/zs32/topology/zs32_4cam_double_side_v1.json` and the topology-bound four serials; serial overrides are not accepted by the live path.
- progress/control communication uses atomic JSON only. The dashboard never parses terminal output and never runs GPU inference on the OpenCV main thread.
- Use `uv` for Python commands. Preserve the dirty worktree; never use checkout, restore, reset, or overwrite unrelated staged/unstaged changes.
- Existing real assets are commissioning inputs: `/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json`, `results/zs32_patchcore_eight_view_seed42/eight_view_summary.csv`, `results/yolo/zs32_eight_view_roi_n640_seed42/weights/best.pt`, and `results/zs32_template_gate_right_eight_view_v1/model.json`.
- `front_secondary` PatchCore sample F1 is about `0.286`; model presence is not production acceptance.

---

## File Structure

```text
src/zs32_inspection/domain/views.py              # dependency-free canonical eight-view tuple
capture_data/zs32_runtime_bundle.py              # source validation, assets publication, final bundle loading
pipeline/37_publish_zs32_runtime_bundle.py       # two-phase publish/finalize CLI
config/fusion/zs32_eight_view_bundle_source.json # replaceable model/ROI source declaration
config/fusion/zs32_runtime_models_eight_view.json# generated-compatible Stage32 assets config
config/fusion/zs32_right_eight_view_24_group_commissioning.json # stable rules/template; versions are generated
src/zs32_inspection/dashboard/live.py             # owned Stage35 process group
```

Existing large modules remain in place; the plan adds the bundle boundary instead of expanding Stage35 with asset parsing logic.

### Dirty-worktree execution protocol

Before every task, save `git status --short`, `git diff -- <task paths>`, and `git diff --cached -- <task paths>` in that task's `.git/sdd/task-N-report.md`. Existing same-path changes are inputs, not disposable noise. The implementer must edit incrementally, must not alter the index to hide pre-existing hunks, and must state honestly if a task commit necessarily includes earlier same-file work. Every reviewer receives the full recorded BASE-to-HEAD package plus that report; if safe scope isolation is impossible, stop rather than reset or reconstruct the user's index.

---

### Task 1: Reconcile and freeze the strict eight-view application contract

**Files:**
- Create: `src/zs32_inspection/domain/views.py`
- Modify: `capture_data/zs32_inspection_orchestrator.py`
- Modify: `capture_data/zs32_model_runtime.py`
- Modify: `pipeline/32_run_zs32_multimodel_inference.py`
- Modify: `src/zs32_inspection/dashboard/contracts.py`
- Modify: `src/zs32_inspection/dashboard/parser.py`
- Modify: `src/zs32_inspection/dashboard/compositor.py`
- Modify: `tests/unit/capture_data/test_zs32_inspection_orchestrator.py`
- Modify: `tests/unit/capture_data/test_zs32_model_runtime.py`
- Modify: `tests/unit/pipeline/test_zs32_multimodel_inference.py`
- Modify: `tests/unit/zs32_refactor/dashboard/conftest.py`
- Modify: `tests/unit/zs32_refactor/dashboard/test_contracts.py`
- Modify: `tests/unit/zs32_refactor/dashboard/test_parser.py`
- Modify: `tests/unit/zs32_refactor/dashboard/test_compositor.py`

**Interfaces:**
- Produces: one shared eight-view tuple through `CANONICAL_VIEWS`, `VIEW_ORDER`, and `MODELED_VIEWS`.
- Produces: Stage32 eight required image arguments, eight PatchCore runs, one ordered eight-image YOLO batch, and exact four dashboard branches per view.

- [ ] **Step 1: Add failing strict-contract tests**

```python
def test_all_eight_views_are_modeled() -> None:
    assert MODELED_VIEWS == VIEW_ORDER


@pytest.mark.parametrize("view", VIEW_ORDER)
def test_each_view_requires_model_support(view: str, view_factory: Callable[..., ViewResult]) -> None:
    with pytest.raises(ValueError, match="model_supported"):
        view_factory(view=view, model_supported=False)


def test_stage32_parser_requires_all_eight_images() -> None:
    parser = build_parser()
    required = {action.dest for action in parser._actions if action.required}
    assert {f"{view}_image" for view in VIEW_ORDER} <= required
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/dashboard/test_contracts.py \
  tests/unit/zs32_refactor/dashboard/test_parser.py \
  tests/unit/zs32_refactor/dashboard/test_compositor.py \
  tests/unit/capture_data/test_zs32_inspection_orchestrator.py \
  tests/unit/capture_data/test_zs32_model_runtime.py \
  tests/unit/pipeline/test_zs32_multimodel_inference.py -q
```

Expected: failures identify the remaining six-modeled/secondary-unsupported assumptions or inconsistent eight-view profile counts.

- [ ] **Step 3: Make one eight-view constant contract authoritative**

```python
VIEW_ORDER = (
    "front", "front_left", "front_right", "front_secondary",
    "back", "back_left", "back_right", "back_secondary",
)
MODELED_VIEWS = VIEW_ORDER
CANONICAL_VIEWS = VIEW_ORDER
```

Define these aliases in the dependency-free `zs32_inspection.domain.views` module and import them into capture, runtime, and dashboard modules. Do not import dashboard code from `capture_data`.

`ViewResult.__post_init__` and parser support checks must require `model_supported is True` for every item. Remove compositor's secondary-specific unsupported return; branch state alone determines notices.

- [ ] **Step 4: Finish Stage32 eight-view invariants**

Require runtime configs to contain eight distinct PatchCore checkpoints, an eight-view ROI, and an ordered eight-result YOLO batch. New commissioning assertions use `24`. Keep legacy six-view production profile assertions and their historical `36` groups unchanged; this plan does not create an eight-view production profile.

- [ ] **Step 5: Run regression and commit only Task 1 paths**

Expected: focused command passes; `uv run --no-sync python pipeline/32_run_zs32_multimodel_inference.py --help` lists both secondary image arguments.

Commit message: `feat: freeze strict ZS32 eight-view contracts`.

---

### Task 2: Add the two-phase versioned runtime bundle publisher

**Files:**
- Create: `capture_data/zs32_runtime_bundle.py`
- Create: `pipeline/37_publish_zs32_runtime_bundle.py`
- Create: `config/fusion/zs32_eight_view_bundle_source.json`
- Modify: `capture_data/zs32_model_runtime.py`
- Create: `tests/unit/capture_data/test_zs32_runtime_bundle.py`
- Create: `tests/unit/pipeline/test_zs32_runtime_bundle_cli.py`

**Interfaces:**
- Produces: `RuntimeAssetsPublication`, `RuntimeBundle`, `publish_runtime_assets(source_path, output_dir)`, `finalize_runtime_bundle(assets_manifest, threshold_artifact, output_dir)`, and `load_runtime_bundle(path)`.
- Produces: `runtime_assets.json` for Stage32 and `runtime_bundle.json` for Stage35.

- [ ] **Step 1: Write failing publication and mutation tests**

```python
def test_runtime_assets_publication_contains_exact_eight_views(source_spec: Path, tmp_path: Path) -> None:
    publication = publish_runtime_assets(source_spec, tmp_path / "assets")
    payload = json.loads(publication.runtime_assets.read_text())
    assert tuple(payload["patchcore"]) == VIEW_ORDER
    assert payload["commissioning_only"] is True
    assert payload["production_release_allowed"] is False


def test_bundle_rejects_asset_hash_drift(published_bundle: Path) -> None:
    payload = json.loads(published_bundle.read_text())
    Path(payload["runtime_assets"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_runtime_bundle(published_bundle)
```

Also test missing view, duplicate checkpoint, ROI/template generation mismatch, non-finite threshold, wrong product/hand, output reuse, path escape, and a threshold artifact not bound to the exact runtime-assets/template hashes.

The PatchCore deploy thresholds copied from `eight_view_summary.csv` are provenance fields named `reported_deploy_threshold`; they do not replace the Stage33 low/high calibration records and must not directly drive Stage18.

- [ ] **Step 2: Confirm the tests fail because the publisher does not exist**

```bash
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/capture_data/test_zs32_runtime_bundle.py \
  tests/unit/pipeline/test_zs32_runtime_bundle_cli.py -q
```

- [ ] **Step 3: Implement immutable assets publication**

The checked-in source declaration contains replaceable paths, not copied checkpoint data:

```json
{
  "schema_version": 1,
  "bundle_id": "zs32-right-eight-view-commissioning-v1",
  "product": "ZS32",
  "hand": "right",
  "view_order": [
    "front", "front_left", "front_right", "front_secondary",
    "back", "back_left", "back_right", "back_secondary"
  ],
  "roi_config": "/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json",
  "template_model_dir": "results/zs32_template_gate_right_eight_view_v1",
  "patchcore_summary": "results/zs32_patchcore_eight_view_seed42/eight_view_summary.csv",
  "yolo_weights": "results/yolo/zs32_eight_view_roi_n640_seed42/weights/best.pt",
  "fusion_profile_template": "config/fusion/zs32_right_eight_view_24_group_commissioning.json",
  "commissioning_only": true,
  "production_release_allowed": false
}
```

The publisher reads each checkpoint path and `reported_deploy_threshold` from the summary, verifies the eight view names and file hashes, and emits the Stage32-compatible config.

```python
@dataclass(frozen=True, slots=True)
class RuntimeAssetsPublication:
    runtime_assets: Path
    assets_manifest: Path
    fusion_profile: Path
    asset_set_sha256: str


def publish_runtime_assets(source_path: Path, output_dir: Path) -> RuntimeAssetsPublication:
    source = _load_source(source_path)
    assets = _validate_and_hash_assets(source)
    asset_set_sha256 = _canonical_sha256(assets["asset_set"])
    runtime_path = _write_new_json(output_dir / "runtime_assets.json", assets["runtime"])
    fusion_profile = _write_new_json(
        output_dir / "fusion_profile.json",
        _generate_expected_versions(assets["fusion_policy"], assets["runtime"], assets["template"]),
    )
    manifest_path = _write_new_json(
        output_dir / "assets_manifest.json",
        {**assets["manifest"], "asset_set_sha256": asset_set_sha256,
         "runtime_assets": {"path": str(runtime_path), "sha256": _sha256(runtime_path)}},
    )
    return RuntimeAssetsPublication(runtime_path, manifest_path, fusion_profile, asset_set_sha256)
```

Publish to a staging directory and rename the directory only after all three JSON files are complete. Never overwrite an existing output directory. The checked-in 24-group file supplies stable branch order, rules, and commissioning flags; the publisher regenerates all 24 `expected_versions` from the selected Template/PatchCore/YOLO assets so changing weights never requires hand-editing version rows.

- [ ] **Step 4: Implement finalization without a circular hash**

```python
@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    path: Path
    bundle_id: str
    asset_set_sha256: str
    runtime_assets: Path
    template_model_dir: Path
    fusion_profile: Path
    threshold_artifact: Path


def finalize_runtime_bundle(assets_manifest: Path, threshold_artifact: Path, output_dir: Path) -> Path:
    assets = _validate_assets_manifest(assets_manifest)
    thresholds = _validate_threshold_binding(threshold_artifact, assets)
    payload = _compose_bundle(assets, thresholds)
    return _write_new_json(output_dir / "runtime_bundle.json", payload)
```

The threshold artifact binds the immutable inner `runtime_assets.json` and Template model hashes. The outer bundle records the already-published threshold artifact hash, so there is no recursive hash dependency.

- [ ] **Step 5: Implement CLI phases and verify**

```text
pipeline/37_publish_zs32_runtime_bundle.py publish-assets \
  --source config/fusion/zs32_eight_view_bundle_source.json \
  --output-dir results/zs32_runtime_assets_eight_view_v1

pipeline/37_publish_zs32_runtime_bundle.py finalize \
  --assets-manifest results/zs32_runtime_assets_eight_view_v1/assets_manifest.json \
  --threshold-artifact results/zs32_24group_commissioning_v1/thresholds.json \
  --output-dir results/zs32_runtime_bundle_eight_view_v1
```

Expected: tests pass, help lists both subcommands, and malformed/hashing cases exit nonzero.

- [ ] **Step 6: Commit Task 2**

Commit message: `feat: publish versioned ZS32 runtime bundles`.

---

### Task 3: Validate the real eight-view Template and publish Stage33 calibration inputs

**Files:**
- Modify: `capture_data/zs32_template_gate.py`
- Modify: `pipeline/train_zs32_template_gate.py`
- Modify: `capture_data/zs32_offline_calibration.py`
- Modify: `pipeline/33_run_zs32_offline_calibration.py`
- Modify: `tests/unit/capture_data/test_zs32_template_gate.py`
- Modify: `tests/unit/capture_data/test_zs32_offline_calibration.py`
- Modify: `tests/unit/pipeline/test_zs32_offline_calibration_cli.py`

**Interfaces:**
- Consumes: Task 2 `runtime_assets.json` and the existing eight-view dataset.
- Produces: an exact eight-group right-hand Template model and Stage33 `template_patchcore_threshold_calibration/thresholds.json` plus `yolo_high_precision_auxiliary/thresholds.json`.

- [ ] **Step 1: Add exact-eight Template and calibration tests**

```python
def test_right_only_training_publishes_exact_eight_groups(trained_model: dict[str, object]) -> None:
    assert tuple(trained_model["required_views"]) == VIEW_ORDER
    assert tuple(trained_model["groups"]) == tuple(f"right/{view}" for view in VIEW_ORDER)


def test_stage33_requires_eight_complete_cases(case_index: list[OfflineCalibrationCase]) -> None:
    assert all(tuple(case.images) == VIEW_ORDER for case in case_index)
```

Add failures for missing secondary normal/defect calibration rows, reused physical part across fit/test, runtime/template ROI-version mismatch, and a YOLO auxiliary record missing one view.

- [ ] **Step 2: Run focused tests and confirm RED for any remaining six-view assumptions**

```bash
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/capture_data/test_zs32_template_gate.py \
  tests/unit/capture_data/test_zs32_offline_calibration.py \
  tests/unit/pipeline/test_zs32_offline_calibration_cli.py -q
```

- [ ] **Step 3: Verify the existing Template artifact before retraining**

```bash
PYTHONPATH=. uv run --no-sync python -c \
  "from pathlib import Path; from capture_data.zs32_template_gate import load_model; m=load_model(Path('results/zs32_template_gate_right_eight_view_v1')); assert len(m['groups']) == 8"
```

If this exact command fails, train a new immutable version with:

```bash
PYTHONPATH=. uv run --no-sync python pipeline/train_zs32_template_gate.py \
  --manifest /home/yunjing/anomalib/dataset/zs32_eight_view_patchcore_roi/crop_manifest.csv \
  --path-root /home/yunjing/anomalib \
  --required-hand right \
  --output-dir results/zs32_template_gate_right_eight_view_bundle_v1 \
  --model-version zs32-right-eight-view-template-bundle-v1 \
  --threshold-version zs32-eight-view-commissioning-v1 \
  --roi-version zs32-eight-view-roi-v1 \
  --template-version zs32-right-eight-view-template-bundle-v1
```

The task is blocked rather than weakened if any view lacks normal and defect calibration rows.

- [ ] **Step 4: Publish runtime assets and run Stage33**

```bash
PYTHONPATH=. uv run --no-sync python pipeline/37_publish_zs32_runtime_bundle.py publish-assets \
  --source config/fusion/zs32_eight_view_bundle_source.json \
  --output-dir results/zs32_runtime_assets_eight_view_v1

PYTHONPATH=. uv run --no-sync python pipeline/33_run_zs32_offline_calibration.py \
  --crop-manifest /home/yunjing/anomalib/dataset/zs32_eight_view_patchcore_roi/crop_manifest.csv \
  --template-calibration-csv results/zs32_template_gate_right_eight_view_v1/calibration_rows.csv \
  --template-model-dir results/zs32_template_gate_right_eight_view_v1 \
  --runtime-config results/zs32_runtime_assets_eight_view_v1/runtime_assets.json \
  --yolo-dataset-root /home/yunjing/anomalib/dataset/zs32_eight_view_roi_yolo \
  --path-root /home/yunjing/anomalib \
  --output-dir results/zs32_stage33_eight_view_commissioning_v1 \
  --accelerator gpu --devices 1 --yolo-device 0 \
  --yolo-aux-min-image-precision 0.95 --commissioning-only
```

Expected: both threshold JSON files contain exact right-hand eight-view groups, their publication sidecars match, and no test split was used for selection.

- [ ] **Step 5: Commit code/test changes and record artifact paths**

Commit message: `feat: calibrate strict ZS32 eight-view assets`.

Do not commit checkpoints, generated templates, crops, or calibration results.

---

### Task 4: Finish the 24-group Stage18 contract and finalize a real bundle

**Files:**
- Modify: `capture_data/zs32_18_group_commissioning.py`
- Modify: `capture_data/fusion_engine.py`
- Modify: `capture_data/inspection_audit.py`
- Modify: `pipeline/18_fuse_inspection_results.py`
- Modify: `pipeline/34_publish_zs32_18_group_commissioning.py`
- Modify: `pipeline/32_run_zs32_multimodel_inference.py`
- Modify: `config/fusion/zs32_right_eight_view_24_group_commissioning.json`
- Modify: `tests/unit/capture_data/test_zs32_18_group_commissioning.py`
- Modify: `tests/unit/pipeline/test_zs32_18_group_fusion.py`
- Modify: `tests/unit/pipeline/test_zs32_multimodel_inference.py`

**Interfaces:**
- Produces: a generated `fusion_profile.json`, named baseline profile `zs32-right-24-commissioning`, and a threshold artifact accepted only with the exact Task 3 runtime-assets and Template hashes.

- [ ] **Step 1: Add failing exact-24 tests**

```python
def test_eight_view_profile_has_exact_twenty_four_groups() -> None:
    profile = json.loads(PROFILE_24.read_text())
    assert tuple(profile["required_branches_by_view"]) == VIEW_ORDER
    assert len(profile["expected_versions"]) == 24
    assert {(r["view"], r["branch"]) for r in profile["expected_versions"]} == {
        (view, branch)
        for view in VIEW_ORDER
        for branch in ("template_match", f"anomaly_{view}", "yolo")
    }
```

Test missing/extra/duplicate secondary evidence, wrong expected version, threshold hash drift, and face helpers omitting the fourth view.

- [ ] **Step 2: Run and confirm RED**

```bash
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/capture_data/test_zs32_18_group_commissioning.py \
  tests/unit/pipeline/test_zs32_18_group_fusion.py -q
```

- [ ] **Step 3: Complete profile-driven 24-group publication**

Keep legacy 18-group behavior unchanged. Map `zs32-right-24-commissioning` to the checked-in baseline profile, accept the bundle-generated profile through `--fusion-config`, and make face-level helpers derive their three- or four-view set from the selected profile rather than one hard-coded tuple. Add Stage32 `--fusion-config PATH`, make it mutually exclusive with nonmatching named profiles, and pass that exact path to Stage18 for live bundle execution.

- [ ] **Step 4: Publish the real 24-group artifact**

```bash
PYTHONPATH=. uv run --no-sync python pipeline/34_publish_zs32_18_group_commissioning.py \
  --profile results/zs32_runtime_assets_eight_view_v1/fusion_profile.json \
  --runtime-config results/zs32_runtime_assets_eight_view_v1/runtime_assets.json \
  --template-model results/zs32_template_gate_right_eight_view_v1/model.json \
  --template-patchcore-thresholds results/zs32_stage33_eight_view_commissioning_v1/template_patchcore_threshold_calibration/thresholds.json \
  --yolo-auxiliary-thresholds results/zs32_stage33_eight_view_commissioning_v1/yolo_high_precision_auxiliary/thresholds.json \
  --output-dir results/zs32_24group_commissioning_v1 \
  --commissioning-only
```

Then finalize the one-file Stage35 entrypoint:

```bash
PYTHONPATH=. uv run --no-sync python pipeline/37_publish_zs32_runtime_bundle.py finalize \
  --assets-manifest results/zs32_runtime_assets_eight_view_v1/assets_manifest.json \
  --threshold-artifact results/zs32_24group_commissioning_v1/thresholds.json \
  --output-dir results/zs32_runtime_bundle_eight_view_v1
```

- [ ] **Step 5: Verify and commit**

Expected: Stage18 loads 24 records, the final bundle validates after a new process start, and changing any source byte causes validation failure.

Commit message: `feat: publish ZS32 eight-view commissioning fusion`.

---

### Task 5: Switch Stage35 to topology-driven four-camera eight-view capture

**Files:**
- Modify: `capture_data/zs32_live_commissioning.py`
- Modify: `pipeline/35_run_zs32_live_commissioning.py`
- Modify: `tests/unit/capture_data/test_zs32_live_commissioning.py`
- Modify: `tests/unit/pipeline/test_zs32_live_commissioning_cli.py`

**Interfaces:**
- Consumes: Task 4 final `runtime_bundle.json`.
- Produces: an exact eight-view `CapturedSample`, a four-camera capture command, an eight-image Stage32 command, and an eight-view final manifest.

- [ ] **Step 1: Write failing topology and command tests**

```python
def test_stage35_uses_topology_capture_and_all_eight_stage32_images(config: LiveRunConfig, sample: CapturedSample) -> None:
    capture = build_capture_command(config, config.capture_root / config.run_id)
    assert capture[1].endswith("pipeline/zs32_bootstrap_capture.py")
    assert "--topology" in capture
    stage32 = build_stage32_command(config, sample, config.output_root / sample.part_id)
    assert all(f"--{view.replace('_', '-')}-image" in stage32 for view in VIEW_ORDER)
    assert "--fusion-config" in stage32
    assert str(load_runtime_bundle(config.runtime_config).fusion_profile) in stage32
```

Cover missing/duplicate view, wrong serial/round, mixed session, unsafe path, incomplete sample row, and bundle/preflight failure before capture.

- [ ] **Step 2: Confirm existing three-camera code fails the new tests**

```bash
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/domain/test_topology.py \
  tests/unit/zs32_refactor/capture_data/test_legacy_dataset_store.py \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py -q
```

- [ ] **Step 3: Build the locked capture command**

```python
return [
    sys.executable, str(config.repo_root / "pipeline/zs32_bootstrap_capture.py"),
    "--topology", str(config.topology_path),
    "--root", str(capture_run_root),
    "--capture-session", config.run_id,
    "--legacy-layout", "--hand", "right", "--label", "normal",
    "--part-id", config.part_id, "--group-count", "1", "--images-per-group", "1",
    "--manual-load", "--hdr",
]
```

Remove front/left/right serial overrides from `LiveRunConfig` and CLI. Load serial identity only from the checked-in topology.

- [ ] **Step 4: Resolve the final bundle before opening cameras**

```python
bundle = load_runtime_bundle(config.runtime_config)
validate_commissioning_source_assets(
    bundle.threshold_artifact,
    bundle.runtime_assets,
    bundle.template_model_dir / "model.json",
)
```

Pass the inner runtime assets, bundle Template, generated fusion-profile path, and threshold paths to Stage32. Publish eight source records without changing Stage32 machine status or errors.

- [ ] **Step 5: Run regressions and commit**

Expected: all five test files pass; Stage35 `--help` exposes `--runtime-config` and `--topology`, and no serial override options.

Commit message: `feat: capture strict ZS32 eight-view live samples`.

---

### Task 6: Connect real capture and inference boundaries to progress/control JSON

**Files:**
- Modify: `src/zs32_inspection/capture/bootstrap.py`
- Modify: `src/zs32_inspection/cli/bootstrap_capture.py`
- Modify: `pipeline/32_run_zs32_multimodel_inference.py`
- Modify: `capture_data/zs32_live_commissioning.py`
- Create: `tests/unit/capture_data/test_zs32_live_progress.py`
- Modify: `tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py`
- Modify: `tests/unit/pipeline/test_zs32_multimodel_inference.py`

**Interfaces:**
- Consumes: dashboard `write_progress`, `consume_confirmation`, `ProgressRecord`, and `ConfirmationCommand`.
- Produces: `--progress-json`, `--control-json`, no-TTY confirmation, and the exact live state sequence.

- [ ] **Step 1: Write failing sequence and token tests**

```python
assert states == [
    "waiting_front", "capturing_front", "waiting_back", "capturing_back",
    "running_template", "running_patchcore_yolo", "running_fusion", "complete",
]
```

Also assert a Template stop ends with `running_template, complete`; wrong/stale/duplicate confirmation never captures; a correct token captures exactly once without stdin/stdout TTY; any exception writes `failed` with the same part identity.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/capture_data/test_zs32_live_progress.py \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py \
  tests/unit/pipeline/test_zs32_multimodel_inference.py -q
```

- [ ] **Step 3: Emit states at real boundaries**

Generate a fresh `uuid4().hex` before each waiting state. Write `capturing_*` only after a matching confirmation is consumed and before triggering cameras. Write `running_template`, `running_patchcore_yolo`, and `running_fusion` immediately before the corresponding real operations.

- [ ] **Step 4: Make progress failure fail closed**

Do not swallow write/replace exceptions. On a business or child-process error, attempt one `failed` write; preserve the original error as the cause if that write also fails.

- [ ] **Step 5: Verify and commit**

Commit message: `feat: report ZS32 eight-view live progress`.

---

### Task 7: Connect the dashboard contextual action to an owned Stage35 process group

**Files:**
- Create: `src/zs32_inspection/dashboard/live.py`
- Modify: `src/zs32_inspection/dashboard/app.py`
- Modify: `src/zs32_inspection/cli/dashboard.py`
- Create: `tests/unit/zs32_refactor/dashboard/test_live.py`
- Modify: `tests/unit/zs32_refactor/dashboard/test_app.py`
- Modify: `tests/unit/zs32_refactor/runtime/test_dashboard_cli.py`

**Interfaces:**
- Produces: `Stage35Controller.start(part_id)`, `poll()`, `confirm()`, and `close(timeout=5.0)`.

- [ ] **Step 1: Write failing process ownership tests**

```python
controller.start("part-1")
with pytest.raises(RuntimeError, match="already running"):
    controller.start("part-1")
controller.confirm()
assert read_control(controller.control_path).round == "front"
controller.close()
fake_killpg.assert_called_once_with(fake_process.pid, signal.SIGTERM)
```

Cover nonzero exit, failed progress, timeout then SIGKILL, non-owned PID no-op, confirm outside waiting no-op, repeated confirm, and dashboard close cleanup.

- [ ] **Step 2: Confirm RED**

```bash
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/dashboard/test_live.py \
  tests/unit/zs32_refactor/dashboard/test_app.py \
  tests/unit/zs32_refactor/runtime/test_dashboard_cli.py -q
```

- [ ] **Step 3: Implement owned process-group lifecycle**

```python
self.process = subprocess.Popen(
    command,
    start_new_session=True,
    stdout=log_file,
    stderr=subprocess.STDOUT,
)
```

On close, signal only `self.process.pid` as the process-group ID, wait up to five seconds, then SIGKILL and wait again. `confirm()` writes a command only for current `waiting_front/back` progress with a non-empty confirmation ID.

- [ ] **Step 4: Connect the one contextual CTA**

`开始检测 [S]` starts Stage35. `确认正面并拍摄 [S]` and `确认背面并拍摄 [S]` call `confirm()`. All other running states disable the action. On complete, reload the final eight-view result through the strict parser.

- [ ] **Step 5: Verify and commit**

Expected: live/dashboard tests pass; headless mode still never calls HighGUI; closing never leaves the fake process group alive.

Commit message: `feat: control ZS32 eight-view live inspection`.

---

### Task 8: Run full regression, real-asset smoke, one-part hardware smoke, and document the result

**Files:**
- Modify: `pipeline/README.md`
- Modify: `docs/ZS32_FOUR_CAMERA_END_TO_END_README.md`
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

**Interfaces:**
- Produces: copyable offline/live commands, measured artifact paths, actual test counts, and a bounded commissioning acceptance statement.

- [ ] **Step 1: Run the full automated regression**

```bash
PYTHONPATH=. uv run --no-sync pytest --confcutdir=tests/unit \
  tests/unit/zs32_refactor/dashboard \
  tests/unit/zs32_refactor/runtime/test_dashboard_cli.py \
  tests/unit/zs32_refactor/domain/test_topology.py \
  tests/unit/zs32_refactor/capture_data/test_legacy_dataset_store.py \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py \
  tests/unit/capture_data/test_zs32_template_gate.py \
  tests/unit/capture_data/test_zs32_model_runtime.py \
  tests/unit/capture_data/test_zs32_18_group_commissioning.py \
  tests/unit/capture_data/test_zs32_live_commissioning.py \
  tests/unit/capture_data/test_zs32_live_progress.py \
  tests/unit/capture_data/test_zs32_runtime_bundle.py \
  tests/unit/pipeline/test_zs32_multimodel_inference.py \
  tests/unit/pipeline/test_zs32_18_group_fusion.py \
  tests/unit/pipeline/test_zs32_live_commissioning_cli.py \
  tests/unit/pipeline/test_zs32_runtime_bundle_cli.py -q
```

Run compileall on all modified Python modules, every affected CLI `--help`, Ruff core/format checks, and `git diff --check`.

- [ ] **Step 2: Run a strict real-asset offline smoke**

Use one existing same-identity eight-view group and the finalized bundle. Run Stage32 in `fuse --fusion-config results/zs32_runtime_assets_eight_view_v1/fusion_profile.json` mode, then run Stage36 headless against the output.

Acceptance assertions:

```python
assert tuple(manifest["views"]) == VIEW_ORDER
assert all(manifest["views"][view]["model_supported"] is True for view in VIEW_ORDER)
assert all(set(manifest["views"][view]["branches"]) == {"template", "patchcore", "yolo", "fusion"} for view in VIEW_ORDER)
assert summary["commissioning_only"] is True
assert summary["production_release_allowed"] is False
```

- [ ] **Step 3: Run one four-camera part through the dashboard**

```bash
PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-} \
uv run --no-sync python pipeline/36_zs32_inspection_dashboard.py \
  --live --part-id live_part_001 \
  --runtime-config results/zs32_runtime_bundle_eight_view_v1/runtime_bundle.json
```

Verify two distinct confirmation IDs were consumed once, legacy manifest contains eight image rows plus one complete sample row, Stage32 received eight image arguments, Stage18 consumed 24 groups, the dashboard loaded eight modeled views, and no owned process remains after exit.

- [ ] **Step 4: Record measured evidence without overclaiming**

Write exact commands, test counts, bundle ID/hash, capture session, group ID, result paths, final status, and whether hardware smoke succeeded. If hardware is unavailable or a model/threshold blocks completion, record the exact blocker and do not write that the smoke passed.

- [ ] **Step 5: Commit documentation and memory only**

Commit message: `docs: record ZS32 eight-view commissioning acceptance`.

---

## Completion Standard

- Eight views are modeled end to end; there is no secondary unsupported path in the new chain.
- Weight replacement is performed by publishing and selecting a new runtime bundle, without Python path edits.
- One immutable bundle binds the exact ROI, Template, eight PatchCore checkpoints, shared YOLO, 24-group profile, and threshold artifact.
- Offline real-asset smoke produces an eight-view, four-branch manifest accepted by the dashboard.
- One hardware run, if devices and operator interaction are available, produces one identity-consistent eight-image sample and cleans up every owned process.
- Every report remains commissioning-only and explicitly excludes production release.
