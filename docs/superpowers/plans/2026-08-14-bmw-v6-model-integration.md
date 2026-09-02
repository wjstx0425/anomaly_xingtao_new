# BMW V6 Model Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one no-overwrite command that publishes the completed 2026-08-14 Template/EfficientAD models into an isolated V6 Demo while retaining V5 masks and numeric thresholds, V5 YOLO, and the new rotated bright-streak candidate.

**Architecture:** A focused publisher validates every source, creates a composite run with symlinks and rebound threshold assets, then emits one V6 experiment JSON. A separate strict bright-streak candidate loader keeps the accepted V5 tracked-profile loader unchanged.

**Tech Stack:** Python 3.13, pathlib, hashlib, JSON, OpenCV, pytest, uv.

## Global Constraints

- Never overwrite or edit the V5 config or V5 model assets.
- Template and EfficientAD retain the exact V5 numeric thresholds and the exact V5 manual ignore-mask asset.
- New model/checkpoint SHA-256 values replace only the old model identities in new rebound assets.
- YOLO remains the V5 checkpoint and its SHA-256 is recorded in the composite receipt.
- V6 uses the V5 public ROI; the publisher must prove its coordinates equal the new training ROI.
- The new rotated bright-streak report remains explicitly candidate-only and must not weaken the accepted V5 loader.
- Publication fails until the training report is complete, all eight models exist, and the final EfficientAD threshold artifact exists.

---

### Task 1: Strict rotated bright-streak candidate runtime

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

**Interfaces:**
- Consumes: candidate `report.json`, its config SHA, the accepted rotated ROI path/SHA.
- Produces: `bright_streak_engine="tracked_profile_v3_manual_rotated_candidate"` and a predictor that returns the existing `ModelOutput` contract.

- [ ] **Step 1: Write failing config and predictor tests**

```python
def test_candidate_engine_requires_exact_report_and_rotated_roi_sha(tmp_path):
    config = load_demo_config(_candidate_config(tmp_path))
    assert config.bright_streak_engine == "tracked_profile_v3_manual_rotated_candidate"

def test_candidate_predictor_uses_report_thresholds(candidate_report, roi_asset):
    predictor = EightViewRotatedTrackedProfileCandidatePredictor(candidate_report, roi_asset)
    output = predictor.predict(sample_hdr)
    assert output.details["candidate_only"] is True
```

- [ ] **Step 2: Run RED tests**

Run: `uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_eight_view_demo.py tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py -k rotated_candidate`

Expected: failure because the engine and predictor do not exist.

- [ ] **Step 3: Implement the minimal strict loader**

Add a candidate predictor that requires schema `bmw.bright_streak_rotated_v3_candidate/1.0`, `status=complete`, `candidate_only=true`, algorithm `tracked_profile_v3_manual_rotated_roi`, exact threshold keys, source/ROI SHA matches, and explicit `no_streak_independent_test_count=0`. Reuse `analyze_tracked_profile`, `classify_tracked_profile`, `rectify_bright_streak_roi`, and the existing evidence renderer; do not copy their math.

- [ ] **Step 4: Run GREEN tests**

Run: `uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_eight_view_demo.py tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py -k 'rotated_candidate or rotated_roi'`

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/bmw_inspection/lab/eight_view_demo.py src/bmw_inspection/lab/eight_view_demo_models.py tests/unit/bmw_inspection/lab/test_eight_view_demo.py tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py
git commit -m "feat: load rotated bright streak candidate in BMW demo"
```

---

### Task 2: Atomic V6 composite publisher

**Files:**
- Create: `src/bmw_inspection/lab/v6_demo_publisher.py`
- Test: `tests/unit/bmw_inspection/lab/test_v6_demo_publisher.py`

**Interfaces:**
- Consumes: `publish_v6_demo(repo_root: Path, *, output_run: Path, output_config: Path) -> dict[str, object]`.
- Produces: composite symlinks, `template_thresholds_v5_rebound.json`, `efficientad_thresholds_v5_rebound.json`, `composition.json`, and V6 JSON.

- [ ] **Step 1: Write failing publisher contract tests**

```python
def test_rejects_incomplete_eight_view_training(fixture):
    fixture.remove_checkpoint("back_secondary")
    with pytest.raises(ValueError, match="八视角"):
        publish_v6_demo(**fixture.arguments)

def test_rebinds_sha_but_preserves_threshold_values(fixture):
    report = publish_v6_demo(**fixture.arguments)
    assert report["template_thresholds_unchanged"] is True
    assert report["efficientad_thresholds_unchanged"] is True
    assert report["yolo_sha256"] == fixture.yolo_sha256

def test_keeps_v5_bytes_unchanged(fixture):
    before = fixture.v5_config.read_bytes()
    publish_v6_demo(**fixture.arguments)
    assert fixture.v5_config.read_bytes() == before
```

- [ ] **Step 2: Run RED tests**

Run: `uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_v6_demo_publisher.py`

Expected: import failure because the publisher module does not exist.

- [ ] **Step 3: Implement source validation and threshold rebinding**

The publisher must require `run_report.status == "complete"`, eight Template `model.json`, eight EfficientAD `model.ckpt`, eight metrics, and `efficientad/score_analysis/part_thresholds.json`. Copy V5 threshold JSON objects, assert numeric mappings are unchanged, update only model SHA fields, and add:

```python
payload["thresholds_recalibrated"] = False
payload["rebound_from"] = str(source_asset)
payload["rebound_from_sha256"] = sha256(source_asset)
```

- [ ] **Step 4: Implement no-overwrite publication**

Create a sibling temporary directory, write both rebound assets and `composition.json`, create relative or absolute directory symlinks for `template`, `efficientad`, and `yolo`, then rename the staging directory to the fresh output. Write the V6 config through a temporary file and `Path.replace`; reject an existing destination before staging.

- [ ] **Step 5: Run GREEN tests**

Run: `uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_v6_demo_publisher.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/bmw_inspection/lab/v6_demo_publisher.py tests/unit/bmw_inspection/lab/test_v6_demo_publisher.py
git commit -m "feat: publish BMW V6 composite demo"
```

---

### Task 3: One-command handoff and integration verification

**Files:**
- Create: `pipeline/bmw_lab_prepare_normal_20260814_v6_demo.py`
- Modify: `AGENTS_MEMORY.md`
- Test: `tests/unit/pipeline/test_bmw_lab_prepare_normal_v6_demo.py`

**Interfaces:**
- Consumes: optional CLI path overrides; defaults to the exact approved 2026-08-14 sources.
- Produces: JSON success/error output and a copy-pastable V6 launch command.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_cli_reports_incomplete_training_without_partial_output(monkeypatch, capsys):
    monkeypatch.setattr(module, "publish_v6_demo", _raise_incomplete)
    assert module.main([]) == 2
    assert "训练尚未完成" in capsys.readouterr().err
```

- [ ] **Step 2: Run RED test**

Run: `uv run --no-sync pytest -q tests/unit/pipeline/test_bmw_lab_prepare_normal_v6_demo.py`

Expected: import failure because the CLI does not exist.

- [ ] **Step 3: Implement the CLI**

Use repository defaults for the new training run, V5 config, training ROI, rotated candidate report, output composite, and output config. Catch `FileExistsError`, `OSError`, `TypeError`, and `ValueError`; print one Chinese JSON error and return 2. On success print the receipt plus:

```text
uv run --no-sync python pipeline/bmw_lab_eight_view_demo.py --config configs/bmw/experiments/bmw_eight_view_demo_v6_right_normal_20260814_v1.json --experiment-mode
```

- [ ] **Step 4: Run focused and integration tests**

Run: `uv run --no-sync pytest -q tests/unit/pipeline/test_bmw_lab_prepare_normal_v6_demo.py tests/unit/bmw_inspection/lab/test_v6_demo_publisher.py tests/unit/bmw_inspection/lab/test_eight_view_demo.py tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

Expected: all tests pass.

- [ ] **Step 5: Run static verification**

Run: `uv run --no-sync python -m compileall -q pipeline/bmw_lab_prepare_normal_20260814_v6_demo.py src/bmw_inspection/lab/v6_demo_publisher.py src/bmw_inspection/lab/eight_view_demo.py src/bmw_inspection/lab/eight_view_demo_models.py && git diff --check`

Expected: exit 0 and no output.

- [ ] **Step 6: Commit Task 3**

```bash
git add pipeline/bmw_lab_prepare_normal_20260814_v6_demo.py tests/unit/pipeline/test_bmw_lab_prepare_normal_v6_demo.py AGENTS_MEMORY.md
git commit -m "feat: add one-command BMW V6 model handoff"
```
