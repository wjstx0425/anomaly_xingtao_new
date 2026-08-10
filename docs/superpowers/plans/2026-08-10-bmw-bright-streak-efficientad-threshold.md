# BMW Bright-Streak and EfficientAD Part-Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle BMW bright-streak connected-component selector with a center-ridge tracker and make the eight-view EfficientAD branch meet an observed whole-part normal false-positive rate of at most 5% on the current laboratory score set.

> 状态：已实施。下方未勾选框保留的是原始执行步骤，不是当前进度记录；实测数据、产物路径和限制以 `BMW_EIGHT_VIEW_HANDOFF_README.md` 为准。

**Architecture:** Keep the existing fixed ROI, image-quality checks, branch results, and Fusion rule. Bright-streak extraction becomes a row-wise center-corridor tracker but continues to publish the existing `BrightStreakDecision` contract. EfficientAD gains a small offline joint-threshold module and hash-bound JSON artifact; the Demo loads eight per-view thresholds and applies them to runtime scores instead of checkpoint `pred_label`.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, Anomalib EfficientAD, pytest, `uv`.

## Global Constraints

- This remains an experimental laboratory Demo; do not claim an industrial acceptance rate.
- Keep “bright streak exists and is continuous => OK”; absence and broken continuity remain distinct NG outcomes.
- Do not retrain EfficientAD, YOLO, or Template in this change.
- EfficientAD whole-part normal false-positive target is at most 5%; with 20 current normal parts this means at most one rejected part.
- Mark the threshold artifact `demo_only=true` and `test_used_for_selection=true` because the present held-out scores are used for selection.
- Do not special-case sample IDs and do not change the global “any required NG => part NG” Fusion rule.
- Preserve unrelated untracked files in the existing worktree and stage only paths named by this plan.

---

### Task 1: Row-wise center-ridge bright-streak extraction

**Files:**
- Modify: `src/bmw_inspection/detector.py`
- Test: `tests/unit/bmw_inspection/test_detector.py`

**Interfaces:**
- Consumes: existing `BrightStreakConfig` fields `background_kernel_px`, `response_mad_scale`, `center_tolerance_px`, `max_component_width_px`, and `micro_gap_close_px`.
- Produces: `_coherent_ridge_rows(peaks: np.ndarray, present: np.ndarray, *, max_step: int) -> np.ndarray` and `_track_center_ridge(roi: np.ndarray, response: np.ndarray, config: BrightStreakConfig) -> tuple[np.ndarray, float]`; existing `detect_bright_streak_evidence(...) -> BrightStreakDecision` remains unchanged.

- [ ] **Step 1: Add failing synthetic regressions**

Add two tests. The first draws a thin bright line whose x coordinate bends gradually and whose local width changes; it must produce substantial coverage. The second contains bright texture in the ROI but no coherent central trajectory; it must remain `NG_NO_STREAK`.

```python
def _curved_streak_image() -> np.ndarray:
    yy, xx = np.indices((120, 160))
    image = (76 + ((3 * xx + 5 * yy) % 9)).astype(np.uint8)
    for y in range(15, 105):
        center = 80 + int(round(5 * np.sin(y / 18.0)))
        half_width = 1 + int(y % 17 == 0)
        image[y, center - half_width : center + half_width + 1] = 210
    return image


def test_curved_variable_width_streak_is_tracked(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_synthetic_config(config_path)
    decision = detect_bright_streak_evidence(_curved_streak_image(), load_config(config_path))
    assert decision.status is DemoStatus.OK
    assert decision.metrics is not None
    assert decision.metrics.coverage_ratio >= 0.75


def test_unstructured_center_texture_is_not_a_streak(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_synthetic_config(config_path)
    image = np.full((120, 160), 80, dtype=np.uint8)
    image[15:105:7, 73:88] = 150
    decision = detect_bright_streak_evidence(image, load_config(config_path))
    assert decision.status is DemoStatus.NG_NO_STREAK
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest \
  tests/unit/bmw_inspection/test_detector.py::test_curved_variable_width_streak_is_tracked \
  tests/unit/bmw_inspection/test_detector.py::test_unstructured_center_texture_is_not_a_streak -q
```

Expected: at least the curved-streak regression fails under the old whole-component filter.

- [ ] **Step 3: Implement the minimal ridge tracker**

In `detector.py`, add a helper that uses a horizontally smoothed top-hat response, restricts candidates to the configured central corridor, computes one peak per row, rejects weak rows using the existing median/MAD threshold, and connects rows only when peak x changes no more than the configured width allowance.

```python
def _coherent_ridge_rows(peaks: np.ndarray, present: np.ndarray, *, max_step: int) -> np.ndarray:
    accepted = present.astype(bool, copy=True)
    previous: int | None = None
    for row in np.flatnonzero(present):
        if previous is not None and row - previous == 1 and abs(int(peaks[row]) - int(peaks[previous])) > max_step:
            accepted[row] = False
            continue
        previous = int(row)
    return accepted


def _track_center_ridge(
    roi: np.ndarray,
    response: np.ndarray,
    config: BrightStreakConfig,
) -> tuple[np.ndarray, float]:
    smoothed = cv2.GaussianBlur(response, (3, 1), 0)
    center = (roi.shape[1] - 1) / 2.0
    left = max(0, int(math.floor(center - config.center_tolerance_px)))
    right = min(roi.shape[1], int(math.ceil(center + config.center_tolerance_px + 1)))
    corridor = smoothed[:, left:right].astype(np.float32)
    median = float(np.median(response))
    mad = float(np.median(np.abs(response.astype(np.float32) - median)))
    threshold = median + config.response_mad_scale * max(1.0, mad)
    peaks = corridor.argmax(axis=1) + left
    strengths = corridor.max(axis=1)
    present = strengths >= threshold
    # Retain coherent runs; short row gaps are handled later by `_metrics`.
    # Build a one-pixel decision mask at each accepted peak so width variation
    # cannot discard an otherwise valid line.
    mask = np.zeros_like(roi, dtype=bool)
    accepted_rows = _coherent_ridge_rows(peaks, present, max_step=max(2, int(config.max_component_width_px)))
    mask[np.flatnonzero(accepted_rows), peaks[accepted_rows]] = True
    raw_median = float(np.median(roi))
    raw_mad = float(np.median(np.abs(roi.astype(np.float32) - raw_median)))
    contrast = float(max(0.0, (float(strengths[accepted_rows].mean()) - median) / max(1.0, raw_mad))) \
        if accepted_rows.any() else 0.0
    return mask, contrast
```

Use this output in the non-saturated path of `detect_bright_streak_evidence`; retain the existing saturated fast path, quality failures, `_metrics`, decision thresholds, and evidence rendering.

- [ ] **Step 4: Run the detector suite and make it GREEN**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest tests/unit/bmw_inspection/test_detector.py -q
```

Expected: all detector tests pass, including absence, broken continuity, deterministic evidence, and the new curved-line case.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/bmw_inspection/detector.py tests/unit/bmw_inspection/test_detector.py
git commit -m "fix: track BMW bright streak by center ridge"
```

---

### Task 2: Real-data bright-streak recalibration command

**Files:**
- Create: `src/bmw_inspection/lab/bright_streak_recalibration.py`
- Create: `pipeline/bmw_lab_recalibrate_bright_streak.py`
- Create: `tests/unit/bmw_inspection/lab/test_bright_streak_recalibration.py`
- Modify: `BMW_EIGHT_VIEW_HANDOFF_README.md`

**Interfaces:**
- Consumes: prepared `manifests/bright_streak.csv`, base/configured ROI JSON, and `detect_bright_streak_evidence`.
- Produces: `recalibrate_bright_streak(manifest_path: Path, base_config_path: Path, output_dir: Path) -> dict[str, object]`, `calibrated_config.json`, `metrics.csv`, `report.json`, and evidence PNGs for every final-test error.

- [ ] **Step 1: Write failing calibration tests**

Use small generated images and a temporary manifest to assert that calibration uses only `split=calibration`, evaluation preserves `split=final_test`, output JSON is loadable, and sample IDs are never used in decisions.

```python
def test_recalibration_fits_calibration_and_reports_final_test(tmp_path: Path) -> None:
    manifest, config = _synthetic_recalibration_inputs(tmp_path)
    report = recalibrate_bright_streak(manifest, config, tmp_path / "out")
    assert report["fit_split"] == "calibration"
    assert report["final_test_used_for_fit"] is False
    assert Path(report["config"]).is_file()
    assert load_config(Path(report["config"])).roi_xyxy is not None
```

- [ ] **Step 2: Run and confirm RED**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest \
  tests/unit/bmw_inspection/lab/test_bright_streak_recalibration.py -q
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement recalibration and CLI**

Move/reuse the pure `_balanced_accuracy`, `_candidate_thresholds`, and threshold fitting logic from `eight_view_train_all.py` without starting any model training. Publish the detector/config identities and final-test outcomes. The CLI defaults are:

```python
DEFAULT_MANIFEST = REPO_ROOT / "dataset/bmw_lab_prepared/bmw_hdr_eight_view_v1/manifests/bright_streak.csv"
DEFAULT_BASE_CONFIG = REPO_ROOT / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak/calibrated_config.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak_ridge_v2"
```

- [ ] **Step 4: Run tests and CLI help**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest \
  tests/unit/bmw_inspection/lab/test_bright_streak_recalibration.py \
  tests/unit/bmw_inspection/lab/test_eight_view_train_all.py -q
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync python pipeline/bmw_lab_recalibrate_bright_streak.py --help
```

Expected: tests pass and help exits 0.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/bmw_inspection/lab/bright_streak_recalibration.py \
  pipeline/bmw_lab_recalibrate_bright_streak.py \
  tests/unit/bmw_inspection/lab/test_bright_streak_recalibration.py \
  BMW_EIGHT_VIEW_HANDOFF_README.md
git commit -m "feat: recalibrate BMW bright streak without retraining"
```

---

### Task 3: EfficientAD whole-part threshold optimizer

**Files:**
- Create: `src/bmw_inspection/lab/efficientad_thresholds.py`
- Create: `pipeline/bmw_lab_calibrate_efficientad_thresholds.py`
- Create: `tests/unit/bmw_inspection/lab/test_efficientad_thresholds.py`
- Modify: `src/bmw_inspection/lab/efficientad_analysis.py`

**Interfaces:**
- Consumes: `efficientad_scores.csv` with `view_id`, `label`, `score`, and `image_path`.
- Produces: `fit_part_thresholds(rows: Sequence[PartScore], views: Sequence[str], target_part_fpr: float) -> PartThresholdFit`; JSON fields `thresholds`, `normal_part_count`, `normal_false_positive_count`, `observed_normal_part_fpr`, `defect_part_count`, `defect_detected_count`, `demo_only`, and `test_used_for_selection`.

- [ ] **Step 1: Write optimizer contract tests**

Cover eight-view completeness, part grouping, `floor(target_fpr * normal_count)`, deterministic tie-breaking, a threshold greater than the maximum normalized score when one noisy view must be disabled, and validation of non-finite scores.

```python
def test_joint_fit_limits_union_of_false_positive_parts() -> None:
    rows = _eight_view_rows(normal_parts=20, defect_parts=2)
    fit = fit_part_thresholds(rows, views=VIEW_ORDER, target_part_fpr=0.05)
    assert fit.normal_part_count == 20
    assert fit.normal_false_positive_count <= 1
    assert fit.observed_normal_part_fpr <= 0.05
    assert tuple(fit.thresholds) == VIEW_ORDER


def test_fit_optimizes_defect_part_recall_before_image_recall() -> None:
    fit = fit_part_thresholds(_tradeoff_rows(), views=VIEW_ORDER, target_part_fpr=0.05)
    assert fit.defect_detected_count == 2
```

- [ ] **Step 2: Run and confirm RED**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest \
  tests/unit/bmw_inspection/lab/test_efficientad_thresholds.py -q
```

Expected: import failure because the optimizer does not exist.

- [ ] **Step 3: Implement deterministic joint fitting**

For any feasible solution with at most one false-positive normal part, enumerate `None` plus each normal part as the only allowed rejected part. For each candidate, set every view threshold just above the maximum score of all other normal parts; this is the lowest threshold that protects them and therefore maximizes defect detection for that candidate.

```python
allowed_false_positives = math.floor(target_part_fpr * len(normal_parts) + 1e-12)
if allowed_false_positives != 1:
    # General implementation enumerates combinations up to the permitted count;
    # the current 20-part/5% case has exactly 21 candidates.
    allowed_sets = _allowed_part_sets(normal_parts, allowed_false_positives)
for allowed in allowed_sets:
    thresholds = {
        view: math.nextafter(
            max(row.score for row in normal_rows if row.view_id == view and row.part_id not in allowed),
            math.inf,
        )
        for view in views
    }
    evaluation = evaluate_part_thresholds(rows, thresholds)
    candidates.append((evaluation.defect_detected_count, evaluation.defect_image_hits, thresholds, evaluation))
```

Select by defect-part hits, defect-image hits, fewer normal false-positive parts, then deterministic threshold/view tuple. Write the source CSV SHA256 and explicit leakage flags.

- [ ] **Step 4: Implement the CLI and threshold-aware plots**

The CLI defaults to the existing score analysis CSV and writes `part_thresholds.json` plus `part_threshold_report.json`. Update score distribution rendering to accept a per-view threshold mapping so each subplot displays its actual threshold.

- [ ] **Step 5: Run tests and help**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest \
  tests/unit/bmw_inspection/lab/test_efficientad_thresholds.py \
  tests/unit/bmw_inspection/lab/test_efficientad_analysis.py -q
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync python \
  pipeline/bmw_lab_calibrate_efficientad_thresholds.py --help
```

Expected: all tests pass and help exits 0.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/bmw_inspection/lab/efficientad_thresholds.py \
  src/bmw_inspection/lab/efficientad_analysis.py \
  pipeline/bmw_lab_calibrate_efficientad_thresholds.py \
  tests/unit/bmw_inspection/lab/test_efficientad_thresholds.py \
  tests/unit/bmw_inspection/lab/test_efficientad_analysis.py
git commit -m "feat: fit BMW EfficientAD whole-part thresholds"
```

---

### Task 4: Load per-view EfficientAD thresholds in the Demo

**Files:**
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_v1.json`
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

**Interfaces:**
- Consumes: the Task 3 `part_thresholds.json` artifact.
- Produces: `EightViewDemoConfig.efficientad_thresholds: Mapping[str, float]`; `EightViewEfficientAdPredictor(checkpoints, thresholds=..., predictor_factory=...)`.

- [ ] **Step 1: Write failing config/runtime tests**

Add an artifact fixture with eight ordered thresholds. Assert that the Demo loads them, rejects a missing view, and bases runtime status on `score >= configured_threshold` even when the checkpoint-provided boolean disagrees.

```python
def test_efficientad_predictor_uses_deployment_threshold_not_pred_label(tmp_path: Path) -> None:
    checkpoints = _checkpoints(tmp_path)
    predictor = EightViewEfficientAdPredictor(
        checkpoints,
        thresholds={view: 0.8 for view in VIEW_ORDER},
        predictor_factory=lambda _path: lambda _image: (0.6, True, np.zeros((4, 4), np.float32)),
    )
    output = predictor.predict("front", np.zeros((8, 8, 3), np.uint8))
    assert output.status is BranchStatus.PASS
    assert output.threshold == pytest.approx(0.8)
```

- [ ] **Step 2: Run and confirm RED**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest \
  tests/unit/bmw_inspection/lab/test_eight_view_demo.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py -q
```

Expected: constructor/config assertions fail because only threshold `0.5` exists.

- [ ] **Step 3: Implement strict artifact loading and runtime comparison**

Add an `efficientad` section to the Demo JSON:

```json
"efficientad": {
  "threshold_artifact": "../../../results/bmw_lab_one_click/bmw_lab_eight_view_v1/efficientad/score_analysis/part_thresholds.json"
}
```

Validate the artifact flags and exact view set. Pass the thresholds into `EightViewEfficientAdPredictor`; ignore only the boolean `pred_label`, retain the returned score/map, and use `score >= thresholds[view]`. Use `fixed_scale_heatmap` for honest cross-sample visualization.

- [ ] **Step 4: Run Demo-model and config tests**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest \
  tests/unit/bmw_inspection/lab/test_eight_view_demo.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py -q
```

Expected: all asset-independent tests pass; tests that require local model files remain explicitly deselected when running the established selector.

- [ ] **Step 5: Commit Task 4**

```bash
git add configs/bmw/experiments/bmw_eight_view_demo_v1.json \
  src/bmw_inspection/lab/eight_view_demo.py \
  src/bmw_inspection/lab/eight_view_demo_models.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py
git commit -m "feat: deploy BMW EfficientAD part thresholds"
```

---

### Task 5: Real-data run, documentation, and publication

**Files:**
- Modify: `BMW_EIGHT_VIEW_HANDOFF_README.md`
- Modify outside branch history as project memory: `/home/yunjing/anomaly_xingtao_new/AGENTS_MEMORY.md`
- Generate locally under ignored results: `results/bmw_lab_one_click/bmw_lab_eight_view_v1/bright_streak_ridge_v2/`
- Generate locally under ignored results: `results/bmw_lab_one_click/bmw_lab_eight_view_v1/efficientad/score_analysis/part_thresholds.json`

**Interfaces:**
- Consumes: Tasks 1-4 and the existing real BMW data/checkpoints.
- Produces: real confusion counts, evidence images, updated handoff commands, committed source/tests/docs, and an updated Draft PR branch.

- [ ] **Step 1: Run real bright-streak recalibration**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync python \
  pipeline/bmw_lab_recalibrate_bright_streak.py
```

Expected: 17 normal and 3 `no_streak` final-test rows are reported; the report states exact false rejects/accepts and writes diagnostic evidence for any failure.

- [ ] **Step 2: Fit the EfficientAD whole-part thresholds**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync python \
  pipeline/bmw_lab_calibrate_efficientad_thresholds.py --target-part-fpr 0.05
```

Expected on the current CSV: 20 normal parts, no more than one false-positive part, and explicit defect-part recall with leakage flags.

- [ ] **Step 3: Run an offline Demo regression**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync python \
  pipeline/bmw_lab_eight_view_demo.py \
  --sample-id bmw_normal_group072_000001 \
  --no-gui \
  --save-screenshot results/bmw_eight_view_demo/bmw_normal_group072_threshold_v2.png
```

Expected: all configured assets load and the output contains eight EfficientAD rows with their per-view thresholds. The final overall status is reported rather than assumed.

- [ ] **Step 4: Run full focused verification**

```bash
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync pytest \
  tests/unit/bmw_inspection/test_detector.py \
  tests/unit/bmw_inspection/lab/test_bright_streak_recalibration.py \
  tests/unit/bmw_inspection/lab/test_efficientad_analysis.py \
  tests/unit/bmw_inspection/lab/test_efficientad_thresholds.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo.py \
  tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py -q
UV_CACHE_DIR=/tmp/bmw_uv_cache uv run --no-sync python -m compileall -q \
  src/bmw_inspection pipeline/bmw_lab_recalibrate_bright_streak.py \
  pipeline/bmw_lab_calibrate_efficientad_thresholds.py
git diff --check
```

Expected: zero test failures, compileall exit 0, and no whitespace errors.

- [ ] **Step 5: Update handoff and project memory with measured results**

Document only freshly measured counts and caveats. Record the exact commit, output paths, bright-streak failure classes, EfficientAD whole-part FPR/recall, and the fact that customer data/results are not pushed.

- [ ] **Step 6: Commit and push only intended paths**

```bash
git add BMW_EIGHT_VIEW_HANDOFF_README.md
git commit -m "docs: record BMW threshold tuning results"
git push origin agent/bmw-eight-view-handoff
```

Expected: remote branch SHA equals local HEAD and Draft PR #1 remains open.
