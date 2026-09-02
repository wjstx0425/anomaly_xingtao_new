# BMW Template Critical-Region Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the influence of operator-selected critical regions inside the existing whole-ROI Template score while preserving the fixed 25-result inspection chain and the legacy score/threshold rollback path.

**Architecture:** A focused pure module parses the existing hand-specific ROI JSON, maps ROI-local rectangles through the same aspect-fit geometry as Template preprocessing, combines weights with the Template ignore mask, and calculates one weighted normalized correlation for the whole Template/alignment already selected by the legacy matcher. The existing Template branch remains the only result row. A small calibration CLI fits only weighted thresholds from 0823 calibration-normal rows and reports final-test normals without changing legacy thresholds.

**Tech Stack:** Python 3.13, NumPy, OpenCV, dataclasses, JSON/CSV, pytest, uv.

## Global Constraints

- Keep exactly 25 inspection results: front 13 plus back 12; add no branch, UI card, shortcut, or fusion input.
- Keep Template selection, `TM_CCOEFF_NORMED` search, `max_shift`, public ROI, ignore masks, EfficientAD, YOLO, bright-streak, HDR, camera settings, and final fusion unchanged.
- Normal pixels have weight `1.0`; selected critical-region pixels use `3.0`; overlaps use maximum; ignore-mask pixels finish at `0.0`.
- Preserve every value under `template.thresholds`; weighted thresholds live only under `template.weighted_regions.thresholds`.
- Missing/disabled weighting restores exact legacy scoring. Views with no selected regions stay on the legacy score and threshold.
- Fit weighted thresholds only from calibration normal as `max(weighted_risk) * 1.10`; final-test normal is reporting-only.
- Keep right/left ROI files, models, manifests, and weighted thresholds separate.
- Add no SHA, receipt, provenance, publisher, schema-version, immutable-release, or rebind logic.
- Use this worktree and uv; preserve unrelated dirty changes; do not push or merge.

---

## File Structure

- Create `src/bmw_inspection/lab/template_region_weighting.py`: ROI parsing, geometry mapping, weight-map creation, weighted correlation, threshold envelope.
- Modify `src/bmw_inspection/lab/eight_view_demo.py`: optional weighted Template configuration.
- Modify `src/bmw_inspection/lab/eight_view_demo_models.py`: use weighted final score inside the existing Template row.
- Create `pipeline/bmw_lab_calibrate_weighted_template.py`: calibration/final-test scoring and config update.
- Modify both active 0820 mixed configs, focused tests, and `AGENTS_MEMORY.md`.

### Task 1: Pure weighting core

**Files:**
- Create: `src/bmw_inspection/lab/template_region_weighting.py`
- Create: `tests/unit/bmw_inspection/lab/test_template_region_weighting.py`

**Interfaces:**
- Produces `TemplateWeightedRegion(region_id: str, roi_xyxy: tuple[int, int, int, int])`.
- Produces `load_template_weighted_regions(path, expected_shapes) -> Mapping[str, tuple[TemplateWeightedRegion, ...]]`.
- Produces `build_aligned_template_weights(input_shape, target_size, max_shift, best_location, regions, region_weight, ignore_mask) -> np.ndarray`.
- Produces `weighted_ccoeff(query, template, weights) -> float`.
- Produces `normal_envelope_threshold(scores, margin_ratio=0.10) -> float`.

- [ ] **Step 1: Write parser and geometry tests that fail before the module exists**

```python
def test_regions_allow_zero_or_many_and_validate_public_roi_bounds(tmp_path: Path) -> None:
    payload = {"views": {view: [] for view in VIEW_ORDER}}
    payload["views"]["front"] = [{"id": "foot", "roi_xyxy": [1, 2, 5, 7]}]
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    regions = load_template_weighted_regions(
        path, expected_shapes={view: (10, 12) for view in VIEW_ORDER}
    )
    assert regions["front"][0].roi_xyxy == (1, 2, 5, 7)
    assert regions["back"] == ()


def test_overlap_uses_max_and_ignore_mask_wins() -> None:
    regions = (
        TemplateWeightedRegion("a", (0, 0, 6, 6)),
        TemplateWeightedRegion("b", (3, 3, 8, 8)),
    )
    ignore = np.zeros((8, 8), dtype=np.uint8)
    ignore[4:6, 4:6] = 255
    weights = build_aligned_template_weights(
        (8, 8), (8, 8), 0, (0, 0), regions, 3.0, ignore
    )
    assert weights[3, 3] == 3.0
    assert weights[4, 4] == 0.0
    assert weights.max() == 3.0
```

- [ ] **Step 2: Run RED**

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/bmw_inspection/lab/test_template_region_weighting.py
```

Expected: collection fails because `template_region_weighting` does not exist.

- [ ] **Step 3: Implement parsing, aspect-fit mapping, weighted correlation, and envelope**

```python
def weighted_ccoeff(query: np.ndarray, template: np.ndarray, weights: np.ndarray) -> float:
    q = np.asarray(query, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if q.shape != t.shape or q.shape != w.shape or np.any(w < 0):
        raise ValueError("Template加权输入形状或权重无效")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Template有效权重为空")
    qc = q - float(np.sum(w * q) / total)
    tc = t - float(np.sum(w * t) / total)
    denominator = math.sqrt(float(np.sum(w * qc**2) * np.sum(w * tc**2)))
    if denominator <= 0:
        raise ValueError("Template加权相关系数分母为零")
    return float(np.clip(np.sum(w * qc * tc) / denominator, -1.0, 1.0))


def normal_envelope_threshold(scores: Sequence[float], margin_ratio: float = 0.10) -> float:
    maximum = max(float(score) for score in scores)
    return math.nextafter(0.0, math.inf) if maximum == 0.0 else maximum * (1.0 + margin_ratio)
```

The loader accepts empty lists, requires integer half-open coordinates within each public ROI, rejects duplicate IDs per view and unknown views, and ignores unrelated JSON fields. The mapping uses the exact scale/rounded fitted size/padding of `_prepare_template_image`; rectangle starts use floor and ends use ceil.

- [ ] **Step 4: Add a synthetic score test**

```python
def test_weight_three_emphasizes_difference_inside_region() -> None:
    template = np.arange(64, dtype=np.float64).reshape(8, 8)
    query = template.copy()
    query[1:3, 1:3] += 50
    base = weighted_ccoeff(query, template, np.ones((8, 8)))
    weights = np.ones((8, 8))
    weights[1:3, 1:3] = 3.0
    weighted = weighted_ccoeff(query, template, weights)
    assert 1.0 - weighted > 1.0 - base
```

- [ ] **Step 5: Run GREEN and commit**

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/bmw_inspection/lab/test_template_region_weighting.py
git add src/bmw_inspection/lab/template_region_weighting.py tests/unit/bmw_inspection/lab/test_template_region_weighting.py
git commit -m "feat: add weighted template scoring core"
```

### Task 2: Optional config and existing Template-row integration

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_lab_config.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

**Interfaces:**
- Consumes Task 1 helpers.
- Produces optional `EightViewDemoConfig.template_weighted_regions`.
- Extends `EightViewTemplatePredictor(..., weighted_regions=...)` without changing `predict(view, image) -> ModelOutput`.

- [ ] **Step 1: Write failing compatibility tests**

```python
def test_weighted_regions_is_optional(config_path: Path) -> None:
    assert load_demo_config(config_path).template_weighted_regions is None


def test_weighted_config_preserves_legacy_thresholds(config_path: Path) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    legacy = dict(payload["template"]["thresholds"])
    payload["template"]["weighted_regions"] = {
        "enabled": True,
        "weight": 3.0,
        "roi_config": "regions.json",
        "thresholds": {view: 0.02 for view in VIEW_ORDER},
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_demo_config(config_path)
    assert dict(config.template_thresholds) == legacy
    assert config.template_weighted_regions.weight == 3.0
```

- [ ] **Step 2: Run RED, then parse only the optional four fields**

Validate finite weight `>=1.0`, ROI file existence, and eight finite non-negative weighted thresholds. Defer ROI-coordinate validation until `build_model_suite()` has public-ROI shapes.

- [ ] **Step 3: Write predictor tests for exact legacy fallback and one weighted row**

```python
def test_disabled_weighting_matches_legacy_exactly(template_fixture) -> None:
    legacy = template_fixture.legacy.predict("front", template_fixture.image)
    disabled = template_fixture.disabled.predict("front", template_fixture.image)
    assert (disabled.status, disabled.score, disabled.threshold) == (
        legacy.status, legacy.score, legacy.threshold
    )


def test_empty_regions_keep_legacy_score_and_threshold(template_fixture) -> None:
    output = template_fixture.empty.predict("back", template_fixture.image)
    assert output.threshold == template_fixture.legacy_threshold
    assert output.details["score_source"] != "weighted_region_ccoeff_normed"


def test_weighted_view_keeps_legacy_and_weighted_diagnostics(template_fixture) -> None:
    output = template_fixture.weighted.predict("front", template_fixture.changed_image)
    assert output.details["score_source"] == "weighted_region_ccoeff_normed"
    assert output.details["region_weight"] == 3.0
    assert output.details["weighted_risk"] == output.score
    assert output.details["weighted_region_count"] == 1
    assert "legacy_risk" in output.details
```

- [ ] **Step 4: Integrate weighting only after current model/alignment selection**

Keep current `similarity`, `risk`, `best`, `best_location`, and aligned ignore-mask computation. For an enabled non-empty view, build aligned weights, calculate weighted correlation over `aligned_query` and `best`, select the weighted threshold, and add the approved diagnostic fields. Draw mapped rectangles on the existing Template overlay. Do not produce a second branch result.

- [ ] **Step 5: Prove the suite remains 13/12/25**

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/bmw_inspection/lab/test_eight_view_lab_config.py tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py
```

Expected: legacy front-13, back-12, final-25 assertions and all weighting tests pass.

- [ ] **Step 6: Commit runtime integration**

```bash
git add src/bmw_inspection/lab/eight_view_demo.py src/bmw_inspection/lab/eight_view_demo_models.py tests/unit/bmw_inspection/lab/test_eight_view_lab_config.py tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py
git commit -m "feat: weight critical regions in template score"
```

### Task 3: Focused calibration CLI

**Files:**
- Create: `pipeline/bmw_lab_calibrate_weighted_template.py`
- Create: `tests/unit/pipeline/test_bmw_lab_calibrate_weighted_template.py`

**Interfaces:**
- Consumes `--config`, `--prepared-manifest`, `--output`, and optional `--write-config`.
- Produces per-view calibration scores/max/proposed threshold and final-test scores/max/false-reject count.
- With `--write-config`, modifies only `template.weighted_regions.thresholds`.

- [ ] **Step 1: Write failing split-isolation and safe-write tests**

```python
def test_final_test_does_not_change_threshold() -> None:
    first = summarize_view([0.1, 0.2], [0.9])
    second = summarize_view([0.1, 0.2], [0.01])
    assert first["threshold"] == second["threshold"] == pytest.approx(0.22)


def test_writer_preserves_legacy_thresholds(tmp_path: Path) -> None:
    target = tmp_path / "demo.json"
    target.write_text(SOURCE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    before = json.loads(target.read_text(encoding="utf-8"))
    write_weighted_thresholds(target, {view: 0.02 for view in VIEW_ORDER})
    after = json.loads(target.read_text(encoding="utf-8"))
    assert after["template"]["thresholds"] == before["template"]["thresholds"]
```

- [ ] **Step 2: Run RED**

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/pipeline/test_bmw_lab_calibrate_weighted_template.py
```

Expected: import failure because the CLI module does not exist.

- [ ] **Step 3: Implement scoring and reporting**

Read only `label=normal` rows whose split is `calibration` or `final_test`. Read each source image, crop the active public ROI, and invoke the same Template predictor as Demo. Fit only enabled non-empty views; report empty views as legacy. Output JSON includes weight, ROI config, image counts, individual scores, maxima, proposed thresholds, and final-test false rejects. It contains no hashes or publishing metadata.

- [ ] **Step 4: Run GREEN and syntax check**

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/pipeline/test_bmw_lab_calibrate_weighted_template.py
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m py_compile pipeline/bmw_lab_calibrate_weighted_template.py
```

- [ ] **Step 5: Commit calibration CLI**

```bash
git add pipeline/bmw_lab_calibrate_weighted_template.py tests/unit/pipeline/test_bmw_lab_calibrate_weighted_template.py
git commit -m "feat: calibrate weighted template thresholds"
```

### Task 4: Calibrate, wire, verify, and document both hands

**Files:**
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_right_0820_mixed_v1.json`
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_left_0820_mixed_v1.json`
- Modify: `tests/unit/bmw_inspection/lab/test_bmw_0820_mixed_configs.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_persistence.py`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes the hand-specific ROI files and Task 3 CLI.
- Produces active weight `3.0`, calibrated thresholds, rollback instructions, and replay evidence.

- [ ] **Step 1: Add failing active-config assertions**

```python
@pytest.mark.parametrize(("hand", "roi_name"), [
    ("right", "bmw_right_0823_key_rois_v1.json"),
    ("left", "bmw_left_0823_key_rois_v1.json"),
])
def test_mixed_config_uses_hand_specific_weighted_regions(hand: str, roi_name: str) -> None:
    payload = json.loads(CONFIGS[hand].read_text(encoding="utf-8"))
    weighted = payload["template"]["weighted_regions"]
    assert weighted["enabled"] is True
    assert weighted["weight"] == 3.0
    assert weighted["roi_config"].endswith(roi_name)
    assert set(weighted["thresholds"]) == set(VIEW_ORDER)
```

- [ ] **Step 2: Add each config block with a temporary copy of legacy thresholds**

Reference `configs/bmw/key_template/bmw_right_0823_key_rois_v1.json` only from the right config and the left file only from the left config. Do not edit any existing `template.thresholds` number.

- [ ] **Step 3: Run real calibration and update only weighted thresholds**

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python pipeline/bmw_lab_calibrate_weighted_template.py --config configs/bmw/experiments/bmw_eight_view_demo_right_0820_mixed_v1.json --prepared-manifest /home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared/bmw_right_front_right_0823_v1/manifests/dataset_manifest.csv --output results/bmw_lab_one_click/bmw_weighted_template_calibration_right_0823_v1/report.json --write-config

UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python pipeline/bmw_lab_calibrate_weighted_template.py --config configs/bmw/experiments/bmw_eight_view_demo_left_0820_mixed_v1.json --prepared-manifest /home/yunjing/anomaly_xingtao_new/dataset/bmw_lab_prepared/bmw_left_front_right_0823_v1/manifests/dataset_manifest.csv --output results/bmw_lab_one_click/bmw_weighted_template_calibration_left_0823_v1/report.json --write-config
```

Expected: four front views receive fitted thresholds because they have regions; four empty back views retain legacy thresholds. Diff must prove the legacy threshold objects are unchanged.

- [ ] **Step 4: Run focused regressions**

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m pytest -q tests/unit/bmw_inspection/lab/test_template_region_weighting.py tests/unit/bmw_inspection/lab/test_eight_view_lab_config.py tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py tests/unit/bmw_inspection/lab/test_eight_view_demo_persistence.py tests/unit/bmw_inspection/lab/test_bmw_0820_mixed_configs.py tests/unit/pipeline/test_bmw_lab_calibrate_weighted_template.py tests/unit/pipeline/test_bmw_lab_eight_view_demo.py
```

Expected: all focused tests pass and persistence remains exactly 25 results.

- [ ] **Step 5: Run syntax and whitespace checks**

```bash
UV_CACHE_DIR=/tmp/bmw-uv-cache uv run --project /home/yunjing/anomaly_xingtao_new --no-sync python -m py_compile src/bmw_inspection/lab/template_region_weighting.py src/bmw_inspection/lab/eight_view_demo.py src/bmw_inspection/lab/eight_view_demo_models.py pipeline/bmw_lab_calibrate_weighted_template.py pipeline/bmw_lab_eight_view_demo.py
git diff --check
```

- [ ] **Step 6: Run one 0823 offline normal replay per hand**

Use each active config with a temporary eight-view capture-set built from its 0823 fused images, because the active config's prepared manifest may not contain the retake sample IDs. Record result count, ERROR count, final status, weighted Template scores, and result directories. Do not run the cameras.

- [ ] **Step 7: Verify rollback in a temporary config**

Copy one config to `/tmp`, change only `template.weighted_regions.enabled` to false, run the same capture-set, and verify Template scores and thresholds return to the legacy values. Do not edit active configs for this check.

- [ ] **Step 8: Update memory and audit scope**

Append the active ROI paths, weight, calibrated thresholds, report paths, rollback switch, test/replay results, and unverified CUDA/camera status to `AGENTS_MEMORY.md`. Then run:

```bash
rg -n "key_template|KEY_TEMPLATE" src/bmw_inspection/lab pipeline/bmw_lab_eight_view_demo.py
rg -n "sha256|receipt|provenance|publisher|rebind" src/bmw_inspection/lab/template_region_weighting.py pipeline/bmw_lab_calibrate_weighted_template.py
git status --short
```

Expected: no independent key-template branch has returned; new code contains no release machinery; unrelated dirty files remain intact.
