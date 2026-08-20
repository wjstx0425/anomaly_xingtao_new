# BMW Six-View Lab Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fast-iteration BMW laboratory inspection application that captures six fixed views from three Hikvision cameras and combines Template, bright-streak rules, one-class YOLO, and per-view PatchCore into one truthful Chinese inspection result.

**Architecture:** Keep the approved single-camera bright-streak Demo intact and add an isolated `bmw_inspection.lab` package. Reuse the existing Hikvision capture primitives and established OpenCV/Anomalib/Ultralytics algorithms, while avoiding the ZS32 eight-view deployment bundle, release, topology, and subprocess contracts. Load models once, reload editable thresholds before each part, run Template on all six views first, short-circuit all downstream branches on any Template NG, and otherwise require every configured bright-streak, YOLO, and PatchCore branch to pass for final OK.

**Tech Stack:** Python 3.10+, `uv`, OpenCV, NumPy, Hikvision MVS adapter, Anomalib PatchCore, Ultralytics YOLO, pytest.

## Global Constraints

- The three exact camera serials are `DA9805574`, `DA9625347`, and `DB0968108`; never fall back to enumeration index.
- Capture the same physical part in two manual rounds: front three views, operator flip, then back three views.
- The six canonical view IDs are `front`, `front_left`, `front_right`, `back`, `back_left`, and `back_right`.
- The initial slot mapping is centre=`DA9805574`, left=`DA9625347`, right=`DB0968108`.
- Camera, part fixture, lighting, exposure, and viewpoint are fixed.
- Template detects wrong part, bad pose, or large structural mismatch; it is not the fine-defect detector.
- YOLO has exactly one class named `defect`, and every YOLO-positive training instance requires a bounding box.
- Keep the existing bright-streak rule semantics: a streak must exist and be continuous to pass.
- The existing ROI `[1872, 1180, 1953, 1793]` remains the initial `front_left` bright-streak ROI only; it is not a Template, YOLO, or PatchCore ROI.
- Bright-streak view routing is configuration-driven so additional views can be enabled without code changes.
- Run all six Template branches first. If any Template branch is NG or ERROR, do not run bright-streak, YOLO, or PatchCore.
- Final OK requires every required branch to execute successfully and pass. A disabled required branch must yield `REVIEW`, never OK.
- Image-quality failures yield `RETAKE`; camera/config/model failures yield `ERROR`; neither may be presented as product NG.
- Keep raw diagnostic candidates separate from business results. YOLO boxes below the final threshold are stored but are not drawn red on Fusion/UI.
- PatchCore heatmaps are diagnostic anomaly maps, not claimed pixel-accurate defect segmentation.
- Models stay resident. Threshold-only edits take effect on the next part; ROI, preprocessing, or checkpoint changes require an explicit model reload action.
- Preserve the current `configs/bmw/bright_streak_demo.json`, `pipeline/bmw_bright_streak_demo.py`, and customer Demo behavior.
- Manage dependencies and all commands with `uv`.
- Do not move or delete the existing 6 OK / 7 NG BMW bright-streak images; treat them as regression evidence from 13 physical parts.
- New train/calibration/test splits must be by physical `part_id` or collection session, never random image rows.

---

## Target File Map

### New BMW laboratory package

- `src/bmw_inspection/lab/contracts.py`: immutable view, branch, evidence, capture-set, and final-result types.
- `src/bmw_inspection/lab/config.py`: strict experiment JSON parsing and per-part threshold reload.
- `src/bmw_inspection/lab/capture.py`: three-camera, two-round, six-view acquisition.
- `src/bmw_inspection/lab/bright_streak.py`: adapter around the existing rule detector with structured decision evidence.
- `src/bmw_inspection/lab/template.py`: per-view multi-template training, loading, scoring, and overlays.
- `src/bmw_inspection/lab/yolo.py`: one resident global one-class YOLO backend.
- `src/bmw_inspection/lab/patchcore.py`: six resident per-view PatchCore backends.
- `src/bmw_inspection/lab/fusion.py`: Template gate and all-required-clear fusion.
- `src/bmw_inspection/lab/publisher.py`: lightweight atomic run/evidence publishing.
- `src/bmw_inspection/lab/runtime.py`: one capture-set execution coordinator and model lifecycle.
- `src/bmw_inspection/lab/evaluation.py`: folder/manifest replay, metrics, and error-sample export.
- `src/bmw_inspection/lab/ui.py`: Chinese Experiment and Presentation modes.

### New commands and configuration

- `configs/bmw/topology/bmw_3cam_double_side_v1.json`
- `configs/bmw/experiments/bmw_lab_v1.json`
- `pipeline/bmw_lab_inspection.py`
- `pipeline/bmw_lab_select_rois.py`
- `pipeline/bmw_lab_build_manifest.py`
- `pipeline/bmw_lab_train_template.py`
- `pipeline/bmw_lab_train_yolo.py`
- `pipeline/bmw_lab_train_patchcore.py`
- `pipeline/bmw_lab_evaluate.py`

### Tests

- `tests/unit/bmw_inspection/lab/test_contracts.py`
- `tests/unit/bmw_inspection/lab/test_config.py`
- `tests/unit/bmw_inspection/lab/test_capture.py`
- `tests/unit/bmw_inspection/lab/test_bright_streak.py`
- `tests/unit/bmw_inspection/lab/test_template.py`
- `tests/unit/bmw_inspection/lab/test_yolo.py`
- `tests/unit/bmw_inspection/lab/test_patchcore.py`
- `tests/unit/bmw_inspection/lab/test_fusion.py`
- `tests/unit/bmw_inspection/lab/test_publisher.py`
- `tests/unit/bmw_inspection/lab/test_runtime.py`
- `tests/unit/bmw_inspection/lab/test_evaluation.py`
- `tests/unit/bmw_inspection/lab/test_ui.py`

---

### Task 1: Define the six-view topology, experiment config, and result contracts

**Files:**
- Create: `src/bmw_inspection/lab/__init__.py`
- Create: `src/bmw_inspection/lab/contracts.py`
- Create: `src/bmw_inspection/lab/config.py`
- Create: `configs/bmw/topology/bmw_3cam_double_side_v1.json`
- Create: `configs/bmw/experiments/bmw_lab_v1.json`
- Test: `tests/unit/bmw_inspection/lab/test_contracts.py`
- Test: `tests/unit/bmw_inspection/lab/test_config.py`

**Interfaces:**
- Produces: `ViewId`, `BranchName`, `BranchStatus`, `FinalStatus`, `CapturedView`, `CaptureSet`, `BranchEvidence`, `InspectionResult`, `validate_final_status(status, required_complete)`, `LabExperimentConfig`, and `load_experiment_config(path)`.
- Consumes: existing `BrightStreakConfig` only through the Task 3 adapter; the lab config must not alter the customer Demo config.

- [ ] **Step 1: Write contract tests for exact six-view identity and all public statuses.**

```python
def test_required_views_are_exactly_six() -> None:
    assert tuple(ViewId) == (
        ViewId.FRONT, ViewId.FRONT_LEFT, ViewId.FRONT_RIGHT,
        ViewId.BACK, ViewId.BACK_LEFT, ViewId.BACK_RIGHT,
    )

def test_disabled_required_branch_cannot_publish_ok() -> None:
    with pytest.raises(ValueError, match="required branch"):
        validate_final_status(FinalStatus.OK, required_complete=False)
```

- [ ] **Step 2: Run the tests and verify the new package is absent.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_contracts.py tests/unit/bmw_inspection/lab/test_config.py`

Expected: collection fails because `bmw_inspection.lab` does not exist.

- [ ] **Step 3: Implement strict dataclasses/enums and JSON parsing.**

The key contract must be equivalent to:

```python
class BranchStatus(StrEnum):
    PASS = "PASS"
    NG = "NG"
    SKIPPED = "SKIPPED"
    REVIEW = "REVIEW"
    ERROR = "ERROR"

@dataclass(frozen=True, slots=True)
class BranchEvidence:
    branch: BranchName
    view_id: ViewId
    status: BranchStatus
    score: float | None
    threshold: float | None
    elapsed_ms: float
    reason: str
    model_id: str | None
    artifact_paths: Mapping[str, str]
```

Reject duplicate JSON keys, non-finite thresholds, unknown views/branches, duplicate serials, missing required views, out-of-bounds ROIs, and model paths that do not exist when a branch is enabled.

- [ ] **Step 4: Create the BMW topology with the confirmed right-camera serial.**

```json
{
  "topology_id": "bmw-3cam-double-side-v1",
  "camera_slots": [
    {"slot_id": "center", "serial": "DA9805574", "front": "front", "back": "back"},
    {"slot_id": "left", "serial": "DA9625347", "front": "front_left", "back": "back_left"},
    {"slot_id": "right", "serial": "DB0968108", "front": "front_right", "back": "back_right"}
  ]
}
```

- [ ] **Step 5: Create an experimental profile with per-view ROI and branch routing.**

The profile must contain `experiment_id`, topology path, capture settings, per-view part ROIs, Template groups, `bright_streak.enabled_views`, YOLO checkpoint/settings, six PatchCore checkpoints/settings, final thresholds, `required_for_ok`, and result root. Initialise bright streak with `enabled_views=["front_left"]`; keep unavailable learned-model assets explicitly disabled until trained.

- [ ] **Step 6: Run focused tests.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_contracts.py tests/unit/bmw_inspection/lab/test_config.py`

Expected: all tests pass.

---

### Task 2: Add three-camera, two-round acquisition and offline six-view loading

**Files:**
- Create: `src/bmw_inspection/lab/capture.py`
- Create: `pipeline/bmw_lab_inspection.py`
- Test: `tests/unit/bmw_inspection/lab/test_capture.py`

**Interfaces:**
- Consumes: `LabExperimentConfig` and existing `HikvisionAdapter`, `open_cameras`, `capture_single_round`, and `GroupedTriggerPacer` from `capture_data/collect_multicamera_dataset.py`.
- Produces: `LabCameraSession.capture_round(round_id) -> Mapping[ViewId, np.ndarray]`, `build_capture_set(front, back) -> CaptureSet`, and offline `load_capture_set(path) -> CaptureSet`.

- [ ] **Step 1: Write fake-adapter tests proving serial binding and round mapping.**

```python
def test_two_rounds_map_three_serials_to_six_views(fake_adapter) -> None:
    with LabCameraSession(config, adapter=fake_adapter) as session:
        front = session.capture_round("front")
        back = session.capture_round("back")
    assert set(front) == {ViewId.FRONT, ViewId.FRONT_LEFT, ViewId.FRONT_RIGHT}
    assert set(back) == {ViewId.BACK, ViewId.BACK_LEFT, ViewId.BACK_RIGHT}
    assert fake_adapter.opened_serials == ["DA9805574", "DA9625347", "DB0968108"]
```

Also test missing camera, duplicate serial, incomplete round, wrong dimensions, and cleanup after read failure.

- [ ] **Step 2: Verify tests fail before implementation.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_capture.py`

Expected: import failure for `bmw_inspection.lab.capture`.

- [ ] **Step 3: Implement one resident three-camera session and two explicit operator rounds.**

Do not auto-capture at startup. The live CLI state machine must be:

```text
WAITING_FRONT -> CAPTURE_FRONT -> WAITING_FLIP -> CAPTURE_BACK -> READY_FOR_INSPECTION
```

Trigger the three cameras before reading them, preserve deterministic slot/view order, and retain the existing per-camera warm-up behavior.

- [ ] **Step 4: Implement offline six-view directory loading.**

The accepted directory filenames are exactly the six view IDs plus `.bmp` or `.png`. Missing or duplicate views must fail before inference.

- [ ] **Step 5: Run focused tests and a CLI help smoke.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_capture.py`

Run: `uv run --no-sync python pipeline/bmw_lab_inspection.py --help`

Expected: tests pass and help lists `--config`, `--capture-set`, `--no-gui`, and `--output-root`.

---

### Task 3: Preserve and structure the bright-streak decision evidence

**Files:**
- Modify: `src/bmw_inspection/detector.py`
- Modify: `src/bmw_inspection/demo_app.py`
- Create: `src/bmw_inspection/lab/bright_streak.py`
- Test: `tests/unit/bmw_inspection/test_detector.py`
- Test: `tests/unit/bmw_inspection/test_demo_app.py`
- Test: `tests/unit/bmw_inspection/lab/test_bright_streak.py`

**Interfaces:**
- Produces: `BrightStreakDecision` containing status, metrics, accepted mask, response image, runs, and gaps; `BrightStreakBackend.predict(view_id, image) -> BranchEvidence`.
- Preserves: existing `detect_bright_streak()` status on all current BMW regression images.

- [ ] **Step 1: Add a regression proving published mask equals the detector's actual decision mask.**

```python
def test_saturated_path_publishes_the_mask_used_for_decision(ok_image, config) -> None:
    decision = detect_bright_streak_evidence(ok_image, config)
    assert decision.result.status is DemoStatus.OK
    assert np.array_equal(decision.mask, decision.mask_used_for_metrics)
```

- [ ] **Step 2: Add timing and structured run/gap tests.**

The detector must no longer encode evidence in `reason` and parse it back with regular expressions. `capture_elapsed_ms`, `processing_elapsed_ms`, and `total_elapsed_ms` must be populated with measured finite values.

- [ ] **Step 3: Implement the evidence-returning detector while keeping the old API as a compatibility wrapper.**

```python
def detect_bright_streak(image, config) -> BrightStreakResult:
    return detect_bright_streak_evidence(image, config).result
```

- [ ] **Step 4: Change `demo_app.py` to publish the returned response/mask instead of recomputing top-hat diagnostics.**

- [ ] **Step 5: Add the lab adapter and per-view routing.**

If a view is not configured for bright streak, emit `SKIPPED` and mark whether it is required for final OK. The initial `front_left` adapter must load the approved existing ROI and thresholds without changing the Demo JSON.

- [ ] **Step 6: Run all existing BMW tests plus the new adapter tests.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection tests/unit/bmw_inspection/lab/test_bright_streak.py`

Expected: all existing classifications and UI behavior remain green.

---

### Task 4: Build the per-view Template trainer and resident gate

**Files:**
- Create: `src/bmw_inspection/lab/template.py`
- Create: `pipeline/bmw_lab_train_template.py`
- Create: `pipeline/bmw_lab_select_rois.py`
- Test: `tests/unit/bmw_inspection/lab/test_template.py`

**Interfaces:**
- Consumes: manifest rows with `split=train|calibration|final_test`, `label=normal|defect`, and one part ROI per view.
- Produces: one model directory per view containing `model.json`, `model.sha256`, `templates/*.png`, `calibration_rows.csv`, and `metrics.json`; runtime `TemplateBackend.predict(view_id, crop) -> BranchEvidence`.

- [ ] **Step 1: Write tests for bounded translation matching and multi-template selection.**

```python
def test_small_translation_matches_but_large_translation_fails() -> None:
    backend = trained_backend(max_shift=12)
    assert backend.predict("front", shifted_reference(dx=8)).status is BranchStatus.PASS
    assert backend.predict("front", shifted_reference(dx=20)).status is BranchStatus.NG
```

Also cover flat images, wrong dimensions, missing/tampered templates, threshold equality, deterministic training, and refusal to overwrite an existing output directory.

- [ ] **Step 2: Implement grayscale, aspect-preserving resize, 3x3 blur, `TM_CCOEFF_NORMED`, reflect padding, and bounded XY search.**

Return `similarity`, `risk=1-similarity`, best template, offset, and elapsed time. Use three to five diverse normal templates per view selected by medoid plus farthest-point sampling.

- [ ] **Step 3: Implement threshold fitting from calibration only.**

Select the threshold that maximises calibration balanced accuracy. When tied, choose the stricter threshold with fewer normal false rejects. Never use final-test rows for fitting.

- [ ] **Step 4: Implement view-by-view ROI selection from a saved six-view capture set.**

The ROI command must scale 4024x3036 images to the screen and map display coordinates back to half-open source coordinates, matching the approved BMW selector behavior.

- [ ] **Step 5: Run Template tests and a synthetic training smoke.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_template.py`

Expected: all tests pass; the smoke produces six independently loadable Template groups.

---

### Task 5: Create the physical-part manifest and data exports

**Files:**
- Create: `src/bmw_inspection/lab/dataset.py`
- Create: `pipeline/bmw_lab_build_manifest.py`
- Modify: `capture_data/prepare_yolo_dataset.py` only if a generic callable export boundary cannot be reused without ZS32 names.
- Test: `tests/unit/bmw_inspection/lab/test_dataset.py`

**Interfaces:**
- Produces: `dataset/bmw_lab/manifests/<dataset_id>.csv`, a YOLO export, and six PatchCore folder exports without moving source images.
- Manifest columns: `sample_id,part_id,session_id,view_id,image_path,image_sha256,label,defect_type,x1,y1,x2,y2,split`.

- [ ] **Step 1: Test part-level split isolation and bbox validation.**

```python
def test_one_part_cannot_cross_splits(rows) -> None:
    assert not find_part_split_leakage(rows)

def test_yolo_positive_requires_valid_defect_box() -> None:
    with pytest.raises(ValueError, match="bounding box"):
        validate_row(defect_row(x1="", y1="", x2="", y2=""))
```

- [ ] **Step 2: Implement manifest building without inferring identity from image count.**

Require explicit `part_id` and `session_id`. Preserve the current 13 single-view images as a bright-streak regression manifest; do not route them into six-view Template/PatchCore/YOLO final metrics.

- [ ] **Step 3: Implement deterministic 60/20/20 part-level split generation.**

For fewer than 30 complete six-view physical parts, refuse to label the result as a final evaluation dataset. Allow an explicit `--experimental-small-data` flag that marks reports `experimental_only=true`.

- [ ] **Step 4: Export YOLO and PatchCore layouts from the manifest.**

YOLO uses one class line `0: defect`. PatchCore exports each view separately and includes only normal train images in the memory-bank training directory.

- [ ] **Step 5: Run dataset tests and a dry-run manifest report.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_dataset.py`

Expected: tests pass; the report shows unique physical parts and zero split leakage.

---

### Task 6: Add one global one-class YOLO train/runtime path

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/bmw_inspection/lab/yolo.py`
- Create: `pipeline/bmw_lab_train_yolo.py`
- Test: `tests/unit/bmw_inspection/lab/test_yolo.py`

**Interfaces:**
- Consumes: generated YOLO `data.yaml`, a base Ultralytics checkpoint, and runtime profile settings.
- Produces: versioned `best.pt`, `args.yaml`, `metrics.json`, and runtime `YoloBackend.predict(view_id, crop) -> BranchEvidence` with all candidates and final detections separated.

- [ ] **Step 1: Declare and lock the existing runtime dependency.**

Add an optional project extra equivalent to:

```toml
bmw-lab = ["ultralytics==8.4.89"]
```

Run: `uv lock`

Expected: `uv.lock` contains Ultralytics 8.4.89 and its resolved dependencies.

- [ ] **Step 2: Write fake-model tests for candidate/final threshold separation.**

```python
def test_subthreshold_candidate_is_saved_but_not_final() -> None:
    evidence = backend(candidate_conf=0.01, final_threshold=0.50).predict(image_with_conf(0.30))
    assert len(evidence.candidates) == 1
    assert evidence.final_boxes == ()
    assert evidence.status is BranchStatus.PASS
```

- [ ] **Step 3: Implement one resident `YOLO(best.pt)` shared across all six views.**

Convert Mono8 to three equal channels, crop using the exact configured per-view ROI, resize with the same profile settings used by training, and reject any predicted class other than `defect`.

- [ ] **Step 4: Implement versioned training.**

Use one global model across all six views, default `imgsz=1280`, fixed seed 42, and a new output directory for every run. Save the best checkpoint and calibration predictions; never replace the active profile automatically.

- [ ] **Step 5: Fit the final confidence threshold on calibration rows only.**

Default selection maximises image-level F1. Save the low candidate floor separately. Keep the threshold editable in the experiment profile.

- [ ] **Step 6: Run unit tests and a one-image fake/small-model smoke.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_yolo.py`

Expected: all tests pass and no below-final-threshold red box appears in the final overlay.

---

### Task 7: Add six per-view PatchCore train/runtime paths

**Files:**
- Create: `src/bmw_inspection/lab/patchcore.py`
- Create: `pipeline/bmw_lab_train_patchcore.py`
- Test: `tests/unit/bmw_inspection/lab/test_patchcore.py`

**Interfaces:**
- Consumes: six per-view normal-train exports plus calibration normal/defect rows.
- Produces: six versioned `model.ckpt` artifacts, per-view thresholds, raw anomaly maps, heatmaps, and runtime `PatchCoreBackend.predict(view_id, crop) -> BranchEvidence`.

- [ ] **Step 1: Write backend tests with an injected fake Anomalib predictor.**

Test exact view-to-checkpoint routing, Mono8-to-RGB conversion, finite score validation, threshold equality, raw-map preservation, load-once behavior, and model-load failure.

- [ ] **Step 2: Implement six resident PatchCore predictors.**

Use `Patchcore.load_from_checkpoint()` and preserve the checkpoint's preprocessing/model parameters. Do not reconstruct a different model at inference. GPU calls run serially through one runtime coordinator.

- [ ] **Step 3: Implement per-view training.**

Default profile: `wide_resnet50_2`, layers `layer2/layer3`, image size 512x512, coreset ratio 0.1, nine neighbours, seed 42, one training epoch. Persist the exact settings with every checkpoint.

- [ ] **Step 4: Fit independent per-view thresholds from calibration rows.**

Maximise image-level calibration F1 and save the complete score table. A view with no calibration defect data remains `REVIEW` and cannot participate in an OK system profile.

- [ ] **Step 5: Verify same-image/same-checkpoint equivalence.**

Run one calibration image through the training evaluation path and resident runtime, then assert identical checkpoint SHA256 and numerically equal raw anomaly score/map within `atol=1e-6` before accepting the backend.

- [ ] **Step 6: Run PatchCore tests.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_patchcore.py`

Expected: all tests pass and raw anomaly maps are written without display normalisation altering decision values.

---

### Task 8: Implement Template-first runtime, truthful fusion, and atomic evidence publishing

**Files:**
- Create: `src/bmw_inspection/lab/fusion.py`
- Create: `src/bmw_inspection/lab/publisher.py`
- Create: `src/bmw_inspection/lab/runtime.py`
- Test: `tests/unit/bmw_inspection/lab/test_fusion.py`
- Test: `tests/unit/bmw_inspection/lab/test_publisher.py`
- Test: `tests/unit/bmw_inspection/lab/test_runtime.py`

**Interfaces:**
- Consumes: six-view `CaptureSet`, four backend interfaces, and a freshly reloaded threshold/config snapshot.
- Produces: `LabRuntime.inspect(capture_set) -> PublishedInspection` containing an immutable `InspectionResult` and run directory.

- [ ] **Step 1: Write fusion truth-table tests.**

```python
def test_any_template_ng_short_circuits_every_downstream_backend() -> None:
    result = runtime(template_results={"back_right": BranchStatus.NG}).inspect(capture_set)
    assert result.final_status is FinalStatus.NG_TEMPLATE
    assert all(row.status is BranchStatus.SKIPPED for row in result.downstream_rows)
    assert yolo.call_count == patchcore.call_count == bright_streak.call_count == 0

def test_ok_requires_every_required_branch_clear() -> None:
    assert fuse(all_pass_evidence).final_status is FinalStatus.OK
```

Also test `RETAKE` precedence, model `ERROR`, multiple simultaneous downstream triggers, disabled required branch => `REVIEW`, and incomplete evidence never => OK.

- [ ] **Step 2: Implement the ordered runtime.**

Order is capture validation, image quality, all six Template branches, global Template gate, configured bright-streak views, six-view YOLO, six-view PatchCore, fusion, publication. Template failures must not invoke downstream backends.

- [ ] **Step 3: Keep models resident and thresholds hot-reloadable.**

At the start of every inspection, reload only the validated profile thresholds and enabled/required flags. If checkpoint path, ROI, or preprocessing identity changed, return `RELOAD_REQUIRED` for the UI action rather than mixing old models with new configuration.

- [ ] **Step 4: Publish one complete run directory by staging then rename.**

```text
results/bmw_inspection/<experiment_id>/runs/YYYYMMDD/<run_id>/
  config.snapshot.json
  result.json
  timing.json
  source/<six-view>.png
  crops/<branch>/<view>.png
  evidence/template/<view>/result.json
  evidence/template/<view>/overlay.png
  evidence/template/<view>/difference.png
  evidence/bright_streak/<view>/result.json
  evidence/bright_streak/<view>/response.png
  evidence/bright_streak/<view>/mask.png
  evidence/bright_streak/<view>/overlay.png
  evidence/yolo/<view>/candidates.json
  evidence/yolo/<view>/overlay.png
  evidence/patchcore/<view>/anomaly_map.npy
  evidence/patchcore/<view>/overlay.png
  evidence/fusion/<view>.png
```

Store experiment ID, config digest, model path/version/SHA256, thresholds, per-branch timings, and all triggered branches. Do not make files read-only or build a ZS32-style release bundle.

- [ ] **Step 5: Run runtime/fusion/publisher tests.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_fusion.py tests/unit/bmw_inspection/lab/test_publisher.py tests/unit/bmw_inspection/lab/test_runtime.py`

Expected: all tests pass, including zero downstream calls after Template NG.

---

### Task 9: Add batch evaluation and configuration comparison

**Files:**
- Create: `src/bmw_inspection/lab/evaluation.py`
- Create: `pipeline/bmw_lab_evaluate.py`
- Test: `tests/unit/bmw_inspection/lab/test_evaluation.py`

**Interfaces:**
- Consumes: manifest path, one or more experiment profiles, and resident offline runtime.
- Produces: `predictions.csv`, `summary.json`, `confusion.csv`, per-branch metrics, latency percentiles, `false_ok/`, and `false_ng/` evidence links.

- [ ] **Step 1: Write tests for part-level metrics and no threshold fitting on final test.**

- [ ] **Step 2: Implement `--split calibration|final_test` and `--compare profile1 profile2`.**

`calibration` mode may report candidate thresholds without editing profiles. `final_test` mode must be read-only and must reject `--write-thresholds`.

- [ ] **Step 3: Add truthful partial-profile reporting.**

If learned models are disabled or untrained, mark the evaluation `experimental_only=true`, list missing branches, and never report a fused OK accuracy as though the four-branch system ran.

- [ ] **Step 4: Run evaluation tests.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_evaluation.py`

Expected: tests pass and same-part split leakage is rejected.

---

### Task 10: Build the Chinese Experiment and Presentation UI

**Files:**
- Create: `src/bmw_inspection/lab/ui.py`
- Modify: `pipeline/bmw_lab_inspection.py`
- Test: `tests/unit/bmw_inspection/lab/test_ui.py`

**Interfaces:**
- Consumes: live/offline capture provider, `LabRuntime`, and published inspection results.
- Produces: one 1600x900 Chinese UI with `--mode experiment|presentation`.

- [ ] **Step 1: Write render-state and input-event tests.**

Cover startup idle with zero capture calls, front capture, flip prompt, back capture, inference progress, Template short-circuit, OK, each NG family, RETAKE, ERROR, reload required, and retry. Preserve the previous result only under an explicit `上次结果` label.

- [ ] **Step 2: Implement the shared layout.**

Presentation mode shows the six view thumbnails, selected enlarged evidence, final status, and four Chinese branch cards. Experiment mode adds score, threshold, elapsed time, experiment/model versions, diagnostic candidates, branch/view selection, batch-evaluation launch, and explicit model reload.

- [ ] **Step 3: Apply truthful visual semantics.**

Use green only for final PASS, red only for final business NG evidence, amber for below-threshold diagnostic candidates/REVIEW, magenta for ERROR, and neutral grey for SKIPPED. Rotate the configured narrow bright-streak ROI clockwise in its evidence panel without altering detector coordinates.

- [ ] **Step 4: Handle live exceptions inside the UI loop.**

Capture, decode, config reload, and inference exceptions must publish/show ERROR and keep Quit/Retry usable instead of escaping the event loop.

- [ ] **Step 5: Run UI tests and save deterministic screenshots.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/lab/test_ui.py`

Run: `uv run --no-sync python pipeline/bmw_lab_inspection.py --capture-set <six-view-dir> --mode presentation --save-screenshot /tmp/bmw-lab.png --no-gui`

Expected: tests pass and `/tmp/bmw-lab.png` is a 1600x900 fully Chinese result screen.

---

### Task 11: Execute the minimum acceptance matrix and hand off commands

**Files:**
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Consumes: completed Tasks 1-10 and actual model/data assets.
- Produces: documented collect, train, evaluate, offline replay, and live UI commands with current acceptance evidence.

- [ ] **Step 1: Run the complete BMW lab unit suite.**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection`

Expected: all original BMW Demo tests and all new lab tests pass.

- [ ] **Step 2: Run code-quality checks only on BMW lab files.**

Run: `uv run --no-sync ruff check src/bmw_inspection/lab pipeline/bmw_lab_*.py tests/unit/bmw_inspection/lab`

Run: `uv run --no-sync python -m compileall -q src/bmw_inspection/lab pipeline/bmw_lab_*.py`

Expected: both commands exit 0.

- [ ] **Step 3: Run a six-view Template-NG short-circuit smoke.**

Expected: final `NG_TEMPLATE`, six Template evidence rows, zero bright-streak/YOLO/PatchCore inference calls, and all downstream rows `SKIPPED`.

- [ ] **Step 4: Run a complete six-view all-pass/defect smoke after models exist.**

Expected: an all-pass set yields OK; injected or real `defect` boxes yield `NG_YOLO`; calibrated anomaly samples yield `NG_ANOMALY`; missing/broken streak yields `NG_BRIGHT_STREAK`.

- [ ] **Step 5: Run saved-image/runtime equivalence checks.**

For one image per view, confirm Template similarity, YOLO candidates, PatchCore raw score/map, and bright-streak metrics are the same in standalone and integrated paths using the same profile/model assets.

- [ ] **Step 6: Perform the site-only live smoke.**

Enumerate exactly `DA9805574`, `DA9625347`, and `DB0968108`; capture front three views, flip the same part, capture back three views, verify six image identities, run inspection, retry once, then close and release all cameras.

- [ ] **Step 7: Document exact operator commands and measured limitations.**

Document that the initial 13 images cover bright-streak regression only; record the six-view physical-part counts, YOLO box counts, split identities, trained model paths, thresholds, and final-test results before describing the four-branch profile as complete.

---

## Delivery Milestones

1. **M1 — Six-view shell:** Tasks 1-3. Live/offline six-view capture plus current bright-streak evidence; learned branches remain visibly unavailable.
2. **M2 — Template gate:** Task 4. Six per-view Template models and confirmed short-circuit behavior.
3. **M3 — Data gate:** Task 5 plus collection/annotation. At least 30 complete six-view physical parts for non-placeholder split reporting, with actual `defect` bounding boxes.
4. **M4 — Learned models:** Tasks 6-7. One global YOLO model and six PatchCore models with independent calibration thresholds.
5. **M5 — Integrated lab system:** Tasks 8-10. Truthful fusion, evidence archive, batch comparison, and dual-mode Chinese UI.
6. **M6 — Acceptance:** Task 11. Offline regression first, then one explicit live three-camera/two-round smoke.

## Explicit Non-Goals

- No PLC, MES, database, authentication, remote service, distributed queue, or production deployment release system.
- No automatic camera discovery fallback, automatic part flipping, or continuous video preview.
- No ZS32 eight-view compatibility layer or industrial approval workflow.
- No claim that PatchCore heatmaps are pixel-accurate segmentation.
- No claim of four-branch performance until independent six-view final-test data and trained model assets exist.
