# BMW Bright-Streak Tracked-Profile v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed vertical raw-profile bright-streak diagnostic with a geometry-aware tracked profile that accepts the confirmed continuous现场 streak while retaining complete no-streak rejection and explainable continuity evidence.

**Architecture:** A new standalone module computes a two-dimensional local-contrast response inside the existing `81x613` ROI, tracks one smooth row-wise path with dynamic programming, and applies strong/weak hysteresis before the existing coverage/run/gap classification. A separate no-overwrite evaluator fits and publishes a SHA-bound v3 report from calibration rows plus the explicitly accepted现场 record; the Demo opts into the new engine without changing Template, YOLO, EfficientAD, or fusion.

**Tech Stack:** Python 3.13, NumPy, OpenCV, pytest, JSON/CSV/NPZ artifacts, existing `uv` environment.

## Global Constraints

- Only the `front_left` bright-streak branch may change; Template, YOLO, EfficientAD, the 25-row result structure, and whole-part fusion remain unchanged.
- Keep the fixed full-image ROI `[1792, 1180, 1873, 1793]` with exact shape `81x613`; track geometry inside it instead of globally relaxing the final gap rule.
- `bmw_demo_20260812_211302` is the only newly user-confirmed现场 normal in this task and must remain explicitly identified in provenance.
- Existing final-test rows are report-only and must not participate in initial fitting.
- Every existing complete `NG_NO_STREAK` row must remain NG; no real broken-but-present samples exist, so synthetic gap tests are safeguards rather than recall evidence.
- Publish a new immutable `tracked_profile_v3` artifact; retain `raw_profile_v2` code and artifacts for rollback and never silently fall back between engines.
- Do not commit `dataset/`, `results/`, models, checkpoints, screenshots, or customer images.
- Evidence must continue to expose coverage, longest run, maximum gap, and gap count, and must distinguish strong path rows, weak bridged rows, and diagnostic gaps.

---

### Task 1: Tracked profile core and synthetic behavior

**Files:**
- Create: `src/bmw_inspection/lab/bright_streak_tracked_profile.py`
- Create: `tests/unit/bmw_inspection/lab/test_bright_streak_tracked_profile.py`

**Interfaces:**
- Produces: `TrackedProfileGeometry`, `TrackedProfileThresholds`, `TrackedProfileMetrics` frozen dataclasses.
- Produces: `analyze_tracked_profile(image: np.ndarray, geometry: TrackedProfileGeometry, thresholds: TrackedProfileThresholds) -> TrackedProfileMetrics`.
- Produces: `classify_tracked_profile(metrics: TrackedProfileMetrics, thresholds: TrackedProfileThresholds) -> str` returning `OK`, `NG_NO_STREAK`, or `NG_BROKEN`.
- Produces: `fit_tracked_profile_thresholds(records: Sequence[Mapping[str, object]], geometry: TrackedProfileGeometry) -> TrackedProfileThresholds` where every record has `label` and `path_scores`.

- [ ] **Step 1: Write failing geometry/path tests**

Add tests that build an `81x613` grayscale ROI with a bright diagonal band whose centre moves by at most two pixels per row. Assert the returned `path_x` follows the known band within two pixels on active rows, all arrays are owned read-only copies, and invalid ROI shapes or even smoothing windows fail with exact `ValueError` messages.

- [ ] **Step 2: Run the new test file and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/bmw-v3-uv uv run --no-sync pytest -q \
  tests/unit/bmw_inspection/lab/test_bright_streak_tracked_profile.py
```

Expected: collection/import failure because `bright_streak_tracked_profile` does not exist.

- [ ] **Step 3: Implement response map and deterministic path tracking**

Use defaults `candidate_width=7`, `background_width=10`, `background_gap=3`, `smooth_window=5`, `max_step=2`, and `step_penalty=1.0`. For every legal centre column, calculate centre-band minus symmetric-background mean. Median-smooth responses along rows. Dynamic programming must use transitions only from `x-max_step ... x+max_step`, subtract `step_penalty * abs(delta_x)`, and choose the lowest column on exact ties. Trace back one full-height path and extract `path_scores` without modifying the input image.

- [ ] **Step 4: Write failing hysteresis and classification tests**

Cover all of these independently:

```python
assert classify_tracked_profile(continuous_diagonal, thresholds) == "OK"
assert classify_tracked_profile(no_strong_seed, thresholds) == "NG_NO_STREAK"
assert classify_tracked_profile(five_soft_rows_between_strong_runs, thresholds) == "OK"
assert classify_tracked_profile(twelve_background_rows_between_strong_runs, thresholds) == "NG_BROKEN"
```

Assert weak components are retained only when they contain a strong row. Assert `strong_mask`, `accepted_mask`, `bridged_mask`, coverage, run, internal gap, and count agree with explicit pixel counts.

- [ ] **Step 5: Implement strong/weak masks and metrics**

Build `strong_mask = path_scores >= strong_row_score` and `weak_mask = path_scores >= weak_row_score`. Retain only weak connected components containing at least one strong row. Define `bridged_mask = accepted_mask & ~strong_mask`. Reuse bounded internal-gap semantics: leading/trailing background is not a continuity gap. Classify presence first, then continuity.

- [ ] **Step 6: Write fitting tests and implement calibration-only threshold fitting**

Tests must use labeled synthetic `normal` and `no_streak` path-score arrays and reject other labels. The fit must require both classes and strict separation of their peak-score envelopes. Set the strong threshold to the midpoint between calibration `max(no_streak peak)` and `min(normal peak)`. Set the weak threshold one quarter of that separation above the no-streak peak. Fit presence/run/gap envelopes from normal masks only, record exact pixel-derived ratios, and never accept `final_test` as an implicit field.

- [ ] **Step 7: Verify and commit Task 1**

Run the new test file, `python -m compileall -q` on the new module, and `git diff --check`. Commit only the new module and tests.

---

### Task 2: Immutable evaluator, real fitting, and replay evidence

**Files:**
- Create: `pipeline/bmw_lab_evaluate_bright_streak_tracked_profile.py`
- Modify: `src/bmw_inspection/lab/bright_streak_tracked_profile.py`
- Modify: `tests/unit/bmw_inspection/lab/test_bright_streak_tracked_profile.py`
- Create: `tests/unit/pipeline/test_bmw_lab_evaluate_bright_streak_tracked_profile.py`

**Interfaces:**
- Produces: `evaluate_bright_streak_tracked_profile(manifest: Path, output_dir: Path, accepted_normal_records: Sequence[Path], roi_xyxy: tuple[int, int, int, int]) -> Mapping[str, object]`.
- Produces ignored artifact: `results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v3_tracked/report.json` plus `metrics.csv`, `profiles/*.npz`, and replay summary.
- Consumes the Task-1 geometry, analyzer, fitter, and classifier APIs.

- [ ] **Step 1: Write failing CLI and no-overwrite tests**

Assert `--help` exposes `--manifest`, `--output-dir`, repeatable `--accepted-normal-record`, and fixed ROI coordinates. Assert an existing output file, directory, or symlink raises before image processing. Assert accepted records must contain all three `front_left_hdr.png`, `front_left_short.png`, and `front_left_long.png` source frames plus `inspection.json`, and that the inspection capture ID matches the directory.

- [ ] **Step 2: Verify RED**

Run the two new test files. Expected: the evaluator module and CLI are missing.

- [ ] **Step 3: Implement provenance-bound evaluation**

Read the existing bright-streak manifest using the v2 field contracts. Fit only rows with `split=calibration`, append each explicitly supplied accepted现场 record as `label=normal` with `provenance_kind=user_confirmed_live_normal`, and keep final-test rows report-only. Save per-row `response_map`, `path_x`, `path_scores`, strong/accepted/bridged masks in compressed NPZ. Hash manifest, every accepted record HDR/source/inspection file, ROI configuration, and algorithm source.

- [ ] **Step 4: Write report-contract and replay tests**

Assert the report has exact top-level identity for `tracked_profile_v3`, geometry, thresholds, calibration counts, accepted live IDs/hashes, final-test outcomes, `final_test_used_for_fit=false`, and `real_broken_samples=0`. Assert unconfirmed recent records can be replayed but are listed as `truth=unknown` and excluded from accuracy.

- [ ] **Step 5: Materialize the real immutable v3 artifact**

Run with:

```bash
UV_CACHE_DIR=/tmp/bmw-v3-uv uv run --no-sync python \
  pipeline/bmw_lab_evaluate_bright_streak_tracked_profile.py \
  --manifest dataset/bmw_lab_prepared/bmw_right_batch_20260810_21_v1/manifests/bright_streak.csv \
  --accepted-normal-record results/bmw_eight_view_demo_v3_ng_evidence_v1/bmw_demo_20260812_211302 \
  --output-dir results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v3_tracked
```

If that output exists, stop and choose a new versioned output ID; never overwrite it.

- [ ] **Step 6: Verify real acceptance evidence**

Require the confirmed record to be `OK`, every existing no-streak row to be `NG_NO_STREAK`, and normal calibration/final-test false rejects to be no worse than v2 with a goal of zero. Replay the 20 latest saved live records and report v2→v3 changes without assigning unknown truth. Record CPU per-image p50/max; do not claim a hardware-cycle benchmark.

- [ ] **Step 7: Verify and commit Task 2**

Run both Task-1/Task-2 tests plus compileall and diff-check. Commit code/tests only; confirm the real artifact remains ignored.

---

### Task 3: Demo engine, overlay, and isolated branch invariance

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`
- Modify: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`
- Modify: `tests/unit/bmw_inspection/lab/test_v3_ng_evidence_demo.py`
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`

**Interfaces:**
- Extends config engine enumeration with exact value `tracked_profile_v3`.
- Produces: `EightViewTrackedProfileBrightStreakPredictor(report_path: Path)` with `predict(image: np.ndarray) -> ModelOutput`.
- Consumes the immutable Task-2 `report.json`, pinned by `bright_streak.config_sha256`.

- [ ] **Step 1: Write failing strict-config tests**

Assert v3 config accepts only a SHA-valid tracked report, rejects changed bytes, and preserves `raw_profile_v2` compatibility. Unknown engines still fail closed. Assert the checked-in v3 Demo config resolves the new immutable report and exact SHA.

- [ ] **Step 2: Write failing predictor/evidence tests**

Use synthetic diagonal, absent, softly bridged, and clearly broken ROIs. Assert Chinese reasons expose presence, coverage, run, max gap, gap count, strong/weak thresholds, and bridged row count. Assert the overlay draws the tracked centreline with distinct strong/bridged/gap colors and retains exact `(613, 81, 3)` shape.

- [ ] **Step 3: Implement engine and predictor**

Load and validate exact report schema, ROI, geometry, thresholds, source identity, and no-real-broken marker. Crop only the fixed ROI from the full HDR `front_left` image. Return branch PASS only for `OK`; map both no-streak and broken to NG. Do not touch other model adapters or fusion.

- [ ] **Step 4: Write and verify branch-invariance tests**

Run the same eight images through the old v3 suite with a stub v2 bright predictor and through the new suite with a stub v3 bright predictor. Serialize every Template, YOLO, and EfficientAD `(branch, view, status, score, threshold, reason)` tuple and assert exact equality. Assert only the one bright-streak row may differ and result row count remains 25.

- [ ] **Step 5: Pin the real artifact and update memories**

Replace only `bright_streak.engine`, `bright_streak.config`, and `bright_streak.config_sha256` in `bmw_eight_view_demo_v3_ng_evidence.json`. Record v3 artifact identity, thresholds, replay metrics, evidence meanings, rollback config, and missing-real-broken boundary in both folder memory files.

- [ ] **Step 6: Verify and commit Task 3**

Run all changed config/model/Demo tests plus v2 raw-profile tests, compile production modules, parse the JSON config, and run `git diff --check`. Commit without data or result artifacts.

---

### Task 4: Real offline acceptance, final review, and live restart

**Files:**
- Create ignored verification output only: `artifacts/bmw_bright_streak_tracked_v3_smoke/`

**Interfaces:**
- Consumes the checked-in v3 Demo config and local immutable v3 report.
- Produces an ignored comparison report and screenshot, then restarts the existing four-camera Demo.

- [ ] **Step 1: Run the focused regression gate**

Run tracked-profile, raw-profile-v2, Demo config/model/UI/persistence/capture, and pipeline entrypoint tests. Zero new failures are allowed; report unrelated optional-dependency failures separately.

- [ ] **Step 2: Run the confirmed现场 record offline**

Re-run `bmw_demo_20260812_211302` from its saved HDR images. Save a 1600x900 screenshot and inspection record under the ignored artifact directory. Assert bright streak is PASS and the displayed path/bridged evidence matches persisted numeric details.

- [ ] **Step 3: Prove non-bright branch invariance**

Serialize all 24 non-bright rows from the same input before and after v3 and require exact equality. Store both lists and their SHA-256 in the ignored verification report.

- [ ] **Step 4: Request whole-branch review and fix findings**

Review from commit `86d7967c` through the final implementation head against the design and this plan. Fix every Blocking/Important finding and re-run covering tests. Record Minor findings and explicit hardware boundaries.

- [ ] **Step 5: Restart the live Demo**

Stop the current session only after offline acceptance passes. Launch:

```bash
cd /home/yunjing/anomaly_xingtao_new
UV_CACHE_DIR=/tmp/bmw-v3-live-uv uv run --no-sync python \
  .worktrees/bmw-eight-view-handoff/pipeline/bmw_lab_eight_view_demo.py \
  --config .worktrees/bmw-eight-view-handoff/configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json \
  --experiment-mode
```

Claim live readiness only if the process stays running, all four cameras open, and a fresh capture record is written. Otherwise report the exact failing layer without rolling back verified offline work.
