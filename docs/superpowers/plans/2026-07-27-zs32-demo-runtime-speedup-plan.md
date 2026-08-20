# ZS32 Demo Runtime Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove repeated image decoding/encoding from the Template-NG path, shorten safe HDR capture pacing, and benchmark-gate PatchCore concurrency.

**Architecture:** Keep immutable result paths and fail-closed contracts. Share a bounded decoded-image cache between parser and renderer, make the worker the only live image decoder, keep crops as arrays through Template, and defer PatchCore-only files until the global gate passes.

**Tech Stack:** Python 3.13, OpenCV, NumPy, pytest, uv, Anomalib, CUDA.

## Global Constraints

- Preserve exact source PNG bytes in the runtime result.
- Preserve all thresholds, model checkpoints, ROI coordinates and branch decisions.
- Keep both front and back operator confirmations.
- Keep PatchCore serial unless same-image GPU A/B proves exact evidence equivalence and lower wall time.
- Do not commit or overwrite unrelated dirty-worktree changes.

---

### Task 1: Cache Dashboard source decodes

**Files:**
- Modify: `src/zs32_inspection/dashboard/compositor.py`
- Modify: `src/zs32_inspection/dashboard/parser.py`
- Test: `tests/unit/zs32_refactor/dashboard/test_compositor.py`
- Test: `tests/unit/zs32_refactor/dashboard/test_parser.py`

**Interfaces:**
- Produces: `load_source_image(path: Path) -> np.ndarray`, a bounded immutable-path cache that returns detached arrays.

- [ ] Add a regression test that counts `cv2.imread` calls while parsing and rendering one unchanged result repeatedly; expect one source decode per view.
- [ ] Run the focused test and confirm it fails because parser and compositor decode independently.
- [ ] Implement the bounded shared cache and use it in parser and compositor.
- [ ] Re-run focused Dashboard tests and confirm they pass.

### Task 2: Decode once and keep Template crops in memory

**Files:**
- Modify: `capture_data/zs32_live_commissioning.py`
- Modify: `capture_data/zs32_demo_runtime.py`
- Test: `tests/unit/capture_data/test_zs32_live_commissioning.py`
- Test: `tests/unit/capture_data/test_zs32_demo_runtime.py`

**Interfaces:**
- Produces: `DemoTemplateMatcher.score_array(image: np.ndarray, view: str)`.
- Produces: source archives copied from request paths and in-memory crop arrays.

- [ ] Add tests proving Stage35 does not decode manifest images, source archive SHA-256 equals the capture PNG, Template receives arrays, Template-NG writes no crop PNGs, and all-PASS writes all PatchCore crop PNGs.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Remove Stage35 decode validation, copy source PNG bytes, add array Template scoring, and defer crop writes until after the global gate.
- [ ] Re-run focused live/Demo tests and confirm they pass.

### Task 3: Shorten safe HDR trigger pacing

**Files:**
- Modify: `capture_data/zs32_live_commissioning.py`
- Test: `tests/unit/capture_data/test_zs32_live_commissioning.py`

**Interfaces:**
- Produces: live capture command with `--capture-interval 0.2`; all other confirmation and HDR quality settings unchanged.

- [ ] Change the command-contract test to require `0.2` and confirm it fails against `0.5`.
- [ ] Change only the live capture interval.
- [ ] Re-run live commissioning and bootstrap capture command tests.

### Task 4: Benchmark-gate PatchCore concurrency

**Files:**
- Modify only if accepted by A/B: `capture_data/zs32_model_runtime.py`
- Test if accepted: `tests/unit/capture_data/test_zs32_model_runtime.py`
- Create benchmark artifact under: `results/zs32_demo_speedup_20260727/`

**Interfaces:**
- Candidate: `AnomalibPatchcoreBackend.predict_all(..., inference_workers=2)` using distinct existing engines/models.

- [ ] Record current eight-view serial same-image wall time and evidence hashes.
- [ ] Prototype bounded two-worker inference outside production code.
- [ ] Compare scores, raw maps, masks, overlays and failure behavior across repeated runs.
- [ ] Promote the two-worker scheduler with a failing test only if evidence is exact and wall time improves; otherwise retain serial and record the measured rejection.

### Task 5: End-to-end verification and memory refresh

**Files:**
- Modify: `pipeline/AGENTS_MEMORY.md`

- [ ] Run all focused suites for Tasks 1-4.
- [ ] Run compile and formatting checks for modified Python files.
- [ ] Replay the saved Template-NG eight-view sample and compare decisions plus source hashes.
- [ ] Measure cold-independent Template-NG runtime and output size against the 4.75 s / 135.72 MiB baseline.
- [ ] Update pipeline memory with exact implementation and measured results.

