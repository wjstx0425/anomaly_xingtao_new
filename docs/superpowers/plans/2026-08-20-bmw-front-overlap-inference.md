# BMW Front Capture Overlap Inference Implementation Plan

> **For agentic workers:** Implement task-by-task with TDD. Do not commit, push, merge, reset, or clean this dirty worktree.

**Goal:** Start the 13 front-view checks immediately after front capture and overlap them with flip/back capture while preserving the existing final 25-check contract.

**Architecture:** Add explicit immutable round results to `EightViewModelSuite`, then let the GUI submit front and back rounds to one resident worker. Finalization remains the only place that fuses, matches trusted OK, builds `EightViewInspection`, and persists.

**Tech Stack:** Python, `concurrent.futures.ThreadPoolExecutor`, OpenCV, pytest, uv.

## Global Constraints

- Do not change algorithms, models, thresholds, ROI, mask, HDR, camera parameters, or fusion.
- Keep camera SDK and HighGUI on the main thread.
- Use exactly one model worker and never run front/back model calls concurrently.
- Preserve the canonical final result order and persist only complete eight-view results.
- Preserve unrelated worktree changes and do not commit.

### Task 1: Explicit front/back model stages

**Files:**
- Modify: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

- [ ] Add failing tests for 13 front calls, 12 back calls, shared per-view crops, full-image light streak, no duplicate front inference, and canonical 25-row final order.
- [ ] Add immutable `RoundInspectionResult` containing round views, rows, ROI crops, and elapsed model time.
- [ ] Add `inspect_round(images, round_name)` and `finalize_rounds(images, front, back, capture_id)` with strict completeness/duplicate validation.
- [ ] Make existing `inspect()` a synchronous compatibility wrapper around the two rounds and finalization.
- [ ] Run focused model tests.

### Task 2: Single-worker live GUI orchestration

**Files:**
- Modify: `pipeline/bmw_lab_eight_view_demo.py`
- Test: `tests/unit/pipeline/test_bmw_lab_eight_view_demo.py`

- [ ] Add failing tests for front submit before back capture, back allowed while front is incomplete, PROCESSING space ignored, stale-generation discard, and one final persistence call.
- [ ] Add a small pending-job state carrying generation, futures, frozen images/sources, and capture id.
- [ ] Submit front and back round calls to `ThreadPoolExecutor(max_workers=1)` and poll from the HighGUI loop without calling `future.result()` until done.
- [ ] Finalize and persist after both rounds; route orchestration exceptions to UI ERROR.
- [ ] Keep offline/no-GUI behavior synchronous and unchanged.
- [ ] Run focused pipeline/UI tests.

### Task 3: Focused verification and handoff memory

**Files:**
- Modify: `AGENTS_MEMORY.md`

- [ ] Run the active-chain pytest files and Python syntax checks with the root uv environment.
- [ ] Run one offline eight-view replay through the single background model worker; do not tune thresholds.
- [ ] Run `git diff --check` and confirm no new SHA dependency.
- [ ] Record behavior, tests, replay result, and unverified live camera/GPU boundary in `AGENTS_MEMORY.md`.
