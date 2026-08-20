# ZS32 Resident Multiprocess PatchCore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce one all-PASS part's eight-view PatchCore wall time with safe resident `spawn` processes.

**Architecture:** A parent backend statically shards views across isolated child processes. Children retain their own Anomalib/Lightning objects, write unique evidence files, and return small validated envelopes; the existing Template gate and outer serial worker remain unchanged.

**Tech Stack:** Python 3.13, multiprocessing spawn/Pipe, Anomalib, Lightning, CUDA, OpenCV, NumPy, pytest, uv.

## Global Constraints

- Optimize single-part latency; increased RAM, VRAM, and CUDA contexts are acceptable.
- Only `spawn` is allowed; never fork an initialized CUDA runtime.
- Preserve score hex, raw arrays, decoded masks, decoded overlays, thresholds, and final decisions.
- Template NG must send zero PatchCore requests.
- Production remains serial unless a zero-error candidate improves median by at least 10% and does not regress p95.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Multiprocess protocol and lifecycle

**Files:**
- Create: `capture_data/zs32_patchcore_multiprocess.py`
- Create: `tests/unit/capture_data/test_zs32_patchcore_multiprocess.py`

**Interfaces:**
- Produces: `ResidentMultiprocessPatchcoreBackend(specs, views, process_count, ..., worker_target=None)`.
- Produces: `predict_all(...)`, `close()`, deterministic `view_shards(...)`.

- [x] Write failing tests for 1/2/4/8 sharding, READY gating, eight-result aggregation, stale/duplicate/missing responses, timeout/EOF/death, PID reuse, and idempotent bounded close.
- [x] Run the focused suite and confirm failures are caused by the missing backend.
- [x] Implement pure-data envelopes, per-child Pipes, `connection.wait`, fail-closed reconstruction, and `spawn` lifecycle.
- [x] Re-run the suite and confirm it passes.

### Task 2: Runtime and worker integration

**Files:**
- Modify: `capture_data/zs32_demo_config.py`
- Modify: `configs/zs32/zs32_demo.json`
- Modify: `capture_data/zs32_demo_runtime.py`
- Modify: `pipeline/zs32_inference_worker.py`
- Modify: `tests/unit/capture_data/test_zs32_demo_config.py`
- Modify: `tests/unit/capture_data/test_zs32_demo_runtime.py`
- Modify: `tests/unit/zs32_refactor/dashboard/test_inference_worker.py`

**Interfaces:**
- Consumes: `ResidentMultiprocessPatchcoreBackend`.
- Produces: validated process count in `{1,2,4,8}` and `ZS32DemoRuntime.close()`.

- [x] Write failing tests for configuration, backend selection, zero calls on Template NG, and `finally` cleanup.
- [x] Run focused tests and verify the expected failures.
- [x] Integrate the backend without changing Template, YOLO, or publication semantics.
- [x] Re-run focused tests and confirm they pass.

### Task 3: Reproducible GPU selection gate

**Files:**
- Create: `results/zs32_demo_speedup_20260727/benchmark_patchcore_multiprocess.py`
- Create at runtime: `results/zs32_demo_speedup_20260727/patchcore_multiprocess_ab.json`

**Interfaces:**
- Consumes: production serial and multiprocess backends.
- Produces: strict 1/2/4/8 timing, stability, equivalence, memory, and acceptance records.

- [x] Add parser/unit tests for acceptance calculations and incomplete-result rejection.
- [x] Implement two warmups, five formal rounds, twenty stability rounds, median/p95, PID stability, and exact artifact comparison.
- [x] Run serial then 2/4/8 on the real saved crops with fresh pools.
- [x] Select the fastest qualified count; update production config only when all gates pass.

### Task 4: Final verification and memory

**Files:**
- Modify: `pipeline/AGENTS_MEMORY.md`
- Modify: `capture_data/AGENTS_MEMORY.md`

- [x] Run all focused multiprocessing, Demo, worker, parser, and capture tests.
- [x] Run `py_compile` and `git diff --check`.
- [x] Replay Template-NG and all-PASS paths.
- [x] Record exact selected/rejected timings, VRAM/RSS, limitations, and rollback value `1`.
