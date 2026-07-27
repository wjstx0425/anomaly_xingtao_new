# ZS32 Resident Multiprocess PatchCore Design

## Goal

Minimize the wall-clock latency from submission of one ZS32 part to completion
of all eight views. RAM, VRAM, CUDA-context count, and resident-process overhead
may increase when that produces a measured latency reduction.

This change does not introduce cross-part pipelining. The outer Dashboard
worker continues to accept one part at a time.

## Current Evidence

The current resident Demo worker already loads Template, eight PatchCore
models, and YOLO once. Template is a global gate: any Template NG completes the
part without invoking PatchCore or YOLO.

The accepted serial PatchCore implementation runs eight different models and
Lightning Engines in one process. Its formal median on the saved real crops was
`5.115354 s`.

A two-thread candidate in the same process was rejected. Every warmup and
formal parallel arm failed the same four views with
`IndexError: pop from empty list`. Its apparent `4.170282 s` median was an
incomplete result and is not a valid speedup. The failure prohibits
same-process threaded `Engine.predict` calls; it does not prohibit
process-isolated inference.

## Considered Approaches

### A. Static view sharding across resident child processes

Each child owns a mutually exclusive subset of PatchCore models and Lightning
Engines. Children run concurrently for one part, while each child processes
its assigned views serially.

This is the selected approach because it isolates Lightning state, preserves
the current inference implementation, and requires no tensor IPC.

### B. Multiple complete Demo runtime replicas

Each process would load all eight PatchCore models plus Template and YOLO.
This improves multi-part throughput but does not shorten the critical path of
one part. It is rejected for this objective.

### C. Direct model forward and batched feature extraction

Bypassing Lightning Trainer/DataLoader setup may offer a larger speedup.
However, it changes the authoritative preprocessing and postprocessing path.
It is deferred until the process-isolated design has been measured.

## Architecture

Add a `ResidentMultiprocessPatchcoreBackend` that implements the existing
`predict_all(crops, evidence_dir, diagnostic_mask_threshold=...)` contract.
`ZS32DemoRuntime` selects it without changing the Template gate, YOLO call, or
result publication logic.

The backend uses only:

```python
multiprocessing.get_context("spawn")
```

`fork` and `forkserver` are forbidden. Every child imports Torch, Anomalib, and
Lightning inside the spawned process and creates its own CUDA context,
PatchCore models, and Engines.

The parent sends one request per child containing:

- request ID;
- assigned view-to-crop absolute paths;
- absolute evidence directory;
- diagnostic mask threshold.

The parent does not send NumPy arrays, tensors, models, Engines, or exception
objects through IPC.

Each child writes only its assigned, view-unique artifacts directly into the
current staging result:

- `evidence/patchcore/<view>.png`;
- `evidence/patchcore/raw_maps/<view>.npy`;
- `evidence/patchcore/masks/<view>.png`.

The child returns a pure-data envelope containing score, evidence paths, raw
and mask shapes, mask source, applied diagnostic threshold, request ID, view,
and optional error text. The parent validates the request ID and exact view set
and reconstructs `ModelEvidence`.

Each child has a dedicated duplex `Pipe`. The parent waits with
`multiprocessing.connection.wait` and one bounded request deadline. A timeout,
EOF, child exit, malformed envelope, stale request ID, duplicate view, or
missing view produces fail-closed exceptions for the affected views.

## Process Counts and Sharding

The implementation supports `1`, `2`, `4`, and `8` PatchCore processes for
benchmarking.

- `1`: all eight views, authoritative serial baseline.
- `2`: each child owns three primary views and one secondary view.
- `4`: two children own two primary views each; two children own one primary
  and one secondary view each. Exact assignment is balanced from measured
  per-view serial latency.
- `8`: one view and one Engine per child.

Models are never duplicated between PatchCore children for a single
configuration. The eight checkpoints total approximately 1.5 GB on disk.
Six primary memory banks are materially larger than the two secondary banks,
so simple consecutive-view slicing is not an acceptable four-process balance.

The production process count is not assumed. It is selected only by the GPU
A/B gate.

## Startup and Shutdown

The PatchCore process pool is created before the outer Unix socket is
published. Every child must:

1. load its assigned models;
2. create its independent Engines and CUDA context;
3. execute one discarded warmup inference per assigned view using the real ROI
   geometry;
4. send `READY`.

`ZS32DemoRuntime` exposes `close()`, which delegates to the PatchCore backend.
`pipeline/zs32_inference_worker.py` calls it in `finally`.

Normal shutdown sends `SHUTDOWN`, joins each child, then terminates and kills
only children that exceed bounded shutdown deadlines. The Dashboard's existing
process-group cleanup remains a final fallback.

No child process is contacted when the global Template gate returns NG or
ERROR.

## Configuration

The single Demo configuration gains an explicit PatchCore process-count
setting restricted to `1`, `2`, `4`, or `8`. Model, threshold, ROI, topology,
and branch semantics remain unchanged.

Benchmark candidates are selected explicitly. Startup does not benchmark and
auto-select on every Dashboard launch. After A/B acceptance, the fastest
qualified count is written to `configs/zs32/zs32_demo.json`.

## GPU Benchmark

Benchmark `1`, `2`, `4`, and `8` processes separately on the same RTX GPU and
the same eight saved real crops.

For every count:

- start a fresh resident pool;
- complete two unmeasured warmup rounds;
- run at least five formal timed rounds;
- run twenty consecutive stability rounds;
- measure parent dispatch through durable completion of all artifacts;
- record median, p95, per-view timing, startup time, peak VRAM, parent/child
  RSS, and errors.

The serial arm remains the authoritative reference. Candidate acceptance
requires:

- zero errors in warmup, formal, and stability rounds;
- all eight scores equal by float hex representation;
- raw anomaly maps equal with `numpy.array_equal(..., equal_nan=True)`;
- decoded masks equal pixel-for-pixel;
- decoded evidence overlays equal pixel-for-pixel;
- child PIDs stable and models loaded once across repeated rounds;
- formal median at least 10% lower than serial;
- formal p95 no slower than serial.

The fastest accepted process count becomes production configuration. If no
candidate passes, production remains serial and the rejection evidence is
retained.

## Testing

Unit tests use spawned fake workers or injectable worker targets and cover:

- exact deterministic view sharding;
- READY gating before outer worker availability;
- request and response schema validation;
- aggregation into all eight `ModelEvidence` records;
- one-view and one-child errors;
- timeout, EOF, stale response, duplicate response, and child death;
- bounded, idempotent close;
- two consecutive requests reuse the same child PIDs;
- Template NG makes zero PatchCore pool requests;
- existing serial backend remains available for rollback.

Integration tests verify that the outer resident worker closes the runtime in
`finally` and does not publish its socket before the pool is ready.

The real GPU benchmark is a separate, reproducible artifact and is mandatory
before changing the production process count.

## Non-Goals

- No simultaneous processing of different parts.
- No change to capture timing or operator confirmations.
- No change to Template thresholds or global gate behavior.
- No change to YOLO execution.
- No model retraining.
- No direct-forward or shared-backbone rewrite in this phase.
- No acceptance based only on GPU utilization or apparent timing from
  incomplete results.
