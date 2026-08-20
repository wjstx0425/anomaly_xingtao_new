# ZS32 Four-Camera Legacy Layout Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit four-camera legacy-layout mode that writes the historical per-hand/per-view dataset tree and unchanged 25-column CSV manifest, extended to eight image rows plus one sample row.

**Architecture:** Keep `HikvisionCameraAdapter`, `CapturePlan`, frame validation, and batch camera lifetime unchanged. Add a focused legacy dataset writer beside the isolated bootstrap store, then select that writer from the bootstrap CLI when `--legacy-layout` is present. Legacy mode publishes only historical paths and uses atomic temporary PNG/CSV replacement without duplicating `_bootstrap` images.

**Tech Stack:** Python 3.10+, standard-library `csv`/`pathlib`/`os`, existing ZS32 capture contracts, pytest, uv.

## Global Constraints

- Preserve serial-bound topology identity; never map views by volatile SDK enumeration index.
- Legacy normal path is `<root>/<hand>/<view>/normal/<session>/images`.
- Legacy defect path is `<root>/<hand>/<view>/defect/<defect_type>/<session>/images`.
- Keep the historical 25 CSV columns in their exact existing order.
- A complete four-camera sample is exactly 8 image rows plus 1 complete sample row.
- `images-per-group` remains exactly `1`.
- `--legacy-layout` writes no duplicate `_bootstrap` image publication.
- Existing isolated bootstrap mode remains unchanged when the flag is absent.
- Do not expand Stage 3, YOLO, Label Studio, or training view contracts in this task.
- Do not overwrite existing images or manifests; conflicts fail before hardware initialization.
- Use `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync` for verification in this checkout.

---

### Task 1: Legacy Dataset Writer

**Files:**
- Create: `src/zs32_inspection/capture/legacy_dataset.py`
- Create: `tests/unit/zs32_refactor/capture_data/test_legacy_dataset_store.py`

**Interfaces:**
- Consumes: `CaptureRequest`, `CapturePlan`, `CaptureFrame`, `RoundConfirmation`, and `BootstrapCaptureMetadata`.
- Produces: `LEGACY_MANIFEST_COLUMNS`, `LegacyCaptureContext`, and `LegacyDatasetCaptureStore` with `preflight()`, `publish_complete()`, and `publish_incomplete()`.

- [x] **Step 1: Write failing tests for normal path, filename, and exact CSV rows**

Create deterministic four-camera frames and assert these paths exist after publication:

```python
expected = (
    tmp_path
    / "left"
    / "front_secondary"
    / "normal"
    / "20260714_120000_000001"
    / "images"
    / "left_front_secondary_normal_part_group001_000001_fused.png"
)
assert expected.read_bytes() == frame_by_view["front_secondary"].image_bytes
```

Read `<root>/manifests/20260714_120000_000001.csv` with `csv.DictReader` and assert:

```python
assert tuple(reader.fieldnames or ()) == LEGACY_MANIFEST_COLUMNS
assert len(rows) == 9
assert [row["record_type"] for row in rows].count("image") == 8
assert rows[-1]["record_type"] == "sample"
assert rows[-1]["sample_status"] == "complete"
```

- [x] **Step 2: Run the target test and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m pytest \
  --confcutdir=tests/unit/zs32_refactor/capture_data -q \
  tests/unit/zs32_refactor/capture_data/test_legacy_dataset_store.py
```

Expected: collection/import failure because `zs32_inspection.capture.legacy_dataset` does not exist.

- [x] **Step 3: Implement immutable legacy context and exact column contract**

Define:

```python
LEGACY_MANIFEST_COLUMNS = (
    "record_type", "session_id", "sample_id", "group_id", "image_index",
    "round", "view", "device_index", "camera_serial", "capture_mode",
    "exposure", "gain", "file", "source_short", "source_long",
    "short_exposure", "long_exposure", "hdr_attempt", "fused_clip_pct",
    "captured_at", "sample_status", "failed_round", "failed_view",
    "failed_device_index", "error",
)

@dataclass(frozen=True, slots=True)
class LegacyCaptureContext:
    label: str
    defect_type: str | None
    part_id: str
```

Validate `label in {"normal", "defect"}` and require a non-empty safe `defect_type` only for defect.

- [x] **Step 4: Implement path and row mapping helpers**

Implement these signatures:

```python
def legacy_image_path(
    root: Path,
    request: CaptureRequest,
    frame: CaptureFrame,
    context: LegacyCaptureContext,
) -> Path: ...

def legacy_image_row(
    request: CaptureRequest,
    frame: CaptureFrame,
    context: LegacyCaptureContext,
    destination: Path,
    *,
    sample_status: str,
) -> dict[str, str]: ...
```

Derive `group_id` and `image_index` from the structured `CaptureRequest` IDs using `context.part_id` as the required prefix, and validate both IDs before creating paths. Do not parse arbitrary destination filenames. Map HDR capture parameters with:

```python
parameters = frame.capture_parameters
short_exposure = parameters.get("short_exposure", "")
long_exposure = parameters.get("long_exposure", "")
hdr_attempt = parameters.get("hdr_attempt", "")
fused_clip_pct = parameters.get("fused_clip_pct", "")
kind = "fused" if frame.capture_mode == "hdr_fused" else "single"
```

- [x] **Step 5: Implement complete atomic publication**

Define:

```python
class LegacyDatasetCaptureStore:
    def __init__(self, root: Path, context: LegacyCaptureContext) -> None: ...

    def preflight(
        self,
        requests: Sequence[CaptureRequest],
        plan: CapturePlan,
    ) -> None: ...

    def publish_complete(
        self,
        request: CaptureRequest,
        plan: CapturePlan,
        frames: Sequence[CaptureFrame],
        confirmations: Sequence[RoundConfirmation],
        metadata: BootstrapCaptureMetadata,
        *,
        started_at: str,
    ) -> Path: ...
```

`preflight()` computes all eight destinations for every request and rejects any existing final path, temporary path, manifest, symlinked root, or duplicate destination before cameras open. `publish_complete()` writes exclusive temporary PNGs, fsyncs, renames all eight, writes the full CSV to a sibling temporary file, fsyncs, and replaces the manifest. On any publication error, remove only files created for the current request.

- [x] **Step 6: Write failing defect and incomplete tests, then implement them**

Assert defect images route through:

```text
left/front_secondary/defect/less/<session>/images/
```

Implement:

```python
def publish_incomplete(
    self,
    request: CaptureRequest,
    plan: CapturePlan,
    frames: Sequence[CaptureFrame],
    confirmations: Sequence[RoundConfirmation],
    metadata: BootstrapCaptureMetadata,
    *,
    started_at: str,
    failed_round: str,
    failure_kind: str,
    failure_reason: str,
) -> Path: ...
```

It writes rows only for complete `CaptureFrame` values plus one `sample_status=incomplete` row. The summary row carries `failed_round` and `error`; `failed_view`/`failed_device_index` are filled when the failure text or partial frame identity provides them. It must never write a complete sample row.

- [x] **Step 7: Run Task 1 tests and verify GREEN**

Run the Step 2 command. Expected: all new store tests pass.

---

### Task 2: CLI Legacy Mode and Pre-Hardware Conflict Gate

**Files:**
- Modify: `src/zs32_inspection/cli/bootstrap_capture.py`
- Modify: `src/zs32_inspection/capture/bootstrap.py`
- Modify: `tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py`

**Interfaces:**
- Consumes: `LegacyCaptureContext` and `LegacyDatasetCaptureStore` from Task 1.
- Produces: `--legacy-layout` CLI selection and historical session IDs.

- [x] **Step 1: Write failing CLI integration tests**

Add tests asserting:

```python
argv.append("--legacy-layout")
assert bootstrap_capture._run(argv) == 0
assert not (tmp_path / "_bootstrap").exists()
assert (tmp_path / "manifests" / "20260714_120000_000001.csv").is_file()
assert len(list(tmp_path.glob("left/*/normal/*/images/*.png"))) == 8
```

Patch the adapter with a forbidden constructor and assert an existing target causes `_run()` to fail before hardware initialization.

- [x] **Step 2: Run the CLI tests and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m pytest \
  --confcutdir=tests/unit/zs32_refactor/capture_data -q \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py
```

Expected: failures because `--legacy-layout` is unknown and no legacy store is selected.

- [x] **Step 3: Add parser and session behavior**

Add:

```python
parser.add_argument(
    "--legacy-layout",
    action="store_true",
    help="Write the historical per-view dataset tree and CSV manifest.",
)
```

Change session generation to:

```python
def _default_session(*, legacy_layout: bool) -> str:
    now = datetime.now()
    if legacy_layout:
        return now.strftime("%Y%m%d_%H%M%S_%f")
    return now.astimezone(timezone.utc).strftime("bootstrap-%Y%m%d-%H%M%S-%f")
```

- [x] **Step 4: Select the store and run preflight before adapter construction**

Use:

```python
if args.legacy_layout:
    store = LegacyDatasetCaptureStore(
        args.output_root,
        LegacyCaptureContext(args.label, defect_type, args.part_id),
    )
    store.preflight(requests, plan)
else:
    store = AtomicBootstrapCaptureStore(args.output_root)
```

Keep the single outer `with HikvisionCameraAdapter(acquisition)` unchanged.

- [x] **Step 5: Generalize BootstrapCaptureService store typing without weakening behavior**

Introduce a `BootstrapCaptureStore` protocol in `capture/bootstrap.py` containing the complete and incomplete signatures already used by the service. Type `BootstrapCaptureService.__init__()` against that protocol so both stores share identical orchestration. Do not add skip-gate behavior to formal `CaptureService`.

- [x] **Step 6: Update command result for legacy mode**

Return these additional fields:

```python
{
    "output_layout": "legacy" if args.legacy_layout else "bootstrap",
    "manifest_path": str(store.manifest_path) if args.legacy_layout else None,
}
```

`published_paths` in legacy mode contains the eight image paths grouped by capture set or the session manifest path, with a stable documented meaning tested exactly.

- [x] **Step 7: Run CLI and store tests together**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m pytest \
  --confcutdir=tests/unit/zs32_refactor/capture_data -q \
  tests/unit/zs32_refactor/capture_data/test_legacy_dataset_store.py \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py \
  tests/unit/zs32_refactor/capture_data/test_capture_service.py \
  tests/unit/zs32_refactor/capture_data/test_hikvision_adapter.py
```

Expected: all selected tests pass and no test touches real hardware.

---

### Task 3: Operator Documentation, Memory, and Full Verification

**Files:**
- Modify: `configs/zs32/topology/README.md`
- Modify: `README.md`
- Modify: `AGENTS_MEMORY.md`
- Modify: `pipeline/AGENTS_MEMORY.md`
- Modify: `docs/designs/2026-07-14-zs32-four-camera-legacy-layout-capture-design.md`

**Interfaces:**
- Consumes: final CLI from Task 2.
- Produces: copy-paste one-group and 120-group commands plus durable project routing notes.

- [x] **Step 1: Add the one-group hardware acceptance command**

Document the exact command with:

```text
--legacy-layout
--group-count 1
--images-per-group 1
--hdr
--short-exposure 1500
--long-exposure 5500
--gain 0
--capture-interval 0.2
--hdr-settle-frames 1
--timeout-ms 2000
```

The expected result is eight PNGs in eight view directories and nine CSV rows for the sample.

- [x] **Step 2: Add the 120-group production command and verification**

Document the same command with `--group-count 120`, then verify with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python - <<'PY'
import csv
import os
from pathlib import Path

manifest_dir = Path("/home/yunjing/anomalib/dataset/test/manifests")
manifest = manifest_dir / f"{os.environ['CAPTURE_SESSION']}.csv"
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
assert sum(row["record_type"] == "image" for row in rows) == 960
assert sum(
    row["record_type"] == "sample" and row["sample_status"] == "complete"
    for row in rows
) == 120
assert len(rows) == 1080
print("legacy four-camera batch verified")
PY
```

- [x] **Step 3: Update design status and memories**

Change the design status from `待用户确认` to `已实现`, record exact tests and limitations, and update both memory files with the final CLI/output contract. Preserve unrelated existing memory edits.

- [x] **Step 4: Run fresh final verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m pytest \
  --confcutdir=tests/unit/zs32_refactor/capture_data -q \
  tests/unit/zs32_refactor/capture_data/test_legacy_dataset_store.py \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py \
  tests/unit/zs32_refactor/capture_data/test_capture_service.py \
  tests/unit/zs32_refactor/capture_data/test_hikvision_adapter.py

UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m pytest \
  --confcutdir=tests/unit/zs32_refactor/domain -q \
  tests/unit/zs32_refactor/domain/test_topology.py

UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m py_compile \
  src/zs32_inspection/capture/legacy_dataset.py \
  src/zs32_inspection/capture/bootstrap.py \
  src/zs32_inspection/cli/bootstrap_capture.py \
  pipeline/zs32_bootstrap_capture.py

UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  pipeline/zs32_bootstrap_capture.py --help

git diff --check -- \
  src/zs32_inspection/capture/legacy_dataset.py \
  src/zs32_inspection/capture/bootstrap.py \
  src/zs32_inspection/cli/bootstrap_capture.py \
  tests/unit/zs32_refactor/capture_data/test_legacy_dataset_store.py \
  tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py \
  configs/zs32/topology/README.md \
  docs/designs/2026-07-14-zs32-four-camera-legacy-layout-capture-design.md \
  docs/plans/2026-07-14-zs32-four-camera-legacy-layout-capture-plan.md \
  AGENTS_MEMORY.md pipeline/AGENTS_MEMORY.md
```

Expected: all tests pass, compilation/help exit 0, and diff check is empty.

- [x] **Step 5: Hardware handoff boundary**

Do not claim a real 120-group capture from offline tests. Hand over the one-group command first; after the user confirms eight images and nine manifest rows, hand over the 120-group command.
