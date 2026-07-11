# ZS32 Six-View Multicamera HDR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify one terminal command that captures a static left-hand ZS32 part from six paired views using three Hikvision cameras and adjustable HDR exposures.

**Architecture:** Add a testable multicamera orchestration module with a narrow Hikvision SDK adapter. Each exposure pass sends software triggers to all three cameras before reading any camera, while the six-view session controller pairs a front round and a back round under one sample ID and writes an explicit completeness manifest.

**Tech Stack:** Python 3.10+, Hikvision MVS Python SDK, OpenCV, NumPy, existing `capture_data.exposure_fusion`, argparse, CSV, pytest-compatible offline fakes.

## Global Constraints

- The part is static during each capture round; software trigger is sufficient and must not be described as hardware synchronization.
- `device 0` maps to `front/back`, `device 1` maps to `front_left/back_left`, and `device 2` maps to `front_right/back_right`.
- The only accepted hand is `left`.
- HDR parameters remain CLI-adjustable, and `--save-hdr-sources` preserves short and long source frames.
- Each exposure pass must issue `trigger x3` before `read x3`.
- A sample is complete only after all six fused views are readable and saved.
- Use `.venv/bin/python` for repository commands; if pytest is unavailable, run deterministic direct-Python harnesses plus compile checks.
- Hardware smoke output must stay under `/tmp`, not under the formal dataset.
- Preserve the existing single-camera stage and all unrelated user changes.

---

## File Structure

- Create `capture_data/collect_multicamera_dataset.py`: pure view mapping and validation, SDK adapter, grouped HDR capture, session storage, manifest, parser, and CLI main.
- Create `pipeline/1_collect_multicamera_data.py`: thin numbered pipeline wrapper.
- Create `tests/unit/capture_data/test_collect_multicamera_dataset.py`: deterministic unit tests with fake cameras and fake image storage.
- Modify `tests/unit/pipeline/test_pipeline_wrappers.py`: wrapper forwarding regression test.
- Modify `pipeline/README.md`: operational command, mapping, flip flow, output paths, and sync boundary.
- Modify `AGENTS_MEMORY.md`: durable repository-local entrypoint and verified hardware results.

### Task 1: Pure Mapping, Validation, and SDK-Free Module Import

**Files:**
- Create: `tests/unit/capture_data/test_collect_multicamera_dataset.py`
- Create: `capture_data/collect_multicamera_dataset.py`

**Interfaces:**
- Produces: `ROUND_VIEWS: dict[str, tuple[str, str, str]]`
- Produces: `validate_devices(devices: Sequence[int]) -> tuple[int, int, int]`
- Produces: `view_for(round_name: str, camera_slot: int) -> str`
- Produces: `build_parser() -> argparse.ArgumentParser`
- Constraint: importing the module must not require loading the Hikvision SDK.

- [ ] **Step 1: Write failing mapping and parser tests**

```python
def test_default_device_order_maps_to_six_views() -> None:
    assert multicam.validate_devices([0, 1, 2]) == (0, 1, 2)
    assert [multicam.view_for("front", slot) for slot in range(3)] == [
        "front", "front_left", "front_right"
    ]
    assert [multicam.view_for("back", slot) for slot in range(3)] == [
        "back", "back_left", "back_right"
    ]


@pytest.mark.parametrize("devices", [[], [0, 1], [0, 1, 2, 3], [0, 0, 2]])
def test_validate_devices_rejects_non_unique_triples(devices: list[int]) -> None:
    with pytest.raises(ValueError, match="exactly three unique"):
        multicam.validate_devices(devices)


def test_parser_rejects_non_left_hand() -> None:
    with pytest.raises(SystemExit):
        multicam.build_parser().parse_args(
            ["--devices", "0", "1", "2", "--hand", "right", "--label", "normal"]
        )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/unit/capture_data/test_collect_multicamera_dataset.py -v
```

Expected: collection/import failure because `capture_data.collect_multicamera_dataset` does not exist. If pytest is unavailable, run `.venv/bin/python -c 'import capture_data.collect_multicamera_dataset'` and expect `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimal pure API and CLI schema**

Add module constants and functions with these exact semantics:

```python
ROUND_VIEWS = {
    "front": ("front", "front_left", "front_right"),
    "back": ("back", "back_left", "back_right"),
}


def validate_devices(devices: Sequence[int]) -> tuple[int, int, int]:
    if len(devices) != 3 or len(set(devices)) != 3:
        msg = "--devices requires exactly three unique device indices"
        raise ValueError(msg)
    return devices[0], devices[1], devices[2]


def view_for(round_name: str, camera_slot: int) -> str:
    return ROUND_VIEWS[round_name][camera_slot]
```

Build an argparse parser with `--devices` (`nargs=3`, integer, default `0 1 2`), `--hand` (`choices=("left",)`, default `left`), required `--label` (`normal/defect`), `--defect-type`, `--part-id`, counts, `--manual-load`, required-on capture `--hdr`, all HDR tuning flags from the design, `--root`, and `--list-devices`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same pytest command. Expected: mapping, validation, parser, and import tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add capture_data/collect_multicamera_dataset.py tests/unit/capture_data/test_collect_multicamera_dataset.py
git commit -m "feat: define ZS32 multicamera capture mapping"
```

### Task 2: Hikvision Adapter and Grouped Software Trigger

**Files:**
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`
- Modify: `capture_data/collect_multicamera_dataset.py`

**Interfaces:**
- Produces: `DeviceDescription(index: int, model: str, serial: str)`
- Produces: `CameraHandle(device: DeviceDescription, cam: object, frame_info: object, data_buf: object, started: bool)`
- Produces: `HikvisionAdapter.load() -> HikvisionAdapter`
- Produces: `HikvisionAdapter.list_devices() -> list[DeviceDescription]`
- Produces: `HikvisionAdapter.open(device: DeviceDescription, gain: float, fps: float) -> CameraHandle`
- Produces: `trigger_and_read(handles: Sequence[CameraHandle], adapter: CameraAdapter, timeout_ms: int) -> list[np.ndarray]`
- Consumes: existing `capture_data.collect_dataset.convert_frame_to_bgr` behavior, loaded lazily after the SDK path is available.

- [ ] **Step 1: Write failing trigger-order and cleanup tests**

Use a `FakeAdapter` that appends events. Assert:

```python
frames = multicam.trigger_and_read(handles, adapter, timeout_ms=3000)

assert adapter.events == [
    "trigger:0", "trigger:1", "trigger:2",
    "read:0", "read:1", "read:2",
]
assert len({id(handle.data_buf) for handle in handles}) == 3
assert len({id(handle.frame_info) for handle in handles}) == 3
assert len(frames) == 3
```

Also test a context-managed `open_cameras()` path where opening or starting camera 2 fails and cameras 0 and 1 still receive stop/close/destroy exactly once in reverse order.

- [ ] **Step 2: Run focused tests and verify RED**

Expected failure: `trigger_and_read`, adapter data classes, or lifecycle helper does not exist.

- [ ] **Step 3: Implement lazy SDK loading and grouped trigger/read**

Implement the adapter so `HikvisionAdapter.load()` appends `/opt/MVS/Samples/64/Python/MvImport` only when invoked and imports SDK symbols inside the method. Configure each camera with:

```python
ExposureAuto = 0
GainAuto = 0
TriggerMode = 1
TriggerSource = 7
```

Allocate one `MV_FRAME_OUT_INFO_EX` and one `ctypes.c_ubyte * (50 * 1024 * 1024)` per handle. `trigger_and_read()` must first call `TriggerSoftware` for every handle and only then call `GetOneFrameTimeout` for every handle. A nonzero return raises an error containing device index and serial.

Implement `close_cameras(handles, adapter)` with independent best-effort stop, close, and destroy calls, and call it from `finally` even when setup partially fails.

- [ ] **Step 4: Verify focused and existing capture-related tests GREEN**

```bash
.venv/bin/python -m pytest tests/unit/capture_data/test_collect_multicamera_dataset.py -v
.venv/bin/python -m compileall capture_data/collect_multicamera_dataset.py
```

Expected: tests pass and compileall exits 0.

- [ ] **Step 5: Commit Task 2**

```bash
git add capture_data/collect_multicamera_dataset.py tests/unit/capture_data/test_collect_multicamera_dataset.py
git commit -m "feat: add grouped Hikvision software triggering"
```

### Task 3: Three-Camera HDR Round Capture

**Files:**
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`
- Modify: `capture_data/collect_multicamera_dataset.py`

**Interfaces:**
- Produces: `HdrViewResult(camera_slot: int, short_image: np.ndarray, long_image: np.ndarray, fused_image: np.ndarray, fused_clip_pct: float, attempt: int)`
- Produces: `capture_hdr_round(handles, adapter, config) -> list[HdrViewResult]`
- Consumes: `capture_data.exposure_fusion.fuse_exposures`
- Consumes: `trigger_and_read()` from Task 2.

- [ ] **Step 1: Write a failing HDR event-order test**

Use three fake cameras, `short_exposure=4000`, `long_exposure=35000`, and `hdr_settle_frames=0`. Assert the event subsequence is:

```python
[
    "exposure:0:4000", "exposure:1:4000", "exposure:2:4000",
    "trigger:0", "trigger:1", "trigger:2", "read:0", "read:1", "read:2",
    "exposure:0:35000", "exposure:1:35000", "exposure:2:35000",
    "trigger:0", "trigger:1", "trigger:2", "read:0", "read:1", "read:2",
]
```

Assert the fusion function receives camera 0 short/long together, then camera 1, then camera 2. Add a timeout case asserting no successful round is returned when any camera read fails.

- [ ] **Step 2: Run the HDR tests and verify RED**

Expected failure: `capture_hdr_round` or `HdrViewResult` is missing.

- [ ] **Step 3: Implement grouped HDR capture**

Implement `capture_exposure_pass()` that sets the same exposure on all handles, performs configured settle passes using grouped triggers, and returns one grouped final pass. Implement retry around the complete short/long pair when any fused image exceeds `hdr_max_clip_pct`, matching the current single-camera meaning. Fuse per camera with:

```python
fuse_exposures(
    [short_image, long_image],
    method="selective",
    align=config.align_hdr,
    short_dark_threshold=config.short_dark_threshold,
    long_clip_threshold=config.long_clip_threshold,
    blend_width=config.blend_width,
    blur_size=config.blur_size,
)
```

- [ ] **Step 4: Run focused tests and compile check GREEN**

Expected: exact event ordering, pairing, retry, timeout, and fusion tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add capture_data/collect_multicamera_dataset.py tests/unit/capture_data/test_collect_multicamera_dataset.py
git commit -m "feat: capture grouped multicamera HDR frames"
```

### Task 4: Six-View Session Storage and Completeness Manifest

**Files:**
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`
- Modify: `capture_data/collect_multicamera_dataset.py`

**Interfaces:**
- Produces: `SessionPaths(session_id: str, root: Path, view_dirs: dict[str, Path], manifest_path: Path)`
- Produces: `create_session(args, now: datetime) -> SessionPaths`
- Produces: `save_round(results, round_name, sample_id, group_id, image_index, handles, args, paths) -> list[dict[str, str]]`
- Produces: `capture_sample(...) -> bool`
- Produces: manifest columns specified in the approved design.

- [ ] **Step 1: Write failing storage and completeness tests**

Test a normal sample with two fake capture rounds and assert exactly these fused view directories receive one image:

```python
{"front", "front_left", "front_right", "back", "back_left", "back_right"}
```

Assert all six manifest rows share one `session_id`, `sample_id`, and `group_id`, and the final manifest state is `complete`. With `--save-hdr-sources`, assert each view has a short and a long source path. Add tests where `cv2.imwrite` returns false and where the back round raises: both must produce an incomplete sample record with the exact failed round/view/device context.

- [ ] **Step 2: Run storage tests and verify RED**

Expected failure: session/storage/manifest functions are missing.

- [ ] **Step 3: Implement atomic round storage and manifest records**

Create one microsecond-resolution `session_id` per program run. Build all view directories from that same ID. Capture all three fused results in memory before writing a round. Check every `imwrite` result. Write per-image rows plus a sample summary row or equivalent explicit `sample_status` field so consumers can reject an incomplete six-view sample without inferring from row count.

For defect data, preserve the existing layout:

```text
<root>/left/<view>/defect/<defect_type>/<session>/images
```

For normal data, use:

```text
<root>/left/<view>/normal/<session>/images
```

- [ ] **Step 4: Run storage tests and verify GREEN**

Run the focused test file and a direct CSV readback assertion. Expected: all session, naming, source, and failure semantics pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add capture_data/collect_multicamera_dataset.py tests/unit/capture_data/test_collect_multicamera_dataset.py
git commit -m "feat: store paired ZS32 six-view sessions"
```

### Task 5: CLI Main and Pipeline Wrapper

**Files:**
- Modify: `capture_data/collect_multicamera_dataset.py`
- Create: `pipeline/1_collect_multicamera_data.py`
- Modify: `tests/unit/pipeline/test_pipeline_wrappers.py`
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`

**Interfaces:**
- Produces: `main(argv: Sequence[str] | None = None) -> int`
- Produces: wrapper `main() -> None` forwarding to `capture_data/collect_multicamera_dataset.py`.

- [ ] **Step 1: Write failing CLI and wrapper tests**

Test that `--list-devices` prints index/model/serial and exits without opening cameras. Test that capture mode requires `--hdr`, validates `defect_type` when label is defect, opens all devices once, prompts once for front and once for back per group, and calls cleanup on `KeyboardInterrupt`.

Load the numbered wrapper with the existing `_load_module()` helper, patch `run_repo_script`, set `sys.argv`, call `main()`, and assert:

```python
run_repo_script.assert_called_once_with(
    "capture_data/collect_multicamera_dataset.py",
    ["--devices", "0", "1", "2", "--hand", "left", "--label", "normal", "--hdr"],
)
```

- [ ] **Step 2: Run CLI/wrapper tests and verify RED**

Expected failure: `main` orchestration or wrapper file is absent.

- [ ] **Step 3: Implement CLI orchestration and thin wrapper**

The wrapper must mirror `pipeline/1_collect_data.py`: print focused help on no arguments/help, otherwise forward arguments unchanged via `run_repo_script`. The core `main()` must return an integer and `raise SystemExit(main())` only under `if __name__ == "__main__"`.

For each group and image index, use one sample ID. Prompt text must distinguish:

```text
放好 ZS32 左手件正面后按 Enter 或 s...
将同一个 ZS32 左手件翻到背面后按 Enter 或 s...
```

- [ ] **Step 4: Verify CLI help, wrapper, and offline tests GREEN**

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py --help
.venv/bin/python -m pytest tests/unit/capture_data/test_collect_multicamera_dataset.py tests/unit/pipeline/test_pipeline_wrappers.py -v
.venv/bin/python -m compileall capture_data/collect_multicamera_dataset.py pipeline/1_collect_multicamera_data.py
```

Expected: help exits 0, tests pass, compileall exits 0. If pytest is absent, run equivalent direct imports and function assertions in `.venv/bin/python` and record that limitation.

- [ ] **Step 5: Commit Task 5**

```bash
git add capture_data/collect_multicamera_dataset.py pipeline/1_collect_multicamera_data.py tests/unit/capture_data/test_collect_multicamera_dataset.py tests/unit/pipeline/test_pipeline_wrappers.py
git commit -m "feat: add single-terminal ZS32 capture entrypoint"
```

### Task 6: README and Repository Memory

**Files:**
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Documents the exact command, view mapping, two-prompt flip flow, output layout, HDR controls, manifest completeness, and software-trigger boundary.

- [ ] **Step 1: Add a documentation expectation check**

Use `rg` to demonstrate the new entrypoint is not yet documented:

```bash
rg -n "1_collect_multicamera_data|front_left|back_right" pipeline/README.md AGENTS_MEMORY.md
```

Expected before edits: no complete operational section exists.

- [ ] **Step 2: Update `pipeline/README.md`**

Add the new stage to the stage table and a section immediately after single-camera collection. Include the approved command, `device 0/1/2` mapping table, two prompts per group, all six output paths, `--list-devices`, HDR adjustment note, incomplete manifest semantics, and the statement that software trigger is appropriate for static parts but is not hardware synchronization.

- [ ] **Step 3: Update `AGENTS_MEMORY.md`**

Add a dated section containing the core module and wrapper paths, six canonical view names, default device mapping, test commands, smoke-test output path, connected camera serials, and measured results. Do not write success values until the real smoke test has run; first record only implementation facts, then fill the verification subsection in Task 7.

- [ ] **Step 4: Verify docs and formatting**

```bash
rg -n "1_collect_multicamera_data|front_left|back_right|软件触发" pipeline/README.md AGENTS_MEMORY.md
git diff --check
```

Expected: exact entrypoint and all boundaries are discoverable; diff check exits 0.

- [ ] **Step 5: Commit Task 6**

```bash
git add pipeline/README.md AGENTS_MEMORY.md
git commit -m "docs: document ZS32 six-view HDR capture"
```

### Task 7: Real Three-Camera HDR Smoke Test and Final Verification

**Files:**
- Modify after measured results: `AGENTS_MEMORY.md`
- No formal dataset writes.

**Interfaces:**
- Consumes: final CLI from Task 5.
- Produces: `/tmp/anomalib_zs32_multicamera_hdr_smoke/<session>/...` and an evidence summary in the final response.

- [ ] **Step 1: List and verify connected devices**

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py --list-devices
```

Expected: three Hikvision devices are listed. Confirm index 0 is physically front-facing, index 1 left-facing, and index 2 right-facing with the user-provided mapping. If device count or mapping is different, stop before capture and report it.

- [ ] **Step 2: Run one complete interactive HDR sample**

```bash
.venv/bin/python pipeline/1_collect_multicamera_data.py \
  --devices 0 1 2 \
  --hand left \
  --label normal \
  --part-id zs32_smoke \
  --group-count 1 \
  --images-per-group 1 \
  --manual-load \
  --hdr \
  --save-hdr-sources \
  --short-exposure 4000 \
  --long-exposure 35000 \
  --gain 0 \
  --root /tmp/anomalib_zs32_multicamera_hdr_smoke
```

Expected: one front prompt and one back prompt; after the physical flip, six fused images and twelve source images are saved under one session.

- [ ] **Step 3: Validate artifacts with OpenCV and CSV parsing**

Run a read-only validation harness that finds the newest session, loads every fused and source path with `cv2.imread`, checks `image is not None`, `height > 0`, `width > 0`, and `channels == 3`, then confirms the manifest contains all six canonical views under one sample ID with `sample_status=complete`.

Expected: 6/6 fused readable, 12/12 HDR source images readable, one complete six-view sample.

- [ ] **Step 4: Record timing and update repository memory**

Record short/long exposure values, trigger offsets for each three-camera exposure pass, total front round duration, total back round duration, and total sample duration. Explicitly state that exposure microseconds and program-side milliseconds are different measurements. Add these measured values and the exact `/tmp` session path to `AGENTS_MEMORY.md`.

- [ ] **Step 5: Run final verification suite**

```bash
.venv/bin/python -m pytest tests/unit/capture_data/test_collect_multicamera_dataset.py tests/unit/pipeline/test_pipeline_wrappers.py -v
.venv/bin/python -m compileall capture_data/collect_multicamera_dataset.py pipeline/1_collect_multicamera_data.py
.venv/bin/python pipeline/1_collect_multicamera_data.py --help
git diff --check
git status --short
```

Expected: all available tests pass, compile/help/diff checks exit 0, and status contains only intended files. If pytest is unavailable, execute and report the direct deterministic harness instead of claiming pytest passed.

- [ ] **Step 6: Commit measured verification memory**

```bash
git add AGENTS_MEMORY.md
git commit -m "docs: record ZS32 multicamera HDR smoke test"
```

## Plan Self-Review

- Spec coverage: all approved mappings, two-round interaction, HDR controls, six-view pairing, error behavior, docs, memory, and real smoke verification have an owning task.
- Placeholder scan: implementation steps name exact functions, files, commands, and expected results; no deferred implementation placeholders remain.
- Type consistency: Tasks 2-5 consume the exact adapter, handle, HDR result, session, and CLI interfaces produced by earlier tasks.
- Scope: the plan leaves the existing single-camera path untouched and does not add hardware-trigger or model-training behavior.
