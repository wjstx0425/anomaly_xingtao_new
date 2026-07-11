# ZS32 Serial-Bound Single-Exposure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make single exposure the safe default for the existing three-camera ZS32 six-view collector, bind views by stable USB serial numbers, restore cameras after every exit path, and preserve explicit HDR mode.

**Architecture:** Keep `pipeline/1_collect_multicamera_data.py` and `capture_data/collect_multicamera_dataset.py`. Resolve the three physical camera roles from one enumeration snapshot by serial number, use one grouped software trigger/read pass for single exposure, leave HDR behind `--hdr`, and separate application pacing from camera `AcquisitionFrameRate` configuration.

**Tech Stack:** Python 3.10+, Hikvision MVS Python SDK, OpenCV, NumPy, argparse, CSV, existing exposure-fusion helpers, pytest-compatible offline fakes.

## Global Constraints

- Default mode is single exposure; HDR remains available only when `--hdr` is explicitly passed.
- Front serial is `DA9805574`, left serial is `DA9625347`, and right serial is `DB0998274`.
- Enumeration indices are internal SDK addresses, never view identities.
- Do not write `AcquisitionFrameRate` or `AcquisitionFrameRateEnable` in the new collector.
- Every single-exposure round must preserve `trigger x3 -> read x3`.
- Normal exit, setup failure, capture failure, and `Ctrl+C` must independently attempt `TriggerMode=Off`, close, and destroy for every configured camera; restoration failure must be reported and must not skip remaining cleanup.
- Existing HDR fusion, source-image saving, six-view completeness, atomic file publication, and single-camera collection remain available.
- Do not connect to real cameras or run a hardware smoke test; hand the command to the user for testing.
- Use `.venv/bin/python`; if pytest is unavailable, run deterministic direct-Python assertions plus compile and CLI checks without claiming pytest passed.
- Preserve unrelated user changes. The only dirty file at plan creation is an interrupted agent's uncommitted pacing-margin test edit in `tests/unit/capture_data/test_collect_multicamera_dataset.py`; remove only that exact abandoned edit before new TDD work.

---

## File Structure

- Modify `capture_data/collect_multicamera_dataset.py`: serial selection, single-exposure capture, safe camera configuration/restore, parser, orchestration, and manifest mode fields.
- Modify `pipeline/1_collect_multicamera_data.py`: default-single help and serial-bound examples.
- Modify `tests/unit/capture_data/test_collect_multicamera_dataset.py`: remove abandoned margin-only edits and add serial/single/restore/HDR regression coverage.
- Modify `pipeline/README.md`: default single-exposure command, serial mapping, explicit HDR command, and recovery semantics.
- Modify `AGENTS_MEMORY.md`: final entrypoint behavior, stable serial mapping, and user-owned hardware test command.

### Task 1: Reconcile Dirty Test State and Define Serial CLI Contract

**Files:**
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`
- Modify: `capture_data/collect_multicamera_dataset.py`

**Interfaces:**
- Produces: `CameraSerials(front: str, left: str, right: str)` and `DEFAULT_SERIALS = CameraSerials("DA9805574", "DA9625347", "DB0998274")`.
- Produces: `select_devices_by_serial(serials: CameraSerials, available_devices: Sequence[DeviceDescription]) -> list[DeviceDescription]` ordered front, left, right.
- Produces parser defaults `--front-serial DA9805574`, `--left-serial DA9625347`, `--right-serial DB0998274`.
- Changes parser behavior so capture no longer requires `--hdr`.

- [ ] **Step 1: Remove only the abandoned pacing-margin test edit**

First compare the dirty diff with the four known abandoned hunks. If and only if it contains the three `0.1 -> 0.12` expectation changes and the new `test_trigger_pass_pacer_adds_device_scheduling_margin_at_10_fps`, use `apply_patch` to restore the three expectations to `0.1` and remove that test. If any other dirty hunk exists, stop and report it. Do not use reset/checkout and do not alter production pacing in this task.

- [ ] **Step 2: Write failing serial-selection and parser tests**

```python
def test_select_devices_by_serial_ignores_enumeration_order() -> None:
    available = [
        DeviceDescription(0, "right-model", "DB0998274"),
        DeviceDescription(1, "front-model", "DA9805574"),
        DeviceDescription(2, "left-model", "DA9625347"),
    ]
    selected = select_devices_by_serial(DEFAULT_SERIALS, available)
    assert [device.serial for device in selected] == ["DA9805574", "DA9625347", "DB0998274"]


def test_single_exposure_is_default_capture_mode() -> None:
    args = build_parser().parse_args(["--label", "normal"])
    assert args.hdr is False
    assert args.exposure == 4000.0
```

Add failures for duplicate requested serials, missing serials, and duplicate serials in enumeration.

- [ ] **Step 3: Run focused RED verification**

Run `.venv/bin/python -m pytest tests/unit/capture_data/test_collect_multicamera_dataset.py -q` if available. Otherwise run a direct import/assertion harness and verify failure because serial APIs/options do not exist and the parser still requires HDR.

- [ ] **Step 4: Implement the minimal serial and parser contract**

Add the three serial options with approved defaults, add `--exposure` default `4000.0`, keep `--hdr` optional, and resolve devices from one enumeration snapshot. Delete the capture option `--devices`; `--list-devices` remains the only enumeration-oriented CLI.

When `--hdr` is absent, reject HDR-only flags that request artifacts or materially change fusion (`--save-hdr-sources`, `--align-hdr`) with a clear parser error instead of silently ignoring them. HDR tuning values may retain defaults but are used only in HDR mode.

The one-snapshot contract is exact: `HikvisionAdapter.list_devices()` caches the SDK `device_list` used to build descriptions; `HikvisionAdapter.open(device, gain)` must use the cached SDK entry and verify its decoded serial still equals `device.serial`. It must not call `MV_CC_EnumDevices` again. Add an SDK-double assertion that capture startup enumerates exactly once.

- [ ] **Step 5: Verify GREEN and commit**

Run focused direct/pytest assertions, `compileall`, and `git diff --check`.

```bash
git add capture_data/collect_multicamera_dataset.py tests/unit/capture_data/test_collect_multicamera_dataset.py
git commit -m "feat: bind ZS32 camera views by serial"
```

### Task 2: Safe Camera Configuration Without Hardware FPS

**Files:**
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`
- Modify: `capture_data/collect_multicamera_dataset.py`

**Interfaces:**
- Changes: `CameraAdapter.open(device, gain) -> CameraHandle` without hardware FPS.
- Produces: `CameraAdapter.restore_continuous(handle: CameraHandle) -> None`, implemented by setting `TriggerMode=0`.
- Changes: `open_cameras(devices, adapter, gain)` and cleanup to restore every configured/partially configured camera.

- [ ] **Step 1: Write failing adapter and lifecycle tests**

Use an SDK double to assert camera open/configuration never calls either `AcquisitionFrameRateEnable` or `AcquisitionFrameRate`. Add exact event assertions that normal exit, `OpenDevice` failure after `CreateHandle`, start failure on camera 2, configuration failure after `TriggerMode=1`, capture exception, and `KeyboardInterrupt` all independently attempt restore/close/destroy for every created handle. Add failure injection where stop fails and restore still runs, and where restore fails and close/destroy still run.

- [ ] **Step 2: Verify RED**

The current implementation must fail because it enables and writes hardware FPS and because a configuration failure inside `open()` can close the raw handle without restoring trigger mode.

- [ ] **Step 3: Implement configuration rollback and restore**

Do not write either frame-rate node. Configure manual exposure/gain and software trigger only after the handle exists. Track the raw handle as soon as `CreateHandle` succeeds. Catch `BaseException` during raw-handle setup so `KeyboardInterrupt` also triggers cleanup, then re-raise it. On any later error, independently attempt `TriggerMode=0`, close, and destroy. During normal `close_cameras`, independently attempt stop (if started), restore, close, and destroy for each handle; one operation or camera failure must not prevent later cleanup.

Add `validate_float_range(cam, name: str, value: float) -> None` using `MV_CC_GetFloatValue` and the SDK float-value structure. Validate gain during open and validate every single/HDR exposure before writing it. Range errors must name the parameter, requested value, minimum, and maximum. Parser-level validation must also reject `group_count <= 0`, `images_per_group <= 0`, `timeout_ms <= 0`, `exposure <= 0`, short/long exposure `<= 0`, and `capture_interval < 0` before opening cameras.

- [ ] **Step 4: Verify GREEN and commit**

Run lifecycle assertions, compile, and diff checks.

```bash
git add capture_data/collect_multicamera_dataset.py tests/unit/capture_data/test_collect_multicamera_dataset.py
git commit -m "fix: restore Hikvision cameras after capture"
```

### Task 3: Default Grouped Single-Exposure Capture

**Files:**
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`
- Modify: `capture_data/collect_multicamera_dataset.py`

**Interfaces:**
- Produces: `CapturedViewResult(camera_slot: int, final_image: np.ndarray, source_short: np.ndarray | None = None, source_long: np.ndarray | None = None, hdr_attempt: int | None = None, fused_clip_pct: float | None = None)`.
- Produces: `capture_single_round(handles: Sequence[CameraHandle], adapter: CameraAdapter, exposure: float, timeout_ms: int, pacer: GroupedTriggerPacer) -> list[CapturedViewResult]`.
- Changes: group orchestration selects `capture_single_round` by default and `capture_hdr_round` only for `args.hdr`.
- Produces: `GroupedTriggerPacer(interval_seconds: float)` and `--capture-interval` in seconds, default `0.2`, validated `>= 0`.
- Changes: `save_round(results: Sequence[CapturedViewResult], ...) -> list[dict[str, str]]` for both modes.

- [ ] **Step 1: Write failing single-exposure behavior tests**

Assert one round emits:

```text
exposure:0, exposure:1, exposure:2,
trigger:0, trigger:1, trigger:2,
read:0, read:1, read:2
```

Assert no fusion function is called, no short/long source paths are created, and the three final images retain camera slots 0/1/2. Add group coverage proving front and back prompts remain once each and six views share the same sample ID. Add a separate regression proving `--hdr` still selects the existing HDR code path and preserves source saving.

- [ ] **Step 2: Verify RED**

The current parser/orchestrator must fail because it always calls `capture_hdr_round`.

- [ ] **Step 3: Implement minimal mode dispatch and shared storage**

Add single capture without changing `trigger_and_read`. Generalize HDR output into `CapturedViewResult` and use the same storage path. Add exact manifest columns `capture_mode`, `exposure`, and `gain` while preserving existing `short_exposure`, `long_exposure`, `hdr_attempt`, and `fused_clip_pct`. Single mode writes `capture_mode=single`, its exposure/gain, empty source paths, and a `single`-kind `.png` name. HDR writes `capture_mode=hdr_fused`, preserves the current `fused`-kind final filename, its optional source filenames, and all existing HDR fields. Sample summary rows also record `capture_mode`. Failure rows retain group/round/device/view context.

Replace FPS-based pacing with one `GroupedTriggerPacer(args.capture_interval)` shared across the complete open-camera session. It applies before every grouped trigger pass after the first: repeated single samples, HDR settle passes, short/long passes, retries, front/back, and groups. Remove `--fps`, `HdrCaptureConfig.fps`, and `TriggerPassPacer`. Add exact tests that `--capture-interval 0.2` yields one immediate first pass and 0.2-second minimum spacing for all later grouped passes in both single and HDR modes.

- [ ] **Step 4: Verify GREEN and HDR regression, then commit**

Run single/HDR/storage tests, compile, CLI help, and diff checks.

```bash
git add capture_data/collect_multicamera_dataset.py tests/unit/capture_data/test_collect_multicamera_dataset.py
git commit -m "feat: add default three-camera single exposure"
```

### Task 4: CLI, Documentation, and Offline Handoff Verification

**Files:**
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`
- Modify: `tests/unit/pipeline/test_pipeline_wrappers.py` only if wrapper behavior changes.
- Modify: `pipeline/1_collect_multicamera_data.py`
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Documents exact single-exposure and explicit HDR commands.
- Leaves hardware execution to the user.

- [ ] **Step 1: Add CLI integration expectations**

Test that `--list-devices` is non-mutating, capture startup prints role/index/model/serial mapping, missing serial exits before opening any camera, default mode is single, and `--hdr` remains accepted. Verify the wrapper forwards arguments unchanged and its own help text describes single exposure first, serial options, and HDR as optional.

- [ ] **Step 2: Update operational documentation**

Document this exact default mapping:

```text
front=DA9805574
left=DA9625347
right=DB0998274
```

Provide a default single-exposure command without `--hdr` or `--fps`, followed by a separate HDR command. Explain exclusive access, that every exit path attempts trigger restoration and reports restoration failure, manifest success criteria, and how the user can reopen MVS after the command exits.

- [ ] **Step 3: Update repository memory**

Record the previous `--fps 1` failure (`0x80000102`), the reason hardware FPS is no longer written, stable serial mapping, safe restore behavior, and that no post-fix hardware smoke test was run by Codex because the user requested self-testing.

README must include an executable user-owned checklist: run `--list-devices`; run one-group single exposure under `/tmp`; validate six readable three-channel images and one `sample_status=complete` row; reopen MVS continuous mode after exit; only then try HDR and verify six final plus twelve source images. State that Codex did not run this post-fix hardware acceptance. Record exposure microseconds separately from total program milliseconds.

Manual prompts must continue accepting Enter or `s`. Invalid input continues waiting. For `images_per_group > 1`, each image index is one sample; a group is operationally successful only when every sample row in that group is complete. Do not add a new group-summary manifest row in this scope.

- [ ] **Step 4: Run final offline verification**

```bash
.venv/bin/python -m pytest tests/unit/capture_data/test_collect_multicamera_dataset.py tests/unit/pipeline/test_pipeline_wrappers.py -q
.venv/bin/python -m compileall capture_data/collect_multicamera_dataset.py pipeline/1_collect_multicamera_data.py
.venv/bin/python pipeline/1_collect_multicamera_data.py --help
git diff --check
git status --short
```

If pytest is missing, execute deterministic direct assertions for serial resolution, single/HDR dispatch, trigger restoration, and wrapper forwarding. Report the missing dependency and do not claim pytest success.

- [ ] **Step 5: Commit documentation and memory**

```bash
git add pipeline/README.md AGENTS_MEMORY.md tests/unit/capture_data/test_collect_multicamera_dataset.py tests/unit/pipeline/test_pipeline_wrappers.py
git commit -m "docs: document serial-bound ZS32 capture"
```

## Plan Self-Review

- Spec coverage: stable serial binding, default single exposure, retained HDR, no hardware FPS writes, trigger restoration, six-view integrity, docs, and user-owned testing each have an implementation task.
- Dirty-state safety: the exact abandoned 120 ms test-only edit is removed with `apply_patch`; no checkout/reset is used and no unrelated file is touched.
- Type consistency: serial selection feeds the existing ordered slot model; both capture modes produce camera-slot-indexed final images for one shared storage path.
- Scope: no hardware test, model training, inference, crop processing, or unrelated refactor is included.
