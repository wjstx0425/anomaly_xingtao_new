# BMW Eight-View Laboratory Demo Integration Implementation Plan

> **For agentic workers:** Implement inline in the current workspace. Do not dispatch subagents, modify the legacy
> six-view BMW runtime, or create broad commits from the dirty worktree.

**Goal:** Connect the trained Template, bright-streak, EfficientAD-S, and YOLO26n assets to one Chinese four-camera
eight-view laboratory Demo.

**Architecture:** Add an isolated string-view eight-view contract and runtime, reuse the proven BMW ROI/HDR assets,
and keep all trained models resident. A separate OpenCV/Pillow dashboard consumes one immutable result and provides
live two-round HDR plus offline sample modes.

**Tech Stack:** Python 3.13, uv, OpenCV, Pillow, Anomalib, Ultralytics, Hikvision MVS SDK.

## Global Constraints

- Canonical views are `front`, `front_left`, `front_right`, `front_secondary`, `back`, `back_left`, `back_right`,
  `back_secondary`.
- Preserve the old six-view files and entrypoint.
- Run every branch even after an earlier NG so the Demo always has complete evidence.
- Use the exact trained run `results/bmw_lab_one_click/bmw_lab_eight_view_v1`.
- Keep implementation and verification laboratory-sized; no industrial integration or full repository test run.

---

### Task 1: Eight-view configuration and immutable results

**Files:**
- Create: `src/bmw_inspection/lab/eight_view_demo.py`
- Create: `configs/bmw/experiments/bmw_eight_view_demo_v1.json`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo.py`

**Interfaces:**
- Produces `EightViewDemoConfig`, `DemoBranchResult`, `EightViewInspection`, `load_demo_config`,
  `load_manifest_sample`, and `load_capture_directory`.

- [ ] Write a failing test for exact eight-view validation, current asset paths, offline sample loading, and final
  status precedence.
- [ ] Run the focused test and confirm it fails because the module is absent.
- [ ] Implement strict but small configuration parsing and immutable result contracts.
- [ ] Run the focused test and confirm it passes.

### Task 2: Resident model adapters and complete runtime

**Files:**
- Create: `src/bmw_inspection/lab/eight_view_demo_models.py`
- Modify: `src/bmw_inspection/lab/eight_view_demo.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_models.py`

**Interfaces:**
- Produces `EightViewModelSuite.inspect(images) -> EightViewInspection`.
- Consumes full 4024x3036 source images and crops the configured part ROI once per view.

- [ ] Write failing tests with injected lightweight predictors proving all 25 checks run: 8 Template, 1 light
  streak, 8 YOLO, and 8 EfficientAD.
- [ ] Implement generic eight-view Template loading, resident YOLO, resident EfficientAD, bright-streak mapping,
  evidence overlays, and error-to-result conversion.
- [ ] Verify any NG yields NG, any model error yields ERROR, and all PASS yields OK without short-circuiting.

### Task 3: Live four-camera HDR and offline entrypoints

**Files:**
- Create: `src/bmw_inspection/lab/eight_view_demo_capture.py`
- Create: `pipeline/bmw_lab_eight_view_demo.py`
- Test: `tests/unit/pipeline/test_bmw_lab_eight_view_demo.py`

**Interfaces:**
- Produces `FourCameraHdrSession.capture_round(round_id) -> dict[str, np.ndarray]`.
- CLI accepts mutually exclusive `--sample-id` and `--capture-set`; neither means live HDR.

- [ ] Write failing CLI/default and four-slot mapping tests.
- [ ] Reuse `HikvisionAdapter`, `open_cameras`, and `capture_hdr_round` with the approved BMW HDR profile.
- [ ] Implement live two-round flow, offline manifest/directory flow, `--no-gui`, and `--save-screenshot`.
- [ ] Run the focused tests without opening cameras.

### Task 4: Chinese eight-view dashboard

**Files:**
- Create: `src/bmw_inspection/lab/eight_view_demo_ui.py`
- Modify: `pipeline/bmw_lab_eight_view_demo.py`
- Test: `tests/unit/bmw_inspection/lab/test_eight_view_demo_ui.py`

**Interfaces:**
- Produces `DemoUiState` and `render_eight_view_dashboard(state) -> np.ndarray` with fixed 1600x900 output.

- [ ] Write a failing render test for startup state, 4x2 real-image layout, Chinese copy, and evidence selection.
- [ ] Implement the existing Swiss visual system, `1-8` and `T/L/Y/E` selection, automatic first-NG evidence,
  blank startup, retry, and quit behavior.
- [ ] Save and inspect one real offline screenshot.

### Task 5: Real-model smoke and handoff

**Files:**
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`

- [ ] Run focused unit tests and compile the new modules.
- [ ] Run `--sample-id bmw_normal_group072_000001 --no-gui --save-screenshot ...` with the actual checkpoints on
  GPU 0 and confirm all four branch types appear.
- [ ] Document the live and offline commands, current model-quality caveats, and evidence output path.
