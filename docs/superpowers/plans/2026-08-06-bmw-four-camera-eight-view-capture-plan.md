# BMW Four-Camera Eight-View HDR Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a BMW-specific four-camera, two-round, eight-view HDR dataset collector without changing the existing ZS32 or BMW detection contracts.

**Architecture:** A versioned JSON owns the exact camera/view/HDR contract. A focused `bmw_inspection.capture` module validates the profile, acquires four serial-bound cameras with the existing Hikvision/HDR primitives, and atomically publishes per-session images plus an auditable raw capture manifest. A thin pipeline CLI supplies normal/defect batch metadata and operator prompts.

**Tech Stack:** Python 3.10+, dataclasses, OpenCV, Hikvision MVS adapter, existing selective HDR fusion, csv/json/hashlib, pytest, uv.

> **2026-08-06 scope reduction:** The user requested the smallest fast implementation. Execute Task 1, a reduced
> Task 4 implemented as a thin wrapper over the existing four-camera bootstrap collector, Task 5 documentation, and
> focused verification only. Tasks 2-3 and HDR source-frame publication are deliberately deferred with the full
> eight-view detection upgrade.

## Global Constraints

- Preserve `pipeline/1_collect_multicamera_data.py` and every existing ZS32 capture behavior.
- Bind center=`DA9805574`, left=`DA9625347`, right=`DB0998274`, secondary=`DB0968108`.
- Produce front/back rounds with exactly eight fused images per complete physical part.
- Use explicit HDR defaults 1500/6000 us, gain 0, interval 0.2 s, settle 1, timeout 3000 ms.
- Do not claim that the existing six-view BMW detection runtime can consume the new data yet.
- Preserve all unrelated dirty-worktree changes.

---

### Task 1: Versioned four-camera capture profile

**Files:**
- Create: `configs/bmw/capture/bmw_4cam_eight_view_hdr_v1.json`
- Create: `src/bmw_inspection/capture/__init__.py`
- Create: `src/bmw_inspection/capture/config.py`
- Test: `tests/unit/bmw_inspection/capture/test_config.py`

**Interfaces:**
- Produces: `load_capture_profile(path: Path) -> BmwCaptureProfile`
- Produces: `BmwCaptureProfile.slots`, `front_views`, `back_views`, `all_views`, and HDR settings.

- [ ] Write failing tests that require the exact four serials, eight unique canonical views, positive HDR values, short exposure below long exposure, and path-safe profile ID.
- [ ] Run `PYTHONPATH=.:src uv run --no-sync pytest -q tests/unit/bmw_inspection/capture/test_config.py` and verify failure because the module does not exist.
- [ ] Add the minimal immutable dataclasses, strict JSON loader, and approved profile JSON.
- [ ] Re-run the focused test and require PASS.

### Task 2: Raw capture publication contract

**Files:**
- Create: `src/bmw_inspection/capture/publisher.py`
- Test: `tests/unit/bmw_inspection/capture/test_publisher.py`

**Interfaces:**
- Consumes: `BmwCaptureProfile` and eight view results.
- Produces: `CapturePublisher.publish_sample(...) -> PublishedSample`.
- Produces: `<root>/sessions/<session_id>/samples/<sample_id>/<view>.png`, optional source frames, `capture_manifest.csv`, and `summary.json`.

- [ ] Write failing tests for eight fused images, optional sixteen HDR sources, SHA256 fields, one sample-completion record, no complete status for partial input, and no overwrite.
- [ ] Run the focused test and verify the expected import/API failure.
- [ ] Implement atomic PNG writes and append-safe manifest/summary publication with explicit columns.
- [ ] Re-run the focused test and require PASS.

### Task 3: Four-camera HDR acquisition session

**Files:**
- Create: `src/bmw_inspection/capture/session.py`
- Test: `tests/unit/bmw_inspection/capture/test_session.py`

**Interfaces:**
- Consumes: `BmwCaptureProfile`, a camera adapter, and existing `open_cameras`, `capture_hdr_round`, `GroupedTriggerPacer` primitives.
- Produces: `BmwFourCameraSession.capture_round("front" | "back") -> Mapping[str, CapturedHdrView]`.

- [ ] Write failing tests proving declared serial order, four images per round, correct front/back view mapping, exact HDR parameter forwarding, persistent handles, and cleanup after errors.
- [ ] Run the focused test and verify failure because the session does not exist.
- [ ] Implement the minimal serial selector and context-managed HDR session without modifying the legacy collector.
- [ ] Re-run the focused test and require PASS.

### Task 4: BMW batch collection CLI

**Files:**
- Create: `pipeline/bmw_lab_collect_data.py`
- Test: `tests/unit/bmw_inspection/capture/test_cli.py`

**Interfaces:**
- Produces CLI options `--config`, `--list-devices`, `--label`, `--defect-type`, `--part-prefix`, `--group-count`, `--root`, and `--save-hdr-sources`.
- Produces one unique `part_id`/`sample_id` per group and exactly two operator confirmations per physical part.

- [ ] Write failing parser and fake-session tests for discovery, normal capture, required defect type, one-part identity per group, interrupted session summary, and eight-view publication.
- [ ] Run the focused test and verify failure because the CLI does not exist.
- [ ] Implement the thin CLI, Chinese prompts, list-devices mode, and nonzero error exits.
- [ ] Re-run the focused test and require PASS.

### Task 5: Operator documentation and project memory

**Files:**
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**
- Documents the one-group smoke command and normal/no-streak batch commands using the exact four serials and HDR profile.

- [ ] Add the BMW four-camera command surface and explicitly mark `pipeline/1_collect_multicamera_data.py` as obsolete for this BMW topology.
- [ ] Record the topology, HDR identity, output layout, and remaining six-view detection boundary in project memory.
- [ ] Run `rg -n "bmw_lab_collect_data|DB0998274|front_secondary" pipeline/README.md AGENTS_MEMORY.md` and verify the new operational surface is present.

### Task 6: Focused and regression verification

**Files:**
- Verify only; no planned production edits.

**Interfaces:**
- Confirms all new tests plus existing BMW tests remain green.

- [ ] Run `PYTHONPATH=.:src UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest -q tests/unit/bmw_inspection/capture tests/unit/bmw_inspection`.
- [ ] Run `PYTHONPATH=.:src UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m compileall -q src/bmw_inspection/capture pipeline/bmw_lab_collect_data.py`.
- [ ] Run `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python pipeline/bmw_lab_collect_data.py --help`.
- [ ] Run `git diff --check` and inspect only the intended files.
- [ ] Report that live four-camera acquisition remains an operator hardware step unless actually executed.
