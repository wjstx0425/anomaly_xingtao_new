# BMW Eight-View Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a reproducible Chinese handoff for the BMW four-camera/eight-view laboratory inspection workflow.

**Architecture:** Build a clean branch from remote `main`, copy only the BMW implementation and its direct shared capture dependency, and add one standalone handoff README. Validate local paths and focused tests before an explicit-path commit, SSH push, and Draft PR.

**Tech Stack:** Python 3.11+, uv, OpenCV, Hikvision MVS, Anomalib EfficientAD, Ultralytics YOLO26n, Label Studio, Git/GitHub

## Global Constraints

- Do not publish datasets, checkpoints, Label Studio databases, credentials, or unrelated ZS32 working-tree changes.
- Keep the accepted camera mapping and HDR settings exact.
- Document this as a laboratory iteration system, not an industrial release.
- Use explicit file staging; never use `git add .` or `git add -A`.

---

### Task 1: Assemble the reproducible BMW code scope

**Files:**
- Create: `src/bmw_inspection/**`
- Create: `configs/bmw/**`
- Create: `pipeline/bmw_*.py`
- Create: `tests/unit/bmw_inspection/**`
- Modify: `capture_data/collect_multicamera_dataset.py`

**Interfaces:**
- Produces the eight-view collection, preparation, training, and Demo CLIs documented by Task 2.

- [ ] Copy only BMW source, configuration, pipeline scripts, focused tests, and the current shared collector implementation into the clean worktree.
- [ ] Run `python -m compileall` over the copied modules and run the focused BMW unit tests.
- [ ] Inspect `git status --short` and confirm that no `dataset/`, `results/`, `.label-studio/`, checkpoint, or unrelated ZS32 file is present.

### Task 2: Write and validate the handoff README

**Files:**
- Create: `BMW_EIGHT_VIEW_HANDOFF_README.md`

**Interfaces:**
- Consumes the exact CLI names and paths published by Task 1.
- Produces the single entry document used by the successor engineer.

- [ ] Write the complete Chinese workflow with copy-pasteable commands and explicit artifact boundaries.
- [ ] Validate every referenced repository path and run every documented CLI with `--help`.
- [ ] Record current measured Template, bright-streak, YOLO, and EfficientAD results plus their laboratory-only limitations.

### Task 3: Publish the isolated branch

**Files:**
- Stage only the explicit paths from Tasks 1 and 2 plus this design and plan.

**Interfaces:**
- Produces branch `agent/bmw-eight-view-handoff` and a Draft PR targeting `main`.

- [ ] Review staged names and diff statistics; reject any unexpected data, result, credential, or ZS32-only path.
- [ ] Commit with message `docs: add BMW eight-view handoff`.
- [ ] Push with SSH to `git@github.com:wjstx0425/anomaly_xingtao_new.git`.
- [ ] Create a Draft PR through the authenticated GitHub connector and report the branch, commit, PR URL, and validation evidence.
