# ZS32 Right-Hand HDR Batch Implementation Plan

> **For agentic workers:** implement this single task with TDD and offline verification only.

**Goal:** Allow the existing serial-bound six-view collector to write right-hand ZS32 normal HDR data under `dataset/right`.

**Architecture:** Extend only the `--hand` parser choice from `left` to `left/right`. Existing session paths already derive from `args.hand`, so no capture, HDR, serial, storage, or camera lifecycle logic changes are required.

**Files:**
- Modify: `tests/unit/capture_data/test_collect_multicamera_dataset.py`
- Modify: `capture_data/collect_multicamera_dataset.py`
- Modify: `pipeline/README.md`
- Modify: `AGENTS_MEMORY.md`

- [ ] Add a failing parser assertion that `--hand right --label normal --hdr` is accepted.
- [ ] Add a failing session-path assertion that every view directory starts under `<root>/right/`.
- [ ] Run the focused assertions and confirm failure because parser choices only contain `left`.
- [ ] Change the parser to `choices=("left", "right")`; do not touch capture or HDR logic.
- [ ] Add the confirmed 100-group command with short exposure `1500`, long exposure `6000`, `images-per-group=1`, and root `/home/yunjing/anomalib/dataset` to `pipeline/README.md` and `AGENTS_MEMORY.md`.
- [ ] Run direct parser/path assertions, compileall, CLI help, `git diff --check`, and verify no camera command ran.
- [ ] Commit as `feat: support right-hand ZS32 capture`.
