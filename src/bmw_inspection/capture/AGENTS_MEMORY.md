# BMW capture module memory

## Unconfigured camera serials in exposure tests — 2026-09-12

- User attempted `--serial DB1624062 --exposures-us 600 750 --repeat 1`, blocked by configured-four whitelist before SDK load. Standalone exposure CLI now accepts explicitly selected unconfigured serials; SDK enumeration still requires actual devices. Duplicate/blank serials fail before SDK load.
- Known serials keep semantic views; unconfigured devices receive positional `camera_01_front`/`camera_01_back` etc. `metadata.selected_slots` records exact mapping. No production profile or normal collector changes. No --serial still selects configured four. Source records/replay maintain serial/view identity.
-45 capture tests passed, existing NVML warning; CLI help checked. New fake-camera test uses user's DB1624062 and600/750us, plus duplicate rejection. No real DB1624062 capture/hardware acceptance this turn. Guide `docs/bmw/exposure_test.md` updated.

## Yellow front image diagnosis — 2026-09-11

- User session20260911_161054_385521:300/400us, gain12. Front DA9805574 MV-CU120-10UC color; other3 MV-CU120-10UM mono confirmed equal PNG channels. Raw front already yellow-green, not HDR; all8 source hashes verified.
- Previous155941 same400us/gain12 more neutral. Midtone(gray30..180) B/G=.980 vs latest.893; not gray-card calibration or identical pixel sets. Do not blame gain alone. WB/actualPixelType not saved or set; light/camera color-state contribution unresolved.
- Synthetic test confirms conditional Bayer bug: PFNC BayerRG8 needs COLOR_BAYER_RGGB2BGR; current COLOR_BAYER_RG2BGR swapsR/B. Actual session branch unknown; swap alone cannot turn neutral gray yellow. No code fix during diagnosis.
- Report `docs/bmw/exposure_color_review_20260911.md`; contact sheets `artifacts/bmw_exposure_color_review_20260911/`. No camera access/settings changes, original image edits, or live color acceptance.

## Exposure scan and fusion comparison — 2026-09-11

- User requested a test program based on existing acquisition. Independent CLI `bmw_inspection.cli.exposure_test`; future installed command `bmw-exposure-test`. User did not answer optional UI preference during implementation; shipped CLI plus local HTML comparison. Guide `docs/bmw/exposure_test.md`.
- Reuses capture profile/serials, `HikvisionAdapter`, `open_cameras`, grouped trigger pacing, and original selective fusion. Existing collector defaults unchanged. Hardware additions: `get_exposure()` SDK readback and backward-compatible `capture_single_round(..., settle_frames=0)` keyword.
- Set exposure once, discard configured settle frames, then retain next frame. Fixed gain/autos disabled via existing adapter. Device readback is ExposureTime setting, not chunk exposure proof; receive timestamp is host batch completion time, not hardware exposure timestamp. Do not claim hardware sync.
- 2–12 increasing exposures, selected configured serials or four by default, one front/back pose per invocation, repetitions, optional pair subset/gain/ROI/alignment. Default exposure pair inherited (currently 1500/6000 us). No clipping retries; keep bad exposures as evidence.
- Save uint8 BGR lossless PNG sources with SHA256 and per-frame readback; fuse exact same pair with current selective logic and OpenCV Mertens weights1/1/1, gamma1, clipped0..1 to8bit. Optional MTB alignment shared by both algorithms; originals unchanged. No radiometric calibrated HDR claim.
- `session.json` tracks capturing/fusing/complete/failed and original records; failures retain evidence and clean up camera context. Output directories must be fresh. Successful sessions replay without SDK or profile, verify source hashes, preserve source capture settings. Replay does not accept capture overrides.
- `exposure_report.py`: per-pair four-column comparison gallery, full-size links, bounded thumbnails, UTF-8 HTML/JSON and BOM CSV. ROI metrics only; dark gray<=5, clip any channel>=250 (different from runtime gray clipping), mean/std/p01/p99/Laplacian variance. No automatic best-exposure ranking. Unicode-safe image file I/O.
- Verification: `UV_CACHE_DIR=/tmp/bmw-exposure-uv-cache uv run --no-sync python -m pytest tests/unit/bmw_inspection -q` ->401 passed, existing NVML warning. New exposure/report tests29 total. CLI help passes. Real subprocess synthetic replay at `artifacts/bmw_exposure_synthetic_20260911/replay/index.html` has3 sources+6 fusions; explicitly synthetic, not camera evidence.
- No real camera, Windows, live GUI, lighting or production acceptance. Existing settings are not restored to pre-run exposure/gain; cleanup restores continuous trigger and closes cameras. Normal collector sets parameters again on next run. No commit/push. Preserve other dirty stamp/hole/contour work.
