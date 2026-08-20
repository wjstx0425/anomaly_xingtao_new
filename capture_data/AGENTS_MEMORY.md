# ZS32 0727 Template release preparation

- `prepare_zs32_template_release.py` creates a new immutable Template-only release from a Stage30
  `crop_manifest.csv`. It requires right-hand, normal rows, exactly eight canonical views per physical part,
  and rejects encoded-file SHA256 reuse across parts.
- Physical identity is `session_id:groupNNN`; seed 42 and part roles `train=11`,
  `model_val=3`, `calibration=3`, `final_test=2` are enforced as the only valid 19-part contract.
- The output uses live image symlinks, copies sibling `roi_config.json` byte-for-byte, writes
  `template_manifest.csv` with the trainer-consumed `split` field, and publishes SHA256 receipts.
  `validate` is fail-closed on any encoded-image duplication, role leakage, missing views/assets, source SHA drift, copied-ROI drift,
  or receipt drift. Never overwrite an existing release root.
- This is a fixed v12 provenance gate, not a generic publisher: accept only `source_name=zs32_0727`, the
  exact `dataset/zs32_0727_template_roi_v1/crop_manifest.csv`, and its sibling ROI config SHA256
  `9412b2838cdb96f722db356714cf9bbbb5ea01810671ccf92b67323de77ebe65`. Every resolved `output_path` and
  release symlink target must remain under that ROI release; every existing raw `source_path` must remain
  under `dataset/zs32_0727`; and raw/output `groupNNN` identities must agree across exactly `group001` to
  `group019`. Resolve relative Stage30 paths from the repository root, not the CLI cwd. Validate the
  temporary staged directory before `os.replace`, mapping final manifest image paths back to staging so a
  failed validation cannot publish a final release root.

## Published v12 runtime-bundle contract (2026-07-27)

- `tests/unit/capture_data/test_zs32_runtime_bundle.py` loads the immutable v12 bundle from
  `results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json` without a skip. It locks
  the full 24 expected/threshold records, including Template records for both secondary views.
- The test verifies that only the Template model path and model SHA differ from v11. It rehashes the real
  bound PatchCore checkpoints, YOLO weights, and shared ROI config and compares their bound identities with
  v11; both releases remain commissioning-only and explicitly disallow production.

## ZS32 clean Demo runtime (2026-07-27)

- `zs32_demo_config.py` owns the hash-free single JSON contract. `zs32_demo_runtime.py` adapts the existing Template,
  PatchCore, and YOLO backends without copying model algorithms.
- The runtime loads geometry from the validated ROI JSON, retains all model objects, reloads only thresholds per part,
  and rejects model/ROI/topology/inference changes until Dashboard restart.
- Every request runs eight Template, eight PatchCore, and eight YOLO branches. Branch errors retain their configured
  threshold, publish a real diagnostic, and force the local fusion to ERROR.
- `zs32_live_commissioning.py` now builds only `pipeline/zs32_demo_inference.py` argv and reads only the unified
  `runtime_manifest.json`; it has no online runtime-bundle, Stage18, threshold-artifact, or fusion-profile dependency.
- Preserve historical strict modules for replay until the user separately approves exact physical deletions.

## ZS32 Demo global Template gate (2026-07-27)

- `zs32_demo_runtime.py` evaluates all eight Template views before any per-part PatchCore/YOLO inference.
- Any `NG_TEMPLATE` ends the part as a complete `NG_TEMPLATE` decision. Every PatchCore and YOLO branch remains
  `state=skipped,status=SKIPPED,score=null`; each Fusion branch is available and mirrors that view's Template
  `NG_TEMPLATE` or `PASS`, with a real local evidence image.
- Template errors produce ERROR and use the same downstream skip path. Only an all-PASS Template gate calls the
  resident PatchCore and YOLO backends.

## ZS32 Demo I/O and pacing optimization (2026-07-27)

- The operating priority is minimum per-part detection latency, not minimum RAM/VRAM/GPU use. Prefer preloading,
  resident duplicated workers, larger caches, and other measured space-for-time trades when results remain complete.
- Online Stage35 calls `load_complete_sample(..., validate_decode=False)` so only the resident worker decodes image
  content. Direct callers retain the fail-closed `validate_decode=True` default.
- `zs32_demo_runtime.py` reads each encoded source once, decodes both color and authoritative OpenCV grayscale arrays,
  copies original source PNG bytes to the result, and crops Template/YOLO inputs in memory. It creates crop PNGs only
  after all eight Template views pass because PatchCore still consumes paths.
- Real Template-NG replay completed in 2.1237 s with no downstream calls, no crop directory, byte-identical source
  copies, and Template records equal to the earlier result. See
  `results/zs32_demo_speedup_20260727/template_ng_replay.json`.
- The capture command now uses a 0.2 s interval while preserving manual front/back confirmation, 1500/5500 us HDR,
  settle 1, timeout 2000, short-dark threshold 80, long-clip threshold 245, blend width 50, blur size 101,
  zero retries, max clip 5%, and HDR alignment disabled. These values intentionally match the actual
  `dataset/zs32_top` collection command. Do not claim hardware wall-time improvement until an operator reruns it.
- A two-thread, distinct-engine PatchCore candidate is unsafe with the current Anomalib/Lightning stack: every
  RTX 4090 A/B parallel arm lost four views with `IndexError: pop from empty list`. Never restore same-process
  threaded inference; the process-isolated implementation below supersedes the former serial-only boundary.

## ZS32 PatchCore process isolation (2026-07-27)

- Threaded `Engine.predict` remains forbidden. The accepted implementation is
  `ResidentMultiprocessPatchcoreBackend` using spawn-only, statically sharded, resident child processes.
- `configs/zs32/zs32_demo.json` selects 8 processes after exact RTX 4090 A/B. Its formal median is `1.4662 s`
  versus serial `4.9488 s` (`70.37%` faster), with 20 additional zero-error stability rounds and exact evidence.
- `ZS32DemoRuntime.close()` and the outer worker `finally` perform bounded cleanup. If YOLO construction fails after
  children start, construction also closes the pool. Serial rollback is the single config change
  `patchcore_process_count: 1`.

## ZS32 top two-view incremental release builder (2026-07-27)

- `prepare_zs32_top_patchcore.py` is intentionally front/back-only. It ignores the other six `zs32_top` directories,
  excludes incomplete group029, requires 28 complete front/back identities, and crops only those 56 raw PNGs.
- It reuses v14 train/calibration assets by file-level symlink, adds new normal roles 20/4/4 with seed 42, publishes
  atomically, hashes every routed image, and rejects any unselected PatchCore view during validation.
- The published release is `dataset/zs32_top_front_back_patchcore_release_v1`; do not broaden its `VIEWS` tuple without
  explicit user approval.

## ZS32 v14 active Demo artifacts (2026-07-27)

- The active Template root is
  `results/zs32_template_gate_right_0727_plus_defect_eight_view_v14`; use its `model.json` `high_threshold` values
  for the Demo's binary Template gate.
- The active eight-view PatchCore root is
  `results/zs32_patchcore_eight_view_0727_plus_defect_seed42_v14`; use `eight_view_summary.csv` `deploy_threshold`
  values with those checkpoints.
- Validation covered Template model SHA256
  `f0f5db9f3e1bd40efd510f47c236e11b4a92472a8378b5181c8bffcc511895ad`, all 40 Template image hashes, eight
  non-empty checkpoint files, production config loading, and a real RTX 4090 eight-process READY/STOP smoke.
- The replacement did not change the ROI config, topology, YOLO settings, or `patchcore_process_count=8`.
