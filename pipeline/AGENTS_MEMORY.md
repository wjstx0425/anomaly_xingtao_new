# Pipeline Memory

## ZS32 EfficientAD six-view benchmark audit blocker (2026-07-21)

- New speed-only candidate evidence: `results/zs32_efficientad_s_six_view_seed42_v1/runtime_benchmark_same_image_20260721.json`. With six checkpoints resident on RTX 4090, one warmup and five formal rounds over the exact current primary-view crops, EfficientAD-S six-view `Engine.predict` is 1.748 s median (1.717-1.787) versus PatchCore 3.301 s, saving 1.553 s / 47.0% in the Engine segment. A simple Stage32 projection is 7.006 -> 5.454 s, but no EfficientAD runtime/evidence/Stage18 integration has been run.
- Current `capture_data/zs32_model_runtime.py` cannot load EfficientAD by path substitution: it constructs `Patchcore`, publishes PatchCore-specific contracts, and v10 binds those model/profile hashes. The new EfficientAD reports provide one threshold per view, not strict low/high thresholds. Keep this as a separate candidate until a new no-clobber bundle supplies an EfficientAD backend, evidence equivalence, dual thresholds, and full same-image benchmark.
- Historical context: the earlier mandatory audit completed without training. The later user-supplied EfficientAD-S checkpoints now enable speed testing, but the data contract remains unresolved: `crop_manifest.csv` has no verified physical-part ID and existing workflow outputs do not provide independent train/calibration/final-test partitions.
- Do not run `pipeline/8_train_custom_models.py` unchanged for this benchmark. It hardcodes EfficientAD `model_size="medium"`, maps validation to test, calibrates thresholds on `normal_test`, and reports FPR on that same set. It also lacks the requested resume, per-type/pixel metrics, and segmented load/read/ROI/forward/postprocess/write timing.
- The strict `pipeline/zs32_train_anomaly.py` path is the preferred training foundation once inputs exist, because it already enforces small/medium parameters, teacher/ImageNette hashes, train-normal-only consumption, and immutable candidate publication. It is currently blocked by the absence of a real EfficientAD recipe, bound dataset release, and materialized Anomalib export.
- Active v10 points to the collision-affected `results/zs32_patchcore_eight_view_all_seed42`, while `_v2` fixes path uniqueness but still has only train plus reused normal-test/test semantics. Preserve both roots and all v10 artifacts unchanged.

## Stage32/35/36 resident-worker speed path (2026-07-17)

- Capture CPU parallelism added 2026-07-21 in `src/zs32_inspection/capture/hikvision.py`: after each four-camera SDK exposure pass is complete, HDR fusion+clip and PNG encoding run in bounded four-worker pools. Futures are collected in topology order, PNG failures retain only the canonical successful prefix, `CaptureFrame` construction/timestamps and `TimingRecorder.add()` stay on the main thread, and SDK calls are not part of these new CPU pools.
- Read `hdr_fusion_parallel_wall` and `image_encoding_parallel_wall` for elapsed capture cost; their unsuffixed counterparts are now overlapping per-worker CPU sums. Offline 4024x3036 four-image microbenchmark: fusion 1.742 -> 0.683 s, encoding 2.290 -> 0.978 s, exact pixels/PNG bytes, about 2.88 GiB peak RSS. A real front/back operator capture is still required to establish the live improvement.
- Final capture-parallel gate: 253 related tests passed, targeted compile/diff checks passed, and code review has no remaining Critical/Important findings. The expanded capture-data suite's 19 unrelated pre-existing failures are not acceptance failures for this change.
- Current lossless parallel path (2026-07-19): Stage32 uses four workers for the eight independent `cv2.imread` calls, then keeps canonical result/error ordering. It writes all eight Template crops serially before using six workers for the six primary Template evaluations; 20/22-group secondary views remain exact `SKIPPED` entries and are never submitted. Workers return timing/results to the main thread because `TimingRecorder` is not thread-safe.
- Authoritative artifact is `results/zs32_timing_worker_parallel_v1/run_01..05` plus `warmup`. Five formal runs are all OK/complete: median total 7.006 s (6.826-7.296), decode 0.299 s, Template aggregate 0.906 s, and Template evaluate wall 0.123 s. Relative to io-v3, total/decode/Template improve 14.1%/66.6%/32.3%. Normalized Template/PatchCore/YOLO/fusion CSVs, all source/crop/evidence PNG bytes, and six raw maps exactly match io-v3 run_02.
- Second lossless I/O pass: Stage32 no longer writes eight temporary YOLO crop PNGs; it passes canonical contiguous BGR ndarray crops directly. PatchCore retains only six required primary-view PNG crops, overlaps evidence writers with later GPU inference, and YOLO evidence uses two bounded writers. Each archived source SHA is computed once, while `sources/` stays an independent copy for immutable audit.
- Authoritative benchmark `results/zs32_timing_worker_io_v3/run_01..03` completes Stage32+Stage18 in 8.171/8.155/8.061 s (median 8.155 s), down 30.7% from worker v2. The target I/O critical path falls from 3.798 s to 1.276 s median (-66.4%). All three runs exactly match worker v2 semantic fields, CSV values, hashes, raw maps, decoded evidence pixels, 20-group decisions, and secondary SKIPPED states.
- `pipeline/zs32_inference_worker.py` is the Dashboard-owned persistent execution process. It imports Stage32 once, preloads six primary PatchCore models plus YOLO and prepared six-view Template assets, then serves serial `run_argv` jobs over a private Unix socket.
- `pipeline/32_run_zs32_multimodel_inference.py` retains standalone CLI compatibility and the immutable output/Stage18 contracts. In a worker it reuses the prepared runtime, while each job still creates a unique generation and writes `timing.json`.
- `pipeline/35_run_zs32_live_commissioning.py` accepts the private `--inference-socket`; without it, the old Stage32 subprocess path remains active. Stage35 merges capture/Stage32/Stage35 timings before publishing `complete`.
- `pipeline/36_zs32_inspection_dashboard.py` command shape is unchanged. Dashboard startup creates the worker automatically and shutdown sends STOP before process-group fallback termination.
- The active 20-group v10 still runs six primary Template/PatchCore branches and eight YOLO branches; `front_secondary/back_secondary` Template and PatchCore remain explicit SKIPPED. Do not call `_ensure_backends()` without the six-view subset in this profile.
- Same-image measured inference improved from 24.260 s cold to 11.765 s prepared-worker without projected result differences. See `docs/ZS32_DEMO_SPEED_OPTIMIZATION_20260717.md` and `results/zs32_timing_worker_v2/worker_benchmark_v2`.
- Final directly related verification: 354 pipeline/model/live tests, 135 Dashboard tests, 28 capture tests, and 5 worker socket tests passed. Broader suites still expose unrelated pre-existing runtime-bundle CLI, strict-fusion fixture, and compositor pixel-equality failures; see the report for exact counts.

## ZS32 four-camera legacy-layout capture design (2026-07-14)

- Implemented design: `docs/designs/2026-07-14-zs32-four-camera-legacy-layout-capture-design.md`; copy-paste hardware commands: `configs/zs32/topology/README.md`.
- Final `pipeline/zs32_bootstrap_capture.py --legacy-layout` mode writes historical per-hand/per-view label directories and one unchanged 25-column `manifests/<session>.csv`, extending each complete sample to 8 image rows plus 1 sample row for `front_secondary/back_secondary`. It publishes no duplicate `_bootstrap` images; omission of the flag preserves isolated bootstrap output.
- Hardware contract is topology `configs/zs32/topology/zs32_4cam_double_side_v1.json`, output root `/home/yunjing/anomalib/dataset/test`, `images-per-group=1`, HDR `1500/5500 us`, gain `0`, interval `0.2 s`, settle `1`, timeout `2000 ms`. Accept 1 group only at 8 PNGs/9 data rows, then run 120 groups and require 960 image plus 120 complete sample rows, 1080 rows total.
- Legacy mode preflights conflicts before camera initialization. Stage 2 can discover all 8 view directories, but Stage 3, YOLO, Label Studio and parts of training remain six-view-only.
- Final offline gate on 2026-07-14: capture-focused `57 passed`, topology `9 passed`, plus successful compilation, CLI help, topology JSON/legacy CSV static checks and diff check. No real one-group or 120-group capture was claimed.

## ZS32 four-camera bootstrap capture (2026-07-14)

- Immediate four-camera entrypoint: `pipeline/zs32_bootstrap_capture.py`; installed console script: `zs32-bootstrap-capture`. It is topology-driven and uses `configs/zs32/topology/zs32_4cam_double_side_v1.json`, including front-mounted serial `DB0968108` as `front_secondary/back_secondary`.
- The CLI intentionally mirrors the legacy collector's `images-per-group=1` batch/HDR command, keeps four camera handles open over the whole batch, and asks for Enter before each front/back round. It rejects `images-per-group>1`, does not save HDR source exposures, and defaults HDR alignment off. The recommended fast HDR settings are short/long `1500/5500 us`, gain `0`, settle frames `1`, timeout `2000 ms`, and `hdr_max_retries=0`.
- Hardware invocation on this host needs `PYTHONPATH=/opt/MVS/Samples/64/Python/MvImport:${PYTHONPATH:-}` because the project `.venv` does not directly contain `MvCameraControl_class`; the prefixed import was verified successfully.
- Successful sets are isolated under `<root>/_bootstrap/<session>/<set>` with eight PNGs and `bootstrap_manifest.json`; partial failures are under `_bootstrap_incomplete`. This path has no gate evidence and never writes formal `capture_sets.csv`, `images.csv`, `capture_gates.json`, or `capture_manifest.json`, so it is not automatically eligible for dataset/release use.
- Operator command and exact output contract are maintained in `configs/zs32/topology/README.md`. Focused tests are in `tests/unit/zs32_refactor/capture_data/test_bootstrap_capture_cli.py`.

## ZS32 Stage35 live 18-group commissioning (2026-07-14)

- Use `/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py --list-devices` for camera preflight and `/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py --part-id live_part_001` for one right-hand part. The entrypoint delegates to `capture_data/zs32_live_commissioning.py` and the existing collector/Stage32 processes under the same interpreter.
- To isolate the template operator, run `/home/yunjing/anomalib/.venv/bin/python pipeline/35_run_zs32_live_commissioning.py --part-id live_part_001 --diagnostic-skip-template`. The same six-view capture feeds Stage32 `infer`; six PatchCore branches and YOLO publish absolute `<output_dir>/patchcore.csv` and `<output_dir>/yolo.csv`. Template loading, `template_match.csv`, fusion, and Stage18 audit are intentionally absent. The contract stays `REVIEW`, incomplete, non-strict, commissioning-only, and `production_release_allowed=false`; it is diagnostic evidence only.
- The live path locks serials `DA9805574` (front/centre), `DA9625347` (left), `DB0998274` (right), one manual-load HDR sample, `1500/6000 us`, gain `0`, settle frames `1`, and the `zs32-right-18-commissioning` profile. The operator confirms the front three views, flips the same right-hand part, then confirms the back three views.
- Defaults are capture `/home/yunjing/anomalib/results/zs32_live_capture/<run_id>`, output `/home/yunjing/anomalib/results/zs32_live_runtime/<capture_session>/<sample_id>`, runtime config `/home/yunjing/anomalib/results/zs32_offline_commissioning/runtime_models.local.json`, template `/home/yunjing/anomalib/results/zs32_template_gate_right_unified_roi_v1`, and thresholds `/home/yunjing/anomalib/results/zs32_18group_commissioning_v1/thresholds.json`.
- Stage35 has no quality, registration, geometry, GUI, alternate profile, hand, or group-count surface. It always reports `commissioning_only=true`, `production_release_allowed=false` and is not a production-release path.
- Full fusion requires the runtime summary plus Stage18 audit and formats all 18 canonical rows. A real Stage32 template short circuit is a separate valid contract: `NG_TEMPLATE` or exception-shaped `REVIEW`, `inspection_complete=false`, verified template rows, and `audit_path=None`. Do not invent or require Stage18 audit after that short circuit. Source-image decode/dimension failures currently make Stage32 nonzero and remain execution errors.
- Test files: `tests/unit/pipeline/test_zs32_multimodel_inference.py`, `tests/unit/capture_data/test_zs32_live_commissioning.py`, `tests/unit/pipeline/test_zs32_live_commissioning_cli.py`, and the collector prompt tests in `tests/unit/capture_data/test_collect_multicamera_dataset.py`.
- Final skip-template diagnostic verification on 2026-07-14: `233 passed`; Ruff/format, compilation, CLI help, diff check, and the Stage35 real-artifact validator passed. `--list-devices` returned `DB0998274`, `DA9805574`, and `DA9625347` with exit code 0.
- A real GPU replay of the saved correct-right-hand capture `20260714_145508_515286` is at `/home/yunjing/anomalib/results/zs32_live_runtime_diagnostic/20260714_145508_515286/live_part_001_group001_000001`. It contains six PatchCore rows and six YOLO rows with 12 evidence files, no model errors, no template/fusion/audit artifacts, and REVIEW/incomplete/non-production policy. PatchCore scores were `0.313845, 0.888506, 1.0, 0.364892, 0.796209, 0.919070` in canonical view order; YOLO scores were all zero for this normal part.
- Hardware/model preflight found one CUDA-visible RTX 4090 under the anomalib venv. Six PatchCore checkpoints and YOLO `best.pt` matched their pinned SHA-256 values; both ROI configs existed and the 18-group artifact source binding validated against the current runtime and template model.

## ZS32 unified-ROI offline commissioning (2026-07-14)

- Template reference images and PatchCore training inputs must come from `/home/yunjing/anomalib/dataset/zs32_patchcore_roi_yolo`; its `crop_manifest.csv` is the template trainer input with `--path-root /home/yunjing/anomalib`.
- Right-only template publication: `/home/yunjing/anomalib/results/zs32_template_gate_right_unified_roi_v1`; six groups, five templates per group, model SHA-256 `db3730a5e6559ef734b77c67c8f82b2b06c0165719533ebf929dd8f0e3ad330d`.
- Fresh unified-ROI PatchCore publication: `/home/yunjing/anomalib/results/six_view_roi_yolo_fixed_seed42`; the aggregate has six rows and no `failed_views.txt`.
- Commissioning runtime config is outside the checkout at `/home/yunjing/anomalib/results/zs32_offline_commissioning/runtime_models.local.json`. Use the old anomalib environment because it imports the local Ultralytics fork: `/home/yunjing/anomalib/.venv/bin/python pipeline/32_run_zs32_multimodel_inference.py infer ...`.
- Real right-normal `group038` GPU smoke output is `/home/yunjing/anomalib/results/zs32_offline_commissioning/group038`. It produced 6 PASS template rows, 6 PatchCore rows, 6 YOLO rows, 12 evidence images, and the intentional `REVIEW/incomplete` terminal state pending locked thresholds plus quality/registration/geometry branches.

## ZS32 Stage 33 dataset commissioning (2026-07-14)

- Batch entrypoint: `pipeline/33_run_zs32_offline_calibration.py`; reusable core: `capture_data/zs32_offline_calibration.py`. The required `--commissioning-only` flag prevents this legacy split from being presented as a formal locked-threshold workflow.
- Source images are selected through `/home/yunjing/anomalib/dataset/zs32_patchcore_roi_yolo/crop_manifest.csv`, but six-view routing is recovered from original source basenames. Do not trust the old `resolved_view` values for deform groups 009-013. Template and PatchCore use their unified ROI assets; YOLO labels come from `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo`.
- Completed output: `/home/yunjing/anomalib/results/zs32_offline_calibration_unified_roi_v1`. It contains 99 case directories, 594 six-view samples and 1782 merged template/PatchCore/YOLO rows with no runtime errors.
- The useful current diagnostic is `template_patchcore_threshold_calibration/`: 12 valid groups, test defect non-clear recall 4/4, normal strong-reject 5/28, review 16/32. The held-out defect set is too small for a release claim.
- For YOLO, `yolo_annotation_calibration_rows.csv` excludes train and uses actual per-image box presence from val/test. `yolo_annotation_thresholds_by_view/summary.json` sets `all_views_operationally_usable=false`: back, back_left, back_right and front_left require zero low thresholds at target recall 1.0 because the model missed labeled positives. Front and front_right are nonzero but each reaches only 2/3 held-out positive non-clear recall under fitted thresholds.
- Never pass the root `threshold_calibration/thresholds.json` or any current YOLO diagnostic threshold into Stage 32 `fuse`. A formal artifact still needs one pre-locked physical-part split across training/calibration/test, improved YOLO recall, and all 36 right-profile template/quality/registration/PatchCore/YOLO/geometry groups.
- Review hardening is part of Stage 33: `commissioning_run_contract.json` binds exact input/config/template/YOLO-label hashes and fit parameters; every derived threshold directory has a `stage33_publication.json` sidecar binding its input CSV and output hashes. Resume also recomputes source hashes and rejects drift, mismatched labels/splits, altered aggregates, or stale threshold publications.
- YOLO split validation is physical-part-level across all six views and verifies one paired image plus valid normalized bbox syntax. The current strengthened summary marks all six views unusable: four have zero thresholds, while front/front_right also fail minimum sample and held-out recall requirements; front_right's equal low/high threshold is explicitly degenerate.
- Preflight now performs the YOLO syntax/split checks before runtime creation and the contract hashes every paired ROI image. The current result's `yolo_runtime_crop_identity.json` confirms pixel-for-pixel equality for all 594 selected runtime crops and labeled ROI images despite different PNG byte encodings.
- `/home/yunjing/anomalib/dataset/zs32_six_view_yolo` and `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo` have exact label-stem/split/box-presence correspondence: 1398 train, 306 val and 306 test labels, with zero missing/extra/presence mismatches. Use ROI labels for Stage 33 because the scored YOLO images are ROI crops.
- Optional complementary YOLO calibration is `--yolo-aux-min-image-precision 0.90`; it writes `yolo_high_precision_auxiliary/` without replacing the old recall-first per-view diagnostics. Threshold selection uses only val/calibration image-level bbox-presence labels and test is evaluation-only. The core independently enforces split-isolated physical IDs and exactly six canonical rows per part.
- Current per-view direct-STRONG candidates are front `0.38068947196006775`, front_left `0.8800162672996521`, front_right `0.8765060305595398`, back `0.025683363899588585`, back_left `0.5230337381362915`, and back_right `0.008334489539265633`. Six-view part OR is val `TP=4 FP=0 FN=1 TN=8` and test `TP=5 FP=0 FN=0 TN=13`; all five test positive triggers are on a view that itself has a GT box.
- This is commissioning/report-only and not a Stage-31 locked runtime bundle. `score>=T` is intended as direct `NG_YOLO` strong evidence after future full-profile integration; `score<T` only clears YOLO so PatchCore/template/other views can cover misses. The measured precision is image bbox-presence precision, not IoU-matched detection precision. `runtime_injection_supported=false` remains mandatory until all 36 strict right-profile groups are locked.

## ZS32 18-group Stage32/18 commissioning (2026-07-14)

- Reduced profile: `config/fusion/zs32_right_unified_roi_18_group_commissioning.json`; alias: `zs32-right-18-commissioning`. It is exact right-only `6×3=18` for template/PatchCore/YOLO and is machine-marked `commissioning_only=true`, `production_release_allowed=false`. Never replace the unchanged production `zs32-right` 36-group profile.
- Publisher: `pipeline/34_publish_zs32_18_group_commissioning.py`; core: `capture_data/zs32_18_group_commissioning.py`; real artifact: `/home/yunjing/anomalib/results/zs32_18group_commissioning_v1/thresholds.json`. It binds runtime/template versions, template model SHA, Stage33 sidecars, exact group coverage, profile SHA and canonical artifact hashes. It rejects invalid calibration, non-ok source records, incomplete YOLO candidates, test-selection leakage and precision-policy violations.
- Stage32 usage is `fuse --fusion-profile zs32-right-18-commissioning` with template model and the 18-group artifact; omit quality/registration/geometry only in this explicit profile. Default production fusion still requires all three. Before GPU loading, Stage32 compares current runtime config and actual template `model.json` bytes with the artifact source SHA bindings. Only a complete finite semantic template REVIEW continues in commissioning; exception REVIEW and NG_TEMPLATE stop with explicit non-production metadata.
- Real normal `group038` GPU result: `/home/yunjing/anomalib/results/zs32_18group_smoke/group038`. All 18 rows are CLEAR; Stage18 returns `OK`, `inspection_complete=true`, with no errors. Post-hardening rerun: `/home/yunjing/anomalib/results/zs32_18group_smoke_v2/group038`, same result. The runtime manifest is updated after fusion and preserves pre-fusion REVIEW separately; all summary/policy/audit surfaces state production release is forbidden.
- Verification: focused publisher/Stage18/Stage32 tests `37 passed`; broader relevant fusion/runtime gate `216 passed`. Ruff, legacy core E/F/I, formatting, compilation, JSON and diff checks pass.

## Cropped six-view PatchCore retraining

- `run_patchcore_roi_six_views.sh` is the minimal user-run entrypoint.
- Defaults: data `dataset/zs32_patchcore_roi`, output `results/six_view_roi_fixed_seed42`, GPU `0`, and `HF_HUB_OFFLINE=1`.
- It verifies `crop_manifest.csv`, then reuses `run_wrn50_fixed_six_views.sh` for serial, resumable right-side six-view PatchCore training with the existing fixed seed-42 configuration.
- Run from the repository root with `bash pipeline/run_patchcore_roi_six_views.sh`.
- Before the directory-preserving recrop, all old training processes/services and old ROI results were stopped/removed. The regenerated dataset contains 2010 images, preserves every original hand/view directory, and records zero corrected views. Start retraining fresh with the runner.

## C789 grayscale multi-template matcher prototype

- `train_template_matcher.py` and `predict.py` implement a standalone OpenCV image-level baseline; they are not part of the numbered pipeline and do not use anomalib `Engine` or checkpoints.
- Training groups already-cropped images by `left_top|left_bottom` and `slotNN`. It selects representative normal templates using a median prototype plus farthest-first thumbnail sampling.
- Matching uses grayscale resize, a `3x3` Gaussian blur, and `cv2.TM_CCOEFF_NORMED` over a reflected `+-shift` translation window. The score is the maximum similarity across a slot's templates; lower than the group threshold means `defect`.
- The default `normal_quantile=0.995` produces a lower-tail threshold at quantile `0.005` from the same `normal_test` images later included in metrics. Those normal metrics are calibration-set results, not an independent test estimate; a missing `normal_test` group silently falls back to `0.8`.
- Outputs are `dataset_index.csv`, template PNGs, `model.json`, `predictions.csv`, and `metrics.json`. `predict.py` requires an already-cropped image plus explicit `--view` and `--slot` and prints one JSON result.
- Keep this baseline distinct from the canonical stage 11-15 geometry workflow in `capture_data/geometry_shape.py`: this prototype compares whole-image grayscale appearance and produces one binary score, while geometry templates compare aligned masks and produce interpretable `missing_mask` / `extra_mask` evidence.
- Known prototype boundaries: no ROI/slot detection, no defect localization/type, no dedicated README/tests/config, no friendly checks for missing groups/templates, direct aspect-ratio coercion on shape mismatch, and direct-script-only import style for `predict.py`.

## ZS32 template-first inspection gate (2026-07-13)

- Production ZS32 gate code is now merged into main and remains independent from the separate C789 prototype scripts.
- `train_zs32_template_gate.py` trains 12 whole-view `(hand, view)` groups from the stage-30 ROI `crop_manifest.csv`; use `--path-root /home/yunjing/anomalib` when the data remains in the main workspace.
- Template-fit normals and threshold normals are disjoint by physical `part_id`. Risk is `1 - similarity`; `T_low` is the minimum calibration defect risk and `T_high=max(T_low, normal quantile)`.
- Training writes `model.sha256` and same-source `calibration_rows.csv`; Stage 31 must combine those template rows with the other five branch families so its locked template thresholds exactly reproduce the online model values.
- `predict_zs32_template_gate.py` returns exit code 0/10/20/2 for PASS/REVIEW/NG_TEMPLATE/invalid gate. It retains similarity, risk, both thresholds, best offset, best-template path and SHA-256, and all deployment versions.
- `capture_data/zs32_inspection_orchestrator.py` is the execution short-circuit. It validates all six inputs, runs `template_match` in canonical order, and calls the injected downstream runner only after six explicit, numerically consistent PASS results.
- The downstream runner receives both `InspectionRequest` and the six normalized template results. Use `template_results_to_branch_rows` or `write_template_match_csv` to preserve them for Stage 18.
- PASS requires four deployment versions plus a readable best-template path whose SHA-256 matches the declared hash. Stage 18 preserves that value as `evidence_hash` and verifies it against the evidence file.
- A downstream `OK` is valid only when `inspection_complete=true`; unknown status or incomplete OK becomes REVIEW.
- Strict Stage 18 requires nonblank 64-hex source/evidence hashes on every required row and recomputes both before release.
- Early-stop audits retain the human-review lifecycle fields (`PENDING` review, null review/release statuses) because Stage 18 is not called after a template non-PASS.
- A non-PASS template result never fabricates downstream CLEAR evidence: it records skipped views/branches and returns REVIEW or NG_TEMPLATE with `inspection_complete=false`.
- Strict fusion now requires six branches per view, including `template_match`; the left/right contract is 72 exact groups. `template_match` is ordinary anomaly evidence, not a quality/registration RETAKE gate.
- `/tmp/zs32-template-gate-smoke-*` models are reduced-width smoke artifacts only and must not be deployed.
- Local merge verification after combining Stage 30 PatchCore and template-first fusion: `333 passed, 1 warning`; the backup branch is `backup/main-before-zs32-fusion-20260713`, and nothing was pushed to GitHub.

## ZS32 right-hand unified multi-model runtime (2026-07-13)

- Unified entrypoint: `pipeline/32_run_zs32_multimodel_inference.py`; core runtime: `capture_data/zs32_model_runtime.py`.
- Checked-in model bundle: `config/fusion/zs32_runtime_models.json`. It pins six checkpoints below `results/six_view_roi_fixed_seed42` and YOLO `/home/yunjing/ultralytics-c789/final_n640_p1_seed42/weights/best.pt` by SHA-256. The assets are local/ignored and must exist on the deployment host.
- Current weights are right-hand only. Reject left requests before model loading; do not reuse the right models or YOLO ROI for left parts.
- PatchCore and YOLO use different ROI files. PatchCore runs once per canonical view with one persistent model/Engine per view; YOLO runs one six-image batch with `candidate_conf=0.001`, which is only a candidate acquisition floor.
- Dashboard Fusion must not visualize sub-threshold YOLO candidates: keep all candidates on the standalone YOLO evidence page, but on Fusion draw only per-box `confidence >=` the view's final YOLO threshold, and draw none when the branch score is below threshold.
- `infer` writes `patchcore.csv`, `yolo.csv`, heatmaps/box overlays, crop files, hashes, `runtime_manifest.json`, and optional Stage-31-compatible `calibration_rows.csv`. Without locked dual thresholds it must remain REVIEW/incomplete.
- `--template-model-dir` uses the PatchCore ROI and runs before any PatchCore/YOLO backend. A non-PASS result short-circuits and publishes no fabricated downstream evidence.
- Right-only strict profile: `config/fusion/zs32_right_six_view.json`, named Stage-18 profile `zs32-right`, exactly 36 versioned groups. Keep the original `zs32` two-hand 72-group profile unchanged.
- `fuse` requires the locked right-profile Stage-31 artifact plus template, quality, registration, geometry, PatchCore, and YOLO evidence. Only Stage 18 may turn the runtime result into a complete OK.
- Production `fuse` deliberately requires `--template-model-dir`; an old `template_match.csv` cannot replace the online first-stage gate or bypass current-image binding.
- Strict fusion profiles are `zs32_six_view_v1` and `zs32_right_six_view_v1`. For both, `fusion_engine` ignores CSV-declared `evidence_level` and recomputes CLEAR/GRAY/STRONG from the continuous score and locked low/high thresholds, preventing forged CLEAR evidence from overriding a strong score.
- YOLO config maps checkpoint class `item` to deployment semantic `defect`; detection evidence retains both `class_name=defect` and `checkpoint_class_name=item`.
- Backend initialization failures publish per-view error JSON and a REVIEW/incomplete generation instead of exiting without diagnostics.
- Final related verification after implementation and safety review fixes: 267 tests passed; config assets and the 36 runtime/profile model groups aligned; compileall, full Ruff on new files, focused F/I on touched legacy files, JSON check, CLI help, and `git diff --check` passed.
- Real CPU smoke used all six `group001` normal images from session `20260711_165347_078120` and wrote `/tmp/zs32-stage32-real-smoke`: 6 PatchCore rows, 6 YOLO rows, 12 calibration rows, 6 heatmaps, and 6 YOLO overlays with no runtime errors. It correctly remained REVIEW because no threshold/template/quality/registration/geometry bundle was supplied. Observed PatchCore scores were `0.39345, 0.49116, 0.38902, 0.32278, 0.30612, 0.10668`; YOLO retained low-confidence candidates in the first three views and explicit empty lists for the last three.
# ZS32 Stage 36 offline dashboard checkpoint (2026-07-14)

## ZS32 Stage35 four-camera live entrypoint (2026-07-15)

- Run `pipeline/35_run_zs32_live_commissioning.py` with a part ID; its defaults select the finalized eight-view bundle and checked-in four-camera topology.
- `--list-devices` is the non-capturing discovery path. A live run uses `pipeline/zs32_bootstrap_capture.py`, waits for front/back operator confirmation, then forwards the exact eight images and bundle-resolved assets to Stage32.

## ZS32 strict eight-view Stage33 Task 3 (2026-07-15)

- `33_run_zs32_offline_calibration.py` now binds runtime and Template ROI/template versions, defaults every Stage31 adapter call to canonical `VIEW_ORDER`, preflights both calibration classes per required view, records required views in publication sidecars, and iterates all eight views for per-view YOLO calibration.
- Code/test commit: `a263ca2f`; clean-HEAD related suite: 73 passed plus compile, CLI help, and full targeted Ruff.
- Do not run Stage33 against a weakened or ad-hoc runtime config. The specified `results/zs32_runtime_assets_eight_view_v1` publication is immutable but predates the current publisher asset-set binding schema, so Task 3 stopped before GPU. See `.git/sdd/task-3-report.md` for the exact mismatch and continuation command placeholders.

- `pipeline/36_zs32_inspection_dashboard.py` is the thin Task 5 entrypoint.
  Offline GUI command:
  `uv run --no-sync python pipeline/36_zs32_inspection_dashboard.py --result-dir /home/yunjing/anomaly_xingtao_new/artifacts/zs32_dashboard_offline_acceptance/result`.
- Use `--no-gui --save-screenshot <path>` for a headless frame; this branch does
  not call OpenCV HighGUI. The accepted screenshot is
  `/home/yunjing/anomaly_xingtao_new/artifacts/zs32_dashboard_offline_acceptance/zs32_dashboard_1600x920.png`.
- Task 5 deliberately leaves `--live` unconnected. Do not add Stage35 process
  lifecycle work before explicit offline-display acceptance.
- Review follow-up `de5d29ef` adds an automated subprocess test that invokes
  this exact wrapper with a valid result and headless screenshot output. The
  accepted image was regenerated after enforcing one-pixel active/CTA borders.

## ZS32 eight-view fixed ROI (2026-07-15)

- `pipeline/29_zs32_fixed_roi.py` supports eight views including `front_secondary/back_secondary` and accepts `--repo-root` for data stored outside this code checkout.
- For the current dataset, pass `--repo-root /home/yunjing/anomalib` plus explicit paths below `/home/yunjing/anomalib/dataset/zs32_eight_view_*`; selecting writes one ROI and preview per view before conversion.

## ZS32 Stage 30 right-hand eight-view conversion (2026-07-15)

- Use `pipeline/30_crop_zs32_patchcore_dataset.py convert` with `--repo-root /home/yunjing/anomalib`, input `/home/yunjing/anomalib/dataset/zs32_new`, ROI config `/home/yunjing/anomalib/dataset/zs32_eight_view_roi_config.json`, output `/home/yunjing/anomalib/dataset/zs32_eight_view_patchcore_roi`, and `--hand right`.
- Repeat `--exclude-session` for `zs32_right_deform_20260714_204302_593214015` and `zs32_right_deform_20260714_201404_438451409`. Filtering is output-only and preserves raw data.
- Stage 30 reuses the Stage 29 top-level eight-view ROI schema for right-only conversion and allows nested source roots below `<repo-root>/dataset` while requiring a separate sibling output.

## ZS32 eight-view PatchCore runner (2026-07-15)

- Run `pipeline/run_patchcore_roi_eight_views.sh DATA_ROOT RUN_BASE GPU` for eight independent right-hand PatchCore models. The wrapper keeps the strong `wide_resnet50_2`, `layer2 layer3`, 256-square, coreset 0.05, float32, batch-16, seed-42 settings.
- The reusable serial runner accepts `PATCHCORE_VIEWS` and `PATCHCORE_SUMMARY_NAME`; the eight-view wrapper sets both and publishes `eight_view_summary.csv`. Existing six-view commands still default to the original six views and `six_view_summary.csv`.

## ZS32 Stage 32 eight-view runtime (2026-07-15)

- `pipeline/32_run_zs32_multimodel_inference.py` now requires explicit image arguments for all eight canonical views, including `--front-secondary-image` and `--back-secondary-image`.
- The runtime validates eight distinct source images, runs eight per-view PatchCore models and one ordered eight-image YOLO batch, and accepts the right-only top-level Stage 29 ROI schema.

## ZS32 Stage 18/34 eight-view commissioning fusion (2026-07-15)

- Stage 18 named alias `zs32-right-24-commissioning` resolves to `config/fusion/zs32_right_eight_view_24_group_commissioning.json` and requires exact right-hand evidence for eight views times three branches.
- Stage 34 accepts the 24-group profile through `--profile` while keeping the existing 18-group default and compatibility API. Publication remains immutable, source-hash-bound, commissioning-only, and forbidden for production release.
- The pinned eight-view runtime config is `config/fusion/zs32_runtime_models_eight_view.json`; the trained template model is `results/zs32_template_gate_right_eight_view_v1/model.json`; the successful real inference smoke is `results/zs32_eight_view_runtime_smoke/group064_v2`.
- Stage 33 completed 90 cases in `results/zs32_offline_calibration_eight_view_v1`, but the strict Stage 34 publication is correctly blocked: YOLO `front` cannot reach the declared 0.90 calibration precision and `front_secondary` has no calibration positive. Collect additional labeled positives and rerun Stage 33; never synthesize a 24-group threshold artifact from the held-out test positive.
- A separately marked, user-authorized fake path now exists for temporary smoke only: Stage 33 `--yolo-aux-use-test-for-selection --reuse-existing-inference`, then Stage 34 `--allow-test-leakage`. Outputs are `results/zs32_offline_calibration_eight_view_v1/yolo_high_precision_auxiliary_test_leakage/` and `results/zs32_24group_commissioning_test_leakage_v1/`; both declare test leakage and remain forbidden for production.
- Real fused GPU smoke `results/zs32_eight_view_24group_test_leakage_smoke/group064` produced all 24 branch rows and completed fusion as `NG_YOLO`; this known-normal image triggered back YOLO at `0.00245661` versus leaked threshold `0.00126685`, so the smoke proves runtime wiring but not model quality.

## ZS32 strict eight-view runtime bundle decision (2026-07-15)

- New Stage32/35/dashboard work uses all eight views as modeled inputs. Do not restore the earlier six-modeled/two-unsupported behavior; `front_secondary` and `back_secondary` require the same four dashboard branches as the other views.
- The user-approved design is `docs/superpowers/specs/2026-07-15-zs32-strict-eight-view-commissioning-design.md`. Stage32 and Stage35 should accept one versioned `--runtime-config`; weight changes publish a new hash-bound bundle and matching 24-group commissioning artifact rather than changing Python paths.
- Acceptance is a complete eight-view commissioning run only. Preserve legacy six-view configs for replay and do not describe the current front-secondary model or the chain as production-ready.
- Execute `docs/superpowers/plans/2026-07-15-zs32-strict-eight-view-commissioning-plan.md`. The bundle publisher must generate all 24 expected-version rows from selected assets; changing weights must not require hand-editing the fusion profile.

## ZS32 Stage 33 review hardening (2026-07-15)

- `--preflight-only` is read-only and validates current Task2, Template, and ROI bindings, per-view fit classes, and any existing contract, case index, and reused cases before model initialization.
- `--reuse-existing-inference` requires `--resume` and complete exact case generations; directory existence alone is never enough.
- Stage31 report files and `stage33_publication.json` publish together through atomic no-replace directory publication. Resume requires the exact five files and matching input, parameters, and four report hashes.
- When optional YOLO auxiliary calibration is enabled, each canonical view requires both box and no-box val labels before execution.

## ZS32 Task 8 minimal dashboard acceptance (2026-07-15)

- Offline acceptance reuses `results/zs32_stage33_eight_view_commissioning_v4/cases/right_normal_group064-b1959d15`; Stage36 headless wrote `artifacts/zs32_dashboard_eight_view_commissioning/group064_stage33_v4_dashboard_1600x920.png` without rerunning GPU inference.
- Strict parser acceptance: canonical eight views, all modeled, exact four dashboard branches, commissioning true, production false. Case status is intentionally `REVIEW` because its Fusion branch is `SKIPPED`.
- Final bundle is `results/zs32_runtime_bundle_eight_view_v2/runtime_bundle.json`; fresh bundle plus four-camera topology preflight passed. Device listing found the four configured serials, but operator hardware capture was not performed.
- Preserve the v2 caveats in user docs: Stage33 v4 test leakage, known normal group064 back `NG_YOLO`, and finite zero-area post-clipping YOLO candidate exclusion for commissioning compatibility.

## ZS32 EfficientAD-S/M six-primary-view runner (2026-07-21)

- Entrypoint: `pipeline/run_efficientad_s_m_six_views.sh DATA_ROOT [SMALL_OUTPUT_ROOT] [MEDIUM_OUTPUT_ROOT] [GPU]`.
- It calls Stage 8 through `uv`, serially runs exactly six primary views for `small` and then `medium`, and passes `--efficientad-model-size` through Stage 8 to the repository workflow. Secondary views are hard-excluded by the runner's fixed view list.
- Default parameters are image size `256,256`, batch size 1, 100 epochs, seed 42, one GPU job, and workers 8. Each view has its own output root and Stage 8 logs/checkpoint/report; each model family gets a SHA256-bearing `six_view_summary.csv`.
- Resume rules mirror the existing PatchCore runner: summary+checkpoint skips, checkpoint without summary evaluates only, and an existing preprocess manifest uses `--skip-preprocess`. Existing aggregate summaries are not overwritten.

## ZS32 0723 retraining release builder (2026-07-26)

- Entrypoint: `pipeline/prepare_zs32_0723_retraining.py`; implementation:
  `capture_data/prepare_zs32_0723_retraining.py`.
- The accepted immutable output is `dataset/zs32_all_plus_0723_retraining_release_v2`, with 2852 PatchCore rows,
  848 Template rows, 1627 YOLO rows, and 106 physical parts split 64/16/13/13 at seed 42.
- The builder materializes live symlinks, preserves PatchCore session-qualified paths, copies and receipts the ROI config,
  preserves legacy YOLO label bytes, and refuses overwrite. Use its `validate` subcommand before training.
- `capture_data/zs32_template_gate.py` supports explicit train/model_val/calibration/final_test roles without changing
  legacy calibration/test behavior. Do not use the older generated `release_v1`.

## ZS32 0723 secondary PatchCore normal-only handling (2026-07-26)

- Secondary views intentionally contain only 0723 normal/normal_test data. `_build_datamodule` must pass
  `abnormal_dir=None` when `<preprocessed-view>/defect` is absent; never create fake defect images and never skip the
  secondary models.
- The v11 runner is resume-safe after this fix: primary summaries skip, and failed secondary views reuse their
  preprocess manifests before training.

## ZS32 0723 v11 platform publication (2026-07-26)

- Source/profile:
  `config/fusion/zs32_eight_view_24group_v11_bundle_source.json` and
  `config/fusion/zs32_right_eight_view_24_group_commissioning_v11.json`.
- Published chain:
  `results/zs32_runtime_assets_eight_view_v11` ->
  `results/zs32_24group_0723_v11_commissioning` ->
  `results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json`.
- The contract is strict 24-group: Template + per-view PatchCore + YOLO on all eight views, including both secondary
  views. Stage35 and Dashboard default to v11; pass v10 explicitly for rollback.
- When adapting old Stage33 thresholds whose ROI version name differs, Stage34 requires both
  `--allow-roi-version-rebind` and `--source-roi-config`. It validates the YOLO-sidecar-signed Stage33 run contract,
  bound source runtime path/SHA and ROI versions, and all old/new PatchCore/YOLO ROI bytes before permitting identity
  rebind, then signs the complete evidence.
- This v11 artifact is commissioning-only/non-production. Its YOLO source retains test leakage/model-rebind warnings,
  all YOLO thresholds are `0.07`, and all PatchCore low/high pairs use the corresponding v11 deploy threshold.
- Template records use current-model calibration counts. Rebound PatchCore/YOLO records use neutral `0/0` counts plus
  `count_provenance=unavailable_for_rebound_model`, never stale Stage33 counts.

## ZS32 0727 Template v12 release builder (2026-07-27)

- Entrypoint: `pipeline/prepare_zs32_template_release.py`; implementation:
  `capture_data/prepare_zs32_template_release.py`. It makes a Template-only immutable release from a Stage30
  `crop_manifest.csv`; it does not alter v11 PatchCore, YOLO, or ROI assets.
- Use `prepare` with source `zs32_0727`, 19 expected parts, seed 42, and explicit counts
  `train=11`, `model_val=3`, `calibration=3`, `final_test=2`. `template_manifest.csv` includes trainer-consumed
  `split` values exactly matching those four roles, and keeps all eight canonical views of each
  `session_id:groupNNN` physical part together.
- The builder rejects non-right/non-normal rows, incomplete or duplicate `(part, view)` records, cross-part encoded
  SHA256 duplication, and overwrite. It symlinks source crops, byte-copies the sibling ROI config, records input and
  artifact SHA256 evidence, and `validate` fails closed on role leakage or any artifact drift.

## ZS32 0727 Template v12 candidate publication

- Model: `results/zs32_template_gate_right_0727_eight_view_v12/model.json`, SHA256
  `5b071bcfb8a603f5bbf20ecf284f590e88e3e65e114f434fbde1eaf81692ff6e`.
- Candidate:
  `results/zs32_runtime_bundle_eight_view_template_0727_v12/runtime_bundle.json`, canonical SHA
  `1b3341ca05ac7e786733690534c8a75ee6c5a493927145cc39860c4448c29133`.
- The v12 profile is strict 24-group; both secondary views require Template/PatchCore/YOLO. PatchCore, YOLO, labels,
  and ROI remain the v11 assets. v11 and v10 rollback bundles remain loadable.
- The user reviewed the final-test visualization and accepted final-test normal scoring of 12 PASS/4 NG_TEMPLATE.
  Stage35 and Dashboard now default to v12. Explicitly pass
  `results/zs32_runtime_bundle_eight_view_v11/runtime_bundle.json` to roll back.
- Operational commands and per-view thresholds are in
  `docs/ZS32_0727_TEMPLATE_V12_REPLACEMENT_20260727.md`.

## ZS32 0727 Template v13 default and rollback chain

- `pipeline/35_run_zs32_live_commissioning.py` and the Dashboard worker now default to
  `results/zs32_runtime_bundle_eight_view_template_0727_v13/runtime_bundle.json`.
- v13 Template SHA256 is `eac211cab4ee1d27adda9933c3cb9c6fde71c4a7eb50fae138201a600fa17ab5`;
  runtime-bundle canonical identity is `94b025f85bb2f529fc36a0f62a941eafe30cbd71b1c2298431c24b539a29e2bf`.
  It contains 24 required records and eight Template groups, including required Template/PatchCore/YOLO on both
  secondary views, while PatchCore, YOLO, and ROI bindings stay equal to v12.
- Fresh-process loads passed independently for v13, v12, v11, and v10. Roll back only by explicitly passing the
  desired bundle to `--runtime-config`; do not mutate or alias the versioned directories.
- The v13 addendum in `docs/ZS32_0727_TEMPLATE_V12_REPLACEMENT_20260727.md` records all exact manual thresholds,
  hashes, the `/opt/MVS/Samples/64/Python/MvImport` live command, and rollback commands. v13 is still
  commissioning-only/non-production with preserved leakage/rebind warnings.

## ZS32 single Demo entry (2026-07-27)

- The previous v13 `--runtime-config` entry above is historical. Current online operation uses only
  `pipeline/36_zs32_inspection_dashboard.py --live --demo-config configs/zs32/zs32_demo.json --part-id ...`.
- `zs32_demo_inference.py` is the unified offline/resident-worker execution API. `zs32_inference_worker.py` prepares
  the Demo runtime once and serially calls `run_argv`; a bad per-part JSON or inference request returns an error but
  does not terminate the worker.
- Stage35 retains topology-driven four-camera capture and two operator confirmations, then sends exactly eight image
  paths plus the same Demo config to the worker. Its online production files no longer reference runtime bundle,
  Stage18, threshold artifact, fusion profile, diagnostic skip, or SHA validation.
- The current docs expose one Dashboard command and label the path `DEMO / 非生产`. Historical Stage32/34/37 commands
  remain only for replay and must not be treated as online instructions.

## ZS32 Demo Template-first short circuit (2026-07-27)

- The current `zs32_demo_inference.py` runtime contract is Template-first: evaluate all eight Template views, aggregate
  the global gate, and invoke PatchCore/YOLO only when every Template result is PASS.
- A Template NG generation has 8 evaluated model branches and 24 planned branches; all 16 downstream model records are
  explicit `SKIPPED`, while Fusion is available per view and mirrors that view's Template `NG_TEMPLATE` or `PASS`.
  It is a valid complete `NG_TEMPLATE` result, not a worker failure.

## ZS32 Demo runtime speed pass (2026-07-27)

- For the actual ZS32 inspection path, optimize single-part detection latency first. Higher RAM, VRAM, GPU occupancy,
  larger resident caches, duplicated resident model processes, and other space-for-time tradeoffs are acceptable when
  measured output remains complete and equivalent.
- Dashboard source decoding is shared by the strict parser and compositor through a bounded eight-entry path cache.
  Callers receive detached copies, so overlays cannot mutate cached pixels. A real eight-view result decoded 8 sources
  on its first parse/render and 0 additional sources across 10 repeated parse/render cycles.
- Stage35 online manifest loading uses `validate_decode=False`; the resident worker is the authoritative decoder.
  The direct `load_complete_sample(...)` API keeps decode validation enabled by default.
- The Demo worker reads each source PNG byte stream once for color and grayscale decode, copies the original PNG bytes
  unchanged into `sources/`, and performs both color and Template grayscale ROI crops in memory. It writes `crops/`
  only after all eight Template views pass.
- Real Template-NG replay evidence is in
  `results/zs32_demo_speedup_20260727/template_ng_replay.json`: 2.1237 s worker wall time, zero PatchCore/YOLO calls,
  no `crops/`, byte-identical sources, exact pre-change Template records, and output size 61,048,298 bytes versus the
  prior 142,312,909-byte generation.
- The live HDR command keeps manual front/back confirmation, 1500/5500 us exposures, settle 1, and
  `--capture-interval 0.2`. On 2026-07-27 it was aligned with the actual `dataset/zs32_top` collection command:
  timeout 2000, short-dark threshold 80, long-clip threshold 245, blend width 50, blur size 101, zero retries,
  max clip 5%, and HDR alignment disabled. Actual camera wall-time improvement still requires an operator hardware run.
- Do not enable two-thread PatchCore inference with the current Lightning/Anomalib engines. The RTX 4090 A/B in
  `results/zs32_demo_speedup_20260727/patchcore_concurrency_ab.json` rejected it: serial median 5.115354 s, parallel
  median 4.170282 s, but every parallel arm failed four views with `IndexError: pop from empty list`. This rejects only
  same-process threaded inference; the later spawn-only process-isolated implementation supersedes the old serial
  production setting.
- Focused executable regressions passed: Demo runtime/Stage35 44, strict parser 55, Dashboard cache 1, capture CLI 13.
  Broader legacy tests still expose unrelated contract drift: four old `ZS32ModelRuntime` manifests omit the now
  required branch `threshold`, and one compositor test expects an unsupported Fusion branch to suppress otherwise
  available downstream overlays.

## ZS32 resident multiprocess PatchCore (2026-07-27)

- The production Demo config now uses `patchcore_process_count=8`. This was selected by the strict RTX 4090 gate in
  `results/zs32_demo_speedup_20260727/patchcore_multiprocess_ab.json`; set it back to `1` for serial rollback.
- The parent uses only `multiprocessing.get_context("spawn")`. Eight long-lived children each own one distinct
  PatchCore model and Lightning Engine. IPC carries paths and small result envelopes; raw maps, masks, and overlays
  are written to unique staging paths by their owning child.
- The protocol requires per-request `DONE -> DONE_ACK -> IDLE`; timeout, stale/duplicate/missing response, child exit,
  malformed payload, and close failure are fail-closed and cannot contaminate the next part.
- Same real crops, 2 warmups + 5 formal + 20 stability rounds per count:
  serial median/p95 `4.9488/4.9768 s`; 2-process `3.3362/3.3506 s`; 4-process `2.1468/2.2291 s`;
  8-process `1.4662/1.4785 s`. All counts had zero errors, stable PIDs, and exact score-hex/raw-map/mask/overlay
  equality. Eight processes improved median PatchCore wall time by `70.37%`.
- A real full worker startup with eight children, ROI warmup, YOLO load, READY publication, STOP, and bounded cleanup
  completed in `12.94 s`; exit code was 0 and the Unix socket was removed.

## ZS32 v14 Demo model replacement (2026-07-27)

- `configs/zs32/zs32_demo.json` now binds Template to
  `results/zs32_template_gate_right_0727_plus_defect_eight_view_v14` and all eight PatchCore views to
  `results/zs32_patchcore_eight_view_0727_plus_defect_seed42_v14`.
- The binary Template gate uses each v14 `model.json` `high_threshold`; PatchCore uses each v14
  `eight_view_summary.csv` `deploy_threshold`. Do not mix either v14 model family with v13/v11 thresholds.
- Template model SHA256 is `f0f5db9f3e1bd40efd510f47c236e11b4a92472a8378b5181c8bffcc511895ad`.
  All 40 declared Template PNG hashes and all eight checkpoint paths were verified.
- A real RTX 4090 worker loaded and warmed all eight v14 PatchCore children plus YOLO, published READY in `11.634 s`,
  then handled STOP with exit code 0 and removed its Unix socket.
- ROI, topology, YOLO, and `patchcore_process_count=8` did not change. Model rollback roots are Template v13
  `results/zs32_template_gate_right_0727_eight_view_v13` and PatchCore v11
  `results/zs32_patchcore_eight_view_all_plus_0723_seed42_v11`, together with their old paired thresholds.

## ZS32 top front/back PatchCore v15 candidate (2026-07-27)

- Prepare/validate with `pipeline/prepare_zs32_top_patchcore.py`; training used
  `pipeline/8_train_custom_models.py --views right_front right_back --models patchcore`, not the eight-view runner.
- The dry-run and real output both contain exactly `right_front` and `right_back`. Candidate root:
  `results/zs32_patchcore_front_back_top_incremental_seed42_v15`.
- Holdout and the prior live-normal probe did not meet the deployment gate, so the candidate was not written into
  `configs/zs32/zs32_demo.json`.

## BMW 四算法卡片点击坐标修复（2026-08-13）

- `c8f56ab0` 移除 `bmw_lab_eight_view_demo.py` 对 Qt 鼠标回调坐标的二次缩放。OpenCV Qt `DefaultViewPort::icvmouseProcessing` 已通过反变换和 `ratioX/ratioY` 返回原图像素坐标，所以点击应直接交给 `dashboard_hit_test()`。
- 四算法卡片中心点的失败回归已验证 red/green；干净快照 UI/V3 回归 `183 passed`。实时相机启动被 `DA9805574` 独占占用 `0x80000203` 阻止，不得把此次修复说成新的四相机拍摄验收；离线交互会话为 `16688`。

## BMW 光痕 V3 与点击 UI 运行整合（2026-08-12）

- `pipeline/bmw_lab_eight_view_demo.py` 于 `37172148` 接入现有点击 UI，`642a37c6` 修复 OpenCV 4.13 Qt 的中文窗口句柄问题。内部必须始终用 `BMW_EIGHT_VIEW_DEMO` 调用 `namedWindow` / `imshow` / `getWindowImageRect` / `setMouseCallback`，可见标题单独用 `setWindowTitle` 设为中文。
- 干净快照回归是 `182 passed`；`211302` 离线重放光痕仍为 `PASS/OK`，非光痕 24 项完全不变。真机启动会话 `8809` 已完成模型/可信 OK 库预热并进入四相机 GUI 主循环，但未自动拍摄零件。

## BMW 光痕 tracked-profile V3 部署验收（2026-08-12）

- 当前光痕后端提交为 `1ca4f471`，配置 `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json` 指向 `tracked_profile_v3` 的不可变 v8 报告（SHA256 `6b43690af67702333646fa7a88a2a5053a0afae6d19cb343bf6d3abb193c17d4`）。回退时只需恢复该配置的三个 `bright_streak` 字段到 `raw_profile_v2` 及 corrected v2 报告。
- 最终干净快照验证：`181 passed`；真实保存记录 `bmw_demo_20260812_211302` 为 `PASS/OK`，并证明引擎切换前后 24 个非光痕结果完全一致。证据保存在忽略目录 `artifacts/bmw_bright_streak_tracked_v3_smoke/final_1ca4f471/`，不要提交截图、inspection 或客户图像。
- V3 只在固定 `[1792,1180,1873,1793]` ROI 中追踪斜向光痕。证据中心线绿色=强响应、橙色=弱桥接、红色=有效范围内的内部断点、灰色=前后背景。
- `real_broken_samples=0`：真实断续召回尚未验证；现阶段只能确认 8/8 完全无光痕样本保持 `NG_NO_STREAK`，不能把合成断续测试描述成真实数据效果。

## BMW trusted-OK comparison final offline gate (2026-08-12)

- Runtime v2 reference identity is fixed at `dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v2`: 50 approved parts, 400 references, 802 files, index SHA-256 `ae7833ab35cbc76cbfef6cfa5163f77ef345d879a6e8e4834fa8cbfcf6023acc`, whitelist SHA-256 `15d9d86d8ffc7a28b55706b4cce2bd84e2ddd60ebdef7bd7b67d2725281745f5`. The root checkout and BMW worktree copies were checked independently and matched exactly.
- `build_model_suite()` performs the one-time trusted-bank SHA validation/preload and tells the operator to allow about 27 seconds. Runtime shortcuts are `O` trusted comparison on/off, `N` next actionable row, and `P` previous actionable row. Bright streak alone matches `front_left/full`; Template, YOLO, and EfficientAD match their fixed part `roi`. Missing or invalid references stay diagnostic and render `无可信OK参考`; they never alter the 25 detector results or fusion.
- Offline command used the existing local Ultralytics 8.4.89 checkout: `PYTHONPATH=/home/yunjing/ultralytics-c789 UV_CACHE_DIR=/tmp/bmw_task5_uv uv run --no-sync python artifacts/bmw_trusted_ok_reference_smoke/task5_verify.py`. Saved HDR-pair sample `bmw_demo_20260812_170451` produced 25 checks, final NG, with `template/front_right` and `efficientad/front_right` NG. `front_right/roi` selected approved `bmw_right_normal_group077` at similarity `0.9715221524`, shift `(0, 0)`; no actionable comparison was missing. Feature-off and feature-on serialized detector tuples were exactly equal.
- The ignored output contains the visually checked 1600x900 trusted-mode screenshot and immutable inspection record under `artifacts/bmw_trusted_ok_reference_smoke/`; all persisted reference-file SHA values were recomputed successfully. Do not commit the local verification script, screenshots, copied HDR sources, reference images, or inspection records.
- Focused regression covered v3 config, trusted reference, UI, capture, persistence, Template, bright-streak, YOLO runtime/evidence, and EfficientAD: 212 tests passed when the three unrelated optional Ultralytics training-package pin/install assertions were omitted; direct YOLO runtime selection was 15 passed. A broader attempt reported 212 passed and those 3 known environment/repository-baseline failures. NVML/GPU was unavailable, and Task 5 intentionally did not open the GUI or four cameras; live hardware readiness remains for the parent launch step.

## BMW v3 NG-evidence Demo entrypoints (2026-08-12)

- `bmw_lab_prepare_v3_ng_evidence_demo.py` creates a no-overwrite composite from 21:00 Template, 21:00 EfficientAD-v2,
  unchanged baseline YOLO, and a deployment threshold asset with an explicit `+0.05` margin. The bright-streak-v2
  report is selected by `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`.
- `bmw_lab_eight_view_demo.py` now persists each completed inspection. Live runs save actual short/long/HDR sources;
  offline samples are labeled `fused_only`. The experiment UI uses `N/P` for the NG/ERROR evidence queue and saves
  structured result reasons, metrics, thresholds, overlays, ROI crops, capture-profile SHA/parameters, and an index
  below the configured result root. Offline UI panels explicitly state that short/long originals are unavailable.

## BMW trusted-OK review package (2026-08-12)

- `bmw_lab_prepare_trusted_ok_review.py` is a PENDING-only, atomic no-overwrite CLI for a human review queue. Its defaults bind the 21:00 prepared manifest and the ignored frozen v2 output `dataset/bmw_trusted_ok_review/bmw_right_20260810_21_train_normal_v2`; `--session-id` is mandatory.
- It delegates strict schema/filter/hash validation to `bmw_inspection.lab.trusted_ok_reference`. The current v2 package has 50 complete training-normal physical parts (400 images) and 50 Chinese-labelled contact sheets; it is the only future candidate package. Historical v1 is retained solely as an immutable audit artifact and must not be regenerated or published.

## BMW trusted-OK approved reference publisher (2026-08-12)

- `bmw_lab_publish_trusted_ok_reference.py` publishes only exact `APPROVED` decisions from the immutable review package. It validates the decision schema, decision-to-candidate binding, all approved source SHA-256 values, fixed eight-view ROI contract, and complete views before atomically creating a no-overwrite release.
- The fixed first release is `dataset/bmw_trusted_ok_reference/bmw_right_20260810_21_train_normal_approved_v1`. Its `trusted_ok_whitelist.json` binds the candidate and decision CSV hashes; `reference_index.json` binds the whitelist/ROI SHA and one copied full image plus RGB PNG ROI crop per part/view.
- User confirmation on 2026-08-12 changed the ignored review decisions for all 50 candidates to `APPROVED`, reviewer `user-confirmed-20260812`, note `用户确认50个全部OK`. Publication independently verified 50 parts, 400 index rows, 50 references per view, and SHA matches for 400 sources, 400 full copies, and 400 ROI files. This is curated normal-reference provenance, not model or production acceptance.
- P1 correction: v1's review package predates the frozen candidate-manifest binding and must not be used for later publishing. The producer now writes `candidate_manifest_sha256`; the publisher validates it before reading decisions. The only future candidate is v2: `dataset/bmw_trusted_ok_review/bmw_right_20260810_21_train_normal_v2`, whose 400 identity/source-SHA rows exactly match v1 and whose frozen SHA is `9ff29f52bf0bb63636558832b8da0ffc7244809b2d5f056aa124ba444559ce8e`.

## BMW 21-point Template-only diagnostic candidate (2026-08-11)

- `bmw_lab_train_template_fixed_thresholds.py` is the isolated offline entrypoint. It accepts only the five
  21-point sessions by default, extracts the source session from every crop filename, trains templates only from
  `train/normal`, reuses each morning `template/<view>/model.json` numerical threshold, and refuses an existing
  output directory.
- The candidate writes `template/<view>` models plus `template_report.json` with normal-only calibration/final-test
  per-view metrics and eight-view physical-part pass rates. It does not update `bmw_eight_view_demo_v1.json`.
- The real fixed-threshold candidate reports 13/21 calibration parts passing and 8/20 final-test parts passing; changing only the template source did not fix the low whole-part pass rate. `front_right` and `front_secondary` are the dominant final-test rejection views.
- The corrected 21:00 bright-streak candidate uses `--roi-xyxy 1792 1180 1873 1793` by default and writes `bmw_right_batch_20260810_21_bright_v2_roi_corrected`. Its held-out result is 19/20 versus 10/20 for the unchanged current detector; broken-but-present evidence is still absent.
- `bmw_lab_prepare_template_21only_demo.py` creates a no-overwrite composite run for live single-variable testing: candidate `template`, baseline `efficientad`, and baseline `yolo`. Launch it with `bmw_eight_view_demo_template_21only_v1.json`; do not edit the default Demo config.
- `bmw_lab_prepare_efficientad_bright_v2_demo.py` creates the second no-overwrite live-test composition: baseline Template/YOLO plus 21:00 EfficientAD. Launch it with `bmw_eight_view_demo_efficientad_bright_v2_v1.json`; this config also selects the SHA-bound corrected-ROI raw-profile bright-streak v2 report.
- The v3 NG-evidence Demo now optionally loads the strictly approved-v2 trusted-OK bank from `trusted_ok_reference.index` plus `index_sha256`. `pipeline/bmw_lab_eight_view_demo.py` prints the Chinese preload status, keeps running the four detector branches if reference preload fails, and uses `O` only to toggle diagnostic comparison after a reference-enabled actionable inspection. `N` remains next and `P` remains previous.
- Trusted-reference output is keyed by `<view>/<mode>` and no-overwrite under each ordinary capture record at `references/<view>/<mode>/`; bright streak uses `front_left/full`, while Template/YOLO/EfficientAD use ROI mode. `inspection.json` records the exact selected part/sample, similarity, shift, original source/reference/index/whitelist hashes, saved artifact hashes, and `reference_is_diagnostic_only=true`. It never changes any of the 25 detector rows or final fusion.

## BMW tracked-profile bright-streak v3 Demo handoff (2026-08-12)

- `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json` pins ignored report
  `results/bmw_lab_one_click/bmw_right_batch_20260810_21_bright_v3_tracked_v8/report.json` at SHA-256
  `6b43690af67702333646fa7a88a2a5053a0afae6d19cb343bf6d3abb193c17d4`; only its `bright_streak.engine`,
  `.config`, and `.config_sha256` fields changed. The rollback engine is `raw_profile_v2` with corrected-v2 report SHA
  `24250c00b8518e3c886673815b696f0a4b6cfd5ffdb0fb1dc7d2fd8a5799268d`.
- Runtime geometry remains fixed to ROI `[1792,1180,1873,1793]`; selected candidate width is 5, max step 2, strong
  threshold `136.85`, weak threshold `128.20`, minimum coverage `0.0848287113`, minimum run `0.0440456770`, maximum
  gap `0.1060358891`, and maximum gap count 2. Strong/bridged/gap path colors are green/orange/red diagnostic
  centreline evidence rather than segmentation truth.
- Real saved HDR `bmw_demo_20260812_211302` is `PASS/OK`: coverage `0.1076672104`, longest run `0.0554649266`,
  max gap `0.0032626427`, one gap, 20 bridged rows. The report records 8/8 no-streak NG, calibration normal false
  rejects 0, final-test normal false rejects 1 (not worse than v2), and a 20-record replay with 19 truth-unknown rows.
- Preserve the 25-row contract and exact equality of all 24 non-bright tuples across the engine switch. The report has
  `real_broken_samples=0`; synthetic broken-path checks do not establish real broken-streak recall.
