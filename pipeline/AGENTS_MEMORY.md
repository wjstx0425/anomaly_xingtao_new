# Pipeline Memory

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
