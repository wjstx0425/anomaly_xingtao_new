# Pipeline Memory

## BMW v3 NG-evidence Demo entrypoints (2026-08-12)

- `bmw_lab_prepare_v3_ng_evidence_demo.py` creates a no-overwrite composite from 21:00 Template, 21:00 EfficientAD-v2,
  unchanged baseline YOLO, and a deployment threshold asset with an explicit `+0.05` margin. The bright-streak-v2
  report is selected by `configs/bmw/experiments/bmw_eight_view_demo_v3_ng_evidence.json`.
- `bmw_lab_eight_view_demo.py` now persists each completed inspection. Live runs save actual short/long/HDR sources;
  offline samples are labeled `fused_only`. The experiment UI uses `N/P` for the NG/ERROR evidence queue and saves
  structured result reasons, metrics, thresholds, overlays, ROI crops, capture-profile SHA/parameters, and an index
  below the configured result root. Offline UI panels explicitly state that short/long originals are unavailable.

## BMW trusted-OK review package (2026-08-12)

- `bmw_lab_prepare_trusted_ok_review.py` is a PENDING-only, atomic no-overwrite CLI for a human review queue. Its defaults bind the 21:00 prepared manifest and the ignored `dataset/bmw_trusted_ok_review/bmw_right_20260810_21_train_normal_v1` output; `--session-id` is mandatory.
- It delegates strict schema/filter/hash validation to `bmw_inspection.lab.trusted_ok_reference`. The current actual package has 50 complete training-normal physical parts (400 images), 50 Chinese-labelled contact sheets, and no automatic approvals; reviewers must update decisions in a later explicit workflow.

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
