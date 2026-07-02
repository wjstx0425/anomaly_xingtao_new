# AGENTS Memory

## GitHub upload guardrails

- On 2026-07-02, before uploading local code to GitHub, the checkout had large local artifacts under `results/` (~70G), `dataset/` (~79G), and `c789_bottom/` (~3G).
- `.gitignore` already ignored `results`, `dataset/`, `datasets`, and training logs such as `wandb/`, `lightning_logs/`, and `mlruns`.
- Added `c789_bottom/` and `*.ckpt` to `.gitignore` so local model checkpoints are not accidentally staged by `git add .`.
- Push local code changes to the `mygithub` remote (`git@github.com:wjstx0425/anomaly_xingtao.git`) rather than the upstream `origin` remote unless the user explicitly wants to contribute to upstream anomalib.

## C789 left_top geometry + AnomalyDINO fusion

- User override for this workflow: use `.venv/bin/python`, not `uv`, even though `AGENTS.md` mentions uv.
- Do not overwrite `results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks` unless the user explicitly asks.
- Pytest is not available in the current `.venv`; use `compileall` and direct Python/CLI harnesses for validation.
- Current manual-mask workflow paths:
  - Mask input: `results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks`
  - Manual templates: `results/c789_100_hardened/left_top_geometry/manual_templates`
  - Stress locked eval: `results/c789_100_hardened/left_top_geometry/manual_stress_locked`
  - Defect fused eval: `results/c789_100_hardened/left_top_geometry/manual_defect_fused`
- Manual masks are complete for `slot01` through `slot06`: each has `expected`, `allowed`, `ignore`, and `watch_edge` PNGs. They appeared generated around the same time as the review sheets on 2026-06-20, so treat them as seed masks unless the user confirms hand edits.
- Fresh commands run on 2026-06-20:
  - `.venv/bin/python pipeline/15_edit_geometry_masks.py --review-pack results/c789_100_hardened/left_top_geometry/manual_review_pack --template-dir results/c789_100_hardened/left_top_geometry/manual_templates --stress-root dataset/c789_stress_normal_group_split/locked/left/top --defect-root dataset/c789_100_left_top_parts/left/top/defect --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv --stress-output-dir results/c789_100_hardened/left_top_geometry/manual_stress_locked --defect-output-dir results/c789_100_hardened/left_top_geometry/manual_defect_fused`
  - `.venv/bin/python pipeline/14_build_manual_geometry_templates.py --mask-dir results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks --output-dir results/c789_100_hardened/left_top_geometry/manual_templates`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_stress_normal_group_split/locked/left/top --template-dir results/c789_100_hardened/left_top_geometry/manual_templates --output-dir results/c789_100_hardened/left_top_geometry/manual_stress_locked --calibrate-thresholds`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_100_left_top_parts/left/top/defect --template-dir results/c789_100_hardened/left_top_geometry/manual_templates --thresholds results/c789_100_hardened/left_top_geometry/manual_stress_locked/geometry_thresholds.csv --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv --output-dir results/c789_100_hardened/left_top_geometry/manual_defect_fused`
- Visual editor validation note: in `pipeline/15_edit_geometry_masks.py`, `v` should rebuild `manual_templates`, recalibrate locked stress normal thresholds, and rerun defect fusion against `manual_defect_fused`; keep masks pure black/white, and rely on editor backups before save overwrites.
- Fresh metrics from those outputs:
  - Stress locked geometry FP: `0/90`.
  - AnomalyDINO defect recall: `8/17`.
  - Geometry defect positives: `13/17`.
  - Fused defect recall: `15/17`.
  - Fused misses: `less_1_2_slot02`, `corner_3_1_slot05`.
  - Key samples:
    - `less_1_2_slot02`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r00_c05`, score `2722.0`, threshold `5817.0`, missing `186`, extra `2722`.
    - `less_2_1_slot03`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r03_c06`, score `986.0`, threshold `954.45`, missing `150`, extra `24449`.
    - `more_2_2_slot04`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r02_c00`, score `3993.0`, threshold `127.05`, missing `0`, extra `11379`.
- Implementation caveat: default geometry threshold calibration is per slot/type/coarse region. A defect in a region/type not seen in stress-normal calibration can be ignored unless a slot or wildcard fallback threshold exists.
- Review-pack visual observations:
  - `slot02` currently has a broad full-outline `watch_edge`; rescuing `less_1_2_slot02` likely needs a more precise watch edge around the actual missing boundary and less tolerance around unrelated bright/fixture areas.
  - `slot03` and `slot04` seed masks include visible non-part fixture/background structures in `allowed`/`watch_edge`; remove these from part geometry or put them into `ignore` before treating the masks as final.
- Fresh verification: `.venv/bin/python -m compileall` on the geometry scripts and pipeline wrappers exited with code 0.

## C789 left_top high-exposure geometry seed

- On 2026-06-21, generated a separate high-exposure seed flow from:
  - `dataset/c789_100/left/top/normal/part001_20260610_105404/raw_exposures/no_hand_top_normal_part001_g001_000000_exp35000.png`
- This flow intentionally does not overwrite the active manual masks under:
  - `results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks`
- High-exposure seed artifacts:
  - Slot crops: `results/c789_100_hardened/left_top_geometry/high_exp_seed_crops`
  - Review pack and editable seed masks: `results/c789_100_hardened/left_top_geometry/high_exp_review_pack`
  - Compiled templates: `results/c789_100_hardened/left_top_geometry/high_exp_manual_templates`
  - Stress calibration: `results/c789_100_hardened/left_top_geometry/high_exp_stress_locked`
  - Defect fusion: `results/c789_100_hardened/left_top_geometry/high_exp_defect_fused`
- Important high-exposure seed rule: when exporting the review pack, do not pass `--template-dir`; otherwise the editable masks are copied from the old templates instead of being generated from the high-exposure reference crop.
- Commands used:
  - `.venv/bin/python pipeline/13_export_geometry_review_pack.py --normal-root results/c789_100_hardened/left_top_geometry/high_exp_seed_crops --stress-root dataset/c789_stress_normal_group_split/locked/left/top --defect-root dataset/c789_100_left_top_parts/left/top/defect --output-dir results/c789_100_hardened/left_top_geometry/high_exp_review_pack --preset c789_left_top_3x2 --samples-per-split 2`
  - `.venv/bin/python pipeline/14_build_manual_geometry_templates.py --mask-dir results/c789_100_hardened/left_top_geometry/high_exp_review_pack/manual_masks --output-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_stress_normal_group_split/locked/left/top --template-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates --output-dir results/c789_100_hardened/left_top_geometry/high_exp_stress_locked --calibrate-thresholds`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_100_left_top_parts/left/top/defect --template-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates --thresholds results/c789_100_hardened/left_top_geometry/high_exp_stress_locked/geometry_thresholds.csv --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv --output-dir results/c789_100_hardened/left_top_geometry/high_exp_defect_fused`
- Fresh high-exposure seed metrics:
  - Stress locked geometry FP: `0/90`.
  - AnomalyDINO defect recall: `8/17`.
  - Geometry defect positives: `10/17`.
  - Fused defect recall: `13/17`.
  - Fused misses: `less_1_2_slot02`, `less_2_1_slot03`, `corner_3_1_slot05`, `surface_3_1_slot05`.
  - Key samples:
    - `less_1_2_slot02`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r00_c06`, score `435.0`, threshold `2197.65`, missing `161`, extra `821`.
    - `less_2_1_slot03`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r02_c02`, score `957.0`, threshold `1004.85`, missing `90`, extra `28888`.
    - `more_2_2_slot04`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r00_c06`, score `512.0`, threshold `385.35`, missing `1415`, extra `19756`.
- Interpretation: high-exposure seed is useful as a cleaner visual/manual-edit starting point, but this unedited seed is not better than the previous best manual output because fused recall is `13/17` instead of `15/17`.

## C789 left_top high-exposure corner-only strategy

- On 2026-06-21, applied the corner-only strategy to the active high-exposure masks:
  - Active mask dir overwritten by request: `results/c789_100_hardened/left_top_geometry/high_exp_review_pack/manual_masks`
  - Backup before overwrite: `results/c789_100_hardened/left_top_geometry/high_exp_review_pack/manual_masks_backup/20260621_170113_corner_strategy_before_apply`
  - Final candidate source: `results/c789_100_hardened/left_top_geometry/high_exp_review_pack/corner_strategy_candidate_v5/manual_masks`
- The non-high-exp manual masks were not targeted:
  - `results/c789_100_hardened/left_top_geometry/manual_review_pack/manual_masks`
- Final mask semantics after applying the strategy:
  - `expected`: unchanged high-exp body masks, still full enough for alignment.
  - `watch_edge`: constrained to four corner windows, no longer full long-edge contour.
  - `allowed`: `expected` plus previous high-exp allowed tolerance only near the new four-corner watch area.
  - `ignore`: unchanged high-exp ignore masks.
- Final high-exp corner-only mask areas:
  - `slot01`: expected `369978`, allowed `400939`, ignore `39738`, watch_edge `234698`.
  - `slot02`: expected `428527`, allowed `457282`, ignore `39738`, watch_edge `225007`.
  - `slot03`: expected `484736`, allowed `515733`, ignore `39738`, watch_edge `237915`.
  - `slot04`: expected `491914`, allowed `520151`, ignore `39738`, watch_edge `230774`.
  - `slot05`: expected `559056`, allowed `584887`, ignore `39738`, watch_edge `195492`.
  - `slot06`: expected `570722`, allowed `597418`, ignore `39738`, watch_edge `199231`.
- Commands rerun after applying masks:
  - `.venv/bin/python pipeline/14_build_manual_geometry_templates.py --mask-dir results/c789_100_hardened/left_top_geometry/high_exp_review_pack/manual_masks --output-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_stress_normal_group_split/locked/left/top --template-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates --output-dir results/c789_100_hardened/left_top_geometry/high_exp_stress_locked --calibrate-thresholds`
  - `.venv/bin/python pipeline/12_geometry_eval.py --data-root dataset/c789_100_left_top_parts/left/top/defect --template-dir results/c789_100_hardened/left_top_geometry/high_exp_manual_templates --thresholds results/c789_100_hardened/left_top_geometry/high_exp_stress_locked/geometry_thresholds.csv --anomaly-predictions results/c789_100_hardened/left_top_anomaly_dino/reports/predictions.csv --output-dir results/c789_100_hardened/left_top_geometry/high_exp_defect_fused`
- Fresh metrics after applying corner-only high-exp masks:
  - Stress locked geometry FP: `0/90`.
  - Geometry defect positives: `12/17`.
  - AnomalyDINO defect recall: `8/17`.
  - Fused defect recall: `14/17`.
  - Fused misses: `less_2_1_slot03`, `more_2_2_slot04`, `crack_3_2_slot06`.
  - Key samples:
    - `less_1_2_slot02`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r00_c02`, score `821.0`, threshold `168.0`.
    - `less_2_1_slot03`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r02_c02`, score `301.0`, threshold `316.05`.
    - `more_2_2_slot04`: anomaly `0`, geometry `0`, final `0`, type `more`, region `r00_c01`, score `8582.0`, threshold `9059.4`.
    - `corner_3_1_slot05`: anomaly `0`, geometry `1`, final `1`, type `more`, region `r00_c06`, score `11062.0`, threshold `4590.6`.
- Interpretation: corner-only high-exp masks are more aligned with the intended edge/corner inspection target and improved fused recall from `13/17` to `14/17` while keeping stress FP at `0/90`, but they no longer rescue `more_2_2_slot04` and still miss `less_2_1_slot03`.
