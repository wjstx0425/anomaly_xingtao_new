# BMW tools memory

## Tool documentation organization — 2026-09-23

- Expanded README from stamp-only heading to a BMW tool index covering contour geometry, holes, algorithm comparison and imaging diagnostics. Script locations and behavior preserved; generated evidence stays local. BMW unit suite: 587 passed.

## Geometry script now uses integration API — 2026-09-23

- `measure_front_geometry.py` delegates reference preparation and per-image pose/geometry/acquisition to `FrontContourInspector`; retains manifest SHA/metadata validation, CSV/NPZ/JSON and visual reports. No separate measurement logic should be added back to the script. API docs: `docs/bmw/front_contour_api.md`. Two-image real parity passed; CLI elapsed_ms includes I/O while API elapsed_ms is computation only.

## Frozen BMW algorithm comparison — 2026-09-12

- Added run_algorithm_comparison.py for resolveddevelopment_only E0–E2 plan. ReusesruntimeTemplate/EfficientAD/YOLO andcontinuousR01cache/nativepixelR02a; existingoutputdirectoryrefused,inputsSHAchecked,failuresERRORpreserve48plannedrecords. No labels passed toinference. User task docs/BMW_第1-10项_算法对照实验_Codex交接包_V1.1. Actual8×6output results/bmw_benchmark/left_front_e2_20260912 has0ERROR;546BMW+pipeline tests pass. Report docs/bmw/algorithm_comparison_e2_20260912.md; detailedsemanticmemory src/bmw_inspection/benchmark/AGENTS_MEMORY.md. R01currentwall iscacheimport,notalgorithmruntime. Templateoverlayrequiresalignedtarget->fullmapping,paddingexcluded; R02normalcandidates1211–1313notcracks. Freezeoriginalassets,noofficialleaderboardwithunknownentity/trainingoverlap/partiallabels.

## Measured reflection reference — 2026-09-12

- User supplied actual pair: `Image_20260912201543028.bmp` HAS PART, `Image_20260912201549452.bmp` is reflection-only; visually confirmed order. New helper `subtract_reflection_template.py` subtracts scale*template in float32, saves clipped PNG + signednpz,80%candidate,saturation/negative masks,HTML/manifest. No registration or inpainting.
- Output `results/bmw_reflection_template/20260912_2015_measured/`; guide `docs/bmw/measured_reflection_template_20260912.md`. Common leftROI[300,500,520,2500] source/template mean42.95/43.37,MAE1.59,correctedmean.63. Background white veil removal effective. True clean-surface comparison unavailable.
- Critical source all25519.08%,any>=25020.33%,template0. Subtraction creates inverted bright-circle ghost in alreadyclipped areas; do not claim recovered detail or fix by filling. ~77.7%ofselected centralbrightresidual was all255 before. Need matched lower-exposure/gain pair,not new exposure image plus old reference.
- Independent background registration small ambiguous dx1–2/dy13–18 gave minorMAE improvement; not adopted. Inputs unchanged,allPNGhashes/formulas/signedresidual exact,synthetic subtraction/negative/zero/invalidscale verified. No live hardware/production acceptance.

## Fixed rectangular reflection field follow-up — 2026-09-12

- User challenged earlier generic CLAHE/normalization limitations; correctly motivated structured additive field fit. `docs/bmw/glare_fixed_field_20260912.md` supersedes blanket no-removal interpretation. Fit soft rectangular15param field on presumed dark background of firstBMP using scipy soft_l1, apply identical field to second. Reproducer/artifacts `artifacts/bmw_glare_fixed_field_20260912/run_model.py` and outputs/index.html.
- Rectangle visibly attenuated. Right background median86->28 at65% subtraction,0 at full,17 with guarded max(I-R,.2I). Full creates26.66%new nearblack pixels(sourcegray>=8/output<=1), because circular region/scene field differs. Guarded creates0%under this threshold but retains reflection and changes contrast. Background holdout error2.62,second-image2.71 counts; not calibrated reflection truth or detection acceptance.
- Exact field.npy,params/zones,masks,originalclipmask,8full-size outputs/formula/hash verification saved; originals unchanged. Do not claim80%reflection recovery from80%background dimming. No inpaint or circle erasure.
- Pending user question: does rectangle remain in same sensor position when part moves? For true fixed additive layer, controlled black-scene reference may estimate full field if reflection source itself unchanged. If follows metal pose, requires pose-specific calibration. No live reference or hardware acceptance yet.

## Rectangular glare feasibility — 2026-09-12

- User BMPs `dataset/Image_20260912195524733.bmp` and `Image_20260912195528653.bmp`. Actual full-res OpenCV bilateral-base illumination normalization and LAB CLAHE tried. Bright rectangle/dark circle persist; normalization dims glare but does not recover surface, CLAHE enhances stamp/texture/noise. No inpainting or image-source modifications.
- All3channels255 about4.82% each;97.07%ofAclipped overlapsB, gray correlation.99928; practically no complementary highlight information. Do not classify all high pixels as glare or erase unknown dark-circle structure.
- Report `docs/bmw/glare_trial_20260912.md`; reproducer `artifacts/bmw_glare_trial_20260912/run_trial.py`; results `outputs/index.html`, fullPNG, crops, source/outputhashes and verification. Source hashes and originalPNG equality verified; no actual hardware or defect/OCR acceptance.

## Additional 6 / 8 / 10 px simulation — 2026-09-12

- Same source and confirmed rightward direction. Helper now accepts `--pixels 6 8 10`; default remains1..5, original0 included. Dynamic report labels/layout preserve actual displacement values.
- Output `results/bmw_motion_blur/20260912_front_750us_right_6_8_10px/` contains0/6/8/10full-size PNGs, same2ROIs/4x sheets,HTML/manifest/verification. Source hash unchanged, all output hashes/shapes and kernel equality verified; synthetic6/8/10impulse support and centroid checked. Earlier1..5outputs retained.

## Horizontal motion simulation — 2026-09-12

- User selected `results/bmw_exposure/20260912_173223_066379/raw/front/r001_e01.png`, explicitly confirmed horizontal right. Source4024x3036, recorded750us/gain5, SHA25692a8469b1324bb6b5e70bb170f968bf849c67e594ec4645150e3567181390f65.
- `simulate_motion_blur.py` generates original plus1..5px added motion using OpenCV filter2D, no new dependencies. Uniform displacement0..L integrated through linear interpolation: L+1tap [.5,1,...,1,.5]/L, anchor(L,0), replicate borders.1px [.5,.5] must blur; centroid shifts rightL/2. This is numerical image-space simulation, not a generative illustration.
- Run: `UV_CACHE_DIR=/tmp/bmw-exposure-uv-cache uv run --no-sync python tools/bmw/simulate_motion_blur.py --image results/bmw_exposure/20260912_173223_066379/raw/front/r001_e01.png --output NEW_DIR --roi 3080 850 3208 978 --roi 1330 1210 1458 1338`.
- Completed output `results/bmw_motion_blur/20260912_front_750us_right_1to5px/`:6full PNG,2ROIs with4x nearest-neighbor six-panel sheets,HTML,manifest,verification.json. Source unchanged/hash checked;0px exact pixel copy; all outputs shape/dtype/hash checked. Synthetic constants, rightward impulse support/centroids and1px blur passed; independent agent verified kernel model.
- Apply in saved PNG values (unknown camera response), includes existing blur/noise and blurs whole frame/background. No new noise/sharpening/white-balance changes; not a calibrated physical prediction or detection acceptance. Do not infer permissible speed without physical pixel scale and real motion validation.
