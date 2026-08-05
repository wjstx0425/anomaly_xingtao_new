# BMW Single-Camera Bright-Streak Demo Design

Date: 2026-08-05

Status: approved design

Scope: customer-facing demo, not production release

## 1. Objective

Build a local OpenCV demo that uses only Hikvision camera serial `DA9625347` to capture one fixed-position,
fixed-lighting image of a BMW metal part and classify the required central bright streak.

The business decision contract is:

- `OK`: the bright streak exists and is continuous;
- `NG_NO_STREAK`: no valid bright streak exists;
- `NG_BROKEN`: a valid bright streak exists but contains a gap larger than the configured tolerance;
- `ERROR`: capture, image quality, configuration, ROI, or execution failed.

`ERROR` is never converted into a part-level NG decision.

The approved reference images are:

- `dataset/bmw/Image_20260805160156065.bmp`: `OK`, continuous bright streak;
- `dataset/bmw/Image_20260805160207779.bmp`: `NG_NO_STREAK`, no bright streak.

The demo uses traditional OpenCV operators only. It does not load a deep-learning model and does not reuse ZS32
model, ROI, threshold, release, or product semantics.

## 2. User Workflow

1. Start the local demo from the repository root.
2. The application enumerates cameras and opens exactly one device with serial `DA9625347`.
3. The interface shows `READY` only after the camera is exclusively open and configured.
4. The operator places the part in the fixed fixture.
5. The operator clicks `拍照检测` or presses Space.
6. The camera acquires one Mono8 image with fixed exposure and gain.
7. The detector evaluates image quality, extracts the configured ROI, measures the streak, and publishes one result.
8. The interface shows the source image, ROI evidence, metrics, final decision, and reason.
9. The application saves the source, evidence, and structured result, then returns to a state that accepts the next part.

The first version deliberately has no live video preview. A single software-triggered frame keeps capture and UI
state deterministic and reduces failure modes during the customer demonstration.

## 3. Architecture and File Boundaries

The BMW demo is isolated from `src/zs32_inspection`:

```text
src/bmw_inspection/
  contracts.py
  camera.py
  detector.py
  roi_selector.py
  demo_app.py

configs/bmw/
  bright_streak_demo.json

pipeline/
  bmw_select_bright_streak_roi.py
  bmw_bright_streak_demo.py

tests/unit/bmw_inspection/
  test_config.py
  test_detector.py
  test_camera.py
  test_demo_app.py
```

Responsibilities:

- `contracts.py`: immutable configuration and result records plus validation;
- `camera.py`: serial-bound single-camera lifecycle and one-frame acquisition;
- `detector.py`: pure image-to-result traditional vision logic, with no UI or SDK dependency;
- `roi_selector.py`: one-time interactive ROI selection and configuration update;
- `demo_app.py`: UI state, click/keyboard handling, evidence rendering, and result persistence;
- `pipeline/*.py`: thin repository entrypoints only.

The camera layer reuses the tested MVS behaviors from `capture_data/collect_multicamera_dataset.py`: delayed SDK
loading, USB/GigE enumeration, serial verification, exposure range checking, software trigger, Mono8 conversion, and
complete cleanup. It does not reuse that file's three-camera CLI or ZS32 view semantics.

## 4. Camera Contract

The camera contract is intentionally strict:

- requested serial is exactly `DA9625347`;
- zero or multiple matches produce `ERROR`;
- there is no device-index fallback;
- `ExposureAuto` and `GainAuto` are disabled;
- exposure and gain come from the JSON configuration;
- trigger mode is enabled with software trigger source;
- one or two warm-up frames may be discarded at startup;
- each inspection triggers and reads exactly one frame;
- the accepted frame is Mono8 with configured dimensions `4024 x 3036`;
- timeout, decode, or save failure produces `ERROR`;
- exit and error paths stop grabbing, restore trigger mode, close the device, and destroy the SDK handle.

The initial exposure is configurable. Historical values from another product are not copied into the BMW contract;
the value is fixed during the on-site image check and then kept unchanged for the demo.

## 5. ROI Selection

`bmw_select_bright_streak_roi.py` loads the approved OK image by default and uses an OpenCV drag rectangle:

1. display the complete image;
2. let the user drag the smallest practical rectangle containing the entire expected streak and limited surrounding
   metal background;
3. reject empty, out-of-bounds, or edge-only regions;
4. show a magnified ROI confirmation;
5. write `roi_xyxy` to `configs/bmw/bright_streak_demo.json` after confirmation.

The ROI must exclude the bright outer silhouette and embossed edges because those are not the business streak. The
configuration stores half-open pixel coordinates `[x1, y1, x2, y2]` and the source image dimensions.

## 6. Detection Algorithm

The fixture and light are fixed, so the first version uses the configured ROI directly. It does not add full-image
registration. A lightweight location guard verifies that expected stable metal edges around the ROI remain within the
configured tolerance; failure is `ERROR`, not streak NG.

Processing steps:

1. validate image type and dimensions;
2. evaluate ROI mean intensity, dark/bright clipping ratio, and Laplacian sharpness;
3. crop the configured bright-streak ROI;
4. normalize low-frequency illumination using a local background estimate;
5. compute a white top-hat or equivalent `gray - local_background` response whose transverse kernel is wider than
   the expected streak;
6. derive a robust threshold from the ROI response median and median absolute deviation;
7. remove small components and components outside the expected center corridor, orientation, width, or area range;
8. apply only a small along-streak closing operation, limited to the explicitly allowed micro-gap;
9. collapse the accepted binary mask across its width into a one-dimensional along-streak presence sequence;
10. calculate the decision metrics and evidence overlay.

Required metrics:

- `contrast_snr`;
- `coverage_ratio`;
- `longest_run_ratio`;
- `max_gap_px` and `max_gap_ratio`;
- `gap_count`;
- `mean_width_px`;
- `lateral_offset_px`;
- image-quality measurements;
- capture, processing, and total elapsed time.

Decision order:

```text
quality or execution failure
  -> ERROR

contrast_snr < min_contrast_snr
or coverage_ratio < min_coverage_ratio
  -> NG_NO_STREAK

max_gap_px > allowed_max_gap_px
or longest_run_ratio < min_longest_run_ratio
or gap_count > max_gap_count
  -> NG_BROKEN

otherwise
  -> OK
```

The JSON configuration owns every threshold. Initial demo-only starting values are
`min_contrast_snr=4.0`, `min_coverage_ratio=0.60`, `min_longest_run_ratio=0.50`,
`max_gap_ratio=0.05`, and `max_gap_count=1`. ROI selection converts the ratio-based gap limit to pixels. These values
are tuning starting points, not production acceptance limits.

## 7. UI Design

The application uses one 1600 x 900 OpenCV window and the same drawn hit-region pattern already used by the ZS32
Dashboard, avoiding dependency on Qt-only OpenCV buttons.

Layout:

```text
+------------------------------------------------------------------+
| BMW bright-streak inspection | DA9625347: connected | result OK  |
+--------------------------------------+---------------------------+
|                                      | magnified ROI             |
| captured source image                | green: accepted streak    |
| yellow: configured ROI               | red: detected gaps        |
|                                      +---------------------------+
|                                      | coverage / longest run    |
|                                      | maximum gap / contrast    |
+--------------------------------------+---------------------------+
| [Capture / Space]       [Retry / R]              [Quit / Q]      |
+------------------------------------------------------------------+
```

UI states are `STARTING`, `READY`, `CAPTURING`, `ANALYZING`, `OK`, `NG_NO_STREAK`, `NG_BROKEN`, and `ERROR`.
Green is used only for `OK`, red for both NG decisions, amber for `ERROR`, and gray for non-terminal states. The first
error or business reason is always visible; the UI never displays a generic success while evidence is incomplete.

## 8. Evidence and Result Persistence

Every inspection is written atomically to:

```text
results/bmw_bright_streak_demo/YYYYMMDD/<timestamp>/
  source.png
  roi.png
  response.png
  mask.png
  evidence.png
  result.json
```

`result.json` contains timestamp, camera serial, exposure, gain, image shape, ROI, configuration identity, all
metrics, all thresholds, final status, reason, artifact paths, and timings. Source and evidence images are lossless.
A temporary directory is renamed to the final result directory only after all required files are written.

## 9. Error Handling

The following produce `ERROR` and a visible retry instruction:

- SDK unavailable;
- target serial missing, duplicated, or exclusively opened by another application;
- trigger or frame timeout;
- unsupported pixel format or incorrect dimensions;
- empty/out-of-bounds ROI;
- excessive underexposure, overexposure, or blur;
- part-position guard failure;
- evidence or result publication failure;
- non-finite metrics or invalid configuration.

Closing Hikvision MVS tools before the demo is an operating prerequisite because the application requires exclusive
camera access.

## 10. Verification and Acceptance

Offline acceptance:

- `Image_20260805160156065.bmp` returns `OK`;
- `Image_20260805160207779.bmp` returns `NG_NO_STREAK`;
- an explicitly test-only copy of the OK image with a removed middle streak segment returns `NG_BROKEN`;
- empty, incorrect-size, blurred, and clipped fixtures return `ERROR`;
- repeated runs of the same image produce identical status and metrics;
- saved evidence matches the source inspection identity.

Camera acceptance:

- only serial `DA9625347` can be opened;
- serial missing/duplicate, timeout, and cleanup paths are tested;
- one click produces exactly one accepted 4024 x 3036 frame;
- repeated inspections do not require restarting the application;
- click-to-result target is at most two seconds on the demonstration host;
- the application can close and reopen the camera cleanly.

The artificial broken-streak image verifies branch mechanics only. Reliable customer acceptance of `NG_BROKEN`
requires real broken-streak examples. Before presenting that branch as validated, collect at least several real
broken examples; 10 or more is preferred. This limitation is displayed in the engineering acceptance report but not
as a distracting operator UI warning.

## 11. Out of Scope

- deep-learning training or inference;
- live video preview;
- multi-camera capture;
- ZS32 runtime bundles or fusion profiles;
- PLC, MES, database, network API, or remote control;
- production threshold certification;
- automatic light-controller or strobe integration.
