# BMW New Dataset ROI Redefinition Design

## Goal

Replace the obsolete BMW bright-streak ROI with one manually selected against the newly captured fixed-camera data,
then use the same ROI and detector configuration for every new OK and NG image.

## Dataset Contract

- OK root: `dataset/bmw/OK` with 6 BMP images.
- NG root: `dataset/bmw/NG` with 7 BMP images.
- Every image is 4024x3036 Mono8.
- OK means the required bright streak is present and continuous.
- NG means the bright streak is completely absent.
- Camera, part, fixture, and light are fixed, so one ROI must serve all 13 images.

## ROI Workflow

1. Use `dataset/bmw/OK/Image_20260805172921398.bmp` as the default manual-selection reference.
2. Clear the obsolete ROI before selection so the runtime cannot silently reuse stale coordinates.
3. Display the reference inside the existing 1280x720 selection viewport.
4. Map the selected display rectangle back to half-open 4024x3036 source coordinates and atomically save it in
   `configs/bmw/bright_streak_demo.json`.
5. Do not create per-image ROIs or add image alignment for this fixed setup.

## Batch Acceptance

After the operator saves the ROI, replay the exact production detector over all 13 images. Acceptance requires:

- all 6 images under `dataset/bmw/OK` produce `OK`;
- all 7 images under `dataset/bmw/NG` produce `NG_NO_STREAK`;
- no image produces `ERROR` or `NG_BROKEN`;
- no filename-specific or path-specific branch is introduced.

Only editable rule thresholds may be tuned after the ROI is fixed. The camera serial remains `DA9625347`, and the
runtime continues to distinguish `ERROR` from part-level NG.

## Scope Boundary

This change updates the BMW selector default, the BMW configuration, focused tests, and project memory only. It does
not change ZS32 code, introduce deep learning, or perform a real USB/MVS capture.
