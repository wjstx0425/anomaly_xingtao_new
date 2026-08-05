# BMW Demo Chinese Industrial UI Redesign

## Goal

Redesign the existing 1600x900 OpenCV customer Demo as a polished, fully Chinese, dark industrial dashboard without
changing camera acquisition, bright-streak decisions, thresholds, or result publication.

## Visual Direction

- Use a dark blue-black background with layered blue-gray cards and restrained cyan accents.
- Use green only for `OK`, red only for part-level NG, and amber for `ERROR`.
- Do not use an official BMW logo or imitate an official BMW application.
- The title is `BMW 零件亮痕检测演示系统`.
- All visible labels, reasons, metrics, status text, instructions, and buttons are Chinese.

## Layout

The canvas remains exactly 1600x900.

1. A 92-pixel header contains the title, connection indicator, camera serial `DA9625347`, and current local time.
2. The left card presents the complete source image with a high-visibility yellow ROI rectangle and the label
   `实时采集图像`.
3. The upper-right card presents a magnified ROI evidence image and the label `亮痕检测区域`.
   The narrow vertical ROI evidence is rotated 90 degrees clockwise for presentation, then fitted proportionally
   inside the card. The UI must not stretch it non-uniformly. Rotation applies only to the dashboard evidence view;
   detector coordinates, masks, saved ROI images, and decision metrics remain in the original camera orientation.
4. The middle-right result card presents one large business result:
   - `OK` -> `检测通过` / `亮痕存在且连续`;
   - `NG_NO_STREAK` -> `检测不通过` / `未检测到有效亮痕`;
   - `NG_BROKEN` -> `检测不通过` / `亮痕存在但不连续`;
   - `ERROR` -> `设备或图像异常` / a concise Chinese error category.
5. The right metric card presents four simplified rows: `亮痕覆盖率`, `连续率`, `最大断点`, and `对比度`.
   Coverage and continuity use progress bars; maximum gap uses pixels and a pass/fail hint; contrast uses
   `良好`, `偏低`, or `不可用` plus the numeric SNR.
6. The footer presents `检测完成 · 结果已保存` and three large buttons: `开始检测`, `重新检测`, and `退出`.
   Internal filesystem paths and raw English detector reasons are not displayed.

## Interaction

- Preserve Space as inspection, R as retry, Q/Escape as quit, and the existing clickable button hit regions.
- Add visual hover feedback without changing button actions.
- Preserve the no-live-preview behavior: one software-triggered image is acquired for each inspection action.
- Keep the OpenCV window resizable while rendering the source dashboard at 1600x900.
- Opening the live Demo must not acquire an image, run detection, publish a result, or show the previous result.
  It first renders a connected `等待检测` screen with an empty source/evidence surface. `开始检测` and Space are
  enabled; `重新检测` and R are disabled until one inspection has completed.
- After an inspection, `开始检测`, `重新检测`, Space, and R all acquire exactly one new frame and replace the current
  presentation. Quitting from the initial waiting screen exits successfully without requiring a result object.

## Rendering Boundary

- Continue rendering with OpenCV and NumPy; add no browser, Qt, web server, or deep-learning dependency.
- Isolate palette, geometry, Chinese status mapping, metric formatting, and text drawing into testable helpers inside
  `src/bmw_inspection/demo_app.py` or a small adjacent UI module if needed.
- Use only fonts already available on the host. Prefer a CJK-capable system font through Pillow when available and
  fail clearly during startup if no Chinese font can be loaded; do not render unreadable OpenCV placeholder glyphs.

## Verification

- Render deterministic OK, no-streak NG, broken NG, and ERROR screenshots at exactly 1600x900.
- Assert Chinese status mapping, simplified metric text, button hit regions, and absence of raw result paths from
  the rendered presentation surface.
- Run the focused BMW suite and one headless screenshot smoke using a labeled OK image.
- Real display scaling and mouse hover receive a manual local smoke; real USB/MVS capture remains site-only.
