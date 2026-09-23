"""OpenCV reference-only GrabCut editor with zoom, pan and undo."""
from __future__ import annotations

import cv2
import numpy as np

from .contracts import ContourInputError


def edit_reference(image, config):
    """Return operator-reviewed mask and ROI; never used during inspection.

    R: rectangle, F/B: definite foreground/background brush, G: GrabCut,
    U: undo, +/- or wheel: zoom, middle drag: pan, S: accept, Esc: cancel.
    """
    bgr = image[:, :, :3].copy()
    height, width = bgr.shape[:2]
    window = "Contour teach | R ROI F foreground B background G solve U undo +/- zoom S save Esc cancel"
    vw, vh = min(1280, width), min(850, height)
    scale = min(vw / width, vh / height)
    offset = np.zeros(2, dtype=float)
    labels = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    roi = config.get("part_roi_xyxy")
    if roi is not None:
        x1, y1, x2, y2 = roi
        labels[y1:y2, x1:x2] = cv2.GC_PR_FGD
    history = []
    state = {"mode": "r", "drawing": False, "start": None, "pan": None}
    def location(x, y):
        return tuple(np.clip(np.rint((np.array([x, y]) - offset) / scale), [0, 0], [width-1, height-1]).astype(int))
    def callback(event, x, y, flags, _):
        nonlocal scale, roi
        if event == cv2.EVENT_MOUSEWHEEL:
            old = scale
            scale = np.clip(scale * (1.2 if flags > 0 else 1/1.2), .05, 8)
            offset[:] = np.array([x, y]) - (np.array([x, y])-offset) * scale/old
        elif event == cv2.EVENT_MBUTTONDOWN:
            state["pan"] = np.array([x, y])
        elif event == cv2.EVENT_MBUTTONUP:
            state["pan"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["pan"] is not None:
            offset[:] += np.array([x, y])-state["pan"]
            state["pan"] = np.array([x, y])
        elif event == cv2.EVENT_LBUTTONDOWN:
            history.append((labels.copy(), roi))
            state["drawing"] = True
            state["start"] = location(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if state["drawing"] and state["mode"] == "r":
                a, b = state["start"], location(x, y)
                roi = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0])+1, max(a[1], b[1])+1]
                labels[:] = cv2.GC_PR_BGD
                labels[roi[1]:roi[3], roi[0]:roi[2]] = cv2.GC_PR_FGD
            state["drawing"] = False
        if state["drawing"] and state["mode"] in ("f", "b"):
            cv2.circle(labels, location(x, y), max(1, int(5/scale)), cv2.GC_FGD if state["mode"] == "f" else cv2.GC_BGD, -1)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, vw, vh)
    cv2.setMouseCallback(window, callback)
    try:
        while True:
            display = bgr.copy()
            foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
            contours, _ = cv2.findContours(foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 1)
            display[labels == cv2.GC_FGD] = (0, 200, 100)
            display[labels == cv2.GC_BGD] = (50, 50, 230)
            matrix = np.array([[scale, 0, offset[0]], [0, scale, offset[1]]])
            cv2.imshow(window, cv2.warpAffine(display, matrix, (vw, vh)))
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                raise ContourInputError("reference teaching cancelled")
            if key in (ord("r"), ord("f"), ord("b")):
                state["mode"] = chr(key)
            elif key == ord("u") and history:
                saved, roi = history.pop()
                labels[:] = saved
            elif key in (ord("+"), ord("="), ord("-")):
                scale = np.clip(scale * (1/1.2 if key == ord("-") else 1.2), .05, 8)
            elif key == ord("g"):
                history.append((labels.copy(), roi))
                cv2.setRNGSeed(0)
                cv2.grabCut(bgr, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
            elif key == ord("s"):
                if roi is None:
                    raise ContourInputError("draw part ROI before saving")
                return foreground.astype(np.uint8) * 255, roi
    finally:
        cv2.destroyWindow(window)
