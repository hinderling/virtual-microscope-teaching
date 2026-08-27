"""Regenerate the documentation screenshots from the live GUI.

Usage:  .venv/bin/python scripts/make_screenshots.py

Captures:
  docs/images/napari_gui.png        — GUI with MM toolbars + preview snap
  docs/images/results_explorer.png  — show_results layer stack (split steering)

Runs a real (briefly visible) napari window; total runtime ~1 min.
"""

import cv2
import numpy as np
from qtpy.QtCore import QTimer

from vmteach import load_microscope, advance
from vmteach.gui import launch_gui, show_results

OUT = "docs/images"
WINDOW = (1500, 950)


def detect(img):
    _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells, seg, lbl = [], np.zeros_like(img, np.int32), 1
    for c in cts:
        if cv2.contourArea(c) < 100:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cells.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
        cv2.drawContours(seg, [c], -1, lbl, -1)
        lbl += 1
    return cells, seg


def shot_gui():
    import napari
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0)
    viewer = launch_gui(core)

    def act():
        viewer.window._qt_window.resize(*WINDOW)
        core.setConfig("Channel", "phase-contrast")
        core.snapImage()
        QTimer.singleShot(1200, lambda: (
            viewer.window.screenshot(f"{OUT}/napari_gui.png", canvas_only=False),
            viewer.close()))

    QTimer.singleShot(3000, act)
    napari.run()


def shot_results():
    import napari
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0)
    core.setConfig("Channel", "phase-contrast")
    imgs, masks, cents, segs = [], [], [], []
    for _ in range(80):
        core.snapImage()
        img = core.getImage()
        cells, seg = detect(img)
        mask = np.zeros((512, 512), np.uint8)
        for cx, cy in cells:
            dy = -15 if cx < 256 else 15
            cv2.circle(mask, (cx, int(np.clip(cy + dy, 0, 511))), 11, 255, -1)
        core.setSLMImage("SLM", mask)
        advance(sim, 1.0)
        imgs.append(img.copy())
        masks.append(mask)
        cents.append(cells)
        segs.append(seg)

    viewer = show_results(imgs, masks=masks, centroids=cents,
                          segmentations=segs, name="split steering")

    def act():
        viewer.window._qt_window.resize(*WINDOW)
        viewer.dims.set_current_step(0, 79)
        QTimer.singleShot(800, lambda: (
            viewer.window.screenshot(f"{OUT}/results_explorer.png",
                                     canvas_only=False),
            viewer.close()))

    QTimer.singleShot(2500, act)
    napari.run()


if __name__ == "__main__":
    shot_gui()
    shot_results()
    print("screenshots written to", OUT)
