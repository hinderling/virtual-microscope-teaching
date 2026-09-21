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

from vmteach import advance, detect_nuclei, load_microscope
from vmteach.gui import launch_gui, show_results

OUT = "docs/images"
WINDOW = (1500, 950)


def segment(nuclei_img):
    """Label mask from the miRFP nuclei image (for the segmentation layer).

    The phase image cannot be thresholded directly (its background is the
    bright well interior), which is exactly why the taught pipeline
    segments the nuclear marker channel."""
    _, b = cv2.threshold(nuclei_img, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg, lbl = np.zeros_like(nuclei_img, np.int32), 1
    for c in cts:
        if cv2.contourArea(c) < 20:
            continue
        cv2.drawContours(seg, [c], -1, lbl, -1)
        lbl += 1
    return seg


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
    imgs, masks, cents, segs = [], [], [], []
    for _ in range(80):
        core.setConfig("Channel", "miRFP")      # robust detection channel
        core.snapImage()
        nuc = core.getImage().copy()
        cells = detect_nuclei(nuc)
        core.setConfig("Channel", "phase-contrast")
        core.snapImage()
        img = core.getImage()
        mask = np.zeros((512, 512), np.uint8)
        for cx, cy in cells:
            dy = -15 if cx < 256 else 15
            cv2.circle(mask, (cx, int(np.clip(cy + dy, 0, 511))), 11, 255, -1)
        core.setSLMImage("SLM", mask)
        core.setConfig("Channel", "CyanStim")   # gated delivery
        advance(sim, 1.0)
        imgs.append(img.copy())
        masks.append(mask)
        cents.append(cells)
        segs.append(segment(nuc))

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
