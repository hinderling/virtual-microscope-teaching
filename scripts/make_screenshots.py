"""Regenerate the documentation screenshots from the live GUI.

Usage:  .venv/bin/python scripts/make_screenshots.py

Captures:
  docs/images/napari_gui.png        — GUI with MM toolbars + preview snap
  docs/images/property_browser.png  — device property browser
  docs/images/results_explorer.png  — show_results layer stack (split steering)

Runs a real (briefly visible) napari window; total runtime ~1 min.
"""

import cv2
import numpy as np
from qtpy.QtCore import QTimer

from vmteach import advance, load_microscope
from vmteach.optogenetic import detect_nuclei
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
    core, sim = load_microscope("optogenetic", mode="stepped", n_cells=20, seed=0)
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


def annotate_button(img, tip, label, k=1.0):
    """Draw an arrow pointing up at ``tip`` (x, y) with a label below it."""
    color = (0, 190, 255)                       # BGR: amber
    x, y = tip
    tail = (x + int(120 * k), y + int(150 * k))
    cv2.arrowedLine(img, tail, (x, y + int(6 * k)), color, max(2, int(4 * k)),
                    cv2.LINE_AA, tipLength=0.18)
    scale, th = 0.9 * k, max(1, int(2 * k))
    (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    org = (tail[0] - tw // 2, tail[1] + tht + int(12 * k))
    pad = int(8 * k)
    cv2.rectangle(img, (org[0] - pad, org[1] - tht - pad),
                  (org[0] + tw + pad, org[1] + pad), (30, 30, 30), -1)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th,
                cv2.LINE_AA)


def shot_property_browser():
    """napari with the Device Property Browser docked on the right: every
    device property (camera Binning, LED and filter-wheel labels that the
    Channel presets set, the core's device roles), in napari's styling."""
    import napari
    from pymmcore_widgets import PropertyBrowser
    core, sim = load_microscope("optogenetic", mode="stepped", n_cells=20, seed=0)
    viewer = launch_gui(core)
    browser = PropertyBrowser(mmcore=core)
    viewer.window.add_dock_widget(browser, name="Device Property Browser",
                                  area="right")

    def act():
        viewer.window._qt_window.resize(1700, 950)
        core.setConfig("Channel", "phase-contrast")
        core.snapImage()
        QTimer.singleShot(1200, capture)

    def capture():
        from qtpy.QtCore import QPoint
        from qtpy.QtWidgets import QPushButton
        win = viewer.window._qt_window
        path = f"{OUT}/property_browser.png"
        viewer.window.screenshot(path, canvas_only=False)
        # the toolbar button that opens the browser, in screenshot pixels
        btn = next(b for b in win.findChildren(QPushButton)
                   if b.toolTip() == "Device Property Browser")
        tl = btn.mapTo(win, QPoint(0, 0))
        img = cv2.imread(path)
        k = img.shape[1] / win.width()
        bx, by = (tl.x() + btn.width() / 2) * k, (tl.y() + btn.height()) * k
        annotate_button(img, (int(bx), int(by)), "Device Property Browser", k)
        cv2.imwrite(path, img)
        viewer.close()

    QTimer.singleShot(3000, act)
    napari.run()


def shot_results():
    import napari
    core, sim = load_microscope("optogenetic", mode="stepped", n_cells=20, seed=0)
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
    shot_property_browser()
    shot_results()
    print("screenshots written to", OUT)
