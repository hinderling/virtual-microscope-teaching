"""Generate the README figures.

Usage:  .venv/bin/python scripts/make_readme_figures.py [out_dir]

Produces (deterministic):
    channel_gallery.png    every channel of the optogenetic sample + composite
    well_layouts.png       whole-sample overviews for different well layouts

The GUI screenshots (napari_gui.png, results_explorer.png) come from
scripts/make_screenshots.py.
"""

import sys

import cv2
import numpy as np

from vmteach import load_microscope
from vmteach.sim import OptoCellSim

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/images"

BLACK = (10, 10, 10)
GUTTER = 14      # white gap between concatenated panels
BORDER = 2       # black border around each image


def frame_panel(img_rgb, title):
    """Panel = white title strip above the image, black border around it."""
    bordered = cv2.copyMakeBorder(img_rgb, BORDER, BORDER, BORDER, BORDER,
                                  cv2.BORDER_CONSTANT, value=BLACK)
    w = bordered.shape[1]
    header = np.full((36, w, 3), 255, np.uint8)
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    sc = 0.55 * min(1.0, (w - 12) / max(tw, 1))
    cv2.putText(header, title, (4, 25), cv2.FONT_HERSHEY_SIMPLEX, sc,
                BLACK, 2, cv2.LINE_AA)
    return np.vstack([header, bordered])


def hcat(panels):
    """Concatenate panels horizontally with a white gutter between them."""
    h = max(p.shape[0] for p in panels)
    padded = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for p in panels]
    gap = np.full((h, GUTTER, 3), 255, np.uint8)
    out = [padded[0]]
    for p in padded[1:]:
        out.extend([gap, p])
    return np.hstack(out)


def colorize(gray, rgb):
    """Map a grayscale image onto a single display color (pseudo-color)."""
    g = gray.astype(np.float32) / 255.0
    return (g[..., None] * np.asarray(rgb, np.float32)).astype(np.uint8)


def fig_channel_gallery():
    """Every channel of the optogenetic sample, plus a composite.

    Display colors are pseudo-colors chosen for contrast, as is common
    practice in fluorescence figures.
    """
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
                                warmup=False)

    def snap(ch):
        core.setConfig("Channel", ch)
        core.snapImage()
        return core.getImage()

    phase = snap("phase-contrast")
    nuc = snap("miRFP")
    mem = snap("mVenus")
    ktr = snap("mScarlet")

    SIZE = 384

    def small(img):
        return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    comp = np.stack([small(ktr), small(mem), small(nuc)], axis=-1)  # RGB
    comp = np.clip(comp.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)

    panels = [
        frame_panel(np.stack([small(phase)] * 3, -1), "phase-contrast"),
        frame_panel(colorize(small(nuc), (90, 90, 255)), "miRFP: H2B (nuclei)"),
        frame_panel(colorize(small(mem), (80, 230, 80)),
                    "mVenus: optoFGFR (membrane)"),
        frame_panel(colorize(small(ktr), (255, 80, 80)),
                    "mScarlet: ERK-KTR (activity)"),
        frame_panel(comp, "composite"),
    ]
    cv2.imwrite(f"{OUT}/channel_gallery.png",
                cv2.cvtColor(hcat(panels), cv2.COLOR_RGB2BGR))


def fig_well_layouts():
    """Whole-sample overviews: the well layout is a constructor argument."""
    HEIGHT = 340  # panel height, px

    configs = [
        (dict(), "default: 2 wells in a row"),
        (dict(n_wells=(2, 2)), "n_wells=(2, 2)"),
        (dict(n_wells=(4, 2), well_size=1024.0, well_gap=128.0,
              corner_radius=128.0), "n_wells=(4, 2), well_size=1024"),
        (dict(corner_radius=16.0), "corner_radius=16 (square wells)"),
    ]
    panels = []
    for kwargs, title in configs:
        sim = OptoCellSim(n_cells=6, seed=0, **kwargs)
        scale = HEIGHT / sim.height
        shape = (HEIGHT, int(round(sim.width * scale)))
        img = sim.renderer.render(sim._cells, 0, (0.0, 0.0), scale, shape)
        panels.append(frame_panel(np.stack([img] * 3, -1), title))
    cv2.imwrite(f"{OUT}/well_layouts.png",
                cv2.cvtColor(hcat(panels), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    fig_channel_gallery()
    fig_well_layouts()
    print("README figures written to", OUT)
