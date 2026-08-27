"""Generate the expected-output figures for the NEUBIAS teaching module.

Usage:  .venv/bin/python scripts/make_module_figures.py [out_dir]

Produces (deterministic — same seeds as the course activities):
    act2_steering_expected.png   before/after of the steer-all-up loop
    act2_split_expected.png      per-object decision: left up, right down
    exercise_letter_expected.png letter assembly result (DAPI routing)

Runtime ~30 s.
"""

import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

from vmteach import load_microscope, advance, letter_mask, overlay

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/images"


def detect_cells(img, min_area=100):
    _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cts:
        if cv2.contourArea(c) < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    return out


def label(panel, text):
    cv2.putText(panel, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def steer_mask(cells, dy=-15, r=11):
    mask = np.zeros((512, 512), np.uint8)
    for cx, cy in cells:
        cv2.circle(mask, (cx, int(np.clip(cy + dy, 0, 511))), r, 255, -1)
    return mask


def fig_steering():
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0)
    core.setConfig("Channel", "phase-contrast")
    sim.reset()
    panels = []
    tracks = []
    for i in range(100):
        core.snapImage()
        img = core.getImage()
        cells = detect_cells(img)
        mask = steer_mask(cells)
        core.setSLMImage("SLM", mask)
        tracks.append(cells)
        if i == 0:
            panels.append(label(overlay(img, mask), "cycle 1 - mask (blue)"))
        advance(sim, 1.0)
    final = overlay(img, np.zeros_like(img))
    # draw simple nearest-neighbour tails
    for t in range(1, len(tracks)):
        for (x1, y1) in tracks[t]:
            best, bd = None, 40
            for (x0, y0) in tracks[t - 1]:
                d = np.hypot(x1 - x0, y1 - y0)
                if d < bd:
                    best, bd = (x0, y0), d
            if best:
                cv2.line(final, best, (x1, y1), (60, 120, 60), 1, cv2.LINE_AA)
    panels.append(label(final, "cycle 100 - tracks (population moved up)"))
    cv2.imwrite(f"{OUT}/act2_steering_expected.png",
                cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))


def fig_split():
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
                                warmup=False)
    core.setConfig("Channel", "phase-contrast")
    sim.reset()
    for i in range(100):
        core.snapImage()
        img = core.getImage()
        cells = detect_cells(img)
        mask = np.zeros((512, 512), np.uint8)
        for cx, cy in cells:
            dy = -15 if cx < 256 else 15
            cv2.circle(mask, (cx, int(np.clip(cy + dy, 0, 511))), 11, 255, -1)
        core.setSLMImage("SLM", mask)
        advance(sim, 1.0)
    panel = overlay(img, mask)
    cv2.line(panel, (256, 0), (256, 511), (255, 255, 255), 1, cv2.LINE_AA)
    label(panel, "cycle 100 - left half up, right half down")
    cv2.imwrite(f"{OUT}/act2_split_expected.png",
                cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def fig_letter():
    core, sim = load_microscope("optogenetic", n_cells=40, seed=0,
                                base_radius=13.0, warmup=False)
    target = letter_mask("L", fill=0.65, thickness=55)
    dist_in = cv2.distanceTransform(target, cv2.DIST_L2, 5)
    deep = (dist_in >= 14).astype(np.uint8) * 255
    deep_pts_all = np.column_stack(np.nonzero(deep)[::-1])

    def build(cells):
        mask = np.zeros((512, 512), np.uint8)
        free = deep.copy()
        for cx, cy in cells:
            cv2.circle(free, (cx, cy), 28, 0, -1)
        pts = np.column_stack(np.nonzero(free)[::-1])
        if len(pts) == 0:
            pts = deep_pts_all
        tree = cKDTree(pts)
        for cx, cy in cells:
            if dist_in[cy, cx] >= 14:
                continue
            _, i = tree.query((cx, cy))
            vx, vy = pts[i] - (cx, cy)
            d = np.hypot(vx, vy)
            if d == 0:
                continue
            s = min(12, d)
            cv2.circle(mask, (int(np.clip(cx + s * vx / d, 0, 511)),
                              int(np.clip(cy + s * vy / d, 0, 511))),
                       11, 255, -1)
        return mask

    sim.reset()
    panels = []
    for i in range(500):
        core.setConfig("Channel", "DAPI")
        core.snapImage()
        cells = detect_cells(core.getImage(), min_area=20)
        core.setSLMImage("SLM", build(cells))
        if i in (0, 149):
            core.setConfig("Channel", "phase-contrast")
            core.snapImage()
            p = overlay(core.getImage(), target, color=(255, 120, 120),
                        alpha=0.22)
            panels.append(label(p, f"cycle {i + 1}"))
        advance(sim, 1.0)
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    img = core.getImage()
    on = sum(1 for cx, cy in cells if target[cy, cx] > 0)
    p = overlay(img, target, color=(255, 120, 120), alpha=0.22)
    panels.append(label(p, f"cycle 500 - {on}/{len(cells)} cells on target"))
    cv2.imwrite(f"{OUT}/exercise_letter_expected.png",
                cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    fig_steering()
    fig_split()
    fig_letter()
    print("module figures written to", OUT)
