"""Generate the expected-output figures for the NEUBIAS teaching module.

Usage:  .venv/bin/python scripts/make_module_figures.py [out_dir]

Produces (deterministic — same seeds as the course activities):
    act2_steering_expected.png   before/after of the steer-all-up loop
    act2_split_expected.png      per-object decision: left up, right down
    exercise_letter_expected.png letter assembly result (DAPI routing)

Detection uses the DAPI/nuclei reference detector and Hungarian track
linking from the package, so the figures show what robust course code
produces. Runtime ~30 s.
"""

import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

from vmteach import (advance, detect_nuclei, letter_mask, link_tracks,
                     load_microscope, overlay)

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/images"

BLACK = (10, 10, 10)


def label(panel, text):
    cv2.putText(panel, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                BLACK, 2, cv2.LINE_AA)
    return panel


def steer_mask(cells, dy=-15, r=11):
    mask = np.zeros((512, 512), np.uint8)
    for cx, cy in cells:
        cv2.circle(mask, (cx, int(np.clip(cy + dy, 0, 511))), r, 255, -1)
    return mask


def draw_tracks(panel, rows, color_fn, thickness=2):
    """Draw linked tracks as smooth polylines onto an RGB panel."""
    for tid in np.unique(rows[:, 0]):
        tr = rows[rows[:, 0] == tid]
        if len(tr) < 5:
            continue
        pts = tr[np.argsort(tr[:, 1])][:, 2:]          # (y, x) time-ordered
        poly = np.column_stack([pts[:, 1], pts[:, 0]]).astype(np.int32)
        cv2.polylines(panel, [poly], False, color_fn(tr), thickness,
                      cv2.LINE_AA)
    return panel


def run_loop(core, sim, decide, n=100):
    """Standard loop: detect nuclei on DAPI, build mask, advance.

    Returns (per-frame detections, final phase image, first mask).
    """
    sim.reset()
    detections = []
    first_mask = None
    for i in range(n):
        core.setConfig("Channel", "DAPI")
        core.snapImage()
        cells = detect_nuclei(core.getImage())
        mask = decide(cells)
        core.setSLMImage("SLM", mask)
        if first_mask is None:
            first_mask = mask
        detections.append(cells)
        advance(sim, 1.0)
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    return detections, core.getImage(), first_mask


def fig_steering():
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0)
    det, img_final, mask0 = run_loop(core, sim, steer_mask)

    # panel 1: first frame with the mask
    sim.reset()
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    p1 = label(overlay(core.getImage(), mask0), "cycle 1 - mask (blue)")

    # panel 2: final frame with linked tracks
    p2 = overlay(img_final, np.zeros_like(img_final))
    rows = link_tracks(det)
    draw_tracks(p2, rows, lambda tr: (30, 110, 30))
    label(p2, "cycle 100 - tracks (population moved up)")

    cv2.imwrite(f"{OUT}/act2_steering_expected.png",
                cv2.cvtColor(np.hstack([p1, p2]), cv2.COLOR_RGB2BGR))


def fig_split():
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
                                warmup=False)

    def decide(cells):
        mask = np.zeros((512, 512), np.uint8)
        for cx, cy in cells:
            dy = -15 if cx < 256 else 15
            cv2.circle(mask, (cx, int(np.clip(cy + dy, 0, 511))), 11, 255, -1)
        return mask

    det, img_final, _ = run_loop(core, sim, decide)
    panel = overlay(img_final, np.zeros_like(img_final))
    rows = link_tracks(det)

    def color(tr):
        x0 = tr[np.argmin(tr[:, 1]), 3]        # starting x decides the group
        return (30, 80, 200) if x0 < 256 else (200, 60, 30)

    draw_tracks(panel, rows, color)
    for y in range(0, 512, 14):                # dashed midline
        cv2.line(panel, (256, y), (256, min(y + 7, 511)), BLACK, 1)
    label(panel, "cycle 100 - left half up (blue), right half down (red)")
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
        cells = detect_nuclei(core.getImage())
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
    on = sum(1 for cx, cy in cells if target[cy, cx] > 0)
    p = overlay(core.getImage(), target, color=(255, 120, 120), alpha=0.22)
    panels.append(label(p, f"cycle 500 - {on}/{len(cells)} cells on target"))
    cv2.imwrite(f"{OUT}/exercise_letter_expected.png",
                cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    fig_steering()
    fig_split()
    fig_letter()
    print("module figures written to", OUT)
