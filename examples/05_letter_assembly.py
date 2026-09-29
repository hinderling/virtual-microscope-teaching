# %%
# Assemble cells into a letter: validated reference solution
# (This is the solution to the course exercise; try it yourself first!)
# See the NEUBIAS module exercise for the step-by-step hint ladder.

import cv2
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from vmteach import load_microscope, run_experiment
from vmteach.gui import launch_gui, show_mask
from vmteach.optogenetic import letter_mask, overlay

# Real cells take minutes to assemble. A simulation can simply run faster:
# with SPEED = 5 the sample evolves five times faster than real time, so
# every wait shrinks by the same factor and the 50 one-second cycles take
# 10 s instead of 50 s. On a real microscope SPEED is 1; the rest of the
# script stays the same. (Your analysis time does not shrink: 30 ms of code
# now cost 0.15 s of cell behaviour. Speeding up time also speeds up how
# old your information is.)
SPEED = 5

# Smaller, more numerous cells + a fat letter: cells are solid objects that
# keep a collision distance (~2 cell radii), so the stroke must fit them.
core, sim = load_microscope("optogenetic", n_cells=75, seed=0, base_radius=13.0,
                            mode="realtime", speed=SPEED)
viewer = launch_gui(core)
target = letter_mask("N", fill=0.85, thickness=60)
show_mask(viewer, target, "target", color="orange")   # where cells should go

# %%
# Routing map: aim cells at the CORE of the letter stroke (distance-transform
# "deep" pixels), not its edge; a cell stopping on the edge is half outside.
dist_in = cv2.distanceTransform(target, cv2.DIST_L2, 5)
DEEP = 14
deep = (dist_in >= DEEP).astype(np.uint8) * 255
deep_pts_all = np.column_stack(np.nonzero(deep)[::-1])


# Detect NUCLEI in the miRFP (H2B) channel: nuclei never touch (cells collide
# first), so simple thresholding stays reliable even when cells crowd the
# letter. This is the same reason real workflows segment nuclei, not cell
# bodies.
def detect_cells(img, min_area=20):
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


def build_letter_mask(cells, step_px=12, spot_r=11, occupied_r=28,
                      shape=(512, 512)):
    mask = np.zeros(shape, np.uint8)
    # Unoccupied core pixels, spaced by the collision distance: every routing
    # target is a spot where a cell can actually fit.
    free_deep = deep.copy()
    for cx, cy in cells:
        cv2.circle(free_deep, (cx, cy), occupied_r, 0, -1)
    pts = np.column_stack(np.nonzero(free_deep)[::-1])
    if len(pts) == 0:
        pts = deep_pts_all
    tree = cKDTree(pts)
    for cx, cy in cells:
        if dist_in[cy, cx] >= DEEP:
            continue                    # settled: no stimulus, stays put
        _, i = tree.query((cx, cy))
        vx, vy = pts[i] - (cx, cy)
        d = np.hypot(vx, vy)
        if d == 0:
            continue
        s = min(step_px, d)
        sx = int(np.clip(cx + s * vx / d, 0, shape[1] - 1))
        sy = int(np.clip(cy + s * vy / d, 0, shape[0] - 1))
        cv2.circle(mask, (sx, sy), spot_r, 255, -1)
    return mask


# %%
# Run the feedback loop in the background (50 cycles, ~10 s at SPEED = 5)
# and watch the letter form in the viewer.


def assemble(run):
    for i in range(50):
        if run.stop_requested:
            break
        core.setConfig("Channel", "miRFP")        # acquire nuclei
        core.snapImage()
        cells = detect_cells(core.getImage())
        core.setSLMImage("SLM", build_letter_mask(cells))
        core.setConfig("Channel", "CyanStim")    # expose the pattern
        core.snapImage()
        run.sleep(1.0 / SPEED)                   # one second of cell time

    core.setConfig("Channel", "mVenus")          # cell outlines for the metric
    core.snapImage()
    outlines = core.getImage()
    core.setConfig("Channel", "phase-contrast")  # final image for display
    core.snapImage()
    return cells, outlines, core.getImage()


sim.reset()
cells, outlines, img = run_experiment(assemble).wait()

# %%
# Metrics: cells-on-target is the fair one, because pixel coverage cannot reach
# 100% because cells keep a collision distance (like real cells).
_, binary = cv2.threshold(outlines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(binary, contours, -1, 255, -1)   # fill the outlines
coverage = (binary & target).sum() / target.sum()
on_target = sum(1 for cx, cy in cells if target[cy, cx] > 0) / len(cells)
print(f"target coverage {coverage:.0%}, cells on target {on_target:.0%}")

plt.figure(figsize=(6, 6))
plt.imshow(overlay(img, target, color=(255, 120, 120), alpha=0.25))
plt.title(f"cells on target: {on_target:.0%}")
plt.axis("off")
