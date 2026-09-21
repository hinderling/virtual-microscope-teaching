"""Generate the expected-output figures for the NEUBIAS teaching module.

Usage:  .venv/bin/python scripts/make_module_figures.py [out_dir]

Produces (deterministic — same seeds as the course activities):
    channel_gallery.png          every channel + pseudo-color composite (module A)
    photoactivation_expected.png KTR before / projected light / after / reversal
    keep_active_expected.png     activity curves: one pulse vs closed loop
    pipeline_explained.png       4-step image analysis on a 3-cell crop
    act2_steering_expected.png   before/after of the steer-all-up loop
    act2_split_expected.png      per-object decision: left up, right down
    exercise_letter_expected.png letter assembly result (nuclei routing)

Detection uses the miRFP/nuclei reference detector and Hungarian track
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
GUTTER = 14      # white gap between concatenated panels
BORDER = 2       # black border around each image


def frame_panel(img_rgb, title):
    """Panel = white title strip above the image, black border around it.

    The title is a single line, scaled down if needed to fit the panel
    width. Text sits above the image so it never covers the data.
    """
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
    """Standard loop: detect nuclei on miRFP (H2B), build mask, advance.

    Returns (per-frame detections, final phase image, first mask).
    """
    sim.reset()
    detections = []
    first_mask = None
    for i in range(n):
        core.setConfig("Channel", "miRFP")      # light off, acquire
        core.snapImage()
        cells = detect_nuclei(core.getImage())
        mask = decide(cells)
        core.setSLMImage("SLM", mask)          # upload
        core.setConfig("Channel", "CyanStim")  # light on: deliver
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
    p1 = frame_panel(overlay(core.getImage(), mask0),
                     "Cycle 1: computed stimulation mask (blue)")

    # panel 2: final frame with linked tracks
    img2 = overlay(img_final, np.zeros_like(img_final))
    rows = link_tracks(det)
    draw_tracks(img2, rows, lambda tr: (30, 110, 30))
    p2 = frame_panel(img2, "Cycle 100: tracks, the population moved up")

    cv2.imwrite(f"{OUT}/act2_steering_expected.png",
                cv2.cvtColor(hcat([p1, p2]), cv2.COLOR_RGB2BGR))


def fig_split():
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
                                warmup=False)

    def decide(cells):
        mask = np.zeros((512, 512), np.uint8)
        for cx, cy in cells:
            dy = -15 if cx < 256 else 15
            cv2.circle(mask, (cx, int(np.clip(cy + dy, 0, 511))), 11, 255, -1)
        return mask

    def midline(panel):
        for y in range(0, 512, 14):            # dashed midline
            cv2.line(panel, (256, y), (256, min(y + 7, 511)), BLACK, 1)

    det, img_final, mask0 = run_loop(core, sim, decide)

    # panel 1: first frame with the mask (same layout as the steering figure)
    sim.reset()
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    img1 = overlay(core.getImage(), mask0)
    midline(img1)
    p1 = frame_panel(img1, "Cycle 1: computed stimulation mask (blue)")

    # panel 2: final frame with linked tracks; green = steered up (as in
    # the steering figure), dark purple = steered down
    img2 = overlay(img_final, np.zeros_like(img_final))
    rows = link_tracks(det)

    def color(tr):
        x0 = tr[np.argmin(tr[:, 1]), 3]        # starting x decides the group
        return (30, 110, 30) if x0 < 256 else (110, 40, 140)

    draw_tracks(img2, rows, color)
    midline(img2)
    p2 = frame_panel(img2,
                     "Cycle 100: green tracks steered up, purple down")
    cv2.imwrite(f"{OUT}/act2_split_expected.png",
                cv2.cvtColor(hcat([p1, p2]), cv2.COLOR_RGB2BGR))


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
        core.setConfig("Channel", "miRFP")
        core.snapImage()
        cells = detect_nuclei(core.getImage())
        core.setSLMImage("SLM", build(cells))
        core.setConfig("Channel", "CyanStim")
        if i in (0, 149):
            core.setConfig("Channel", "phase-contrast")
            core.snapImage()
            p = overlay(core.getImage(), target, color=(255, 120, 120),
                        alpha=0.22)
            panels.append(frame_panel(p, f"Cycle {i + 1}"))
        advance(sim, 1.0)
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    on = sum(1 for cx, cy in cells if target[cy, cx] > 0)
    p = overlay(core.getImage(), target, color=(255, 120, 120), alpha=0.22)
    panels.append(frame_panel(p, f"Cycle 500: {on}/{len(cells)} cells on target"))
    cv2.imwrite(f"{OUT}/exercise_letter_expected.png",
                cv2.cvtColor(hcat(panels), cv2.COLOR_RGB2BGR))


def fig_pipeline():
    """From pixels to decisions, on a small crop with ~3 cells.

    All panels show the experiment start (t=0); the tracks panel overlays
    where the cells went during the following 25 steering cycles.
    """
    # seed 23 has a clean, isolated 3-cell neighbourhood for a tight crop
    core, sim = load_microscope("optogenetic", n_cells=20, seed=23,
                                warmup=False)

    def snap(ch):
        core.setConfig("Channel", ch)
        core.snapImage()
        return core.getImage()

    # ── capture everything at t = 0 ─────────────────────────────────────
    phase0 = snap("phase-contrast")
    dapi0 = snap("miRFP")
    _, binary0 = cv2.threshold(dapi0, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # include border-clipped nuclei here: the crop search counts every
    # nucleus near a window, clipped or not
    cells0 = detect_nuclei(dapi0, exclude_border=False)

    # crop window with exactly 3 nuclei, clear of the label area (top ~50 px)
    box = 135
    candidates = []
    for y in range(0, 512 - box, 5):
        for x in range(0, 512 - box, 5):
            inside = [(cx, cy) for cx, cy in cells0
                      if x + 26 < cx < x + box - 26
                      and y + 48 < cy < y + box - 26]
            n_total = sum(1 for cx, cy in cells0
                          if x - 20 < cx < x + box + 20
                          and y - 20 < cy < y + box + 20)
            if len(inside) == 3:
                candidates.append((n_total, x, y))
    assert candidates, "no 3-cell crop found — adjust box/seed"
    # the window with 3 well-centered nuclei and the fewest neighbours
    _, cx0, cy0 = min(candidates)

    SC = 3  # upscale factor for legibility

    def crop(img):
        c = img[cy0:cy0 + box, cx0:cx0 + box]
        return cv2.resize(c, (box * SC, box * SC),
                          interpolation=cv2.INTER_NEAREST)

    def to_rgb(img, invert=True):
        g = 255 - img if invert else img
        return np.stack([g] * 3, axis=-1)

    p1 = frame_panel(to_rgb(crop(phase0)), "1. Acquire: phase contrast")
    p2 = frame_panel(to_rgb(crop(dapi0)), "2. Acquire: nuclei (miRFP, H2B)")
    p3 = frame_panel(to_rgb(crop(binary0)), "3. Threshold (Otsu)")

    # panel 4: connected components + centroids (at t = 0)
    n_lbl, lbl_img = cv2.connectedComponents((binary0 > 0).astype(np.uint8))
    lbl_rgb = np.full((512, 512, 3), 255, np.uint8)
    palette = [(200, 60, 60), (60, 140, 60), (60, 80, 200), (180, 140, 40)]
    for i in range(1, n_lbl):
        lbl_rgb[lbl_img == i] = palette[(i - 1) % len(palette)]
    img4 = cv2.resize(lbl_rgb[cy0:cy0 + box, cx0:cx0 + box],
                      (box * SC, box * SC), interpolation=cv2.INTER_NEAREST)
    for cx, cy in cells0:
        if cx0 < cx < cx0 + box and cy0 < cy < cy0 + box:
            cv2.drawMarker(img4, ((cx - cx0) * SC, (cy - cy0) * SC), BLACK,
                           cv2.MARKER_CROSS, 18, 2)
    p4 = frame_panel(img4, "4. Label and measure centroids")

    strip = hcat([p1, p2, p3, p4])
    cv2.imwrite(f"{OUT}/pipeline_explained.png",
                cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


def fig_stim_logic():
    """The stimulation logic in three stages: segmentation + planned spots →
    binary mask on the SLM/DMD (gray: white = light) → actual projected
    light imaged in the CyanStim channel (cyan = light)."""
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
                                warmup=False)
    core.setConfig("Channel", "miRFP")
    core.snapImage()
    cells = detect_nuclei(core.getImage())
    mask = steer_mask(cells)

    # 1. planned: segmentation centroids + spots over phase-contrast
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    img1 = overlay(core.getImage(), mask)
    for cx, cy in cells:
        cv2.drawMarker(img1, (cx, cy), (0, 150, 150), cv2.MARKER_CROSS, 12, 2)
    p1 = frame_panel(img1, "1. Segment and place spots (software overlay)")

    # 2. the binary pattern as uploaded to the SLM/DMD (white = light on)
    p2 = frame_panel(np.stack([mask] * 3, axis=-1),
                     "2. Binary mask on the SLM/DMD, white = light")

    # 3. the projected light, imaged in the CyanStim channel (cyan = light)
    core.setSLMImage("SLM", mask)
    core.setConfig("Channel", "CyanStim")   # light on: deliver + image
    core.snapImage()
    stim = core.getImage().astype(np.float32)
    img3 = np.stack([np.zeros_like(stim), stim, stim],
                    axis=-1).astype(np.uint8)        # cyan on black
    p3 = frame_panel(img3, "3. Projected light in the CyanStim channel")

    cv2.imwrite(f"{OUT}/stimulation_logic.png",
                cv2.cvtColor(hcat([p1, p2, p3]), cv2.COLOR_RGB2BGR))


def colorize(gray, rgb):
    """Map a grayscale image onto a single display color (pseudo-color)."""
    g = gray.astype(np.float32) / 255.0
    return (g[..., None] * np.asarray(rgb, np.float32)).astype(np.uint8)


def fig_channel_gallery():
    """Module A: each channel of the optogenetic sample, plus a composite.

    Display colors are pseudo-colors chosen for contrast (miRFP blue,
    mVenus green, mScarlet red), as is common practice in fluorescence
    figures; they are unrelated to the emission wavelengths.
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


def fig_photoactivation():
    """Module B activity 1: image -> mask -> stimulate -> image.

    ERK-KTR before, the projected light, the response 5 s later, and the
    reversal 20 s after that. Matches the activity script (seed 0)."""
    from vmteach import measure_activity

    core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
                                warmup=False)

    def snap(ch):
        core.setConfig("Channel", ch)
        core.snapImage()
        return core.getImage()

    cells = detect_nuclei(snap("miRFP"))
    ktr0 = snap("mScarlet")
    n0 = sum(a > 0.5 for a in measure_activity(ktr0, cells))

    mask = np.zeros((512, 512), np.uint8)
    targets = [(x, y) for x, y in cells if x < 256]
    for x, y in targets:
        cv2.circle(mask, (x, y), 25, 255, -1)
    core.setSLMImage("SLM", mask)
    proj = snap("CyanStim")
    advance(sim, 5)

    cells1 = detect_nuclei(snap("miRFP"))
    ktr1 = snap("mScarlet")
    act1 = measure_activity(ktr1, cells1)
    n1 = sum(a > 0.5 for a in act1)

    core.setSLMImage("SLM", np.zeros((512, 512), np.uint8))
    advance(sim, 20)
    cells2 = detect_nuclei(snap("miRFP"))
    ktr2 = snap("mScarlet")
    n2 = sum(a > 0.5 for a in measure_activity(ktr2, cells2))

    def ktr_rgb(img):
        return np.stack([img] * 3, axis=-1)

    proj_rgb = np.stack([np.zeros_like(proj), proj, proj], -1)  # cyan

    panels = [
        frame_panel(ktr_rgb(ktr0), f"ERK-KTR before: {n0}/{len(cells)} active"),
        frame_panel(proj_rgb, "projected light (CyanStim)"),
        frame_panel(ktr_rgb(ktr1),
                    f"5 s after the pulse: {n1}/{len(cells1)} active"),
        frame_panel(ktr_rgb(ktr2), f"20 s later: {n2}/{len(cells2)} active"),
    ]
    cv2.imwrite(f"{OUT}/photoactivation_expected.png",
                cv2.cvtColor(hcat(panels), cv2.COLOR_RGB2BGR))


def fig_keep_active():
    """Module B activity 1 extension: why a loop? One pulse decays; the
    closed loop re-stimulates whenever activity drops and holds the state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vmteach import measure_activity

    def targeted_mean(core, sim, closed_loop, seconds=60):
        sim.reset()
        t, mean_left, mean_right = [], [], []
        pulsed = False
        for i in range(seconds):
            core.setConfig("Channel", "miRFP")
            core.snapImage()
            cells = detect_nuclei(core.getImage())
            core.setConfig("Channel", "mScarlet")
            core.snapImage()
            act = measure_activity(core.getImage(), cells)
            left = [a for (x, _), a in zip(cells, act) if x < 256]
            right = [a for (x, _), a in zip(cells, act) if x >= 256]
            t.append(i)
            mean_left.append(np.mean(left) if left else 0.0)
            mean_right.append(np.mean(right) if right else 0.0)

            mask = np.zeros((512, 512), np.uint8)
            if closed_loop:
                # re-stimulate targeted cells whose activity has dropped
                for (x, y), a in zip(cells, act):
                    if x < 256 and a < 0.7:
                        cv2.circle(mask, (x, y), 25, 255, -1)
            elif not pulsed and i == 2:
                for x, y in cells:
                    if x < 256:
                        cv2.circle(mask, (x, y), 25, 255, -1)
                pulsed = True
            if mask.any():
                core.setSLMImage("SLM", mask)
                core.setConfig("Channel", "CyanStim")   # deliver
                core.setSLMImage("SLM", np.zeros((512, 512), np.uint8))
            advance(sim, 1.0)
        return t, mean_left, mean_right

    core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
                                warmup=False)
    t1, pulse_left, _ = targeted_mean(core, sim, closed_loop=False)
    t2, loop_left, loop_right = targeted_mean(core, sim, closed_loop=True)

    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=150)
    ax.plot(t2, loop_left, color="#1268b3", lw=2,
            label="targeted cells, closed loop")
    ax.plot(t1, pulse_left, color="#1268b3", lw=2, ls="--",
            label="targeted cells, single pulse")
    ax.plot(t2, loop_right, color="#888888", lw=2,
            label="untargeted cells")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean ERK activity")
    ax.set_ylim(-0.05, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/keep_active_expected.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_channel_gallery()
    fig_photoactivation()
    fig_keep_active()
    fig_steering()
    fig_split()
    fig_letter()
    fig_pipeline()
    fig_stim_logic()
    print("module figures written to", OUT)
