"""Image-analysis helpers for the ``optogenetic`` sample's readouts.

Reference implementations for the course exercises, tailored to this
sample's labels (H2B nuclei, ERK-KTR reporter). None of this is
microscope API: the functions take plain images, so they run the same
on frames from the virtual microscope and on a real sample labelled the
same way.

    from vmteach.optogenetic import detect_nuclei, measure_activity
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["detect_nuclei", "cn_ratio", "measure_activity", "link_tracks",
           "overlay", "letter_mask"]


def _nuclei_binary(img: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def detect_nuclei(img: np.ndarray, min_area: int = 20,
                  exclude_border: bool = True) -> list:
    """Reference detector: nuclei centroids from a nuclear-marker image.

    Snap the nuclear channel (H2B-miRFP on the virtual microscope) and
    pass the frame here.

    Nuclei are bright, compact, and, unlike cell bodies, never touch
    (cells collide before their nuclei can), so a plain Otsu threshold
    stays reliable even in crowded fields. Use this as the robust
    detection for feedback loops and tracking; write your own detector
    in the activities to understand what it does.

    Nuclei cut off by the image border are dropped by default (standard
    practice: a clipped object has a biased centroid, and intensity
    measurements around it sample the background).

    Returns a list of ``(x, y)`` integer centroids.
    """
    binary = _nuclei_binary(img)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        if exclude_border:
            x0, y0, bw, bh = cv2.boundingRect(c)
            if x0 <= 0 or y0 <= 0 or x0 + bw >= w or y0 + bh >= h:
                continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    return out


def cn_ratio(ktr_img: np.ndarray, nuclei_img: np.ndarray, centroids,
             ring: tuple[int, int] = (1, 4), match_px: int = 8) -> list:
    """Cytoplasm-to-nucleus (C/N) ratio of a translocation reporter.

    The standard readout for kinase translocation reporters (KTRs): the
    reporter sits in the nucleus while the kinase is inactive and is
    exported to the cytoplasm when it is active, so C/N rises with
    activity. As a *ratio* it does not depend on the cell's expression
    level or on the exposure, so it works on real images as well.

    For each nucleus (segmented from ``nuclei_img`` with the same Otsu
    threshold as :func:`detect_nuclei`), the nuclear signal is the median
    of the nucleus eroded by 1 px, and the cytoplasmic signal the median
    of a thin ring just outside it (``ring`` = inner and outer distance
    in px from the nuclear edge), excluding other nuclei and background.

    Args:
        ktr_img: a frame from the reporter channel (mScarlet, ERK-KTR).
        nuclei_img: the nuclear-marker frame of the same field (miRFP).
        centroids: ``(x, y)`` nucleus positions, e.g. from
            :func:`detect_nuclei` on ``nuclei_img``.
        ring: cytoplasmic ring (inner, outer) distance from the nucleus,
            px. Scale it with the magnification at higher objectives.
        match_px: a centroid outside every nucleus (e.g. detected in an
            earlier frame, before the cell moved) is matched to the
            nearest nucleus within this distance.

    Returns:
        One C/N ratio per centroid (``nan`` if it could not be measured,
        e.g. no nucleus within ``match_px``).
    """
    nuc_bin = _nuclei_binary(nuclei_img)
    n, labels = cv2.connectedComponents(nuc_bin)
    img = ktr_img.astype(np.float32)
    # background: the darkest few percent of the frame, plus a noise margin
    bg = float(np.percentile(img, 5))
    cell_thr = bg + 8.0
    h, w = img.shape[:2]
    r_in, r_out = ring
    pad = r_out + 2
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_in + 1,) * 2)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_out + 1,) * 2)
    k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = []
    for x, y in centroids:
        x, y = int(round(x)), int(round(y))
        lab = labels[min(max(y, 0), h - 1), min(max(x, 0), w - 1)]
        if lab == 0:
            # a centroid from an earlier frame may sit just outside its
            # (moved) nucleus: take the nearest nucleus within match_px
            wy0, wy1 = max(0, y - match_px), min(h, y + match_px + 1)
            wx0, wx1 = max(0, x - match_px), min(w, x + match_px + 1)
            win = labels[wy0:wy1, wx0:wx1]
            iy, ix = np.nonzero(win)
            if len(iy) == 0:
                out.append(float("nan"))
                continue
            k = np.argmin((iy + wy0 - y) ** 2 + (ix + wx0 - x) ** 2)
            lab = win[iy[k], ix[k]]
        ys, xs = np.nonzero(labels == lab)
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
        crop_lab = labels[y0:y1, x0:x1]
        mine = (crop_lab == lab).astype(np.uint8)
        others = ((crop_lab > 0) & (crop_lab != lab)).astype(np.uint8)
        others = cv2.dilate(others, k_in)
        core_px = cv2.erode(mine, k_er).astype(bool)
        ring_px = (cv2.dilate(mine, k_out).astype(bool)
                   & ~cv2.dilate(mine, k_in).astype(bool)
                   & ~others.astype(bool))
        patch = img[y0:y1, x0:x1]
        ring_px &= patch > cell_thr
        if core_px.sum() < 3 or ring_px.sum() < 3:
            out.append(float("nan"))
            continue
        nuc = float(np.median(patch[core_px])) - bg
        cyt = float(np.median(patch[ring_px])) - bg
        out.append(cyt / max(nuc, 1.0))
    return out


def measure_activity(ktr_img: np.ndarray, nuclei_img: np.ndarray, centroids,
                     lo: float = 0.3, hi: float = 1.2) -> list:
    """Per-cell pathway activity in [0, 1] from the ERK-KTR C/N ratio.

    Maps :func:`cn_ratio` linearly between ``lo`` (resting cells) and
    ``hi`` (fully activated cells). The defaults are calibrated for the
    virtual microscope's reporter; on real data, calibrate them from
    resting and maximally stimulated control cells.

    Returns one value per centroid; threshold at 0.5 for a binary
    active/inactive call. Unmeasurable cells return ``nan``.
    """
    out = []
    for r in cn_ratio(ktr_img, nuclei_img, centroids):
        out.append(r if np.isnan(r)
                   else float(np.clip((r - lo) / (hi - lo), 0.0, 1.0)))
    return out


def link_tracks(detections, max_dist: float = 40.0,
                memory: int = 2) -> np.ndarray:
    """Link per-frame detections into tracks (Hungarian assignment).

    Globally optimal frame-to-frame matching (scipy
    ``linear_sum_assignment``) with distance gating: a detection farther
    than ``max_dist`` from every track starts a new track instead of
    producing a jumpy link. Tracks survive up to ``memory`` missed frames
    (detector dropouts, cells briefly merging).

    Args:
        detections: sequence over frames; each frame a sequence of (x, y).
        max_dist: gate, the maximum linking distance in pixels per frame.
        memory: frames a lost track is kept alive for re-linking.

    Returns:
        Array of rows ``(track_id, t, y, x)``, the napari Tracks format.
    """
    from scipy.optimize import linear_sum_assignment

    BIG = 1e9
    active: dict = {}          # tid -> (x, y, last_t)
    rows = []
    next_id = 0
    for t, pts in enumerate(detections):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        tids = list(active.keys())
        matched = set()
        if tids and len(pts):
            prev = np.array([active[i][:2] for i in tids])
            cost = np.linalg.norm(prev[:, None, :] - pts[None, :, :], axis=2)
            cost[cost > max_dist] = BIG
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] >= BIG:
                    continue
                tid = tids[r]
                active[tid] = (pts[c, 0], pts[c, 1], t)
                rows.append((tid, t, pts[c, 1], pts[c, 0]))
                matched.add(c)
        for c, (x, y) in enumerate(pts):
            if c in matched:
                continue
            active[next_id] = (x, y, t)
            rows.append((next_id, t, y, x))
            next_id += 1
        active = {i: v for i, v in active.items() if t - v[2] <= memory}
    return np.asarray(rows, dtype=float)


def overlay(img: np.ndarray, mask: np.ndarray,
            color: tuple = (80, 140, 255), alpha: float = 0.4) -> np.ndarray:
    """Overlay a stimulation mask on a grayscale image as a colored wash.

    Returns an RGB uint8 image (inverted grayscale, matplotlib ``gray_r``
    convention) with ``mask > 0`` pixels tinted in ``color``.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        norm = (img.astype(np.float32) - lo) / (hi - lo) * 255
    else:
        norm = img.astype(np.float32)
    inv = 255 - norm
    rgb = np.stack([inv, inv, inv], axis=-1)
    m = mask > 0
    for c in range(3):
        rgb[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * color[c]
    return rgb.astype(np.uint8)


def _thin(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: the one-pixel centre line of each stroke."""
    img = np.pad((binary > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, -dy, 0), -dx, 1) for dy, dx in
                 ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1))]         # P2..P9, clockwise from N
            n = sum(p)
            t = sum((p[i] == 0) & (p[(i + 1) % 8] == 1) for i in range(8))
            if step == 0:
                c = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                c = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            kill = (img == 1) & (n >= 2) & (n <= 6) & (t == 1) & c
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1]


def letter_mask(char: str, shape: tuple = (512, 512),
                fill: float = 0.6, thickness: int = 40) -> np.ndarray:
    """Binary target image of a letter, centered, for the assembly exercise.

    Args:
        char: A single character (e.g. ``"N"``).
        shape: Output image shape ``(height, width)``.
        fill: Fraction of the smaller image dimension the letter spans
            (its larger side, as drawn).
        thickness: Stroke thickness in pixels. Keep it wider than a cell
            so cells fit on the stroke.

    Returns:
        uint8 array, 255 inside the letter, 0 elsewhere.
    """
    if len(char) != 1:
        raise ValueError("letter_mask takes a single character")

    # cv2 places text by its baseline and nominal size, not by the pixels
    # it draws, and its stroke width follows the font scale (OpenCV 5). So
    # take the glyph's centre line, which scales with the font, size it to
    # span ``fill`` of the frame, widen it to ``thickness`` and centre it
    def centre_line(scale):
        (w, h), base = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX,
                                       scale, 1)
        glyph = np.zeros((h + base + 20, w + 20), np.uint8)
        cv2.putText(glyph, char, (10, h + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, 255, 1, cv2.LINE_8)
        ys, xs = np.nonzero(_thin(glyph))
        return ys - ys.min(), xs - xs.min()

    ys, xs = centre_line(4.0)
    span = max(ys.max(), xs.max()) + 1
    ys, xs = centre_line(4.0 * max(min(shape) * fill - thickness, 1) / span)
    line = np.zeros(shape, np.uint8)
    oy = int(round((shape[0] - 1 - ys.max()) / 2))
    ox = int(round((shape[1] - 1 - xs.max()) / 2))
    keep = ((ys + oy >= 0) & (ys + oy < shape[0])
            & (xs + ox >= 0) & (xs + ox < shape[1]))
    line[ys[keep] + oy, xs[keep] + ox] = 255
    r = max(thickness // 2, 1)
    mask = cv2.dilate(line, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return mask
