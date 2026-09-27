"""Image-analysis helpers for the course exercises.

None of this is microscope API: these functions take plain images and
work the same on frames from the virtual microscope and from real
hardware. They are reference implementations to compare your own
pipeline against, kept separate from the simulator itself
(:mod:`vmteach`).

    from vmteach.analysis import detect_nuclei, measure_activity
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


def letter_mask(char: str, shape: tuple = (512, 512),
                fill: float = 0.6, thickness: int = 40) -> np.ndarray:
    """Binary target image of a letter, centered, for the assembly exercise.

    Args:
        char: A single character (e.g. ``"N"``).
        shape: Output image shape ``(height, width)``.
        fill: Approximate fraction of the smaller image dimension the
            letter should span.
        thickness: Stroke thickness in pixels. Keep it wider than a cell
            so cells fit on the stroke.

    Returns:
        uint8 array, 255 inside the letter, 0 elsewhere.
    """
    if len(char) != 1:
        raise ValueError("letter_mask takes a single character")

    target_px = int(min(shape) * fill)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    (w, h), _ = cv2.getTextSize(char, font, scale, thickness)
    scale = scale * target_px / max(h, 1)
    (w, h), _ = cv2.getTextSize(char, font, scale, thickness)

    mask = np.zeros(shape, dtype=np.uint8)
    org = ((shape[1] - w) // 2, (shape[0] + h) // 2)
    cv2.putText(mask, char, org, font, scale, 255, thickness, cv2.LINE_8)
    return mask
