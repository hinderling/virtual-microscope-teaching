"""Fast cell renderer.

Replaces the original spline-based renderer (~64 ms/frame) with vectorized
polygon smoothing + cv2 fills (~2 ms/frame). Renders the full world once per
snap for the requested channel; the sim crops the viewport afterwards.

Channels (mode):
    0  phase-contrast:  dark background, gray cell body, brighter rim,
                          darker nucleus
    1  DAPI:            nuclei bright on black
    2  membrane:        cell outline bright on black

Stimulated cells get a slightly brighter rim in phase-contrast, so learners
can see which cells received light.
"""

from __future__ import annotations

import cv2
import numpy as np

# Upsample the 24 cell vertices to this many polygon points (visual smoothness)
_SMOOTH_PTS = 72


def _smooth_polygon(center: np.ndarray, angles: np.ndarray, radii: np.ndarray,
                    scale: float = 1.0) -> np.ndarray:
    """Periodic linear upsampling of the vertex radii → smooth int32 polygon."""
    n = len(angles)
    # extend periodically so interpolation wraps
    ang_ext = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    r_ext = np.concatenate([radii, [radii[0]]])
    ang_f = np.linspace(0.0, 2 * np.pi, _SMOOTH_PTS, endpoint=False)
    r_f = np.interp(ang_f, ang_ext, r_ext) * scale
    pts = np.empty((_SMOOTH_PTS, 2), np.float64)
    pts[:, 0] = center[0] + np.cos(ang_f) * r_f
    pts[:, 1] = center[1] + np.sin(ang_f) * r_f
    return pts.astype(np.int32)


class CellRenderer:
    """Draws the cell population into a world-sized uint8 image."""

    # gray levels per channel
    PHASE_BG, PHASE_BODY, PHASE_RIM, PHASE_NUCLEUS = 18, 60, 105, 38
    PHASE_RIM_STIM = 150          # stimulated cells: brighter rim
    FLUO_BG, DAPI_NUCLEUS, MEMBRANE_RIM = 6, 170, 150

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)

    def render(self, cells, mode: int) -> np.ndarray:
        h, w = self.height, self.width
        if mode == 1:      # DAPI
            img = np.full((h, w), self.FLUO_BG, np.uint8)
            polys = [_smooth_polygon(c.center, c.angles, c.r, scale=0.5)
                     for c in cells]
            cv2.fillPoly(img, polys, self.DAPI_NUCLEUS, lineType=cv2.LINE_AA)
            return img
        if mode == 2:      # membrane
            img = np.full((h, w), self.FLUO_BG, np.uint8)
            polys = [_smooth_polygon(c.center, c.angles, c.r) for c in cells]
            cv2.polylines(img, polys, True, self.MEMBRANE_RIM, 3,
                          lineType=cv2.LINE_AA)
            return img

        # phase-contrast (mode 0)
        img = np.full((h, w), self.PHASE_BG, np.uint8)
        bodies, rims, rims_stim, nuclei = [], [], [], []
        for c in cells:
            poly = _smooth_polygon(c.center, c.angles, c.r)
            bodies.append(poly)
            (rims_stim if getattr(c, "is_stimulated", False) else rims).append(poly)
            nuclei.append(_smooth_polygon(c.center, c.angles, c.r, scale=0.45))
        cv2.fillPoly(img, bodies, self.PHASE_BODY, lineType=cv2.LINE_AA)
        if rims:
            cv2.polylines(img, rims, True, self.PHASE_RIM, 3,
                          lineType=cv2.LINE_AA)
        if rims_stim:
            cv2.polylines(img, rims_stim, True, self.PHASE_RIM_STIM, 3,
                          lineType=cv2.LINE_AA)
        cv2.fillPoly(img, nuclei, self.PHASE_NUCLEUS, lineType=cv2.LINE_AA)
        return img
