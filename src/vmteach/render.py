"""Fast cell renderer: draws the population straight into camera pixels.

Cells are polygons, so instead of rasterizing the whole world and cropping,
the renderer maps each visible cell into the camera frame (world -> sensor
pixels, objective magnification and binning folded into one ``scale``)
and lets cv2 fill it. Cost is per *visible* cell and per output pixel,
independent of the world size and of the magnification (~2 ms for a
512x512 frame).

Channels (mode):
    0  phase-contrast:  neutral gray background, slightly darker cell body,
                        bright halo at the edge, darker nucleus; the well
                        wall (plastic) is darker with a bright edge
    1  miRFP (H2B):     nuclei bright on black (wall invisible)
    2  mVenus (optoFGFR): cell outline bright on black; the membrane-bound
                        optogenetic tool looks like a membrane stain
    4  mScarlet (ERK-KTR): kinase translocation reporter; inactive cells
                        show a bright nucleus in a dim cytoplasm, active
                        cells a dark nucleus in a bright cytoplasm

Stimulated cells get a brighter halo in phase-contrast, so learners can see
which cells received light.
"""

from __future__ import annotations

import cv2
import numpy as np

# Upsample the 24 cell vertices to this many polygon points (visual smoothness)
_SMOOTH_PTS = 72
_ANG_F = np.linspace(0.0, 2 * np.pi, _SMOOTH_PTS, endpoint=False)
_COS_F, _SIN_F = np.cos(_ANG_F), np.sin(_ANG_F)


def _smooth_polygon(center: np.ndarray, angles: np.ndarray, radii: np.ndarray,
                    scale: float = 1.0) -> np.ndarray:
    """Periodic linear upsampling of the vertex radii -> smooth int32 polygon.

    ``center`` is in output pixels, ``radii`` in world px; ``scale``
    converts radii to output pixels.
    """
    ang_ext = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    r_ext = np.concatenate([radii, [radii[0]]])
    r_f = np.interp(_ANG_F, ang_ext, r_ext) * scale
    pts = np.empty((_SMOOTH_PTS, 2), np.float64)
    pts[:, 0] = center[0] + _COS_F * r_f
    pts[:, 1] = center[1] + _SIN_F * r_f
    return np.rint(pts).astype(np.int32)


def _rounded_square(center_px, half_px: float, corner_px: float,
                    n_arc: int = 12) -> np.ndarray:
    """int32 polygon of a rounded square in output pixels."""
    cx, cy = center_px
    hx = half_px - corner_px
    pts = []
    for sx, sy, a0 in ((1, 1, 0.0), (-1, 1, np.pi / 2),
                       (-1, -1, np.pi), (1, -1, 3 * np.pi / 2)):
        ang = np.linspace(a0, a0 + np.pi / 2, n_arc)
        pts.append(np.stack([cx + sx * hx + corner_px * np.cos(ang),
                             cy + sy * hx + corner_px * np.sin(ang)], 1))
    return np.rint(np.concatenate(pts)).astype(np.int32)


class CellRenderer:
    """Draws the wells and the cell population into a camera-sized image."""

    # gray levels per channel
    PHASE_BG, PHASE_BODY, PHASE_NUCLEUS = 128, 112, 98
    PHASE_HALO, PHASE_HALO_STIM = 180, 220
    PHASE_WALL, PHASE_WALL_EDGE = 88, 205     # plastic, bright well edge
    WALL_EDGE_UM = 6.0                         # edge line width, world um
    WALL_BLUR_UM = 5.0                         # softness of the wall edge
    FLUO_BG, DAPI_NUCLEUS, MEMBRANE_RIM = 6, 170, 150
    # ERK-KTR (mode 4): intensities interpolate with cell.activity
    KTR_CYTO_LO, KTR_CYTO_HI = 45, 110      # cytoplasm: dim -> bright
    KTR_NUC_LO, KTR_NUC_HI = 40, 170        # nucleus:  bright -> dark
    LINE_UM = 2.5                 # halo / membrane line width, in world um

    def __init__(self, wells, well_half: float, corner_radius: float):
        self.wells = np.asarray(wells, dtype=float)   # (n, 2) centres, um
        self.well_half = float(well_half)
        self.corner_radius = float(corner_radius)

    # ── geometry ────────────────────────────────────────────────────────

    def _visible(self, cells, origin, scale, shape):
        """Yield (cell, centre_px) for every cell that touches the frame."""
        h, w = shape
        for c in cells:
            m = float(c.r.max()) * scale + self.LINE_UM * scale + 2
            x = (c.center[0] - origin[0]) * scale
            y = (c.center[1] - origin[1]) * scale
            if -m <= x <= w + m and -m <= y <= h + m:
                yield c, np.array([x, y])

    def _draw_wells(self, img: np.ndarray, origin, scale: float) -> None:
        """Phase contrast: plastic outside the wells, bright well edge."""
        h, w = img.shape
        img[:] = self.PHASE_WALL
        half_px = self.well_half * scale
        polys = []
        for cx, cy in self.wells:
            x, y = (cx - origin[0]) * scale, (cy - origin[1]) * scale
            if x + half_px < 0 or x - half_px > w or y + half_px < 0 or y - half_px > h:
                continue
            polys.append(_rounded_square((x, y), half_px,
                                         self.corner_radius * scale))
        if polys:
            cv2.fillPoly(img, polys, self.PHASE_BG, lineType=cv2.LINE_AA)
            cv2.polylines(img, polys, True, self.PHASE_WALL_EDGE,
                          max(1, int(round(self.WALL_EDGE_UM * scale))),
                          lineType=cv2.LINE_AA)
            # the wall is out of the focal plane: soften its edge
            sigma = self.WALL_BLUR_UM * scale
            if sigma > 0.3:
                cv2.GaussianBlur(img, (0, 0), sigma, dst=img)

    # ── drawing ─────────────────────────────────────────────────────────

    def render(self, cells, mode: int, origin, scale: float,
               shape: tuple[int, int]) -> np.ndarray:
        """Render ``cells`` into a ``shape`` (h, w) uint8 frame.

        Args:
            cells: the population (objects with ``center``, ``angles``, ``r``).
            mode: channel (0 phase-contrast, 1 miRFP/H2B nuclei,
                2 mVenus/optoFGFR membrane, 4 mScarlet/ERK-KTR).
            origin: world position (px) of output pixel (0, 0).
            scale: output pixels per world px (magnification / binning).
            shape: output (height, width).
        """
        h, w = shape
        thick = max(1, int(round(self.LINE_UM * scale)))
        vis = list(self._visible(cells, origin, scale, shape))

        if mode == 1:      # miRFP: H2B nuclei
            img = np.full((h, w), self.FLUO_BG, np.uint8)
            polys = [_smooth_polygon(p, c.angles, c.r, scale * 0.5) for c, p in vis]
            if polys:
                cv2.fillPoly(img, polys, self.DAPI_NUCLEUS, lineType=cv2.LINE_AA)
            return img
        if mode == 2:      # mVenus: membrane-bound optoFGFR
            img = np.full((h, w), self.FLUO_BG, np.uint8)
            polys = [_smooth_polygon(p, c.angles, c.r, scale) for c, p in vis]
            if polys:
                cv2.polylines(img, polys, True, self.MEMBRANE_RIM, thick,
                              lineType=cv2.LINE_AA)
            return img
        if mode == 4:      # mScarlet: ERK-KTR translocation reporter
            img = np.full((h, w), self.FLUO_BG, np.uint8)
            # draw per cell: cytoplasm then nucleus, levels from activity
            for c, p in vis:
                a = float(getattr(c, "activity", 0.0))
                cyto = int(round(self.KTR_CYTO_LO
                                 + a * (self.KTR_CYTO_HI - self.KTR_CYTO_LO)))
                nuc = int(round(self.KTR_NUC_HI
                                - a * (self.KTR_NUC_HI - self.KTR_NUC_LO)))
                body = _smooth_polygon(p, c.angles, c.r, scale)
                nucleus = _smooth_polygon(p, c.angles, c.r, scale * 0.5)
                cv2.fillPoly(img, [body], cyto, lineType=cv2.LINE_AA)
                cv2.fillPoly(img, [nucleus], nuc, lineType=cv2.LINE_AA)
            return img

        # phase-contrast (mode 0)
        img = np.empty((h, w), np.uint8)
        self._draw_wells(img, origin, scale)
        bodies, halos, halos_stim, nuclei = [], [], [], []
        for c, p in vis:
            poly = _smooth_polygon(p, c.angles, c.r, scale)
            bodies.append(poly)
            (halos_stim if getattr(c, "is_stimulated", False) else halos).append(poly)
            nuclei.append(_smooth_polygon(p, c.angles, c.r, scale * 0.45))
        if bodies:
            cv2.fillPoly(img, bodies, self.PHASE_BODY, lineType=cv2.LINE_AA)
            cv2.fillPoly(img, nuclei, self.PHASE_NUCLEUS, lineType=cv2.LINE_AA)
        if halos:
            cv2.polylines(img, halos, True, self.PHASE_HALO, thick,
                          lineType=cv2.LINE_AA)
        if halos_stim:
            cv2.polylines(img, halos_stim, True, self.PHASE_HALO_STIM, thick,
                          lineType=cv2.LINE_AA)
        return img
