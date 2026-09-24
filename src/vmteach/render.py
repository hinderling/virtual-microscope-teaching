"""Cell renderer: draws the population straight into camera pixels.

Cells are polygons, so instead of rasterizing the whole world and cropping,
the renderer maps each visible cell into the camera frame (world -> sensor
pixels, objective magnification and binning folded into one ``scale``)
and lets cv2 fill it. Cost is per *visible* cell and per output pixel,
independent of the world size and of the magnification.

Image model
-----------
Every channel is derived from a few per-frame fields instead of flat
fills, so cells read as 3D objects rather than drawings:

* **thickness** ``T``: a dome per cell (nested outlines scaled toward the
  nucleus, blurred), ~1 over the nucleus, thin at the lamellipodia.
  Phase contrast darkens with it, its halo comes from its gradient (strong
  around thick parts, faint at thin protrusions), and cytoplasmic
  fluorescence scales with it.
* **nucleus**: a soft mask plus a lens-shaped dome, 1 to 3 dark nucleoli,
  and chromatin texture.
* **texture**: a tileable noise pattern in world units, pasted per cell
  at a per-cell offset, so it moves with the cell and blurs away when the
  magnification cannot resolve it (vesicles, granules, chromatin).
* per-cell **expression** levels per fluorophore (lognormal-like, kept in
  a band that segmentation copes with). No photobleaching: the sample is
  meant to run indefinitely without a reset.
* **retraction fibres** behind stretched, adhered tails, and a little
  static **debris** on the substrate (phase contrast only).

Channels (mode):
    0  phase-contrast:  neutral gray, cells darker with thickness, bright
                        halo around thick parts, lighter nucleus with dark
                        nucleoli, perinuclear granules; the well wall
                        (plastic) is darker with a bright edge
    1  miRFP (H2B):     textured, domed nuclei on black
    2  mVenus (optoFGFR): membrane-bound tool: dim whole-cell signal,
                        brighter rim and ruffles, perinuclear Golgi and
                        vesicles
    4  mScarlet (ERK-KTR): kinase translocation reporter; inactive cells
                        show a bright nucleus in a dim cytoplasm, active
                        cells a dark nucleus in a bright cytoplasm
"""

from __future__ import annotations

import cv2
import numpy as np

from vmteach.optics import blur

# Upsample the 24 cell vertices to this many polygon points (visual smoothness)
_SMOOTH_PTS = 72
_ANG_F = np.linspace(0.0, 2 * np.pi, _SMOOTH_PTS, endpoint=False)
_COS_F, _SIN_F = np.cos(_ANG_F), np.sin(_ANG_F)

# thickness dome: outline scaled toward the nucleus -> thickness level
_T_SCALES = (1.0, 0.74, 0.52, 0.34)
_T_LEVELS = (0.22, 0.5, 0.78, 1.0)

_EXPR_MAX = 1.6          # expression factors are painted as value / max
_TEX_UM = 192.0          # texture tile side, world um
_TEX_PX_PER_UM = 4.0     # texture tile generation resolution


def _smooth_pts(center: np.ndarray, angles: np.ndarray, radii: np.ndarray,
                scale: float = 1.0) -> np.ndarray:
    """Periodic linear upsampling of the vertex radii -> float polygon.

    ``center`` is in output pixels, ``radii`` in world px; ``scale``
    converts radii to output pixels.
    """
    ang_ext = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    r_ext = np.concatenate([radii, [radii[0]]])
    r_f = np.interp(_ANG_F, ang_ext, r_ext) * scale
    pts = np.empty((_SMOOTH_PTS, 2), np.float64)
    pts[:, 0] = center[0] + _COS_F * r_f
    pts[:, 1] = center[1] + _SIN_F * r_f
    return pts


def _smooth_polygon(center, angles, radii, scale: float = 1.0) -> np.ndarray:
    """int32 version of :func:`_smooth_pts` (for cv2 drawing)."""
    return np.rint(_smooth_pts(center, angles, radii, scale)).astype(np.int32)


def _toward(pts: np.ndarray, anchor: np.ndarray, s: float) -> np.ndarray:
    """int32 polygon ``pts`` scaled by ``s`` toward ``anchor``."""
    return np.rint(anchor + s * (pts - anchor)).astype(np.int32)


def _nucleus(c, center_px: np.ndarray, scale: float):
    """Nucleus of cell ``c``: (centre px, float polygon px, mean radius px).

    Slightly wobbly (it follows the mildly irregular *rest* shape, so it
    reads as round without being a perfect circle), clamped inside the
    local cell outline, and lagging a little behind the centre of a
    moving cell, as a fibroblast nucleus does.
    """
    base = float(c.base_r)
    rest = getattr(c, "rest_r", None)
    if rest is None:
        r_n = np.full(len(c.angles), 0.5 * base)
    else:
        r_n = 0.5 * (0.55 * base + 0.45 * rest)
    # never poke through the membrane: cap by the local body radius
    r_n = np.minimum(r_n, 0.8 * np.asarray(c.r))
    pos = center_px
    m = getattr(c, "motion", None)
    if m is not None:
        speed = float(np.hypot(m[0], m[1]))
        if speed > 1e-3:
            lag = 0.08 * base * min(1.0, speed / 1.5) * scale
            pos = center_px - (m / speed) * lag
    return pos, _smooth_pts(pos, c.angles, r_n, scale), float(r_n.mean()) * scale


def _nucleus_polygon(c, center_px: np.ndarray, scale: float) -> np.ndarray:
    """int32 nucleus outline (see :func:`_nucleus`)."""
    return np.rint(_nucleus(c, center_px, scale)[1]).astype(np.int32)


_W_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _interp_matrix(nv: int, npts: int) -> np.ndarray:
    """(nv, npts) periodic linear interpolation weights: vertex radii at
    ``nv`` equally spaced angles -> ``npts`` equally spaced angles."""
    key = (nv, npts)
    W = _W_CACHE.get(key)
    if W is None:
        t = np.arange(npts) * nv / npts
        i0 = np.floor(t).astype(int) % nv
        fr = t - np.floor(t)
        W = np.zeros((nv, npts))
        W[i0, np.arange(npts)] += 1.0 - fr
        W[(i0 + 1) % nv, np.arange(npts)] += fr
        _W_CACHE[key] = W
    return W


def _geometry(vis, scale: float, npts: int):
    """All visible cells at once: body and nucleus outlines (n, npts, 2)
    float px, nucleus centres (n, 2) and mean nucleus radii (n,) px.

    Vectorized version of :func:`_smooth_pts` + :func:`_nucleus` (the
    cells' vertices sit at equally spaced angles)."""
    cells = [c for c, _ in vis]
    centers = np.array([p for _, p in vis])
    r = np.array([c.r for c in cells])
    nv = r.shape[1]
    base = np.array([float(c.base_r) for c in cells])
    rest = np.array([getattr(c, "rest_r", np.full(nv, 0.0)) for c in cells])
    motion = np.array([getattr(c, "motion", np.zeros(2)) for c in cells])
    ang = np.arange(npts) * 2 * np.pi / npts
    cs, sn = np.cos(ang), np.sin(ang)
    W = _interp_matrix(nv, npts)

    rb = (r @ W) * scale
    bodies = np.empty((len(cells), npts, 2))
    bodies[..., 0] = centers[:, :1] + rb * cs
    bodies[..., 1] = centers[:, 1:] + rb * sn

    r_n = np.where(rest > 0, 0.5 * (0.55 * base[:, None] + 0.45 * rest),
                   0.5 * base[:, None])
    r_n = np.minimum(r_n, 0.8 * r)
    speed = np.hypot(motion[:, 0], motion[:, 1])
    lag = 0.08 * base * np.minimum(1.0, speed / 1.5) * scale
    unit = motion / np.maximum(speed, 1e-9)[:, None]
    ncent = centers - np.where(speed[:, None] > 1e-3, unit * lag[:, None], 0.0)
    rn = (r_n @ W) * scale
    nuclei = np.empty_like(bodies)
    nuclei[..., 0] = ncent[:, :1] + rn * cs
    nuclei[..., 1] = ncent[:, 1:] + rn * sn
    return bodies, nuclei, ncent, r_n.mean(axis=1) * scale


def _scaled(pts: np.ndarray, anchors: np.ndarray, s: float) -> list:
    """(n, k, 2) outlines scaled by ``s`` toward per-outline anchors ->
    list of int32 polygons for cv2."""
    if s == 1.0:
        return list(np.rint(pts).astype(np.int32))
    a = anchors[:, None, :]
    return list(np.rint(a + s * (pts - a)).astype(np.int32))


def _paint(canvas: np.ndarray, polys, values, bins: int = 32) -> None:
    """Fill each polygon with its own value in [0, 1] on a uint8 canvas.

    Values are quantized to ``bins`` levels and polygons sharing a level
    are filled in one cv2 call, so the cost is per level, not per cell.
    """
    if not polys:
        return
    q = np.clip(np.rint(np.asarray(values, float) * bins), 0, bins).astype(int)
    for k in np.unique(q):
        if k == 0:
            continue
        group = [polys[i] for i in np.flatnonzero(q == k)]
        cv2.fillPoly(canvas, group, int(round(k * 255 / bins)),
                     lineType=cv2.LINE_AA)


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


class _Look:
    """Static per-cell appearance (expression, nucleoli, texture offset).

    Drawn from the cell's own seed, so it is reproducible and survives
    ``reset()`` identically.
    """

    __slots__ = ("expr", "nucleoli", "tex_off", "golgi")

    def __init__(self, seed: int):
        rng = np.random.RandomState((int(seed) * 7919 + 17) % (2 ** 32))
        # expression: lognormal-like spread, clipped to a band that a
        # plain threshold still segments (no non-expressing cells)
        self.expr = {
            1: float(np.clip(np.exp(rng.normal(0.0, 0.22)), 0.7, 1.45)),  # miRFP
            2: float(np.clip(np.exp(rng.normal(0.0, 0.30)), 0.55, 1.6)),  # mVenus
            4: float(np.clip(np.exp(rng.normal(0.0, 0.12)), 0.85, 1.2)),  # KTR
        }
        # 1-3 nucleoli: (angle, distance as fraction of nuclear radius,
        # radius as fraction of nuclear radius); kept off-centre so the
        # centroid sampling disc of measure_activity stays mostly clear
        k = rng.randint(1, 4)
        self.nucleoli = [(rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 0.55),
                          rng.uniform(0.14, 0.22)) for _ in range(k)]
        self.tex_off = rng.uniform(0.0, 0.33, size=2)   # fraction of tile
        self.golgi = rng.uniform(0.35, 0.55)            # offset, x base_r


class CellRenderer:
    """Draws the wells and the cell population into a camera-sized image."""

    # phase contrast (gray levels)
    PHASE_BG = 128
    PHASE_THICK = 26          # darkening at full thickness
    PHASE_NUC_LIGHT = 13      # nucleus reads lighter than the cytoplasm
    PHASE_NUCLEOLUS = 18      # nucleoli darkening
    PHASE_GRANULE = 22        # perinuclear granules darkening
    PHASE_HALO = 300          # halo gain on the thickness gradient
    HALO_UM = 1.6             # halo width
    PHASE_WALL, PHASE_WALL_EDGE = 88, 205     # plastic, bright well edge
    WALL_EDGE_UM = 6.0                         # edge line width, world um
    WALL_BLUR_UM = 5.0                         # softness of the wall edge
    # fluorescence (gray levels at expression 1)
    FLUO_BG = 6
    H2B = 160
    MEMBRANE = 150
    # ERK-KTR (mode 4): intensities interpolate with cell.ktr / activity
    KTR_CYTO_LO, KTR_CYTO_HI = 45, 110      # cytoplasm: dim -> bright
    KTR_NUC_LO, KTR_NUC_HI = 40, 170        # nucleus:  bright -> dark
    LINE_UM = 2.0                 # membrane rim width, in world um

    def __init__(self, wells, well_half: float, corner_radius: float,
                 debris: np.ndarray | None = None):
        self.wells = np.asarray(wells, dtype=float)   # (n, 2) centres, um
        self.well_half = float(well_half)
        self.corner_radius = float(corner_radius)
        # static substrate debris: (n, 3) world x, y, radius um
        self.debris = (np.zeros((0, 3)) if debris is None
                       else np.asarray(debris, float))
        self._debris_shapes = None
        self.last_visible: list = []
        rng = np.random.default_rng(4242)
        n = int(_TEX_UM * _TEX_PX_PER_UM)
        tile = cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32),
                                (0, 0), 0.9 * _TEX_PX_PER_UM,
                                borderType=cv2.BORDER_REFLECT)
        self._tile = (tile - tile.mean()) / tile.std()
        self._tile_cache: dict[float, np.ndarray] = {}

    # ── appearance helpers ──────────────────────────────────────────────

    @staticmethod
    def look(c) -> _Look:
        """The (lazily created) static appearance of cell ``c``."""
        lk = getattr(c, "_look", None)
        if lk is None:
            lk = _Look(getattr(c, "seed", 0))
            c._look = lk
        return lk

    def _texture_tile(self, scale: float) -> np.ndarray:
        """Texture tile resampled to ``scale`` output px per um (cached).

        Downsampling averages the pattern away, so structure finer than
        the pixel size disappears at low magnification, as it should.
        """
        key = round(scale, 4)
        tile = self._tile_cache.get(key)
        if tile is None:
            n = max(8, int(round(_TEX_UM * scale)))
            interp = (cv2.INTER_AREA if scale < _TEX_PX_PER_UM
                      else cv2.INTER_LINEAR)
            tile = cv2.resize(self._tile, (n, n), interpolation=interp)
            self._tile_cache[key] = tile
        return tile

    def _texture(self, shape, cells_px, scale) -> np.ndarray:
        """Per-frame texture field: the tile pasted over each cell's box."""
        h, w = shape
        tex = np.zeros((h, w), np.float32)
        tile = self._texture_tile(scale)
        n = tile.shape[0]
        for c, pts in cells_px:
            x0 = max(0, int(pts[:, 0].min()) - 1)
            y0 = max(0, int(pts[:, 1].min()) - 1)
            x1 = min(w, int(pts[:, 0].max()) + 2)
            y1 = min(h, int(pts[:, 1].max()) + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            ox, oy = (self.look(c).tex_off * n).astype(int)
            # offset of the box's frame-clipped corner inside the cell box
            bx = x0 - int(pts[:, 0].min()) + 1
            by = y0 - int(pts[:, 1].min()) + 1
            tx, ty = ox + max(0, bx), oy + max(0, by)
            cw = min(x1 - x0, n - tx)
            ch = min(y1 - y0, n - ty)
            if cw > 0 and ch > 0:
                tex[y0:y0 + ch, x0:x0 + cw] = tile[ty:ty + ch, tx:tx + cw]
        return tex

    # ── geometry ────────────────────────────────────────────────────────

    def _visible(self, cells, origin, scale, shape):
        """Yield (cell, centre_px) for every cell that touches the frame."""
        h, w = shape
        for c in cells:
            m = float(c.r.max()) * scale * 1.3 + 3 * scale + 2
            x = (c.center[0] - origin[0]) * scale
            y = (c.center[1] - origin[1]) * scale
            if -m <= x <= w + m and -m <= y <= h + m:
                yield c, np.array([x, y])

    def _draw_wells(self, img: np.ndarray, origin, scale: float) -> None:
        """Phase contrast: plastic outside the wells, bright well edge.

        The result depends only on the stage position and magnification,
        so it is cached: a live view at a fixed position pays for it once.
        """
        key = (round(float(origin[0]), 3), round(float(origin[1]), 3),
               round(scale, 6), img.shape)
        cached = getattr(self, "_wall_cache", None)
        if cached is not None and cached[0] == key:
            img[:] = cached[1]
            return
        self._draw_wells_uncached(img, origin, scale)
        self._wall_cache = (key, img.copy())

    def _draw_wells_uncached(self, img: np.ndarray, origin, scale: float) -> None:
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
            img[:] = np.clip(blur(img.astype(np.float32),
                                  self.WALL_BLUR_UM * scale), 0, 255)

    def _fibres(self, c, center_px, scale):
        """Retraction fibres: thin membrane tethers behind a stretched,
        adhered tail, as (p0, p1) float segments in output px."""
        adh = getattr(c, "adhesion", None)
        rest = getattr(c, "rest_r", None)
        m = getattr(c, "motion", None)
        if adh is None or rest is None or m is None:
            return []
        speed = float(np.hypot(m[0], m[1]))
        if speed < 0.15:
            return []
        back = np.arctan2(-m[1], -m[0])
        out = []
        for v in np.flatnonzero(adh > 0):
            stretch = (c.r[v] - rest[v]) / rest[v]
            if stretch < 0.25 or np.cos(c.angles[v] - back) < 0.6:
                continue
            a = c.angles[v]
            r0 = c.r[v] * scale
            length = (4.0 + 12.0 * min(stretch, 1.0)) * scale
            p0 = center_px + r0 * np.array([np.cos(a), np.sin(a)])
            for da in (-0.12, 0.1):
                p1 = p0 + length * np.array([np.cos(a + da), np.sin(a + da)])
                out.append((p0, p1))
        return out

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
        vis = list(self._visible(cells, origin, scale, shape))
        self.last_visible = [c for c, _ in vis]

        # per-cell geometry, shared by every field (vectorized; coarser
        # outlines when cells are small on screen)
        looks = [self.look(c) for c, _ in vis]
        if vis:
            npts = _SMOOTH_PTS if scale >= 0.8 else _SMOOTH_PTS // 2
            bodies, nuclei, ncent, nrad = _geometry(vis, scale, npts)
            body_i = _scaled(bodies, ncent, 1.0)
            nuc_i = _scaled(nuclei, ncent, 1.0)
        else:
            bodies = nuclei = np.zeros((0, 1, 2))
            ncent, nrad, body_i, nuc_i = np.zeros((0, 2)), np.zeros(0), [], []
        # sub-pixel texture is invisible at low magnification: skip it
        use_tex = scale >= 0.6
        f = np.float32

        def expr(ch):
            return np.array([lk.expr[ch] for lk in looks])

        def nucleus_fields(values):
            """Soft nucleus mask and a lens-shaped dome weighted by values."""
            canvas = np.zeros((h, w), np.uint8)
            _paint(canvas, nuc_i, 0.72 * values)
            _paint(canvas, _scaled(nuclei, ncent, 0.6), values)
            return blur(canvas.astype(f) / 255.0, 0.9 * scale)

        def nucleoli():
            canvas = np.zeros((h, w), np.uint8)
            for lk, a, nr in zip(looks, ncent, nrad):
                for ang, dist, rad in lk.nucleoli:
                    rpx = rad * nr
                    if rpx < 0.8:
                        continue
                    cx = a[0] + np.cos(ang) * dist * nr
                    cy = a[1] + np.sin(ang) * dist * nr
                    cv2.circle(canvas, (int(round(cx * 4)), int(round(cy * 4))),
                               int(round(rpx * 4)), 255, -1,
                               lineType=cv2.LINE_AA, shift=2)
            return blur(canvas.astype(f) / 255.0, 0.35 * scale)

        def thickness():
            canvas = np.zeros((h, w), np.uint8)
            for s_, lv in zip(_T_SCALES, _T_LEVELS):
                cv2.fillPoly(canvas, _scaled(bodies, ncent, s_),
                             int(round(lv * 255)), lineType=cv2.LINE_AA)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, body_i, 255, lineType=cv2.LINE_AA)
            dome = blur(canvas.astype(f) / 255.0, 2.2 * scale)
            return dome * (mask.astype(f) / 255.0), mask

        if mode == 1:      # miRFP: H2B nuclei
            nd = nucleus_fields(expr(1) / _EXPR_MAX) * _EXPR_MAX
            if not vis:
                return np.full((h, w), self.FLUO_BG, np.uint8)
            tex = self._texture(shape, list(zip(self.last_visible, nuclei)), scale) if use_tex else 0.0
            sig = self.H2B * nd * (1.0 - 0.3 * nucleoli()) * (1.0 + 0.10 * tex)
            return np.clip(self.FLUO_BG + sig, 0, 255).astype(np.uint8)

        if mode == 2:      # mVenus: membrane-bound optoFGFR
            if not vis:
                return np.full((h, w), self.FLUO_BG, np.uint8)
            e = expr(2) / _EXPR_MAX
            fill = np.zeros((h, w), np.uint8)
            _paint(fill, body_i, e)
            rim = np.zeros((h, w), np.uint8)
            q = np.clip(np.rint(e * 32), 0, 32).astype(int)
            thick = max(1, int(round(self.LINE_UM * scale)))
            for k in np.unique(q):
                group = [body_i[i] for i in np.flatnonzero(q == k)]
                cv2.polylines(rim, group, True, int(round(k * 255 / 32)),
                              thick, lineType=cv2.LINE_AA)
            golgi = np.zeros((h, w), np.uint8)
            for c, lk, a, ev in zip(self.last_visible, looks, ncent, e):
                pol = float(c.polarity[0]) if hasattr(c, "polarity") else 0.0
                d = lk.golgi * c.base_r * scale
                g = a + d * np.array([np.cos(pol), np.sin(pol)])
                cv2.circle(golgi, (int(round(g[0] * 4)), int(round(g[1] * 4))),
                           int(round(0.22 * c.base_r * scale * 4)),
                           int(round(ev * 255)), -1, lineType=cv2.LINE_AA,
                           shift=2)
            fib = np.zeros((h, w), np.uint8)
            for c, p in vis:
                for p0, p1 in self._fibres(c, p, scale):
                    cv2.line(fib, tuple(np.rint(p0 * 4).astype(int)),
                             tuple(np.rint(p1 * 4).astype(int)), 255,
                             max(1, int(round(0.5 * scale))),
                             lineType=cv2.LINE_AA, shift=2)
            tex = self._texture(shape, list(zip(self.last_visible, bodies)), scale) if use_tex else 0.0
            fill_f = fill.astype(f) / 255.0 * _EXPR_MAX
            vesicles = np.clip(tex - 1.8, 0.0, None) * fill_f
            sig = (0.30 * fill_f
                   + 0.45 * blur(rim.astype(f) / 255.0, 0.7 * scale) * _EXPR_MAX
                   + 0.45 * blur(golgi.astype(f) / 255.0, 0.8 * scale)
                   * _EXPR_MAX * (1.0 + 0.5 * tex)
                   + 0.35 * vesicles
                   + 0.35 * fib.astype(f) / 255.0)
            return np.clip(self.FLUO_BG + self.MEMBRANE * sig, 0, 255).astype(np.uint8)

        if mode == 4:      # mScarlet: ERK-KTR translocation reporter
            if not vis:
                return np.full((h, w), self.FLUO_BG, np.uint8)
            a = np.array([float(getattr(c, "ktr", getattr(c, "activity", 0.0)))
                          for c in self.last_visible])
            e = expr(4)
            cyto_lv = e * (self.KTR_CYTO_LO + a * (self.KTR_CYTO_HI - self.KTR_CYTO_LO))
            nuc_lv = e * (self.KTR_NUC_HI - a * (self.KTR_NUC_HI - self.KTR_NUC_LO))
            vmax = 255.0
            T, _ = thickness()
            cyto = np.zeros((h, w), np.uint8)
            _paint(cyto, body_i, cyto_lv / vmax, bins=64)
            nmask = nucleus_fields(np.ones(len(vis)))
            nd = np.zeros((h, w), np.uint8)
            _paint(nd, nuc_i, 0.8 * nuc_lv / vmax, bins=64)
            _paint(nd, _scaled(nuclei, ncent, 0.6), nuc_lv / vmax, bins=64)
            nd_f = blur(nd.astype(f), 0.9 * scale)
            tex = self._texture(shape, list(zip(self.last_visible, bodies)), scale) if use_tex else 0.0
            sig = (cyto.astype(f) * (0.45 + 0.55 * T) * (1.0 - 0.85 * np.minimum(nmask, 1.0))
                   + nd_f * (1.0 - 0.35 * nucleoli())) * (1.0 + 0.06 * tex)
            return np.clip(self.FLUO_BG + sig, 0, 255).astype(np.uint8)

        # phase-contrast (mode 0)
        base = np.empty((h, w), np.uint8)
        self._draw_wells(base, origin, scale)
        img = base.astype(f)
        extra = np.zeros((h, w), f)          # thickness-like, halo only
        if len(self.debris):
            dbr = np.zeros((h, w), np.uint8)
            for x, y, r in self.debris:
                px, py = (x - origin[0]) * scale, (y - origin[1]) * scale
                rp = r * scale
                if -rp < px < w + rp and -rp < py < h + rp and rp >= 0.6:
                    cv2.circle(dbr, (int(round(px * 4)), int(round(py * 4))),
                               int(round(rp * 4)), 255, -1,
                               lineType=cv2.LINE_AA, shift=2)
            d = blur(dbr.astype(f) / 255.0, 0.4 * scale)
            img -= 30.0 * d
            extra += 0.9 * d
        if vis:
            T, mask = thickness()
            mask_f = blur(mask.astype(f) / 255.0, 0.4 * scale)
            nm = nucleus_fields(np.ones(len(vis)))
            nm = np.minimum(nm, 1.0)
            tex = self._texture(shape, list(zip(self.last_visible, bodies)), scale) if use_tex else 0.0
            granules = np.clip(tex - 1.0, 0.0, None) * T * T * (1.0 - nm)
            fib = np.zeros((h, w), np.uint8)
            for c, p in vis:
                for p0, p1 in self._fibres(c, p, scale):
                    cv2.line(fib, tuple(np.rint(p0 * 4).astype(int)),
                             tuple(np.rint(p1 * 4).astype(int)), 255,
                             max(1, int(round(0.5 * scale))),
                             lineType=cv2.LINE_AA, shift=2)
            fib_f = fib.astype(f) / 255.0
            img += (-self.PHASE_THICK * T
                    + self.PHASE_NUC_LIGHT * nm
                    - self.PHASE_NUCLEOLUS * nucleoli()
                    - self.PHASE_GRANULE * granules
                    - 16.0 * np.clip(blur(nm, 1.1 * scale) - nm, 0.0, None) * 4.0
                    - 12.0 * fib_f)
            extra += 0.45 * mask_f + 0.6 * T + 0.35 * fib_f
        if vis or len(self.debris):
            halo = blur(extra, self.HALO_UM * scale) - extra
            img += self.PHASE_HALO * np.clip(halo, 0.0, None)
        return np.clip(img, 0, 255).astype(np.uint8)
