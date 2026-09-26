"""Cell physics (numba) and the optogenetic cell.

The population lives in flat arrays owned by :class:`vmteach.sim.OptoCellSim`
(``centers``, ``velocities``, ``radii``, ``adhesions``, ...); each
:class:`OptogeneticCell` holds *views* into its own row, so physics,
collisions, stimulation and rendering all see one copy of the state and
nothing has to be synced.

Migration model (stick-slip crawling)
-------------------------------------
Cells are star polygons (24 vertices at fixed angles, per-vertex radius)
that move the way fibroblasts do, not by gliding:

* every vertex can **adhere** to the substrate. Adhered vertices act as
  springs anchored at their tip: when the cell body moves, they stay
  (approximately) world-anchored, so the radius behind a moving cell
  stretches into a dragging tail while overrun front adhesions slacken.
* a cell maintains at most ``N_FRONTS`` **protrusion fronts**: coherent
  lamellipodial arcs spanning several adjacent vertices that grow for a
  few seconds and adhere, so they pull the centre forward. New fronts
  spawn stochastically, aimed near an internal polarity direction (a
  persistent random walk, re-pointed by light).
* the centre moves by **force balance**: sum of the tension in all adhered
  vertices, divided by a drag that grows with the number of adhesions. A
  fully adhered resting cell is sticky and barely moves; motion happens
  when front protrusions build enough tension to matter.
* rear adhesions **detach under tension** (stretch beyond a threshold rips
  them off; the freed vertex retracts toward its rest radius), so
  displacement comes in lurches: protrude, stick, pull the tail off.

The rest shape itself is irregular (per-cell random low-frequency radial
modes), so even idle cells are lumpy rather than circular.

Cells live inside wells: rounded squares with a hard boundary that the
cells collide with (1 world px = 1 um; nothing wraps).
"""
import numpy as np
from numba import njit, prange

# ── crawling / adhesion model parameters ──────────────────────────────
FRICTION = 3.0          # /s: decay of the stimulation impulse velocity
POL_DIFF = 0.7          # rad/sqrt(s): polarity direction diffusion
REPOL_TAU = 35.0        # s: mean time between spontaneous repolarizations
N_FRONTS = 3            # protrusion fronts a cell can maintain at once
FRONT_SPAWN = 0.8       # /s per free slot: new spontaneous front
FRONT_DRIVE_SPAWN = 2.0  # extra spawn rate at full signalling activity
FRONT_TTL_MIN = 2.0     # s: front lifetime, uniform in [MIN, MAX]
FRONT_TTL_MAX = 5.0
FRONT_W = 0.9           # rad: angular half-width of a front (~±50 deg)
FRONT_GROW = 3.0        # um/s: radial growth at the front apex
FRONT_DRIVE_AMP = 0.45  # growth boost at full signalling activity
FRONT_SPREAD_REST = 2.6  # rad: spawn direction noise around polarity, resting
FRONT_SPREAD_DRIVE = 0.25  # rad: spawn direction noise at full activity
MAX_PROTR = 0.5         # cap on protrusion beyond rest, fraction of base_r
TRACTION = 0.5          # /s: spring constant of adhered vertices
DRAG_BODY = 2.0         # baseline drag on the cell body
DRAG_ADH = 0.6          # extra drag per adhered vertex
K_ON = 1.5              # /s: re-adhesion rate of relaxed free vertices
K_OFF = 0.04            # /s: spontaneous adhesion release
K_OFF_TENSION = 8.0     # /s: extra release rate ~ excess stretch^2
DETACH_STRETCH = 0.35   # relative stretch where tension release kicks in
REAR_RELEASE = 0.5      # /s: rear adhesion destabilization at full activity
K_OFF_SLACK = 1.0       # /s: release rate of overrun (compressed) adhesions
RETRACT = 1.2           # /s: relaxation of free vertices toward rest
CREEP = 0.05            # /s: slow slippage of adhered vertices toward rest
CURVE_RELAX = 0.14      # Laplacian smoothing of free vertices, per step
CURVE_RELAX_ADH = 0.03  # gentler smoothing of adhered (anchored) vertices
RUFFLE_STD = 0.008      # membrane noise on free vertices, fraction of base_r
AREA_GAIN = 0.5         # soft area conservation gain, per step
AREA_W_ADH = 0.3        # share of the area correction on adhered vertices
AREA_STEP_MAX = 0.015   # max radius change per step from area correction
SHAPE_MODES = 0.34      # amplitude of the random rest-shape modes
CONTACT_SURFACE = 0.92  # membranes conform at this fraction of the
                        # neighbour's local radius (contact inhibition)
BODY_GUARD = 0.7        # cell bodies (nucleus scale) may not merge
PRESSURE = 1.2         # /s: elastic push-back of a compressed cell


# ------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------ #
@njit(cache=True)
def well_sdf(px: float, py: float, half: float, corner: float) -> tuple:
    """Signed distance (negative inside) from a point, relative to the well
    centre, to the boundary of a rounded square of half-size ``half`` and
    corner radius ``corner``; plus the outward unit normal."""
    hx = half - corner
    qx = abs(px) - hx
    qy = abs(py) - hx
    ox = max(qx, 0.0)
    oy = max(qy, 0.0)
    outer = np.sqrt(ox * ox + oy * oy)
    d = outer + min(max(qx, qy), 0.0) - corner
    sx = 1.0 if px >= 0 else -1.0
    sy = 1.0 if py >= 0 else -1.0
    if outer > 0.0:
        gx, gy = sx * ox / outer, sy * oy / outer
    elif qx > qy:
        gx, gy = sx, 0.0
    else:
        gx, gy = 0.0, sy
    return d, gx, gy


@njit(cache=True)
def confine(center: np.ndarray, vel: np.ndarray, rmax: float,
            wx: float, wy: float, half: float, corner: float) -> None:
    """Keep a cell (in place) inside its well: push the centre back so the
    membrane touches the wall at most, and drop the outward velocity."""
    d, gx, gy = well_sdf(center[0] - wx, center[1] - wy, half, corner)
    push = d + rmax
    if push > 0.0:
        center[0] -= push * gx
        center[1] -= push * gy
        vn = vel[0] * gx + vel[1] * gy
        if vn > 0.0:
            vel[0] -= vn * gx
            vel[1] -= vn * gy


@njit(cache=True)
def polygon_area(pts: np.ndarray) -> float:
    """Calculate polygon area (shoelace)."""
    x, y = pts[:, 0], pts[:, 1]
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= y[i] * x[j]
    return abs(area) * 0.5


@njit(cache=True)
def calculate_vertices(center: np.ndarray, angles: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Calculate vertex position."""
    n = len(angles)
    vertices = np.empty((n, 2), dtype=np.float64)
    vertices[:, 0] = center[0] + np.cos(angles) * r
    vertices[:, 1] = center[1] + np.sin(angles) * r
    return vertices


@njit(cache=True)
def update_cell_physics(center: np.ndarray, vel: np.ndarray, r: np.ndarray,
                        rest: np.ndarray, adh: np.ndarray,
                        pol_arr: np.ndarray, motion: np.ndarray,
                        front_ang: np.ndarray, front_ttl: np.ndarray,
                        angles: np.ndarray, base_r: float, area0: float,
                        drive: float, dt: float, seed: int = 0) -> None:
    """One stick-slip crawling step for one cell (in place).

    ``adh[i] > 0`` marks vertex *i* as adhered to the substrate,
    ``pol_arr[0]`` is the polarity angle, ``motion`` receives the body
    velocity of this step (used by the renderer for the nucleus lag).
    ``front_ang``/``front_ttl`` (N_FRONTS slots) are the cell's active
    protrusion fronts: coherent lamellipodial arcs that grow and adhere
    for a few seconds. ``drive`` (0..1, the cell's signalling activity)
    spawns fronts more often and aims them at the polarity direction, so
    an optogenetically activated cell crawls where the light pointed it.
    """
    np.random.seed(seed)
    n = len(r)
    sq = np.sqrt(dt)

    # 1. polarity: persistent random walk, occasional repolarization
    pol = pol_arr[0] + np.random.normal(0.0, POL_DIFF) * sq
    if np.random.random() < dt / REPOL_TAU:
        pol = np.random.uniform(0.0, 2.0 * np.pi)
    pol_arr[0] = pol

    # 2. protrusion fronts: at most N_FRONTS coherent lamellipodial arcs.
    #    An active front pushes a contiguous group of vertices outward
    #    (cos^2 apex profile) and adheres them; expired slots respawn
    #    stochastically, aimed near the polarity direction when driven
    grow0 = FRONT_GROW * (1.0 - FRONT_DRIVE_AMP + FRONT_DRIVE_AMP * drive) * dt
    cap = MAX_PROTR * base_r
    for f in range(len(front_ttl)):
        if front_ttl[f] > 0.0:
            front_ttl[f] -= dt
            for i in range(n):
                delta = angles[i] - front_ang[f]
                while delta > np.pi:
                    delta -= 2.0 * np.pi
                while delta < -np.pi:
                    delta += 2.0 * np.pi
                if abs(delta) < FRONT_W:
                    w = np.cos(0.5 * np.pi * delta / FRONT_W)
                    if r[i] < rest[i] + cap:
                        r[i] += grow0 * w * w
                    if w * w > 0.3:
                        adh[i] = 1.0
        else:
            rate = FRONT_SPAWN * (1.0 + FRONT_DRIVE_SPAWN * drive)
            if np.random.random() < rate * dt:
                spread = (FRONT_SPREAD_REST * (1.0 - drive)
                          + FRONT_SPREAD_DRIVE * drive)
                front_ang[f] = pol + np.random.normal(0.0, spread)
                front_ttl[f] = FRONT_TTL_MIN + (
                    (FRONT_TTL_MAX - FRONT_TTL_MIN) * np.random.random())

    # 3. force balance: adhered vertices under tension pull the centre,
    #    every adhesion adds drag (a fully stuck resting cell barely moves).
    #    Signalling activity boosts contractility, so an activated cell
    #    translates its protrusions into motion much more effectively.
    tr = TRACTION * (0.9 + 0.6 * drive)
    fx = 0.0
    fy = 0.0
    n_adh = 0.0
    for i in range(n):
        if adh[i] > 0.0:
            n_adh += 1.0
            s = r[i] - rest[i]
            if s > 0.0:            # slack adhesions do not push
                fx += tr * s * np.cos(angles[i])
                fy += tr * s * np.sin(angles[i])
    mob = 1.0 / (DRAG_BODY + DRAG_ADH * n_adh)
    vx = fx * mob + vel[0]         # crawl + stimulation impulse channel
    vy = fy * mob + vel[1]
    dx = vx * dt
    dy = vy * dt
    center[0] += dx
    center[1] += dy
    motion[0] = vx
    motion[1] = vy
    vel[0] *= max(0.0, 1.0 - FRICTION * dt)
    vel[1] *= max(0.0, 1.0 - FRICTION * dt)

    # 4. adhered vertices stay world-anchored: compensate the body motion
    #    radially. The radius behind a moving cell grows (dragging tail),
    #    overrun front adhesions slacken.
    for i in range(n):
        if adh[i] > 0.0:
            r[i] -= dx * np.cos(angles[i]) + dy * np.sin(angles[i])

    # 5. adhesion turnover: tension rips vertices off (stick-slip), overrun
    #    ones release quietly, relaxed free vertices re-stick. In a driven
    #    (signalling) cell, front-rear polarization destabilizes the rear
    #    adhesions, so the tail lets go instead of anchoring the cell:
    #    that release sets the migration speed
    for i in range(n):
        s_rel = (r[i] - rest[i]) / rest[i]
        if adh[i] > 0.0:
            k = K_OFF
            over = s_rel - DETACH_STRETCH
            if over > 0.0:
                k += K_OFF_TENSION * over * over
            if s_rel < -0.10:
                k += K_OFF_SLACK
            if drive > 0.0 and np.cos(angles[i] - pol) < -0.2:
                k += REAR_RELEASE * drive
            if np.random.random() < k * dt:
                adh[i] = 0.0
        elif abs(s_rel) < 0.12:
            if np.random.random() < K_ON * dt:
                adh[i] = 1.0

    # 6. membrane mechanics: free vertices retract toward the rest shape
    #    and ruffle; curvature smoothing damps the *deviation* from the
    #    rest shape (so the lumpy rest silhouette itself is preserved),
    #    gently on anchored vertices so the tail is not smoothed away
    for i in range(n):
        dev_p = r[(i + 1) % n] - rest[(i + 1) % n]
        dev_m = r[(i - 1) % n] - rest[(i - 1) % n]
        lap = dev_p + dev_m - 2.0 * (r[i] - rest[i])
        if adh[i] <= 0.0:
            r[i] += RETRACT * dt * (rest[i] - r[i]) + CURVE_RELAX * lap
            r[i] += np.random.normal(0.0, RUFFLE_STD * base_r)
        else:
            # adhesion creep: standing tension slips slowly back toward
            # rest, so only fresh protrusions keep a cell moving
            r[i] += CURVE_RELAX_ADH * lap + CREEP * dt * (rest[i] - r[i])

    # 7. soft area conservation, spread over the whole outline (mostly the
    #    free vertices, a little on the anchored ones) and capped per step:
    #    dumping it on the few free vertices made a vertex that had just
    #    let go snap inward. Then keep radii in a sane band
    verts = calculate_vertices(center, angles, r)
    area = polygon_area(verts)
    if area > 0.0:
        target = AREA_GAIN * (area0 - area) / area0   # relative area change
        wsum = 0.0
        for i in range(n):
            wsum += 1.0 if adh[i] <= 0.0 else AREA_W_ADH
        # scaling radius i by (1 + f_i) changes the area by ~2 f_i / n
        k = target * n / (2.0 * wsum)
        for i in range(n):
            wi = 1.0 if adh[i] <= 0.0 else AREA_W_ADH
            fi = min(max(k * wi, -AREA_STEP_MAX), AREA_STEP_MAX)
            r[i] *= 1.0 + fi
    for i in range(n):
        r[i] = min(max(r[i], 0.5 * base_r), 2.2 * base_r)


@njit(parallel=True, cache=True)
def update_all_cells_parallel(centers: np.ndarray, velocities: np.ndarray,
                              radii: np.ndarray, rest_radii: np.ndarray,
                              adhesions: np.ndarray, polarities: np.ndarray,
                              motions: np.ndarray, front_angs: np.ndarray,
                              front_ttls: np.ndarray, angles: np.ndarray,
                              base_radii: np.ndarray, areas: np.ndarray,
                              drives: np.ndarray,
                              cell_well: np.ndarray, wells: np.ndarray,
                              half: float, corner: float, dt: float,
                              step_count: int = 0) -> None:
    """Update all cells in parallel (in place) using Numba prange, then
    confine each to its well (``wells``: (n_wells, 2) centres)."""
    n_cells = len(centers)
    for i in prange(n_cells):  # parallel loop
        update_cell_physics(
            centers[i], velocities[i], radii[i], rest_radii[i],
            adhesions[i], polarities[i:i + 1], motions[i],
            front_angs[i], front_ttls[i],
            angles, base_radii[i], areas[i], drives[i], dt,
            seed=i + step_count * n_cells)
        w = cell_well[i]
        confine(centers[i], velocities[i], radii[i].max(),
                wells[w, 0], wells[w, 1], half, corner)


@njit(cache=True)
def build_grid(centers: np.ndarray, cell_size: float):
    """Uniform spatial grid over the cell centres (CSR layout).

    Returns ``(order, start, bx, by, nx, ny)``: the cells sorted by bin
    (stable, so deterministic), per-bin start offsets into ``order``, and
    each cell's bin coordinates. Neighbour searches then scan the 3 x 3
    surrounding bins instead of every cell, which makes the contact
    passes ~linear in the population size.
    """
    n = centers.shape[0]
    x0 = centers[:, 0].min()
    y0 = centers[:, 1].min()
    nx = int((centers[:, 0].max() - x0) / cell_size) + 1
    ny = int((centers[:, 1].max() - y0) / cell_size) + 1
    bx = np.empty(n, np.int64)
    by = np.empty(n, np.int64)
    key = np.empty(n, np.int64)
    for i in range(n):
        bx[i] = int((centers[i, 0] - x0) / cell_size)
        by[i] = int((centers[i, 1] - y0) / cell_size)
        key[i] = by[i] * nx + bx[i]
    order = np.argsort(key, kind="mergesort")
    start = np.zeros(nx * ny + 1, np.int64)
    for i in range(n):
        start[key[i] + 1] += 1
    for b in range(nx * ny):
        start[b + 1] += start[b]
    return order, start, bx, by, nx, ny


@njit(parallel=True, cache=True)
def conform_membranes(centers: np.ndarray, radii_in: np.ndarray,
                      radii_out: np.ndarray, rest_radii: np.ndarray,
                      base_radii: np.ndarray, angles: np.ndarray,
                      cell_well: np.ndarray, pushes: np.ndarray,
                      dt: float, order: np.ndarray, start: np.ndarray,
                      bx: np.ndarray, by: np.ndarray, gnx: int,
                      gny: int) -> None:
    """Contact inhibition of protrusion plus contact pressure (in place).

    Any vertex that would sit inside a neighbouring cell's outline is
    clamped back to the contact surface (``radii_out``): membranes press
    against each other and conform instead of overlapping, and a
    protrusion that runs into a neighbour stops advancing, which stalls
    the traction it provides and thereby the cell's migration.

    A cell compressed by its neighbours pushes back: every contacting
    vertex squeezed below its rest radius contributes an elastic force
    on the cell's centre, away from the contact (``pushes``, a
    displacement for this step). Weak enough that crowded cells still
    snuggle up and fill voids, strong enough that a crowd cannot squash
    cells far below their own area.

    Reads only the ``radii_in`` snapshot and writes only row *i* of
    ``radii_out`` and ``pushes``, so the parallel loop is race-free and
    deterministic.
    """
    n, nv = radii_in.shape
    two_pi = 2.0 * np.pi
    for i in prange(n):
        cx, cy = centers[i, 0], centers[i, 1]
        rmax_i = radii_in[i].max()
        pushes[i, 0] = 0.0
        pushes[i, 1] = 0.0
        for gy in range(max(0, by[i] - 1), min(gny, by[i] + 2)):
          for gx in range(max(0, bx[i] - 1), min(gnx, bx[i] + 2)):
            b = gy * gnx + gx
            for kk in range(start[b], start[b + 1]):
              j = order[kk]
              if j == i or cell_well[j] != cell_well[i]:
                continue
              dx = centers[j, 0] - cx
              dy = centers[j, 1] - cy
              dc = np.sqrt(dx * dx + dy * dy)
              if dc >= rmax_i + radii_in[j].max():
                  continue
              for v in range(nv):
                  rv = radii_out[i, v]
                  px = cx + np.cos(angles[v]) * rv
                  py = cy + np.sin(angles[v]) * rv
                  ddx = px - centers[j, 0]
                  ddy = py - centers[j, 1]
                  d = np.sqrt(ddx * ddx + ddy * ddy)
                  kj = int(round(np.arctan2(ddy, ddx) / two_pi * nv)) % nv
                  lim = CONTACT_SURFACE * radii_in[j, kj]
                  if d < lim:
                      rv = max(rv - (lim - d), 0.5 * base_radii[i])
                      radii_out[i, v] = rv
                      comp = rest_radii[i, v] - rv
                      if comp > 0.0:
                          pushes[i, 0] -= np.cos(angles[v]) * comp
                          pushes[i, 1] -= np.sin(angles[v]) * comp
        pushes[i, 0] *= PRESSURE * dt
        pushes[i, 1] *= PRESSURE * dt


@njit(cache=True)
def resolve_all_collisions(centers: np.ndarray, velocities: np.ndarray,
                           radii: np.ndarray, base_radii: np.ndarray,
                           cell_well: np.ndarray,
                           wells: np.ndarray, half: float, corner: float,
                           order: np.ndarray, start: np.ndarray,
                           bx: np.ndarray, by: np.ndarray, gnx: int,
                           gny: int) -> None:
    """Pairwise body-guard resolution (in place).

    Membrane contact is handled by ``conform_membranes`` (membranes press
    and conform, they do not repel); this pass only keeps the cell BODIES
    (nucleus scale, ``BODY_GUARD`` x base radius) from merging.
    Overlapping bodies are pushed apart symmetrically along the line of
    centres and both velocities are zeroed; the pair is then confined to
    its wells again. Deterministic pair order (i ascending, neighbours
    j > i in grid order), so a rerun gives bit-identical positions.
    """
    n = centers.shape[0]
    maxr = np.empty(n)
    guard = np.empty(n)
    for i in range(n):
        maxr[i] = radii[i].max()
        guard[i] = BODY_GUARD * base_radii[i]
    for i in range(n):
      for gy in range(max(0, by[i] - 1), min(gny, by[i] + 2)):
        for gx in range(max(0, bx[i] - 1), min(gnx, bx[i] + 2)):
          b = gy * gnx + gx
          for kk in range(start[b], start[b + 1]):
            j = order[kk]
            if j <= i:
                continue
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist == 0.0:
                continue
            overlap = guard[i] + guard[j] - dist
            if overlap <= 0.0:
                continue
            s = 0.5 * (overlap + 0.01)
            nx, ny = dx / dist, dy / dist
            centers[i, 0] -= s * nx
            centers[i, 1] -= s * ny
            centers[j, 0] += s * nx
            centers[j, 1] += s * ny
            velocities[i, 0] = 0.0
            velocities[i, 1] = 0.0
            velocities[j, 0] = 0.0
            velocities[j, 1] = 0.0
            wi, wj = cell_well[i], cell_well[j]
            confine(centers[i], velocities[i], maxr[i],
                    wells[wi, 0], wells[wi, 1], half, corner)
            confine(centers[j], velocities[j], maxr[j],
                    wells[wj, 0], wells[wj, 1], half, corner)


# signalling state columns (one row per cell)
SIG_SINCE, SIG_ACTIVITY, SIG_NOISE, SIG_KTR, SIG_MOTILITY = range(5)
ACTIVITY_RISE = 5.0     # s from a pulse to full activity
ACTIVITY_DECAY = 15.0   # s from full activity back to 0 (after RISE)
MOTILITY_TAU = 4.0      # s: decay of the migration drive after a pulse
KTR_NOISE_TAU = 10.0    # s: timescale of the baseline activity noise
KTR_NOISE_STD = 0.03    # stationary std of the baseline noise
KTR_BASE = 0.05         # mean baseline activity (0..0.1 band)


def update_signalling(sig: np.ndarray, dt: float, normals: np.ndarray) -> None:
    """Advance the signalling state of all cells by ``dt`` (in place).

    Activity rises to 1 within ``ACTIVITY_RISE`` s of a pulse and decays
    back to 0 over ``ACTIVITY_DECAY`` s once pulses stop. Migration
    responds faster than the reporter (the crawl drive decays with
    ``MOTILITY_TAU``), so a cell that stops being targeted comes to rest
    instead of overshooting. The rendered reporter adds a per-cell
    baseline fluctuation (an OU process, ~10 s timescale), as real
    reporters show. ``normals``: one standard normal draw per cell.
    """
    since = sig[:, SIG_SINCE]
    since += dt
    rising = since <= ACTIVITY_RISE
    act = sig[:, SIG_ACTIVITY]
    act[:] = np.where(rising, np.minimum(1.0, act + dt / ACTIVITY_RISE),
                      np.maximum(0.0, act - dt / ACTIVITY_DECAY))
    sig[:, SIG_MOTILITY] = np.exp(-since / MOTILITY_TAU)
    noise = sig[:, SIG_NOISE]
    noise += (-noise / KTR_NOISE_TAU * dt
              + KTR_NOISE_STD * np.sqrt(2.0 * dt / KTR_NOISE_TAU) * normals)
    sig[:, SIG_KTR] = np.clip(act + KTR_BASE + noise, 0.0, 1.0)


# ------------------------------------------------------------- #
# Cell
# ------------------------------------------------------------- #
class CellBase:
    """Base cell: geometry (centre, vertex radii, rest shape) and the
    crawling state (adhesion, polarity).

    ``center``, ``vel``, ``r``, ``rest_r``, ``adhesion``, ``polarity`` and
    ``motion`` are plain arrays at construction; the sim replaces them with
    views into its population arrays (see ``OptoCellSim._init_arrays``).
    All updates must therefore be in place.
    """

    def __init__(self, base_radius: float, vertices: int = 24, seed: int = 0):
        self.vertices = vertices
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.base_r = base_radius * (0.78 + 0.44 * rng.random())
        self.angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)

        # irregular rest shape: random low-frequency radial modes, so even
        # an idle cell is lumpy rather than a circle
        profile = np.ones(vertices)
        for m in (2, 3, 4, 5):
            profile += (rng.uniform(0.0, SHAPE_MODES / m)
                        * np.cos(m * self.angles + rng.uniform(0, 2 * np.pi)))
        profile = np.clip(profile, 0.72, 1.35)
        self.rest_r = self.base_r * profile
        self.r = self.rest_r.copy()
        self.area0 = polygon_area(
            calculate_vertices(np.zeros(2), self.angles, self.rest_r))

        self.center = np.zeros(2, dtype=np.float64)   # placed by the sim
        self.vel = np.zeros(2, dtype=np.float64)
        self.adhesion = np.ones(vertices, dtype=np.float64)  # starts stuck
        self.polarity = np.array([rng.uniform(0, 2 * np.pi)])
        self.motion = np.zeros(2, dtype=np.float64)
        self.front_ang = np.zeros(N_FRONTS, dtype=np.float64)
        self.front_ttl = np.zeros(N_FRONTS, dtype=np.float64)  # all inactive
        self._rng = rng

    @property
    def vertices_positions(self) -> np.ndarray:
        """Absolute vertex positions (world px)."""
        return calculate_vertices(self.center, self.angles, self.r)

    def _conserve_area(self) -> None:
        """Rescale the radii in place so the polygon keeps its rest area."""
        area = polygon_area(self.vertices_positions)
        if area > 0:
            self.r *= np.sqrt(self.area0 / area)


class OptogeneticCell(CellBase):
    """A cell that protrudes toward, and migrates toward, projected light.

    Besides moving, a stimulated cell activates its signalling pathway:
    ``activity`` rises from 0 to 1 within ``ACTIVITY_RISE`` seconds of a
    stimulation pulse and, once the pulses stop, falls back to 0 over
    ``ACTIVITY_DECAY`` further seconds (one pulse: fully active at 5 s,
    fully inactive again at 20 s). The ERK-KTR channel renders this state
    as nuclear-to-cytosolic translocation of the reporter, so a
    stimulation response is visible in a single snapshot, without
    timelapse imaging or tracking.
    """

    ACTIVITY_RISE = ACTIVITY_RISE
    ACTIVITY_DECAY = ACTIVITY_DECAY

    def __init__(self, *args, protrusion_gain: float = 0.22,
                 impulse: float = 30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.protrusion_gain = protrusion_gain
        self.impulse = impulse
        self.is_stimulated = False
        # signalling state, one row of the sim's ``signals`` array once the
        # cell belongs to a sim (see SIG_* for the columns)
        self._sig = np.array([np.inf, 0.0, 0.0, 0.0, 0.0])

    # s since the last stimulation pulse
    _since_stim = property(lambda self: float(self._sig[SIG_SINCE]),
                           lambda self, v: self._sig.__setitem__(SIG_SINCE, v))
    # pathway activity in [0, 1] (the pure light response)
    activity = property(lambda self: float(self._sig[SIG_ACTIVITY]),
                        lambda self, v: self._sig.__setitem__(SIG_ACTIVITY, v))
    # rendered reporter state: activity + baseline noise
    ktr = property(lambda self: float(self._sig[SIG_KTR]))
    # migration drive (decays faster than the reporter)
    motility = property(lambda self: float(self._sig[SIG_MOTILITY]))

    def update_activity(self, dt: float) -> None:
        """Advance this cell's signalling state by ``dt`` seconds."""
        update_signalling(self._sig[None, :], dt,
                          self._rng.standard_normal(1))

    def stimulate(self, mask: np.ndarray, origin=(0.0, 0.0),
                  scale: float = 1.0) -> None:
        """Apply optogenetic stimulation from a light pattern.

        ``mask`` is the pattern *as projected on the sample*, in camera
        sensor pixels. ``origin`` is the world position (px) of sensor
        pixel (0, 0) and ``scale`` the number of sensor pixels per world
        px (the objective magnification), so vertex world positions map
        to sensor pixels as ``p = (x - origin) * scale``.
        """
        if mask is None or not mask.any():
            self.is_stimulated = False
            return

        vertices = self.vertices_positions
        vx = (vertices[:, 0] - origin[0]) * scale
        vy = (vertices[:, 1] - origin[1]) * scale

        # vertices that fall on the sensor at all
        inside = ((vx >= -0.5) & (vx < mask.shape[1] - 0.5)
                  & (vy >= -0.5) & (vy < mask.shape[0] - 0.5))
        if not inside.any():
            self.is_stimulated = False
            return

        # round (not truncate) to avoid a systematic sub-pixel bias in the
        # force direction; clamp after rounding (511.6 -> 512 is out of bounds)
        ix = np.clip(np.round(vx[inside]).astype(int), 0, mask.shape[1] - 1)
        iy = np.clip(np.round(vy[inside]).astype(int), 0, mask.shape[0] - 1)
        hit = mask[iy, ix] > 0
        if not hit.any():
            self.is_stimulated = False
            return

        self.is_stimulated = True
        self._since_stim = 0.0      # pulse received: activity starts rising
        idx = np.where(inside)[0][hit]

        # light polarizes the cell: it points the polarity at the
        # illuminated region and spawns (or renews) ONE protrusion front
        # there, so the cell fans out toward the light instead of blebbing
        # under the whole illuminated cap. The rear stays adhered and
        # stretches into a dragging tail until tension rips it off (see
        # update_cell_physics).
        target = np.mean(vertices[idx], axis=0)
        direction = target - self.center
        norm = np.linalg.norm(direction)
        if norm > 0:
            ang = float(np.arctan2(direction[1], direction[0]))
            self.polarity[0] = ang
            self.front_ang[0] = ang
            self.front_ttl[0] = max(float(self.front_ttl[0]), 3.0)

            # impulse toward the illuminated region, scaled by the
            # illuminated fraction. Sets (does not add) the velocity
            # component toward the light, so the result is frame-rate
            # independent; the perpendicular component is preserved.
            stim_fraction = len(idx) / len(self.r)
            direction_unit = direction / norm
            desired_speed = self.impulse * stim_fraction
            current_proj = np.dot(self.vel, direction_unit)
            self.vel += direction_unit * (desired_speed - current_proj)
