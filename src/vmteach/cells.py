"""Cell physics (numba) and the optogenetic cell.

The population lives in flat arrays owned by :class:`vmteach.sim.OptoCellSim`
(``centers``, ``velocities``, ``radii``); each :class:`OptogeneticCell` holds
*views* into its own row, so physics, collisions, stimulation and rendering
all see one copy of the state and nothing has to be synced.

Cells live inside wells: rounded squares with a hard boundary that the
cells collide with (1 world px = 1 um; nothing wraps).
"""
import numpy as np
from numba import njit, prange


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
                        angles: np.ndarray, base_r: float, area0: float,
                        dt: float,
                        friction: float = 3.0, brownian_d: float = 15.0,
                        curvature_relax: float = 0.12, radial_relax: float = 0.08,
                        ruffle_std: float = 0.01, seed: int = 0) -> tuple:
    """One physics step for one cell (Brownian motion, friction, membrane
    ruffling and relaxation, area conservation)."""
    np.random.seed(seed)

    # Brownian motion
    amp = np.random.normal(0, np.sqrt(2 * brownian_d * dt))
    ang = np.random.uniform(0, 2 * np.pi)
    vel += amp * np.array([np.cos(ang), np.sin(ang)])

    # Update position
    center += vel * dt

    # Apply friction
    vel *= max(0.0, 1.0 - friction * dt)

    # Membrane ruffling
    r += np.random.normal(0, ruffle_std * base_r, len(r))

    # Curvature relaxation (Laplacian smoothing)
    lap = np.roll(r, -1) + np.roll(r, 1) - 2 * r
    r += curvature_relax * lap + radial_relax * (base_r - r)

    # Constrain radius
    r = np.clip(r, 0.7 * base_r, 1.3 * base_r)

    # conserve area
    vertices = calculate_vertices(center, angles, r)
    area = polygon_area(vertices)
    if area > 0:
        r *= np.sqrt(area0 / area)

    return center, vel, r


@njit(parallel=True, cache=True)
def update_all_cells_parallel(centers: np.ndarray, velocities: np.ndarray,
                              radii: np.ndarray, angles: np.ndarray,
                              base_radii: np.ndarray, areas: np.ndarray,
                              cell_well: np.ndarray, wells: np.ndarray,
                              half: float, corner: float, dt: float,
                              friction: float = 3.0, brownian_d: float = 15.0,
                              step_count: int = 0) -> None:
    """Update all cells in parallel (in place) using Numba prange, then
    confine each to its well (``wells``: (n_wells, 2) centres)."""
    n_cells = len(centers)
    for i in prange(n_cells):  # parallel loop
        centers[i], velocities[i], radii[i] = update_cell_physics(
            centers[i], velocities[i], radii[i], angles, base_radii[i], areas[i],
            dt, friction, brownian_d,
            seed=i + step_count * n_cells
        )
        w = cell_well[i]
        confine(centers[i], velocities[i], radii[i].max(),
                wells[w, 0], wells[w, 1], half, corner)


@njit(cache=True)
def resolve_all_collisions(centers: np.ndarray, velocities: np.ndarray,
                           radii: np.ndarray, cell_well: np.ndarray,
                           wells: np.ndarray, half: float, corner: float) -> None:
    """Pairwise overlap resolution (in place).

    Overlapping cells are pushed apart symmetrically along the line of
    centres and both velocities are zeroed; the pair is then confined to
    its wells again. Deterministic pair order (i < j), so a rerun gives
    bit-identical positions.
    """
    n = centers.shape[0]
    maxr = np.empty(n)
    for i in range(n):
        maxr[i] = radii[i].max()
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist == 0.0:
                continue
            overlap = maxr[i] + maxr[j] - dist
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


# ------------------------------------------------------------- #
# Cell
# ------------------------------------------------------------- #
class CellBase:
    """Base cell: geometry (centre, vertex radii) and physics parameters.

    ``center``, ``vel`` and ``r`` are plain arrays at construction; the sim
    replaces them with views into its population arrays (see
    ``OptoCellSim._init_arrays``). All updates must therefore be in place.
    """

    def __init__(self, base_radius: float, vertices: int = 24, seed: int = 0):
        self.vertices = vertices
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.base_r = base_radius * (0.85 + 0.3 * rng.random())
        self.r = np.full(vertices, self.base_r, dtype=np.float64)
        self.angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
        self.area0 = np.pi * self.base_r ** 2

        self.center = np.zeros(2, dtype=np.float64)   # placed by the sim
        self.vel = np.zeros(2, dtype=np.float64)
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

    ACTIVITY_RISE = 5.0    # s from a pulse to full activity
    ACTIVITY_DECAY = 15.0  # s from full activity back to 0 (after RISE)

    def __init__(self, *args, protrusion_gain: float = 0.05,
                 impulse: float = 24.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.protrusion_gain = protrusion_gain
        self.impulse = impulse
        self.is_stimulated = False
        self.activity = 0.0
        self._since_stim = np.inf   # s since the last stimulation pulse

    def update_activity(self, dt: float) -> None:
        """Advance the signalling state by ``dt`` seconds (deterministic)."""
        self._since_stim += dt
        if self._since_stim <= self.ACTIVITY_RISE:
            self.activity = min(1.0, self.activity + dt / self.ACTIVITY_RISE)
        else:
            self.activity = max(0.0, self.activity - dt / self.ACTIVITY_DECAY)

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

        # protrusion of the illuminated vertices (in place: r is a view)
        self.r[idx] += self.protrusion_gain * self.base_r
        np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r, out=self.r)
        self._conserve_area()

        # impulse toward the illuminated region, scaled by the illuminated
        # fraction. Sets (does not add) the velocity component toward the
        # light, so the result is frame-rate independent; the perpendicular
        # Brownian component is preserved.
        target = np.mean(vertices[idx], axis=0)
        direction = target - self.center
        norm = np.linalg.norm(direction)
        if norm > 0:
            stim_fraction = len(idx) / len(self.r)
            direction_unit = direction / norm
            desired_speed = self.impulse * stim_fraction
            current_proj = np.dot(self.vel, direction_unit)
            self.vel += direction_unit * (desired_speed - current_proj)
