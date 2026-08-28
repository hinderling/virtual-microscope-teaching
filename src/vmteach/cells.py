"""Base cell module with core physics and properties."""
import numpy as np
from numba import njit, prange
from numba.types import float64 as no_float64
import numba.types as nbt


# ------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------ #
@njit(cache=True)
def wrap_position(pos: np.ndarray, width: float, height: float) -> np.ndarray:
    """Wrap position for periodic boundaries conditions."""
    return np.array([pos[0] % width, pos[1] % height])

@njit(cache=True)
def polygon_area(pts: np.ndarray) -> float:
    """Calculate polygon area"""
    x, y = pts[:, 0], pts[:, 1]
    #return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
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
                         width: float, height: float, dt: float,
                         friction: float = 3.0, brownian_d: float = 15.0,
                         curvature_relax: float = 0.12, radial_relax: float = 0.08,
                         ruffle_std: float = 0.01, seed: int = 0) -> tuple:
    """Update cell physics"""
    np.random.seed(seed)

    # Brownian motion
    amp = np.random.normal(0, np.sqrt(2 * brownian_d * dt))
    ang = np.random.uniform(0, 2 * np.pi)
    vel += amp * np.array([np.cos(ang), np.sin(ang)])

    # Update position
    center += vel * dt
    center = wrap_position(center, width, height)

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


@njit(cache=True)
def check_collision(center1: np.ndarray, center2: np.ndarray,
                    r1: np.ndarray, r2: np.ndarray,
                    width: float, height: float) -> tuple:
    """Fast collision detection and resolution"""
    # Calculate wrapped distance
    dvec = center2 - center1
    dvec[0] -= width * np.round(dvec[0] / width)
    dvec[1] -= height * np.round(dvec[1] / height)

    dist = np.sqrt(dvec[0]**2 + dvec[1]**2)
    if dist == 0:
        return False, center1, center2
    
    overlap = np.max(r1) + np.max(r2) - dist
    if overlap <= 0:
        return False, center1, center2
    
    # resolve collision
    n = dvec / dist
    shift = 0.5 * (overlap + 0.01) * n
    new_center1 = center1 - shift
    new_center2 = center2 + shift

    # Wrap position
    new_center1 = wrap_position(new_center1, width, height)
    new_center2 = wrap_position(new_center2, width, height)

    return True, new_center1, new_center2


@njit(parallel=True, cache=True)
def update_all_cells_parallel(centers: np.ndarray, velocities: np.ndarray,
                              radii: np.ndarray, angles: np.ndarray,
                              base_radii: np.ndarray, areas: np.ndarray,
                              width: float, height: float, dt: float,
                              friction: float = 3.0, brownian_d: float = 15.0,
                              step_count: int = 0) -> None:
    """Update all cells in parallel using Numba prange"""
    n_cells = len(centers)
    for i in prange(n_cells): # parallel loop
        centers[i], velocities[i], radii[i] = update_cell_physics(
            centers[i], velocities[i], radii[i], angles, base_radii[i], areas[i],
            width, height, dt, friction, brownian_d,
            seed=i + step_count * n_cells
        )






# ------------------------------------------------------------- #
# Cell
# ------------------------------------------------------------- #
class CellBase:
    """Base cell class with core physics"""
    def __init__(self, width: float, height: float, base_radius: float, vertices: int = 24, seed: int = 0):
        self.width = width
        self.height = height
        self.vertices = vertices
        self.seed = seed

        # Initialize with random variations (# TODO checks later)
        rng = np.random.RandomState(seed)
        self.base_r = base_radius * (0.85 + 0.3 * rng.random())
        self.r = np.full(vertices, self.base_r, dtype=np.float64)
        self.angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
        self.area0 = np.pi * self.base_r ** 2

        # Position and velocity
        self.center = np.array([rng.uniform(0, width), rng.uniform(0, height)], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.z_position: float = 0.0

        # Physics parameters
        self.friction: float = 3.0
        self.brownian_d: float = 15.0
        self.curvature_relax: float = 0.15
        self.radial_relax: float = 0.10
        self.ruffle_std: float = 0.03

        # Fluorescence properties
        self.nucleus_fluorescence = 0.0
        self.membrane_fluorescence = np.zeros(vertices, dtype=np.float64)

        self._rng = rng


    @property
    def vertices_positions(self) -> np.ndarray:
        """Get absolute vertex positions."""
        return calculate_vertices(self.center, self.angles, self.r)
    
    def update_physics(self, dt: float) -> None:
        """Update cell physics (movement, shape deformation)"""
        self.center, self.vel, self.r = update_cell_physics(
            self.center, self.vel, self.r, self.angles, self.base_r,
            self.area0, self.width, self.height, dt, self.friction, self.brownian_d,
            self.curvature_relax, self.radial_relax, self.ruffle_std, self.seed
            )

    
    def check_collision(self, other: 'CellBase') -> bool:
        """Check and resolve collision with another cell."""
        collided, new_center1, new_center2 = check_collision(
            self.center, other.center, self.r, other.r, self.width, self.height
        )

        if collided:
            self.center = new_center1
            other.center = new_center2
            self.vel[:] = 0
            other.vel[:] = 0

        return collided
    
    def _conserve_area(self) -> None:
        """Conserve cell area after deformation."""
        pts = self.vertices_positions
        area = polygon_area(pts)
        if area > 0:
            self.r *= np.sqrt(self.area0 / area)

"""Optogenetic cell behaviour module."""
import numpy as np




class OptogeneticCell(CellBase):

    def __init__(self, *args, protrusion_gain: float = 0.05, impulse: float = 24.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.protrusion_gain = protrusion_gain
        self.impulse = impulse
        self.is_stimulated = False


    def stimulate(self, mask: np.ndarray, camera_offset: tuple[float, float] = (0.0, 0.0)) -> None:
        """Apply optogenetic stimulation based on mask.

        The mask is in camera/viewport space (matching the rendered image).
        Vertex world positions are converted to viewport coordinates using
        *camera_offset* before being checked against the mask.
        """
        if mask is None or not mask.any():
            self.is_stimulated = False
            return

        # Convert vertex world positions to viewport (camera-relative)
        # coords. The world is periodic and the camera crop wraps around
        # its boundary, so the delta must wrap too: viewport pixel p shows
        # world x = (x0 + p) mod W, hence p = (x - x0) mod W. Without the
        # wrap, cells visible across the world seam could never be
        # stimulated.
        vertices = self.vertices_positions
        vx = (vertices[:, 0] - camera_offset[0]) % self.width
        vy = (vertices[:, 1] - camera_offset[1]) % self.height

        # Check which vertices fall within the mask bounds
        inside = (vx < mask.shape[1]) & (vy < mask.shape[0])

        if not inside.any():
            self.is_stimulated = False
            return

        # Get pixel indices for vertices inside bounds (round, not truncate,
        # to avoid systematic sub-pixel bias that skews the force direction)
        ix = np.round(vx[inside]).astype(int)
        iy = np.round(vy[inside]).astype(int)
        # Clamp after rounding (a vertex at 511.6 rounds to 512 which is out of bounds)
        ix = np.clip(ix, 0, mask.shape[1] - 1)
        iy = np.clip(iy, 0, mask.shape[0] - 1)

        # Check mask at vertex position
        hit = mask[iy, ix] > 0

        if not hit.any():
            self.is_stimulated = False
            return

        self.is_stimulated = True

        # Find which vertices to protrude (boolean index into inside-subset)
        idx = np.where(inside)[0][hit]

        # Apply protrusion
        self.r[idx] += self.protrusion_gain * self.base_r
        self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
        self._conserve_area()

        # Apply impulse toward stimulated region, scaled by fraction of
        # vertices illuminated so partial stimulation gives proportional force.
        # Sets the velocity component toward the light (not accumulative) so
        # the result is frame-rate independent.  Perpendicular Brownian jitter
        # is preserved for natural-looking motion.
        hit_vertices = vertices[idx]
        target = np.mean(hit_vertices, axis=0)
        direction = target - self.center
        norm = np.linalg.norm(direction)

        if norm > 0:
            stim_fraction = len(idx) / len(self.r)
            direction_unit = direction / norm
            desired_speed = self.impulse * stim_fraction
            current_proj = np.dot(self.vel, direction_unit)
            self.vel += direction_unit * (desired_speed - current_proj)

"""Spatial indexing for fast collision detection."""
import numpy as np
from typing import List, Tuple, Set



class SpatialGrid:
    """Spatial grid for efficient collision detection."""

    def __init__(self, width: float, height: float, cell_size: float = 50.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        self.grid_width = int(np.ceil(width / cell_size))
        self.grid_height = int(np.ceil(height / cell_size))
        self.grid = {}


    def clear(self) -> None:
        """Clear the grid"""
        self.grid.clear()

    def add_cell(self, cell: CellBase, index: int) -> None:
        """Add a cell to the grid."""
        grid_pos = self._get_grid_position(cell.center)

        # Add to multiple grid cells if cell is large
        radius = np.max(cell.r)
        cell_to_check = int(np.ceil(radius / self.cell_size))

        for dx in range(-cell_to_check, cell_to_check + 1):
            for dy in range(-cell_to_check, cell_to_check+1):
                gx = (grid_pos[0] + dx) % self.grid_width
                gy = (grid_pos[1] + dy) % self.grid_height
                key = (gx, gy)

                if key not in self.grid:
                    self.grid[key] = []
                
                self.grid[key].append(index)
    

    def get_potential_collisions(self, cell: CellBase, index: int) -> Set[int]:
        """Get indices of cells that might collide with given cell."""
        grid_pos = self._get_grid_position(cell.center)
        potential = set()

        radius = np.max(cell.r)
        cells_to_check = int(np.ceil(radius / self.cell_size))

        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                gx = (grid_pos[0] + dx) % self.grid_width
                gy = (grid_pos[1] + dy) % self.grid_height
                key = (gx, gy)

                if key in self.grid:
                    for other_index in self.grid[key]:
                        if other_index != index:
                            potential.add(other_index)

        return potential

    def _get_grid_position(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert position to grid coordinates."""
        gx = int(pos[0] / self.cell_size) % self.grid_width
        gy = int(pos[1] / self.cell_size) % self.grid_height
        return (gx, gy)