"""Optogenetic cell behaviour module."""
import numpy as np
from vmteach.sims.cell.cell import CellBase



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

        # Convert vertex world positions to viewport (camera-relative) coords
        vertices = self.vertices_positions
        vx = vertices[:, 0] - camera_offset[0]
        vy = vertices[:, 1] - camera_offset[1]

        # Check which vertices fall within the mask bounds
        inside = ((vx >= 0) & (vx < mask.shape[1]) &
                  (vy >= 0) & (vy < mask.shape[0]))

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