"""OptoCellSim — the simulated specimen.

Vertex-model cells (numba physics, see cells.py) that respond to SLM light
with protrusion and directed migration. Exposes exactly the surface the
device bridge needs:

    snap_frame(mask, exposure, intensity)   render one camera frame
    step(dt)                                advance physics
    reset(seed)                             deterministic restart
    camera_offset / set_focal_plane / state_devices / viewport_*

Coordinates: the world is ``world_size`` × ``world_size`` px with periodic
boundaries. The camera views a 512×512 window whose top-left corner is
``camera_offset`` (moved by the XY stage). Objectives crop a smaller window
(10x: 512, 20x: 256, 40x: 128, 100x: 64 world px) and resize to 512×512.
"""

from __future__ import annotations

import cv2
import numpy as np

from vmteach.cells import (
    CellBase,
    OptogeneticCell,
    SpatialGrid,
    update_all_cells_parallel,
)
from vmteach.optics import Optics
from vmteach.render import CellRenderer

# objective label → (fov in world px)
_FOV = {"10x": 512, "20x": 256, "40x": 128, "100x": 64}

# (Filter Wheel label, LED label) → render mode
_MODE_MAP = {
    ("Electra1(402/454)", "CYAN"): 0,      # phase-contrast
    ("SCFP2(434/474)", "UV"): 1,           # DAPI
    ("mScarlet3(569/582)", "ORANGE"): 2,   # membrane
    ("TagGFP2(483/506)", "BLUE"): 3,       # CyanStim: projected SLM light
}


class OptoCellSim:
    """Light-responsive cell population behind a camera/stage/SLM interface."""

    continuous = True             # realtime mode may auto-start an engine
    world_pixel_size_um = 1.0

    def __init__(self, n_cells: int = 20, world_size: int = 600,
                 base_radius: float = 20.0, seed: int = 0,
                 viewport_width: int = 512, viewport_height: int = 512):
        self.n_cells = n_cells
        self.width = self.height = int(world_size)
        self.base_radius = float(base_radius)
        self.viewport_width = int(viewport_width)
        self.viewport_height = int(viewport_height)

        # device-facing state
        self.state_devices: dict = {}
        self.camera_offset = np.array([
            self.width / 2.0 - viewport_width / 2.0,
            self.height / 2.0 - viewport_height / 2.0,
        ])
        self.focal_plane = 0.0
        self.mode = 0
        self.current_objective = "10x"
        self._stim_mask = None

        self.renderer = CellRenderer(self.width, self.height)
        self.spatial_grid = SpatialGrid(self.width, self.height,
                                        base_radius * 3)
        # per-channel optics (independent seeded noise streams)
        self._optics = {
            0: Optics(seed + 300, psf_sigma=0.7, read_std=2.0, photon_k=0.30),
            1: Optics(seed + 301, psf_sigma=1.0, read_std=2.5, photon_k=0.45),
            2: Optics(seed + 302, psf_sigma=1.0, read_std=2.5, photon_k=0.45),
            # stimulation-light channel: strong halo, bright, noisy
            3: Optics(seed + 303, psf_sigma=1.6, read_std=3.0, photon_k=0.50),
        }

        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._cells = self._create_cells()
        self._resolve_initial_overlaps()
        self._init_arrays()
        self._step_count = 0

    # ── population setup ────────────────────────────────────────────────

    def _create_cells(self) -> list[OptogeneticCell]:
        return [
            OptogeneticCell(self.width, self.height, self.base_radius,
                            vertices=24, seed=self._rng.randint(0, 10000))
            for _ in range(self.n_cells)
        ]

    def _resolve_initial_overlaps(self, min_gap: float = 2.0,
                                  max_attempts: int = 500) -> None:
        """Rejection-sample positions so no two cells start overlapping."""
        placed: list[tuple[np.ndarray, float]] = []
        w, h = self.width, self.height
        for cell in self._cells:
            for _ in range(max_attempts):
                cand = np.array([self._rng.uniform(0, w),
                                 self._rng.uniform(0, h)])
                ok = True
                for oc, orr in placed:
                    d = cand - oc
                    d[0] -= w * round(d[0] / w)
                    d[1] -= h * round(d[1] / h)
                    if np.hypot(d[0], d[1]) < cell.base_r + orr + min_gap:
                        ok = False
                        break
                if ok:
                    cell.center = cand
                    break
            placed.append((cell.center.copy(), cell.base_r))

    def _init_arrays(self) -> None:
        n = self.n_cells
        self.centers = np.zeros((n, 2))
        self.velocities = np.zeros((n, 2))
        self.radii = np.zeros((n, 24))
        self.base_radii = np.zeros(n)
        self.areas = np.zeros(n)
        self.angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        for i, c in enumerate(self._cells):
            self.centers[i] = c.center
            self.velocities[i] = c.vel
            self.radii[i] = c.r
            self.base_radii[i] = c.base_r
            self.areas[i] = c.area0

    # ── physics ─────────────────────────────────────────────────────────

    def step(self, dt: float = 0.05) -> None:
        """Advance all cells by *dt* (deterministic given _step_count)."""
        update_all_cells_parallel(
            self.centers, self.velocities, self.radii, self.angles,
            self.base_radii, self.areas, self.width, self.height, dt,
            step_count=self._step_count)
        self._step_count += 1
        # sync arrays → cell objects, handle collisions, sync back
        for i, c in enumerate(self._cells):
            c.center = self.centers[i].copy()
            c.vel = self.velocities[i].copy()
            c.r = self.radii[i].copy()
        self._handle_collisions()
        for i, c in enumerate(self._cells):
            self.centers[i] = c.center
            self.velocities[i] = c.vel

    def _handle_collisions(self) -> None:
        self.spatial_grid.clear()
        for i, c in enumerate(self._cells):
            self.spatial_grid.add_cell(c, i)
        checked: set = set()
        for i, c in enumerate(self._cells):
            for j in self.spatial_grid.get_potential_collisions(c, i):
                pair = (min(i, j), max(i, j))
                if pair in checked:
                    continue
                checked.add(pair)
                c.check_collision(self._cells[j])

    # ── stimulation ─────────────────────────────────────────────────────

    def _handle_mask(self, mask: np.ndarray | None) -> None:
        """Apply optogenetic stimulation. Mask is in viewport coordinates."""
        if mask is None or not mask.any():
            for c in self._cells:
                c.is_stimulated = False
            return
        offset = tuple(self.camera_offset)
        for c in self._cells:
            c.stimulate(mask, camera_offset=offset)
        for i, c in enumerate(self._cells):
            self.radii[i] = c.r
            self.velocities[i] = c.vel

    # ── device hooks ────────────────────────────────────────────────────

    def set_focal_plane(self, z: float) -> None:
        self.focal_plane = float(z)

    def _update_from_devices(self) -> None:
        led = self.state_devices.get("LED", {}).get("label", "CYAN")
        filt = self.state_devices.get("Filter Wheel", {}).get(
            "label", "Electra1(402/454)")
        self.mode = _MODE_MAP.get((filt, led), 0)
        obj = self.state_devices.get("Objective", {}).get("label", "10x")
        if obj in _FOV:
            self.current_objective = obj

    # ── imaging ─────────────────────────────────────────────────────────

    def snap_frame(self, mask: np.ndarray | None = None,
                   exposure: float = 50.0, intensity: float = 1.0,
                   **kwargs) -> np.ndarray:
        self._update_from_devices()
        if mask is not None:
            self._handle_mask(mask)

        if self.mode == 3:
            # CyanStim: image the SLM pattern projected onto the sample.
            # What the camera sees is the stimulation light itself (bright,
            # with a halo from the optics) plus a faint reflection of the
            # cells — enough to check mask–sample alignment, exactly as on
            # a real microscope.
            faint = self.renderer.render(self._cells, 0)
            view = self._crop_view(faint).astype(np.float32) * 0.15
            if mask is not None and mask.shape == view.shape:
                view += (mask > 0).astype(np.float32) * 200.0
            return self._optics[3].apply(
                np.clip(view, 0, 255).astype(np.uint8),
                exposure=exposure, intensity=intensity,
                defocus=self.focal_plane * 0.4)

        world = self.renderer.render(self._cells, self.mode)
        view = self._crop_view(world)

        return self._optics[self.mode].apply(
            view, exposure=exposure, intensity=intensity,
            defocus=self.focal_plane * 0.4)

    def _crop_view(self, world: np.ndarray) -> np.ndarray:
        """Crop the objective's field of view and resize to the viewport."""
        fov = _FOV[self.current_objective]
        cx = self.camera_offset[0] + self.viewport_width / 2.0
        cy = self.camera_offset[1] + self.viewport_height / 2.0
        x0 = int(round(cx - fov / 2.0))
        y0 = int(round(cy - fov / 2.0))
        ix = (np.arange(fov) + x0) % self.width      # periodic world → wrap
        iy = (np.arange(fov) + y0) % self.height
        view = world[np.ix_(iy, ix)]
        if fov != self.viewport_width:
            view = cv2.resize(view, (self.viewport_width,
                                     self.viewport_height),
                              interpolation=cv2.INTER_LINEAR)
        return view

    # ── reset ───────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        """Restore the initial population and all noise streams.

        Deterministic: after ``reset()``, rerunning the same loop yields
        bit-identical images and positions. Pass ``seed`` for a different
        (but equally reproducible) population.
        """
        if seed is not None:
            self._seed = seed
        self._rng = np.random.RandomState(self._seed)
        for o in self._optics.values():
            o.reset()
        self._stim_mask = None
        # clear any leftover SLM mask held by the device bridge
        try:
            import vmteach.bridge as _b
            if _b.GLOBAL_BRIDGE is not None and _b.GLOBAL_BRIDGE._sim is self:
                _b.GLOBAL_BRIDGE._current_slm_mask = None
        except Exception:
            pass
        self._cells = self._create_cells()
        self._resolve_initial_overlaps()
        self._init_arrays()
        self._step_count = 0
