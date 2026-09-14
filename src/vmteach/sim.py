"""OptoCellSim: the simulated specimen and the optics in front of the camera.

Vertex-model cells (numba physics, see cells.py) that respond to SLM light
with protrusion and directed migration. Exposes exactly the surface the
device bridge needs:

    snap_frame(mask, exposure, intensity, binning)   render one camera frame
    step(dt)                                          advance physics
    reset(seed)                                       deterministic restart
    stage / set_focal_plane / state_devices / slm_affine

Geometry
--------
* Sample: ``n_wells`` square wells (rounded corners, ``well_size`` um,
  pitch ``well_size + well_gap``) side by side in a plastic slide. Default:
  two 2048 um wells (4 x 4 fields at 10x each), for multiwell workflows.
  The well wall is a hard boundary the cells collide with; it shows in
  phase contrast (plastic + bright edge) and in fluorescence only as the
  cell-free zone. Nothing wraps.
* Stage: ``stage`` (x, y) in um; (0, 0) is the centre of the first well,
  ``well_positions`` lists the stage coordinates of every well centre.
  Travel is limited to ``stage_limits`` (just inside the wells); moves
  beyond it stop at the limit, like a stage with soft limits.
* Camera: a fixed ``sensor_size`` x ``sensor_size`` sensor (512 x 512).
  Changing the objective changes the *pixel size*, never the image size,
  exactly as on a real microscope: 10x = 1.0 um/px (512 um field), 40x =
  0.25 um/px (128 um field). Binning b sums b x b sensor pixels into one
  output pixel (512/b square image, b^2 more signal).
* SLM: 512 x 512 pixels mapped 1:1 onto the sensor, at any objective.
  ``slm_affine`` (default ``None`` = identity) is the SLM -> sensor
  calibration hook for the DMD calibration exercise: set it to a 2x3
  affine to simulate a misaligned projector.
"""

from __future__ import annotations

import cv2
import numpy as np

from vmteach.cells import (
    OptogeneticCell,
    resolve_all_collisions,
    update_all_cells_parallel,
    well_sdf,
)
from vmteach.optics import Optics
from vmteach.render import CellRenderer

# objective label -> camera pixel size in um (unbinned). 10x is the reference
# (1 um/px); the same numbers are declared as pixel-size configs in
# optogenetic.cfg so core.getPixelSizeUm() reports them.
OBJECTIVES = {"4x": 2.5, "10x": 1.0, "20x": 0.5, "40x": 0.25, "60x": 0.1667}

SENSOR_SIZE = 512          # camera sensor, px (square)
SLM_SHAPE = (512, 512)     # SLM / DMD pixels, mapped 1:1 onto the sensor
FIELD_UM = SENSOR_SIZE * OBJECTIVES["10x"]   # reference 10x field of view

# (Filter Wheel label, LED label) -> render mode
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

    def __init__(self, n_cells: int = 20, well_size: float = 2048.0,
                 n_wells: int = 2, well_gap: float = 256.0,
                 corner_radius: float = 256.0, base_radius: float = 20.0,
                 seed: int = 0, sensor_size: int = SENSOR_SIZE):
        """
        Args:
            n_cells: cell density, as cells per 10x field of view
                (512 x 512 um). Each well holds ``n_cells * (well area /
                field area)`` cells, so every field looks the same.
            well_size: side of each (rounded) square well, um.
            n_wells: wells side by side along x.
            well_gap: plastic between neighbouring wells, um.
            corner_radius: rounding of the well corners, um.
            base_radius: mean cell radius, um.
            seed: population and noise seed (deterministic).
            sensor_size: camera sensor side, px.
        """
        self.well_size = float(well_size)
        self.well_half = self.well_size / 2.0
        self.well_gap = float(well_gap)
        self.corner_radius = float(corner_radius)
        self.n_wells = int(n_wells)
        self.base_radius = float(base_radius)
        self.sensor_size = int(sensor_size)
        pitch = self.well_size + self.well_gap
        # world coordinates (um): well centres along x, margin all round
        margin = self.well_size / 4.0
        self.wells = np.array([[margin + self.well_half + i * pitch,
                                margin + self.well_half]
                               for i in range(self.n_wells)])
        self.width = 2 * margin + self.n_wells * self.well_size \
            + (self.n_wells - 1) * self.well_gap
        self.height = 2 * margin + self.well_size
        fields = (self.well_size / FIELD_UM) ** 2
        self.cells_per_field = n_cells
        self.cells_per_well = max(1, int(round(n_cells * fields)))
        self.n_cells = self.cells_per_well * self.n_wells

        # device-facing state
        self.state_devices: dict = {}
        self.stage = np.zeros(2)          # um; (0, 0) = centre of well 0
        self.focal_plane = 0.0
        self.mode = 0
        self.current_objective = "10x"
        self.binning = 1
        self.slm_affine: np.ndarray | None = None   # SLM -> sensor, 2x3

        self.renderer = CellRenderer(self.wells, self.well_half,
                                     self.corner_radius)
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

    # ── geometry ────────────────────────────────────────────────────────

    @property
    def pixel_size_um(self) -> float:
        """Camera pixel size (unbinned) for the current objective."""
        return OBJECTIVES[self.current_objective]

    @property
    def scale(self) -> float:
        """Sensor pixels per world px (= objective magnification / 10)."""
        return self.world_pixel_size_um / self.pixel_size_um

    @property
    def fov_um(self) -> float:
        """Side of the imaged field for the current objective, um."""
        return self.sensor_size * self.pixel_size_um

    @property
    def view_origin(self) -> np.ndarray:
        """World position (px) of sensor pixel (0, 0)."""
        half = self.fov_um / 2.0
        return self.wells[0] + self.stage - half

    # stage travel stops slightly inside the well extent (um)
    STAGE_INSET_X, STAGE_INSET_Y = 8.0, 24.0

    @property
    def stage_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Stage travel range ((x_min, x_max), (y_min, y_max)) in um, so the
        camera cannot leave the wells: (-1016, 3320) x (-1000, 1000) for the
        default two-well slide."""
        x_last = float(self.wells[-1, 0] - self.wells[0, 0])
        rx = self.well_half - self.STAGE_INSET_X
        ry = self.well_half - self.STAGE_INSET_Y
        return ((-rx, x_last + rx), (-ry, ry))

    def clamp_stage(self, x: float, y: float) -> tuple[float, float]:
        """Clamp a requested stage position to the travel range."""
        (x0, x1), (y0, y1) = self.stage_limits
        return (float(min(max(x, x0), x1)), float(min(max(y, y0), y1)))

    @property
    def well_positions(self) -> list[tuple[float, float]]:
        """Stage coordinates (um) of the well centres, first well = (0, 0)."""
        return [(float(w[0] - self.wells[0, 0]), float(w[1] - self.wells[0, 1]))
                for w in self.wells]

    def well_distance(self, x: float, y: float, well: int = 0) -> float:
        """Signed distance (um, negative inside) from world point to the
        wall of ``well``."""
        c = self.wells[well]
        return well_sdf(x - c[0], y - c[1], self.well_half,
                        self.corner_radius)[0]

    # kept for scripts written against the earlier single-objective sim
    camera_offset = view_origin

    # ── population setup ────────────────────────────────────────────────

    def _create_cells(self) -> list[OptogeneticCell]:
        return [
            OptogeneticCell(self.base_radius, vertices=24,
                            seed=self._rng.randint(0, 10000))
            for _ in range(self.n_cells)
        ]

    def _resolve_initial_overlaps(self, min_gap: float = 2.0,
                                  max_attempts: int = 500) -> None:
        """Rejection-sample positions inside each well so no two cells
        start overlapping and none touches the wall."""
        self.cell_well = np.repeat(np.arange(self.n_wells), self.cells_per_well)
        placed = np.empty((0, 2))
        placed_r = np.empty(0)
        for cell, w in zip(self._cells, self.cell_well):
            cx, cy = self.wells[w]
            lim = self.well_half - cell.base_r - min_gap
            for _ in range(max_attempts):
                cand = np.array([cx + self._rng.uniform(-lim, lim),
                                 cy + self._rng.uniform(-lim, lim)])
                if self.well_distance(cand[0], cand[1], w) > -(cell.base_r + min_gap):
                    continue        # in a rounded corner
                if len(placed) == 0:
                    break
                d = cand - placed
                if np.all(np.hypot(d[:, 0], d[:, 1])
                          >= cell.base_r + placed_r + min_gap):
                    break
            cell.center = cand
            placed = np.vstack([placed, cand])
            placed_r = np.append(placed_r, cell.base_r)

    def _init_arrays(self) -> None:
        """Population arrays; each cell then holds views into its own row."""
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
            c.center = self.centers[i]
            c.vel = self.velocities[i]
            c.r = self.radii[i]
            c.angles = self.angles

    # ── physics ─────────────────────────────────────────────────────────

    def step(self, dt: float = 0.05) -> None:
        """Advance all cells by *dt* (deterministic given _step_count)."""
        update_all_cells_parallel(
            self.centers, self.velocities, self.radii, self.angles,
            self.base_radii, self.areas, self.cell_well, self.wells,
            self.well_half, self.corner_radius, dt,
            step_count=self._step_count)
        self._step_count += 1
        resolve_all_collisions(self.centers, self.velocities, self.radii,
                               self.cell_well, self.wells,
                               self.well_half, self.corner_radius)

    # ── stimulation ─────────────────────────────────────────────────────

    @property
    def stim_light_on(self) -> bool:
        """True while the stimulation light path is engaged (CyanStim)."""
        return self.state_devices.get("LED", {}).get("label") == "BLUE"

    def slm_on_sensor(self, mask: np.ndarray) -> np.ndarray:
        """The SLM pattern as it lands on the sensor (uint8, sensor shape).

        Identity mapping by default (a perfectly calibrated projector);
        ``slm_affine`` (2x3, SLM px -> sensor px) models a misaligned one.
        """
        m = (np.asarray(mask) > 0).astype(np.uint8) * 255
        size = (self.sensor_size, self.sensor_size)
        if self.slm_affine is not None:
            m = cv2.warpAffine(m, np.asarray(self.slm_affine, np.float64)[:2],
                               size, flags=cv2.INTER_NEAREST)
        elif m.shape != size:
            m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
        return m

    def _handle_mask(self, mask: np.ndarray | None) -> None:
        """Apply optogenetic stimulation. ``mask`` is in SLM pixels.

        Gated on the light path: the SLM only *modulates* light, so the
        pattern is delivered to the sample only while the stimulation
        LED is on (Channel "CyanStim"), exactly like real hardware.

        Delivery is an impulse at the delivery events (light-on transition
        and snaps while lit); time advancing with the light engaged does
        not stimulate again. This models pulsed stimulation protocols.
        """
        if mask is None or not mask.any() or not self.stim_light_on:
            for c in self._cells:
                c.is_stimulated = False
            return
        sensor = self.slm_on_sensor(mask)
        origin = tuple(self.view_origin)
        scale = self.scale
        for c in self._cells:
            c.stimulate(sensor, origin=origin, scale=scale)

    # ── device hooks ────────────────────────────────────────────────────

    def set_focal_plane(self, z: float) -> None:
        self.focal_plane = float(z)

    def _update_from_devices(self) -> None:
        led = self.state_devices.get("LED", {}).get("label", "CYAN")
        filt = self.state_devices.get("Filter Wheel", {}).get(
            "label", "Electra1(402/454)")
        self.mode = _MODE_MAP.get((filt, led), 0)
        obj = self.state_devices.get("Objective", {}).get("label", "10x")
        if obj in OBJECTIVES:
            self.current_objective = obj

    # ── imaging ─────────────────────────────────────────────────────────

    def snap_frame(self, mask: np.ndarray | None = None,
                   exposure: float = 50.0, intensity: float = 1.0,
                   binning: int | None = None, **kwargs) -> np.ndarray:
        """Render one camera frame (uint8, ``sensor_size // binning`` square)."""
        self._update_from_devices()
        if binning is not None:
            self.binning = int(binning)
        b = self.binning
        if mask is not None:
            self._handle_mask(mask)

        out = self.sensor_size // b
        shape = (out, out)
        scale = self.scale / b
        origin = self.view_origin
        defocus = self.focal_plane * 0.4

        if self.mode == 3:
            # CyanStim: image the SLM pattern projected onto the sample.
            # What the camera sees is the stimulation light itself (bright,
            # with a halo from the optics) plus a faint reflection of the
            # cells, enough to check the mask to sample alignment, exactly as
            # on a real microscope.
            faint = self.renderer.render(self._cells, 0, origin, scale, shape)
            view = faint.astype(np.float32) * 0.05
            if mask is not None:
                proj = self.slm_on_sensor(mask)
                if proj.shape != shape:
                    proj = cv2.resize(proj, shape, interpolation=cv2.INTER_AREA)
                view += proj.astype(np.float32) * (200.0 / 255.0)
            return self._optics[3].apply(
                np.clip(view, 0, 255).astype(np.uint8),
                exposure=exposure, intensity=intensity,
                defocus=defocus, scale=scale, binning=b)

        view = self.renderer.render(self._cells, self.mode, origin, scale, shape)
        return self._optics[self.mode].apply(
            view, exposure=exposure, intensity=intensity,
            defocus=defocus, scale=scale, binning=b)

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
