"""SimulationBridge: glue between pymmcore devices and the simulation.

Device adapters (camera, stage, SLM, state devices) hold no simulation
logic; they translate pymmcore calls into the small interface below. This
is the only file that knows both sides.
"""

from __future__ import annotations

import threading

import numpy as np

GLOBAL_BRIDGE: "SimulationBridge | None" = None

# State devices wait for this before pushing their initial state
# (device initialization runs on parallel threads).
bridge_ready = threading.Event()


def set_global_bridge(bridge: "SimulationBridge") -> None:
    global GLOBAL_BRIDGE
    GLOBAL_BRIDGE = bridge
    bridge_ready.set()


class SimulationBridge:
    """Routes device calls to an :class:`vmteach.optogenetic.sim.OptoCellSim`."""

    def __init__(self, sim):
        self._sim = sim
        self._current_slm_mask: np.ndarray | None = None
        self._shutter_open = False
        self._engine = None          # set by teach.load_microscope (realtime)

    # ── camera ──────────────────────────────────────────────────────────

    def snap(self, exposure: float, brightness: float, gain: float = 1.0,
             binning: int = 1, **kwargs) -> np.ndarray:
        if self._engine is not None:
            self._engine.touch()
        img = self._sim.snap_frame(mask=self.get_slm_mask(),
                                   exposure=exposure, intensity=brightness,
                                   binning=binning)
        if gain != 1.0:
            img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return img

    # ── stage / focus ───────────────────────────────────────────────────

    def set_stage(self, x: float, y: float) -> tuple[float, float]:
        """Move the stage (um; (0, 0) = centre of the first well).

        Clamped to the travel range; returns the position actually reached.
        """
        x, y = self._sim.clamp_stage(x, y)
        self._sim.stage[:] = (x, y)
        return x, y

    def set_focus(self, z: float) -> None:
        self._sim.set_focal_plane(z)

    # ── state devices (LED / Filter Wheel / Objective) ──────────────────

    def update_state(self, dict_state: dict) -> None:
        self._sim.state_devices.update(dict_state)
        # switching the LED to BLUE only selects the light path; light
        # reaches the sample while the shutter is open (see set_shutter)
        if self._shutter_open:
            self._apply_mask(self._current_slm_mask)

    # ── shutter ─────────────────────────────────────────────────────────

    def set_shutter(self, open_: bool) -> None:
        """Called by the shutter device (``core.setShutterOpen``).

        Snaps and live frames illuminate the sample by themselves (the
        core's auto-shutter), so this is only needed to hold the light on
        by hand. Opening it with the CyanStim channel selected delivers the
        loaded SLM pattern, like a manual exposure on real hardware.
        """
        self._shutter_open = bool(open_)
        if self._shutter_open:
            self._apply_mask(self._current_slm_mask)

    # ── SLM ─────────────────────────────────────────────────────────────

    def set_slm_mask(self, mask: np.ndarray) -> None:
        """Called by the SLM device: upload the pattern.

        The SLM only shapes light, so uploading does not stimulate by
        itself: the pattern reaches the sample when the stimulation light
        shines through it, i.e. on a snap (or live frame) in the CyanStim
        channel, or while the shutter is held open in CyanStim.
        """
        self._current_slm_mask = mask
        if self._shutter_open:
            self._apply_mask(mask)

    def _apply_mask(self, mask) -> None:
        """Push the pattern into the sim (gated there on the light path).

        In realtime mode the engine steps the sim on a background thread,
        so the update takes the engine lock (callers may be on yet another
        thread, e.g. an MDA frameReady callback).
        """
        if mask is None:
            return
        if self._engine is not None:
            with self._engine.lock:
                self._sim._update_from_devices()
                self._sim._handle_mask(mask)
        else:
            self._sim._update_from_devices()
            self._sim._handle_mask(mask)

    def get_slm_mask(self) -> np.ndarray:
        if self._current_slm_mask is not None:
            return self._current_slm_mask
        from vmteach.optogenetic.sim import SLM_SHAPE
        return np.zeros(SLM_SHAPE, dtype=np.uint8)
