"""SimulationBridge — connects pymmcore device adapters to simulation backends.

The bridge is a singleton (GLOBAL_BRIDGE) that all device adapters reference.
Swap it to swap the entire simulated sample without reloading devices.

All simulation backends must implement:
  - snap_frame(mask=None, exposure=1.0, intensity=1.0) -> np.ndarray
  - camera_offset: np.ndarray  (x, y) rendering origin
  - state_devices: dict        {name: {state, label}}
  - set_focal_plane(z: float)
  - viewport_width, viewport_height: int
"""

import threading
from typing import Optional

import numpy as np

# Singleton — all device adapters import this module and read GLOBAL_BRIDGE
GLOBAL_BRIDGE = None

# Event set by SimServer.initialize() when GLOBAL_BRIDGE is ready.
# State devices wait on this during their own initialize() so they can push
# initial state to the bridge even when initializeAllDevices() runs in parallel.
bridge_ready = threading.Event()


class BudgetExceededError(RuntimeError):
    """Raised by ``SimulationBridge.snap()`` when the cumulative exposure
    or bleach budget has been exceeded.

    Subclass of RuntimeError so existing test scenarios (which never set
    a budget) keep passing — only scenarios that opt in via
    ``set_dose_budget()`` ever see this raised. Mandal 2025
    sleepwalking-mode prevention: the simulator stops the agent before
    they bleach the sample beyond recovery.
    """


class SimulationBridge:
    """Bridge between pymmcore device adapters and simulation backends.

    Acts as a duck-typed façade: any sim with snap_frame(), set_focal_plane(),
    camera_offset, state_devices, viewport_width/height is compatible.
    """

    # Half-width of the base 10x render in µm (512 px at 1 µm/px).
    _BASE_HALF = 256

    def __init__(self, microscope_sim, factory=None) -> None:
        if microscope_sim is None:
            raise ValueError("The microscope simulation must be initialized.")
        self._sim = microscope_sim
        self._current_slm_mask = None
        self._engine = None  # RealtimeEngine, set by load_cfg / SimServer
        self._slm_processor = None  # lazy-init SLMProcessor
        self._factory = factory  # callable(**kw) -> new sim instance
        self._factory_kwargs: dict = {}  # params used to create current sim

        # Bleach / exposure budget watchdog (Mandal 2025 sleepwalking-mode
        # prevention). Defaults are infinity so existing scenarios keep
        # passing byte-equivalent. Scenarios opt in by calling
        # ``set_dose_budget(exposure_ms=…, bleach_dose=…)`` after sim
        # creation. On exceed: raises BudgetExceededError. Counters are
        # surfaced as Camera virtual properties for agent visibility.
        self._cum_exposure_ms: float = 0.0
        self._cum_bleach_dose: float = 0.0
        self._budget_exposure_ms: float = float("inf")
        self._budget_bleach_dose: float = float("inf")
        # Per-frame deltas — populated on every snap() so agents can
        # query the dose contributed by THE LAST frame (Mahecic 2022,
        # Jackson 2009 event-driven scoring).
        self._last_frame_exposure_ms: float = 0.0
        self._last_frame_bleach_dose: float = 0.0
        self._last_frame_coverage: float = 1.0
        # One-shot pixel-acquisition mask (Ye 2025 sparse scanning).
        # When set, the next ``snap()`` consumes it: zeros unselected
        # pixels in the output AND scales dose accounting by coverage.
        # Cleared after each snap — re-arm by calling
        # ``set_pixel_mask`` again. Mirrors the Reset device pattern.
        self._next_pixel_mask: Optional[np.ndarray] = None
        self._budget_on_exceed: str = "raise"  # "raise" | "warn"
        # Stim-step dose: per-bridge-step bleach charge when the SLM
        # mask is active (Vicidomini 2018 STED bleach budgeting; Mahecic
        # 2022 event-driven dose accounting). Defaults to 0.0 — every
        # MPC scenario before this mechanic shipped (ch624 / ch628 /
        # ch629) treats bridge.step as bleach-free, which is the
        # current contract. Scenarios opt in by calling
        # ``set_stim_step_dose(d)`` and counters charge ``d × dt`` per
        # step where SLM is active.
        self._stim_step_dose: float = 0.0
        self._last_step_bleach_dose: float = 0.0
        # Sample drift on light-sheet alignment axes (McDole 2018 active
        # imaging). When non-zero, a hidden drift vector accumulates per
        # bridge.step and per snap, additively biasing the effective
        # LightSheetY / LightSheetTiltX / LightSheetTiltY offsets so the
        # alignment optimum walks under the agent's feet between snaps.
        # Defaults to 0.0 = no drift (backwards-compatible; ch664/665
        # remain byte-equivalent).
        self._drift_dy_per_step: float = 0.0
        self._drift_dtx_per_step: float = 0.0
        self._drift_dty_per_step: float = 0.0
        self._cum_drift_dy: float = 0.0
        self._cum_drift_dtx: float = 0.0
        self._cum_drift_dty: float = 0.0

    @property
    def sim(self):
        """Direct access to the underlying simulation object."""
        return self._sim

    def set_pixel_mask(self, mask: Optional[np.ndarray]) -> None:
        """Arm a one-shot pixel-acquisition mask for the *next* snap.

        Maps to Ye 2025 (uncertainty-driven adaptive scanning) /
        Jackson 2009 / Kandel 2023 sparse-scanning archetypes. When
        the next ``snap()`` runs, it zeros pixels not selected by the
        mask AND scales dose accounting by coverage so a 25%-coverage
        frame charges 25% of the bleach dose. Mask is consumed and
        cleared on that snap; pass ``None`` (or just take another
        snap) to disarm.

        ``mask`` must be a 2D boolean (or 0/1) array matching the
        camera viewport shape. Shape mismatch raises ValueError on
        the next snap().
        """
        self._next_pixel_mask = mask

    def snap(self, exposure: float, brightness: float, gain: float = 1.0,
             pixel_mask: Optional[np.ndarray] = None,
             **kwargs) -> np.ndarray:
        # Advance hidden sample drift before the snap renders, so the
        # snapped frame reflects the post-drift effective alignment.
        self._advance_drift(n_steps=1.0)
        slm_mask = self.get_slm_mask()
        img = self._sim.snap_frame(mask=slm_mask, exposure=exposure,
                                   intensity=brightness, **kwargs)
        # Pull the bridge-armed pixel_mask if no kwarg was passed.
        # One-shot semantics: clear after consumption.
        if pixel_mask is None and self._next_pixel_mask is not None:
            pixel_mask = self._next_pixel_mask
            self._next_pixel_mask = None
        # Analog gain: amplifies signal AND noise (electronic amplification).
        if gain != 1.0:
            img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        # Pixel-uncertainty / sparse-scanning support (Ye 2025, Jackson
        # 2009, Kandel 2023). When pixel_mask is provided, unselected
        # pixels are zeroed in the output AND dose accounting scales
        # by coverage so 25%-coverage frames cost 25% of the bleach.
        coverage = 1.0
        if pixel_mask is not None:
            pm = np.asarray(pixel_mask)
            if pm.shape[:2] != img.shape[:2]:
                raise ValueError(
                    f"pixel_mask shape {pm.shape} does not match "
                    f"frame shape {img.shape}")
            pm_bool = pm.astype(bool)
            coverage = float(pm_bool.mean())
            # Zero unselected pixels in the output (uint8-friendly).
            if img.ndim == 3 and pm_bool.ndim == 2:
                pm_bool = pm_bool[:, :, None]
            img = np.where(pm_bool, img, 0).astype(img.dtype)
        self._last_frame_coverage = coverage

        # Dose-budget watchdog. Increment cumulative counters and check
        # against the published budgets. Cheap when budgets are inf
        # (the default) — just two float adds + two finite-comparisons.
        # Coverage scaling: a 25%-coverage frame contributes 25% of the
        # exposure-time + bleach-dose accounting.
        eff_exposure = float(exposure) * coverage
        self._cum_exposure_ms += eff_exposure
        self._last_frame_exposure_ms = eff_exposure
        # Estimated bleach dose: exposure × brightness × per-pipeline
        # photobleach_rate × per-fluorophore bleach_kx. Pulls from the
        # active mode's pipeline if present; otherwise approximates
        # as exposure × brightness. Per-fluorophore kx (Weigert 2018 /
        # Durand 2018) is fetched via sim._active_fluorophore() —
        # mRFP1 bleaches 2× EGFP, DAPI 0.3× — so dose accounting
        # tracks the realistic per-channel cost rather than treating
        # all fluorophores as equally bleach-prone.
        try:
            mode = getattr(self._sim, "mode", 0)
            pipe = self._sim._pipeline.get(mode) if hasattr(self._sim, "_pipeline") else None
            rate = float(getattr(pipe, "photobleach_rate", 0.0)) if pipe else 0.0
        except Exception:
            rate = 0.0
        try:
            fluor = (self._sim._active_fluorophore()
                     if hasattr(self._sim, "_active_fluorophore") else None)
            kx = float(fluor.bleach_kx) if fluor is not None else 1.0
        except Exception:
            kx = 1.0
        frame_bleach = eff_exposure * float(brightness) * max(rate, 1e-6) * kx
        self._cum_bleach_dose += frame_bleach
        self._last_frame_bleach_dose = frame_bleach

        if (self._cum_exposure_ms > self._budget_exposure_ms or
                self._cum_bleach_dose > self._budget_bleach_dose):
            msg = (f"Dose budget exceeded: cum_exposure_ms="
                   f"{self._cum_exposure_ms:.1f} (limit "
                   f"{self._budget_exposure_ms:.1f}), cum_bleach_dose="
                   f"{self._cum_bleach_dose:.4f} (limit "
                   f"{self._budget_bleach_dose:.4f}). Mandal 2025 "
                   f"sleepwalking-mode prevention.")
            if self._budget_on_exceed == "raise":
                raise BudgetExceededError(msg)
            elif self._budget_on_exceed == "warn":
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return img

    def set_dose_budget(self, exposure_ms: float = float("inf"),
                         bleach_dose: float = float("inf"),
                         on_exceed: str = "raise") -> None:
        """Set cumulative-snap budgets enforced by ``snap()``.

        Defaults to infinity = no enforcement (backwards-compatible).
        Scenarios opt in by calling this after sim creation:

            bridge.set_dose_budget(exposure_ms=5000, bleach_dose=0.5)

        ``on_exceed``:
          - ``"raise"`` (default): raise BudgetExceededError on next snap
          - ``"warn"``: issue a RuntimeWarning, allow the snap

        Resets the cumulative counters to 0 — call once at scenario start.
        """
        self._budget_exposure_ms = float(exposure_ms)
        self._budget_bleach_dose = float(bleach_dose)
        self._budget_on_exceed = on_exceed
        self._cum_exposure_ms = 0.0
        self._cum_bleach_dose = 0.0
        self._last_frame_exposure_ms = 0.0
        self._last_frame_bleach_dose = 0.0
        self._last_step_bleach_dose = 0.0

    def set_sample_drift_rate(self, dy: float = 0.0,
                              dtx: float = 0.0,
                              dty: float = 0.0) -> None:
        """Configure hidden sample drift on the LightSheet alignment axes.

        Drift accumulates one step per ``snap()``, additively biasing
        the effective LightSheetY / LightSheetTiltX / LightSheetTiltY
        offsets so the alignment optimum walks under the agent's feet
        between snaps (McDole 2018 mouse-embryo adaptive imaging —
        the "optimum moves while you image" abstraction). Each snap
        is the unit of drift progress; non-snap clock-time does not
        advance drift (the agent-facing ``bridge.step`` RPC was
        retired separately).

        Drift values are normalised offset-units per simulation step
        (matching the LightSheet device offset space [-1, +1]); a
        rate of 0.05 dy means after 20 steps the effective Y-offset
        has drifted by +1.0 (one full quantised step).

        Resets cumulative counters to 0. Call once at scenario start
        AFTER device initialisation (in generate(), pre-serve).

        Defaults to 0.0 = no drift (backwards-compat: ch664/665
        snaps remain byte-equivalent).
        """
        self._drift_dy_per_step = float(dy)
        self._drift_dtx_per_step = float(dtx)
        self._drift_dty_per_step = float(dty)
        self._cum_drift_dy = 0.0
        self._cum_drift_dtx = 0.0
        self._cum_drift_dty = 0.0

    def _advance_drift(self, n_steps: float = 1.0) -> None:
        """Advance the cumulative drift on each axis by n_steps."""
        if (self._drift_dy_per_step == 0.0
                and self._drift_dtx_per_step == 0.0
                and self._drift_dty_per_step == 0.0):
            return  # identity short-circuit
        self._cum_drift_dy += self._drift_dy_per_step * n_steps
        self._cum_drift_dtx += self._drift_dtx_per_step * n_steps
        self._cum_drift_dty += self._drift_dty_per_step * n_steps

    def get_effective_lightsheet_offsets(self) -> tuple:
        """Return (dy, dtx, dty) = device offset + cumulative drift.

        Used by the renderer hook in SimBase via ``_active_lightsheet_
        offsets``; backwards-compatible (returns zeros if no drift +
        no devices).
        """
        sim = self._sim
        if not hasattr(sim, "_active_lightsheet_offsets"):
            return (self._cum_drift_dy, self._cum_drift_dtx,
                    self._cum_drift_dty)
        device_dy, device_dtx, device_dty = sim._active_lightsheet_offsets()
        return (device_dy + self._cum_drift_dy,
                device_dtx + self._cum_drift_dtx,
                device_dty + self._cum_drift_dty)

    def set_stim_step_dose(self, dose_per_step: float) -> None:
        """Vestigial config setter — kept so old scenarios that call it
        during ``generate()`` don't crash on import. The corresponding
        per-step bleach charging path was removed when the agent-facing
        ``bridge.step`` RPC was retired (transferability fix). New
        scenarios should not rely on this; per-step stim cost belongs on
        a proper illumination device.
        """
        self._stim_step_dose = float(dose_per_step)

    def _slm_active(self) -> bool:
        """True iff the current SLM mask has any non-zero pixel."""
        mask = getattr(self, "_current_slm_mask", None)
        if mask is None:
            return False
        try:
            return bool(np.any(mask))
        except Exception:
            return False

    def _enforce_budget_gate(self) -> None:
        """Re-use the same budget-exceed gate as ``snap()`` for
        stim-step dose accounting. Identity when budgets are inf."""
        if (self._cum_exposure_ms > self._budget_exposure_ms or
                self._cum_bleach_dose > self._budget_bleach_dose):
            msg = (f"Dose budget exceeded (stim-step): "
                   f"cum_bleach_dose={self._cum_bleach_dose:.4f} "
                   f"(limit {self._budget_bleach_dose:.4f}). "
                   f"Mandal 2025 sleepwalking-mode prevention.")
            if self._budget_on_exceed == "raise":
                raise BudgetExceededError(msg)
            elif self._budget_on_exceed == "warn":
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def set_stage(self, x: float, y: float) -> None:
        """Stage position relative to world center; convert to rendering origin.

        (0, 0) centres the viewport on the world.  The stage coordinate
        is added to the world centre so positive values pan right/down.
        """
        sim = self._sim
        world_cx = sim.width / 2.0
        world_cy = sim.height / 2.0
        self._sim.camera_offset = np.array([
            world_cx + x - self._BASE_HALF,
            world_cy + y - self._BASE_HALF,
        ])

    def set_focus(self, z: float) -> None:
        self._sim.set_focal_plane(z)

    def update_state(self, dict_state: dict) -> None:
        self._sim.state_devices.update(dict_state)
        # Modality change must invalidate any cached render — voronoi-
        # family backends pre-render at sim init and would otherwise
        # serve stale frames after a modality switch. Other state
        # devices (Filter Wheel, Channel, Objective) keep their existing
        # behavior because they only affect the render path through
        # `_apply_pipeline` which fires per-snap.
        if ("Modality" in dict_state or "DeformableMirror" in dict_state
                or "RotationStage" in dict_state
                or "STEDDepletion" in dict_state
                or "LightSheetY" in dict_state
                or "LightSheetTiltX" in dict_state
                or "LightSheetTiltY" in dict_state):
            invalidate = getattr(self._sim, "_invalidate_render_cache", None)
            if callable(invalidate):
                invalidate()

    @property
    def slm_processor(self):
        """Lazily-initialized SLMProcessor for centralized mask handling."""
        if self._slm_processor is None:
            from vmteach.engine.slm_processor import SLMProcessor
            sim = self._sim
            w = getattr(sim, 'width', getattr(sim, 'world_size', 512))
            h = getattr(sim, 'height', getattr(sim, 'world_size', 512))
            self._slm_processor = SLMProcessor(w, h)
        return self._slm_processor

    def set_slm_mask(self, mask: np.ndarray) -> None:
        """Called by SLM device when pattern changes.

        Updates the SLM processor's stimulation field and also propagates
        to the sim's ``_stim_mask`` so that background-thread step() calls
        (RealtimeEngine) pick up the mask immediately.
        """
        self._current_slm_mask = mask

        # Update SLM processor
        if mask is not None:
            sim = self._sim
            self.slm_processor.update_mask(
                mask,
                camera_offset=sim.camera_offset,
                objective_mag=getattr(sim, 'current_objectiv', 10),
                viewport_width=sim.viewport_width,
                viewport_height=sim.viewport_height,
            )

        # Backward compat: propagate to sim._stim_mask. Prefer the sim's
        # own _handle_mask() if it has one (calcium / RD downsample to a
        # coarse PDE grid, otherwise step_autonomous() blows up with a
        # shape-mismatch broadcast error). Fall back to raw_mask for
        # backends that just want the world-resolution mask as-is.
        if mask is None:
            if hasattr(self._sim, '_stim_mask'):
                self._sim._stim_mask = None
        else:
            handler = getattr(self._sim, '_handle_mask', None)
            if callable(handler):
                handler(self.slm_processor.raw_mask)
            elif hasattr(self._sim, '_stim_mask'):
                self._sim._stim_mask = self.slm_processor.raw_mask

    def get_slm_mask(self) -> np.ndarray:
        """Called by camera device when capturing."""
        if self._current_slm_mask is not None:
            return self._current_slm_mask
        # Default: no stimulation
        return np.zeros(
            (self._sim.viewport_height, self._sim.viewport_width), dtype=bool
        )

    # ── Experiment lifecycle ──

    def reset_simulation(self, seed: int | None = None) -> None:
        """Soft reset: call sim.reset(seed) and clear SLM state.

        Keeps the same sim instance — just resets internal state.
        """
        if hasattr(self._sim, 'reset'):
            self._sim.reset(seed)
        if self._slm_processor is not None:
            self._slm_processor.reset()
        self._current_slm_mask = None

    def recreate_simulation(self, seed: int | None = None) -> None:
        """Hard reset: create a new sim via factory, transfer device state.

        Requires ``factory`` to have been passed at construction.
        """
        if self._factory is None:
            # Fallback to soft reset if no factory
            self.reset_simulation(seed)
            return

        # Build new sim with same params + updated seed
        kwargs = dict(self._factory_kwargs)
        if seed is not None:
            kwargs['seed'] = seed
        new_sim = self._factory(**kwargs)

        # Transfer device state and camera
        new_sim.state_devices = dict(self._sim.state_devices)
        new_sim.camera_offset = self._sim.camera_offset.copy()
        new_sim.focal_plane = self._sim.focal_plane

        # Swap sim reference
        old_sim = self._sim
        self._sim = new_sim

        # Update engine's sim reference if running
        if self._engine is not None:
            self._engine._sim = new_sim
            # Re-patch snap_frame
            self._engine.patch_snap_frame()

        # Reset SLM state
        if self._slm_processor is not None:
            self._slm_processor.reset()
        self._current_slm_mask = None


def set_global_bridge(bridge: SimulationBridge) -> None:
    """Replace the global bridge, stopping any running engine on the old one.

    Use this instead of assigning ``GLOBAL_BRIDGE`` directly so that
    a running RealtimeEngine is properly shut down before the sim reference
    becomes stale.
    """
    global GLOBAL_BRIDGE
    old = GLOBAL_BRIDGE
    if old is not None and hasattr(old, '_engine') and old._engine is not None:
        old._engine.stop()
        old._engine = None
    GLOBAL_BRIDGE = bridge
    bridge_ready.set()
