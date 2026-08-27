"""SimBase — Abstract base class for microscope simulation backends.

Extracts ~1800 lines of duplicated boilerplate shared across 20+ standalone
sims into a single ABC.  Every microscope-style backend (i.e. one that has
objectives, FOV cropping, defocus, etc.) should inherit from SimBase and
implement :meth:`_render_for_mode`.

Non-microscope sims (PlateReaderSim, FlowCytometrySim, GelDocSim,
HemocytometerSim, ColonySim) stay duck-typed and do NOT need this base.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class SimBase(ABC):
    """Abstract base class providing the SimulationBridge-compatible interface.

    Subclasses must implement :meth:`_render_for_mode`.  The canonical
    ``snap_frame()`` pipeline is provided as a concrete template method.
    Override ``_handle_mask()`` for SLM / optogenetic / drug stimulation,
    and ``_finalize_output()`` to change the output format (e.g. RGB).

    Class attributes:
        continuous: Set to ``True`` on dynamic backends whose simulation
            should advance in real-time between snaps.  ``load_cfg()`` uses
            this to auto-start a :class:`RealtimeEngine`.
    """

    continuous: bool = False

    _FOV_MAP: dict = {100: 64, 40: 128, 20: 256}
    _default_temperature: float = 37.0

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        seed: int = 42,
        internal_scale: int = 4,
        mode_map: dict | None = None,
        fixed_dt: float = 0.0,
        auto_step: bool = False,
        snaps_per_step: int = 1,
    ):
        # ── World dimensions ──
        self.width = width
        self.height = height
        self.internal_scale = internal_scale
        self._iw = width * internal_scale
        self._ih = height * internal_scale
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        # ── Camera / state (SimulationBridge interface) ──
        self.camera_offset = np.array([
            (width - viewport_width) / 2.0,
            (height - viewport_height) / 2.0,
        ])
        self.focal_plane = 0.0
        self.tissue_z = 0.0
        self.state_devices: dict = {}
        self.mode = 0
        self.current_objectiv = 10

        # ── Objective / DOF tables ──
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40, "100x": 100}
        self._dof_table = {10: 6.0, 20: 4.0, 40: 1.5, 100: 0.6}
        self._dof = 6.0
        self._blur_scale_table = {10: 0.3, 20: 0.5, 40: 1.0, 100: 2.0}

        # ── Channel routing ──
        self._extra_channels: dict = {}
        self._mode_map: dict = mode_map if mode_map is not None else {}

        # ── Auto-step control ──
        self.auto_step = auto_step
        self.snaps_per_step = snaps_per_step
        self.fixed_dt = fixed_dt

        # ── Timing / snap counter ──
        self._snap_count = 0
        self._time = 0.0

        # ── Z-drift ──
        self.z_drift_rate = 0.0
        self.z_drift_noise = 0.0

        # ── Stage drift ──
        self.stage_drift_rate = 0.0
        self.stage_drift_noise = 0.0
        self._drift_accumulator = np.array([0.0, 0.0])

        # ── Optical pipelines (subclasses populate) ──
        self._pipeline: dict = {}

        # ── RNG ──
        self.rng = np.random.default_rng(seed)

    # ──────────────────────────────────────────────────────────
    # Coordinate scaling
    # ──────────────────────────────────────────────────────────

    def _s(self, v):
        """Scale world coordinate to internal resolution (int)."""
        return int(round(v * self.internal_scale))

    def _sf(self, v):
        """Scale world coordinate to internal resolution (float)."""
        return v * self.internal_scale

    # ──────────────────────────────────────────────────────────
    # Device-state helpers
    # ──────────────────────────────────────────────────────────

    def _get_temperature(self) -> float:
        """Read temperature from the Temperature state device (°C).

        Returns ``_default_temperature`` when the device is absent.
        Subclasses override ``_default_temperature`` at the class level.
        """
        if "Temperature" not in self.state_devices:
            return self._default_temperature
        return float(
            self.state_devices["Temperature"].get(
                "label", str(self._default_temperature)
            )
        )

    def _update_mode(self):
        """Update rendering mode via ``_mode_map`` lookup."""
        if "Filter Wheel" not in self.state_devices or "LED" not in self.state_devices:
            return  # keep current mode when no devices are registered
        filt = self.state_devices["Filter Wheel"]
        led = self.state_devices["LED"]
        filter_label = filt.get("label", filt.get("Label", ""))
        led_label = led.get("label", led.get("Label", ""))

        key = (filter_label, led_label)
        if key in self._mode_map:
            self.mode = self._mode_map[key]
            return
        for mode_id, ch_info in self._extra_channels.items():
            if filter_label == ch_info["filter"] and led_label == ch_info["led"]:
                self.mode = mode_id
                return
        self.mode = 0

    def _update_objectif(self):
        """Update objective magnification from device state."""
        if "Objective" not in self.state_devices:
            return
        obj = self.state_devices["Objective"]
        lbl = obj.get("Label", obj.get("label", ""))
        if lbl in self._objectif_dict:
            mag = self._objectif_dict[lbl]
            dof = self._dof_table.get(mag, 6.0)
            self.current_objectiv = mag
            self._dof = dof
            self._on_objective_changed(mag, dof)

    def _on_objective_changed(self, mag: int, dof: float):
        """Hook called when the objective changes. Override in subclasses.

        Base behaviour: walk every OpticalPipeline whose chromatic_cfg is
        non-empty and update its ``magnification`` to the new value, so
        100x snaps see ~3x heavier lateral chromatic aberration than 10x
        snaps (the LCA scales with objective magnification per the
        physical optics — `optical_pipeline.py:_apply_chromatic_lateral`
        line 264). Backends that didn't opt into chromatic (chromatic_cfg
        empty) are unaffected.
        """
        # Pattern 1: backend-style _pipeline dict {0: bf, 1: nuc, 2: mem}.
        pipes = []
        if hasattr(self, "_pipeline") and isinstance(self._pipeline, dict):
            pipes.extend(self._pipeline.values())
        # Pattern 2: voronoi-style named attributes _bf_pipeline / _nuc_
        # pipeline / _mem_pipeline (sims/voronoi/voronoi.py).
        for attr in ("_bf_pipeline", "_nuc_pipeline", "_mem_pipeline"):
            pipe = getattr(self, attr, None)
            if pipe is not None:
                pipes.append(pipe)
        for pipe in pipes:
            if getattr(pipe, "chromatic_cfg", None):
                pipe.chromatic_cfg["magnification"] = mag

    def set_focal_plane(self, z: float):
        """Set focal plane position (µm)."""
        self.focal_plane = z

    def update_state(self, dict_state: dict):
        """Replace device-state dict (called by SimulationBridge)."""
        self.state_devices = dict_state

    # ──────────────────────────────────────────────────────────
    # FOV cropping
    # ──────────────────────────────────────────────────────────

    def _crop_fov(self, full):
        """Crop FOV from internal-resolution buffer, resize to viewport."""
        s = self.internal_scale
        ih, iw = full.shape[:2]
        out_w, out_h = self.viewport_width, self.viewport_height
        obj = self.current_objectiv

        fov_world = self._FOV_MAP.get(obj, min(512, self.width))
        fov_int = fov_world * s

        # Stage center in world coords → internal coords
        cx_world = int(self.camera_offset[0] + self._drift_accumulator[0]) + out_w // 2
        cy_world = int(self.camera_offset[1] + self._drift_accumulator[1]) + out_h // 2
        cx_int = int(cx_world * s)
        cy_int = int(cy_world * s)

        half = fov_int // 2
        x0 = max(0, min(cx_int - half, iw - fov_int))
        y0 = max(0, min(cy_int - half, ih - fov_int))

        crop = full[y0:y0 + fov_int, x0:x0 + fov_int].copy()

        # Pad if crop extends beyond full image
        ch = crop.shape[2] if crop.ndim == 3 else 0
        if crop.shape[0] < fov_int or crop.shape[1] < fov_int:
            bg = self._get_pad_bg()
            if ch > 0:
                padded = np.full((fov_int, fov_int, ch), bg, dtype=crop.dtype)
            else:
                padded = np.full((fov_int, fov_int), bg, dtype=crop.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        if crop.shape[0] > out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        elif crop.shape[0] < out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return crop

    def _get_pad_bg(self) -> int:
        """Background value for out-of-bounds padding in ``_crop_fov``."""
        return 140 if self.mode == 0 else 0

    # ──────────────────────────────────────────────────────────
    # Defocus
    # ──────────────────────────────────────────────────────────

    def _apply_defocus(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian defocus blur based on distance from focal plane."""
        dz = abs(self.focal_plane - self.tissue_z)
        half_dof = self._dof / 2.0
        if dz <= half_dof:
            return img
        sigma = min((dz - half_dof) * self._blur_scale_table.get(
            self.current_objectiv, 0.5), 30.0)
        if sigma < 0.3:
            return img
        return cv2.GaussianBlur(img, (0, 0), sigma)

    # Per-µm lateral shift coefficient, units = pixels per µm of
    # signed defocus, at 100x reference (scaled per objective via
    # _blur_scale_table). Matches the Pinkard 2019 single-shot AF
    # geometry where an off-axis Köhler LED translates an out-of-
    # focus sample plane by an amount linear in defocus and in the
    # LED's offset angle.
    OFF_AXIS_SHIFT_PER_UM: float = 1.5

    def _active_rotation_angle_rad(self) -> float:
        """Resolve the active RotationStage state to an angle in radians.
        Returns 0.0 when no RotationStage device is registered (the common
        case) or when the device is at state 0 ("0"). The renderer hook
        short-circuits on |angle| < 1e-6, so this is backwards-compatible
        across all 36 backends.

        The RotationStageDevice pushes the parsed angle at the
        ``angle_rad`` key alongside ``state``/``label`` on every state
        change, so this resolver is just dict-lookup.
        """
        rs = (self.state_devices.get("RotationStage")
              if hasattr(self, "state_devices") else None)
        if not rs:
            return 0.0
        a = rs.get("angle_rad")
        if a is not None:
            return float(a)
        # Fallback: parse from label.
        label = rs.get("label") or rs.get("Label")
        try:
            from vmteach.devices.state import RotationStageDevice
            return RotationStageDevice.parse_label(label)
        except Exception:
            return 0.0

    def _apply_sim_pattern(self, img: np.ndarray) -> np.ndarray:
        """Multiply viewport by a sinusoidal SIM illumination pattern.

        Maps to Jin 2020 (deep-learning SIM). The active SIMPattern
        device's (orientation, phase, k_lp_per_px) tuple defines a
        cosine grating `0.5 * (1 + cos(2π k (x cos θ + y sin θ) + φ))`
        applied multiplicatively. Identity (k = 0) for state
        ``"off"`` or when no SIMPattern device is registered → all
        backends without the device wired see no behaviour change.

        Brightfield mode (mode == 0) bypasses — structured
        illumination only modulates fluorescence emission.

        Hook lives between ``_apply_sample_rotation`` and
        ``_apply_pipeline`` so noise / chromatic / bleach see the
        modulated frame (they should: real SIM cameras observe the
        modulated emission, including its noise).
        """
        if getattr(self, "mode", 0) == 0:
            return img
        sim_state = (self.state_devices.get("SIMPattern")
                     if hasattr(self, "state_devices") else None)
        if not sim_state:
            return img
        k = float(sim_state.get("k_lp_per_px", 0.0))
        if abs(k) < 1e-9:
            return img
        orient = float(sim_state.get("orientation_rad", 0.0))
        phase = float(sim_state.get("phase_rad", 0.0))
        h, w = img.shape[:2]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        # 0.5*(1+cos(...)) ∈ [0, 1]; multiplied with img.
        proj = (xx * np.cos(orient) + yy * np.sin(orient))
        pattern = 0.5 * (1.0 + np.cos(2.0 * np.pi * k * proj + phase))
        if img.ndim == 3 and pattern.ndim == 2:
            pattern = pattern[:, :, None]  # broadcast over channel axis
        out = img.astype(np.float32) * pattern
        info = (np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer)
                else None)
        if info is not None:
            return np.clip(out, 0, info.max).astype(img.dtype)
        return np.clip(out, 0.0, 1.0).astype(img.dtype)

    def _active_lightsheet_offsets(self) -> tuple:
        """Return (offset_y, tilt_x, tilt_y) in [-1, +1] from the 3
        light-sheet alignment-axis devices.  All zero when no devices
        are registered.

        offset_y: lateral peak shift along Y (Royer 2016 AutoPilot Y axis)
        tilt_x:   sheet pitched about X -> brightness gradient along Y
        tilt_y:   sheet pitched about Y -> brightness gradient along X
        """
        if not hasattr(self, "state_devices"):
            return (0.0, 0.0, 0.0)
        sd = self.state_devices

        def _read(name):
            d = sd.get(name)
            if not d:
                return 0.0
            v = d.get("offset")
            if v is not None:
                return float(v)
            label = d.get("label") or d.get("Label")
            try:
                from vmteach.devices.state import _LightSheetAxisDevice
                return _LightSheetAxisDevice.parse_label(label)
            except Exception:
                return 0.0

        return (_read("LightSheetY"), _read("LightSheetTiltX"),
                _read("LightSheetTiltY"))

    def _apply_lightsheet_envelope(self, img: np.ndarray) -> np.ndarray:
        """Multiply viewport by a light-sheet illumination envelope.

        Composes a 2D modulation:
          envelope(x, y) = gauss(y - cy_off, sigma) × (1 + tiltY × x_n)
                                                  × (1 + tiltX × y_n)
        where ``x_n``, ``y_n`` are normalised viewport coordinates
        in [-1, +1] and ``cy_off = (h/2) + offset_y * h * 0.45`` shifts
        the Gaussian peak along Y.  ``sigma = h * 0.6`` (broad).

        Brightfield (mode == 0) bypasses — light-sheet illumination
        only modulates fluorescence emission.

        Effective offsets compose the LightSheet device offsets with
        any cumulative bridge-side drift (McDole 2018) — drift
        biases the alignment optimum off the device-default state.

        Identity at all-zero offsets + zero drift so backends without
        LightSheet devices wired (or with all 3 devices at default
        state 2 = 0.0 and no drift) see byte-equivalent renders.
        """
        if getattr(self, "mode", 0) == 0:
            return img
        # Try bridge-effective offsets first (device + drift composed).
        try:
            from vmteach.engine.simulation_bridge import GLOBAL_BRIDGE
            if GLOBAL_BRIDGE is not None and GLOBAL_BRIDGE._sim is self:
                offset_y, tilt_x, tilt_y = (
                    GLOBAL_BRIDGE.get_effective_lightsheet_offsets()
                )
            else:
                offset_y, tilt_x, tilt_y = self._active_lightsheet_offsets()
        except Exception:
            offset_y, tilt_x, tilt_y = self._active_lightsheet_offsets()
        if (abs(offset_y) < 1e-6 and abs(tilt_x) < 1e-6
                and abs(tilt_y) < 1e-6):
            return img
        h, w = img.shape[:2]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        # Normalised viewport coords in [-1, +1]
        x_n = (xx - w / 2.0) / (w / 2.0)
        y_n = (yy - h / 2.0) / (h / 2.0)
        # Gaussian Y envelope (peak shifted by offset_y * 0.45)
        sigma = h * 0.6
        cy_off = (h / 2.0) + offset_y * h * 0.45
        env = np.exp(-((yy - cy_off) ** 2) / (2.0 * sigma ** 2))
        # Multiplicative tilt gradients (small linear terms; max ~30%)
        env = env * (1.0 + 0.3 * tilt_y * x_n)
        env = env * (1.0 + 0.3 * tilt_x * y_n)
        env = np.clip(env, 0.0, 2.0)
        if img.ndim == 3 and env.ndim == 2:
            env = env[:, :, None]
        out = img.astype(np.float32) * env
        info = (np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer)
                else None)
        if info is not None:
            return np.clip(out, 0, info.max).astype(img.dtype)
        return np.clip(out, 0.0, 1.0).astype(img.dtype)

    def _apply_sample_rotation(self, img: np.ndarray) -> np.ndarray:
        """Rotate the viewport in 2D about its centre by the active
        RotationStage angle. Foundation-only: a 2D affine rotation is
        geometrically right for thin-layer specimens (voronoi tissue,
        yeast colonies) but does NOT model true 3D sample-orientation
        rotation. Z-stack backends (volvox, zebrafish, organoid) will
        eventually need a per-depth re-projection — tracked as a
        follow-up wakeup; until those backends opt in, the
        RotationStage is only wired on voronoi.

        No-op when angle ~ 0 so all 36 backends without a RotationStage
        device see byte-equivalent behavior.
        """
        angle_rad = self._active_rotation_angle_rad()
        if abs(angle_rad) < 1e-6:
            return img
        h, w = img.shape[:2]
        angle_deg = float(np.degrees(angle_rad))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _apply_off_axis_lateral_shift(self, img: np.ndarray) -> np.ndarray:
        """Translate the viewport by a vector proportional to signed
        defocus and the off-axis-LED unit direction. No-op when the
        OffAxisLED device is absent or at state ``Off``, so all
        backends without the device see byte-equivalent behavior.

        Simplification vs a faithful Pinkard 2019 render: a real rig
        would superpose an in-focus component with a shifted
        out-of-focus blur. We translate the whole viewport. For
        single-thin-layer specimens (voronoi tissue, yeast colonies,
        etc.) this is realistic enough to encode signed defocus in
        one frame; thick-sample backends may want a follow-up
        per-depth render.
        """
        dx_unit, dy_unit = self._active_off_axis_led_direction()
        if dx_unit == 0.0 and dy_unit == 0.0:
            return img
        # Signed defocus: focal_plane > tissue_z → looking *above* the
        # sample plane → shift toward +(unit) when LED biases that way.
        dz = float(self.focal_plane - self.tissue_z)
        if abs(dz) < 1e-6:
            return img
        scale = self._blur_scale_table.get(self.current_objectiv, 0.5)
        tx = dz * self.OFF_AXIS_SHIFT_PER_UM * scale * dx_unit
        ty = dz * self.OFF_AXIS_SHIFT_PER_UM * scale * dy_unit
        if abs(tx) < 0.05 and abs(ty) < 0.05:
            return img
        h, w = img.shape[:2]
        M = np.float32([[1.0, 0.0, tx], [0.0, 1.0, ty]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # ──────────────────────────────────────────────────────────
    # Exposure & pipeline helpers
    # ──────────────────────────────────────────────────────────

    def _apply_exposure(self, viewport: np.ndarray, exposure: float,
                        intensity: float) -> np.ndarray:
        """Apply exposure and intensity scaling.

        BF (mode 0) uses 2x base so default exposure=50 gives full contrast.
        Fluorescence uses standard 1x photon-collection model.

        Per-fluorophore brightness (Weigert 2018 SNR realism): for
        fluorescence modes, multiply by the active fluorophore's
        brightness relative to EGFP (eps × qy / 1000 / 33.6).
        Identity at EGFP-baseline channels and at no-fluor; >1 for
        mScarlet3 / FITC / Cy5; <1 for Electra1 / miRFP670.
        """
        if self.mode == 0:
            scale = min(intensity * 0.02 * exposure, 2.0)
        else:
            scale = intensity * 0.01 * exposure
            scale *= self._brightness_scale_for_active_fluorophore()
        return (viewport.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)

    def _apply_pipeline(self, viewport: np.ndarray,
                        exposure: float) -> np.ndarray:
        """Apply optical pipeline (noise, PSF, vignette, bleaching).

        Looks up the active emission wavelength from the Filter Wheel
        device label so OpticalPipeline can apply per-channel chromatic
        aberration (LCA + axial-chromatic defocus). Pipelines without
        a ``chromatic={...}`` config short-circuit on the wavelength
        internally, so this is backwards-compatible — only backends
        that opt-in get the new behavior.

        Modality device (if registered): looks up the active modality
        profile and applies psf_sigma_mult / exposure_ceiling_ms /
        bleach_rate_mult / channels_allowed. Backends without a
        Modality device or without registered profiles see identity.
        """
        if self.mode in self._pipeline:
            # Resolve the modality profile (identity if none registered)
            modality = self._active_modality_profile()
            # Channel lockout: modality may exclude this channel entirely
            if modality.get("channels_allowed") is not None:
                fw = (self.state_devices.get("Filter Wheel")
                      if hasattr(self, "state_devices") else None)
                label = (fw.get("label") or fw.get("Label") or "") if fw else ""
                if label not in modality["channels_allowed"]:
                    return np.zeros_like(viewport)

            pipe = self._pipeline[self.mode]
            # mode=0 is brightfield (broadband illumination, no
            # single-wavelength chromatic effect). Fluorescence modes
            # (mode>=1) get the per-channel emission wavelength.
            wl = self._active_wavelength_nm() if self.mode > 0 else 0.0

            # AO: build axis-distinguishable per-axis PSF parameters.
            # The all-zero identity tuple yields psf_params with
            # sigma_x == sigma_y == base_sigma and zero coma — produces
            # output byte-equivalent to the pre-AO scalar PSF path.
            zernike = self._active_zernike_coeffs()
            modality_mult = modality.get("psf_sigma_mult", 1.0)
            # STED depletion multiplier (Hell 1994 / Vicidomini 2018):
            # composes with modality multiplier so a sted_burst modality
            # AND a saturated STEDDepletion device both narrow the PSF.
            # Identity (1.0) when no STEDDepletion device is registered.
            sted_sat = self._active_sted_saturation()
            if sted_sat > 0.0:
                from vmteach.pipeline.optical_pipeline import (
                    sted_psf_sigma_mult)
                modality_mult = modality_mult * sted_psf_sigma_mult(sted_sat)
            base_sigma = pipe.psf_sigma * modality_mult
            psf_params = None
            if any(abs(c) > 1e-9 for c in zernike):
                from vmteach.pipeline.optical_pipeline import (
                    zernike_to_psf_params)
                psf_params = zernike_to_psf_params(zernike, base_sigma)

            # Per-fluorophore bleach kinetics (Weigert 2018 / Durand 2018).
            # When the active fluorophore record is registered, multiply
            # the pipeline's bleach rate by its EGFP-relative kx. mRFP1
            # bleaches 2x faster than EGFP; DAPI 0.3x; mScarlet3 0.5x.
            # Identity at kx=1.0 (or no fluorophore record).
            fluor = self._active_fluorophore()
            bleach_kx = float(fluor.bleach_kx) if fluor is not None else 1.0

            # Apply modality knobs to the pipeline before snap.
            # Save and restore so other channels are unaffected.
            saved_psf = pipe.psf_sigma
            saved_bleach = pipe.photobleach_rate
            try:
                pipe.psf_sigma = base_sigma
                pipe.photobleach_rate = (saved_bleach
                                         * modality.get("bleach_rate_mult", 1.0)
                                         * bleach_kx)
                exposure_capped = min(
                    exposure, modality.get("exposure_ceiling_ms", float("inf")))
                if pipe.photobleach_rate > 0 and self.mode > 0:
                    viewport = pipe.apply_with_bleach(
                        viewport, exposure_ms=exposure_capped,
                        wavelength_nm=wl, psf_params=psf_params)
                else:
                    viewport = pipe.apply(
                        viewport, exposure_ms=exposure_capped,
                        wavelength_nm=wl, psf_params=psf_params)
            finally:
                pipe.psf_sigma = saved_psf
                pipe.photobleach_rate = saved_bleach
        return viewport

    _DEFAULT_MODALITY_PROFILE = {
        "psf_sigma_mult": 1.0,
        "exposure_ceiling_ms": float("inf"),
        "bleach_rate_mult": 1.0,
        "channels_allowed": None,
    }

    def _register_modality_profiles(self, profiles: dict) -> None:
        """Backends call this from ``__init__`` to register their
        modality property bundles, e.g.::

            self._register_modality_profiles({
                "widefield_scout": {"psf_sigma_mult": 1.0, ...},
                "sted_burst":      {"psf_sigma_mult": 0.25,
                                     "exposure_ceiling_ms": 50,
                                     "bleach_rate_mult": 8.0,
                                     "channels_allowed": ["TagGFP2(483/506)"]},
            })

        Each profile dict overrides any of the four
        ``_DEFAULT_MODALITY_PROFILE`` keys.
        """
        if not hasattr(self, "_modality_profiles"):
            self._modality_profiles = {}
        self._modality_profiles.update(profiles)

    def _active_modality_profile(self) -> dict:
        """Resolve the currently-active modality label to a property
        bundle. Returns the default identity profile when no Modality
        device is present or no profile is registered for the active
        label."""
        if not hasattr(self, "_modality_profiles"):
            return self._DEFAULT_MODALITY_PROFILE
        mod = (self.state_devices.get("Modality")
               if hasattr(self, "state_devices") else None)
        if not mod:
            return self._DEFAULT_MODALITY_PROFILE
        label = mod.get("label") or mod.get("Label") or ""
        return self._modality_profiles.get(
            label, self._DEFAULT_MODALITY_PROFILE)

    _DEFAULT_ZERNIKE = (0.0, 0.0, 0.0, 0.0, 0.0)

    def _active_zernike_coeffs(self) -> tuple:
        """Resolve the active DeformableMirror state to a 5-tuple of
        Zernike coefficients (defocus, astig_x, astig_y, coma_x, coma_y)
        in radians-rms wavefront. Returns the all-zero identity tuple
        when no DeformableMirror device is registered (the common case
        — most backends don't have one), so renderer hooks that read
        this resolver are backwards-compatible.

        The DeformableMirrorDevice pushes the parsed tuple at the
        ``zernike`` key alongside ``state``/``label`` on every state
        change, so this resolver is just dict-lookup. If a non-DM
        producer puts a ``DeformableMirror`` entry in state_devices
        without a ``zernike`` key we fall back to the device's
        label-parser as a safety net.
        """
        dm = (self.state_devices.get("DeformableMirror")
              if hasattr(self, "state_devices") else None)
        if dm is not None:
            z = dm.get("zernike")
            if z is None:
                # Fallback: parse from label (non-device producers).
                label = dm.get("label") or dm.get("Label")
                try:
                    from vmteach.devices.state import (
                        DeformableMirrorDevice)
                    z = DeformableMirrorDevice.parse_label(label)
                except Exception:
                    z = self._DEFAULT_ZERNIKE
        else:
            z = self._DEFAULT_ZERNIKE
        # Sample-induced aberration: a baseline wavefront error from
        # specimen heterogeneity (refractive-index mismatch, agarose
        # bowing, etc.) that the DM has to *cancel*. Without a sample
        # aberration the optimal DM is always flat; with one, the
        # optimal DM equals the negative of the sample aberration.
        sample = getattr(self, "_sample_aberration_zernike", None)
        if sample is None:
            return tuple(z)
        return tuple(zi + si for zi, si in zip(z, sample))

    def _active_sted_saturation(self) -> float:
        """Resolve the active STEDDepletion state to a saturation factor
        ζ ∈ [0, 0.95]. Returns 0.0 when no STEDDepletion device is
        registered (the common case — most backends don't have one),
        so renderer hooks that read this resolver are backwards-
        compatible.

        The STEDDepletionDevice pushes the parsed saturation float at
        the ``saturation_factor`` key alongside ``state``/``label`` on
        every state change, so this resolver is just dict-lookup. If a
        non-device producer puts an ``STEDDepletion`` entry in
        state_devices without a ``saturation_factor`` key, fall back
        to the device's label-parser as a safety net.
        """
        sd = (self.state_devices.get("STEDDepletion")
              if hasattr(self, "state_devices") else None)
        if sd is None:
            return 0.0
        sat = sd.get("saturation_factor")
        if sat is not None:
            return float(sat)
        # Fallback: parse from label.
        label = sd.get("label") or sd.get("Label")
        try:
            from vmteach.devices.state import STEDDepletionDevice
            return float(STEDDepletionDevice.parse_label(label))
        except Exception:
            return 0.0

    def _active_off_axis_led_direction(self) -> tuple:
        """Resolve the active OffAxisLED state to a (dx_unit, dy_unit)
        unit-vector tuple. Returns (0.0, 0.0) when no OffAxisLED
        device is registered (the common case — most backends don't
        have one) or when the device is at state 0 ("Off"). The
        renderer hook short-circuits on the zero vector, so this is
        backwards-compatible across all 36 backends.

        The OffAxisLEDDevice pushes the parsed unit-vector at the
        ``direction`` key alongside ``state``/``label`` on every state
        change, so this resolver is just dict-lookup.
        """
        oa = (self.state_devices.get("OffAxisLED")
              if hasattr(self, "state_devices") else None)
        if not oa:
            return (0.0, 0.0)
        d = oa.get("direction")
        if d is not None:
            return tuple(d)  # type: ignore[return-value]
        # Fallback: parse from label (covers any non-device producer).
        label = oa.get("label") or oa.get("Label")
        try:
            from vmteach.devices.state import OffAxisLEDDevice
            return OffAxisLEDDevice.parse_label(label)
        except Exception:
            return (0.0, 0.0)

    def set_sample_aberration(self, coeffs):
        """Pre-load a baseline 5-element Zernike-coefficient tuple
        representing sample-induced wavefront error. Read by
        ``_active_zernike_coeffs`` and added to whatever the
        DeformableMirror device contributes, so the optimal DM
        state is the one that *cancels* this baseline rather than the
        identity.

        ``coeffs`` may be any iterable of 5 floats (defocus, astig_x,
        astig_y, coma_x, coma_y). Pass ``None`` to clear.
        """
        if coeffs is None:
            self._sample_aberration_zernike = None
        else:
            self._sample_aberration_zernike = tuple(float(c) for c in coeffs)
        # Rendered images cached by voronoi-family backends must
        # rebuild after this change so the aberration takes effect.
        invalidate = getattr(self, "_invalidate_render_cache", None)
        if callable(invalidate):
            invalidate()

    # EGFP normalises the brightness multiplier so identity at the
    # GFP-baseline channel; brighter fluorophores (mScarlet3) scale
    # the per-snap signal up, dimmer ones (Electra1, mRFP1) scale down.
    _BRIGHTNESS_BASELINE_EGFP = 33.6  # EGFP eps×qy/1000 = 56*0.6/1000

    def _brightness_scale_for_active_fluorophore(self) -> float:
        """Per-fluorophore signal-scaling multiplier (Weigert 2018 SNR
        realism). Returns 1.0 when no fluorophore is registered or
        when the active fluorophore matches EGFP brightness; >1 for
        bright FPs (mScarlet3 ~2.1×), <1 for dim ones (Electra1 ~0.34×).

        Backends opt out by setting
        ``self._brightness_scaling_enabled = False``.
        """
        if not getattr(self, "_brightness_scaling_enabled", True):
            return 1.0
        fluor = self._active_fluorophore()
        if fluor is None:
            return 1.0
        return float(fluor.brightness) / self._BRIGHTNESS_BASELINE_EGFP

    def disable_brightness_scaling(self) -> None:
        """Opt out of per-fluorophore brightness scaling.

        Useful for backends where existing scenario thresholds were
        calibrated to the pre-change per-channel intensity. Default
        is enabled (identity at EGFP-baseline channels).
        """
        self._brightness_scaling_enabled = False

    def _active_fluorophore(self):
        """Resolve the active Filter Wheel label to a Fluorophore record.

        Returns ``None`` when the mode is brightfield, the label can't
        be parsed, or no Filter Wheel device is registered. Backends
        that opt into per-fluorophore brightness or bleach kinetics
        read this and apply ``brightness`` (eps×qy/1000 vs EGFP) or
        ``bleach_kx`` to scale the relevant pipeline knob — wakeup 2
        of the fluorophore-registry arc will wire one backend into
        this path; until then the resolver is just available for
        smoke tests and the optional callers it grows.
        """
        try:
            self._update_mode()
        except Exception:
            pass
        if getattr(self, "mode", 0) == 0:
            return None
        fw = (self.state_devices.get("Filter Wheel")
              if hasattr(self, "state_devices") else None)
        if not fw:
            return None
        label = fw.get("label") or fw.get("Label") or ""
        from vmteach.pipeline.fluorophores import (
            fluorophore_for_label)
        return fluorophore_for_label(label)

    def _active_wavelength_nm(self) -> float:
        """Resolve the active Filter Wheel label to an emission wavelength.

        Returns 0.0 (the OpticalPipeline's chromatic-off sentinel) when:
          - the active mode is 0 (brightfield is broadband, no single
            chromatic wavelength applies — even though the Channel
            preset for BF still ships a Filter Wheel label like
            Electra1/454, the underlying physics is broadband)
          - no Filter Wheel device is registered
          - the active label is BF / phase-contrast / DIC
          - the label can't be parsed

        Refreshes ``self.mode`` from the live (Filter Wheel, LED) labels
        before checking — the property is read pre-snap by the camera
        adapter and the cached mode would otherwise reflect the previous
        snap's channel rather than the configured one. ``_update_mode``
        is idempotent and side-effect-only on ``self.mode``, which is
        the same value the next ``snap_frame`` would compute anyway.

        Imports the registry lazily so ``base/`` doesn't pull in
        ``pipeline/`` at module-import time.
        """
        try:
            self._update_mode()
        except Exception:
            pass  # gracefully degrade if state_devices not yet populated
        if getattr(self, "mode", 0) == 0:
            return 0.0
        fw = (self.state_devices.get("Filter Wheel")
              if hasattr(self, "state_devices") else None)
        if not fw:
            return 0.0
        label = fw.get("label") or fw.get("Label") or ""
        from vmteach.pipeline.fluorophores import wavelength_for_label
        wl = wavelength_for_label(label)
        return float(wl) if wl is not None else 0.0

    # ──────────────────────────────────────────────────────────
    # Auto-step
    # ──────────────────────────────────────────────────────────

    def _auto_step_tick(self):
        """Call ``step()`` if auto-step is enabled and snap count matches."""
        if (self.auto_step and self._snap_count > 0
                and self._snap_count % self.snaps_per_step == 0):
            self.step()

    # ──────────────────────────────────────────────────────────
    # Photobleaching
    # ──────────────────────────────────────────────────────────

    def _all_pipelines(self):
        """Yield every OpticalPipeline this sim owns.

        Voronoi-family backends keep their fluorescence pipelines on
        attribute names (`_nuc_pipeline`, `_mem_pipeline`, `_bf_pipeline`)
        rather than in `self._pipeline`, so iterating only `_pipeline`
        misses them — `enable_photobleaching` / `reset_photobleaching`
        silently no-op'd on voronoi for that reason.
        """
        seen = set()
        for pipe in self._pipeline.values():
            if id(pipe) not in seen:
                seen.add(id(pipe))
                yield pipe
        for attr in ("_nuc_pipeline", "_mem_pipeline", "_bf_pipeline"):
            pipe = getattr(self, attr, None)
            if pipe is not None and id(pipe) not in seen:
                seen.add(id(pipe))
                yield pipe

    def enable_photobleaching(self, rate: float = 0.001):
        """Enable photobleaching on fluorescence channels."""
        # Voronoi-family: nuc + mem are the fluorescence pipelines (BF
        # is mode 0). Other backends register channel pipelines in
        # `_pipeline[1]` / `_pipeline[2]`. Both layouts get updated.
        for ch in [1, 2]:
            if ch in self._pipeline:
                self._pipeline[ch].photobleach_rate = rate
        for attr in ("_nuc_pipeline", "_mem_pipeline"):
            pipe = getattr(self, attr, None)
            if pipe is not None:
                pipe.photobleach_rate = rate

    def reset_photobleaching(self):
        """Reset accumulated photobleaching on all channels."""
        for pipe in self._all_pipelines():
            pipe.reset_bleach()

    # ──────────────────────────────────────────────────────────
    # Z-drift
    # ──────────────────────────────────────────────────────────

    def _accumulate_z_drift(self, dt: float) -> None:
        """Advance tissue Z-drift by *dt* seconds."""
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

    def get_z_drift(self) -> float:
        """Return cumulative Z-drift (µm)."""
        return self.tissue_z

    def reset_z_drift(self):
        """Reset Z-drift to zero."""
        self.tissue_z = 0.0

    # ──────────────────────────────────────────────────────────
    # Template method — snap_frame pipeline
    # ──────────────────────────────────────────────────────────

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0,
                   **kwargs) -> np.ndarray:
        """Capture a rendered frame (template method).

        Orchestrates the canonical 9-step pipeline.  Subclasses should
        override ``_render_for_mode`` (required), ``_handle_mask``, or
        ``_finalize_output`` rather than replacing this method.
        """
        self._update_mode()
        self._update_objectif()
        if mask is not None:
            self._handle_mask(mask)
        self._auto_step_tick()
        self._snap_count += 1

        self._current_exposure = exposure
        full = self._render_for_mode(self.mode)

        viewport = self._crop_fov(full)
        viewport = self._apply_defocus(viewport)
        viewport = self._apply_off_axis_lateral_shift(viewport)
        viewport = self._apply_sample_rotation(viewport)
        viewport = self._apply_lightsheet_envelope(viewport)
        viewport = self._apply_sim_pattern(viewport)
        viewport = self._apply_pipeline(viewport, exposure)
        viewport = self._apply_exposure(viewport, exposure, intensity)
        return self._finalize_output(viewport)

    # ──────────────────────────────────────────────────────────
    # Abstract / stubs
    # ──────────────────────────────────────────────────────────

    @abstractmethod
    def _render_for_mode(self, mode: int) -> np.ndarray:
        """Render full-resolution image for the given channel *mode*.

        Must return a BGR ``uint8`` image at internal resolution
        (``self._iw × self._ih``).  The base-class pipeline handles
        FOV cropping, defocus, noise, exposure, and grayscale conversion.
        """
        ...

    def _map_slm_to_world(self, mask: np.ndarray) -> np.ndarray:
        """Map viewport-space SLM mask to world-coordinate bool array."""
        obj = self.current_objectiv
        fov_world = self._FOV_MAP.get(obj, min(512, self.width))

        cx = int(self.camera_offset[0]) + self.viewport_width // 2
        cy = int(self.camera_offset[1]) + self.viewport_height // 2

        mask_fov = cv2.resize(
            mask.astype(np.uint8), (fov_world, fov_world),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        world_mask = np.zeros((self.height, self.width), dtype=bool)
        half = fov_world // 2
        x0 = max(0, min(cx - half, self.width - fov_world))
        y0 = max(0, min(cy - half, self.height - fov_world))

        wx1 = min(self.width, x0 + fov_world)
        wy1 = min(self.height, y0 + fov_world)
        mw = wx1 - x0
        mh = wy1 - y0
        world_mask[y0:y0 + mh, x0:x0 + mw] = mask_fov[:mh, :mw]
        return world_mask

    def _handle_mask(self, mask: np.ndarray) -> None:
        """Process an SLM / stimulation mask.  Override in subclasses."""

    def _finalize_output(self, viewport: np.ndarray) -> np.ndarray:
        """Convert viewport to final output format (BGR → grayscale)."""
        if viewport.ndim == 2:
            return viewport
        return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)

    def step(self, dt: float = 1.0):
        """Advance simulation by *dt*.  Override in dynamic sims."""

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics step (no imaging side effects)."""
        self.step(dt)

    def reset(self, seed: int | None = None):
        """Reset simulation state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def get_ground_truth(self) -> dict:
        """Return ground truth data.  Override in subclasses."""
        return {}
