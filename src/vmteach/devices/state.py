from pymmcore_plus.experimental.unicore import StateDevice
import vmteach.engine.simulation_bridge as bridge_module


class GenericStateDevice(StateDevice):
    """A state device with caller-defined labels and name.

    All state devices in the simulation use this base. The three
    microscope-specific subclasses (LED, Filter Wheel, Objective)
    just supply default label dicts.

    Example:
        dev = GenericStateDevice("Channel", {0: "570nm", 1: "690nm-ref"})
    """

    def __init__(self, name: str, labels: dict[int, str]) -> None:
        super().__init__(labels)
        self._current_state = 0
        self._current_label = self._state_to_label.get(self._current_state)
        self._name = name
        if bridge_module.GLOBAL_BRIDGE is not None:
            self.update_microscope_simulation()

    def initialize(self) -> None:
        """Push current state to bridge after SimServer has set GLOBAL_BRIDGE.

        initializeAllDevices() runs all devices in parallel threads, so we
        wait for the bridge_ready event (set by SimServer.initialize()) before
        pushing state. Timeout of 5s prevents hangs if SimServer fails.
        """
        bridge_module.bridge_ready.wait(timeout=5.0)
        self.update_microscope_simulation()

    def get_state(self) -> int:
        return self._current_state

    def set_state(self, position: int | str) -> None:
        if isinstance(position, str):
            position = int(position)
        self._current_state = position
        self._current_label = self._state_to_label.get(self._current_state)
        self.update_microscope_simulation()

    def update_microscope_simulation(self) -> None:
        bridge = bridge_module.GLOBAL_BRIDGE
        if bridge is not None:
            bridge.update_state({
                self._name: {
                    "state": str(self._current_state),
                    "label": self._current_label,
                }
            })


class FilterWheelDevice(GenericStateDevice):
    """Fluorescence microscope filter wheel (7 emission filters)."""

    def __init__(self) -> None:
        super().__init__("Filter Wheel", {
            0: "Electra1(402/454)",
            1: "SCFP2(434/474)",
            2: "TagGFP2(483/506)",
            3: "obeYFP(514/528)",
            4: "mRFP1-Q667(549/570)",
            5: "mScarlet3(569/582)",
            6: "miRFP670(642/670)",
        })


class LEDDevice(GenericStateDevice):
    """Fluorescence microscope LED excitation source (7 wavelengths)."""

    def __init__(self) -> None:
        super().__init__("LED", {
            0: "UV",
            1: "BLUE",
            2: "CYAN",
            3: "GREEN",
            4: "YELLOW",
            5: "ORANGE",
            6: "RED",
        })


class ObjectiveDevice(GenericStateDevice):
    """Microscope objective turret (4 magnifications)."""

    def __init__(self) -> None:
        super().__init__("Objective", {
            0: "10x",
            1: "20x",
            2: "40x",
            3: "100x",
        })


class TemperatureControllerDevice(GenericStateDevice):
    """Stage-top incubator / temperature controller.

    Provides discrete temperature presets covering key biological ranges:
    cold storage (4°C), room temp (20°C), standard incubation (25-37°C),
    and heat shock (42°C).  Backends that support temperature modulation
    read the current label from ``self.state_devices["Temperature"]``
    and adjust dynamics accordingly.
    """

    # Mapping: state index → label string (°C value embedded)
    TEMPS = {
        0: "20",   # room temperature (default)
        1: "4",    # cold — stops most activity
        2: "25",   # C. elegans / yeast optimal
        3: "30",   # many microbes
        4: "37",   # mammalian body temperature
        5: "42",   # heat shock
        6: "15",   # cool — C. elegans slow but active
        7: "22",   # standard room temp
        8: "10",   # cold — minimal activity
        9: "28",   # zebrafish optimal
    }

    def __init__(self) -> None:
        super().__init__("Temperature", self.TEMPS)


class StretchDevice(GenericStateDevice):
    """Uniaxial substrate stretch device (flexible PDMS membrane).

    Applies cyclic uniaxial strain to the culture substrate.
    Fibroblasts reorient perpendicular to the stretch axis over time.
    Stretch direction is horizontal (x-axis).

    State 0 = off, states 1-4 = increasing strain magnitudes.
    Backends read ``self.state_devices["Stretch"]`` to get strain %.
    """

    MODES = {
        0: "Off",     # no stretch
        1: "5%",      # gentle
        2: "10%",     # moderate
        3: "15%",     # strong
        4: "20%",     # high
    }

    def __init__(self) -> None:
        super().__init__("Stretch", self.MODES)


class PerfusionPumpDevice(GenericStateDevice):
    """Microfluidic perfusion pump controller.

    Controls flow speed and reagent delivery in microfluidic backends.
    States 0-4 are the original single-drug protocol. States 5-8 add
    multi-reagent labels for scenarios that need to alternate between
    treatments (e.g. Almada 2019 NanoJ-Fluidics, Conrad 2011 Micropilot,
    Lugagne 2024 MPC) — backends that don't know about a label fall
    back to the default flow behaviour, so the additions are
    backwards-compatible.

    Backends read ``self.state_devices["Perfusion"]`` (a dict with
    'state' and 'label' keys) and dispatch on the label string.
    """

    MODES = {
        0: "Off",       # no flow
        1: "Slow",      # 1 px/s
        2: "Medium",    # 3 px/s (default)
        3: "Fast",      # 10 px/s
        4: "Drug",      # 3 px/s + primary drug enabled (back-compat)
        5: "Drug_B",    # 3 px/s + secondary drug enabled
        6: "Wash",      # high flow + no drug (washout)
        7: "Probe",     # 1 px/s + probe reagent (e.g. STORM imaging buffer)
        8: "Fix",       # stop dynamics + apply fixative (live → fixed)
    }

    def __init__(self) -> None:
        super().__init__("Perfusion", self.MODES)


class AnesthesiaDevice(GenericStateDevice):
    """Tricaine (MS-222) anesthesia delivery for zebrafish.

    Controls anesthesia concentration in the fish water.
    State 0 = none, states 1-3 = increasing Tricaine concentration.
    Higher concentrations suppress cardiac activity and blood flow.
    Backends read ``self.state_devices["Anesthesia"]`` to get suppression level.
    """

    MODES = {
        0: "None",       # no anesthesia
        1: "0.01%",      # light — mild bradycardia (~50% HR reduction)
        2: "0.02%",      # standard — strong bradycardia (~80% HR reduction)
        3: "0.04%",      # deep — near cardiac arrest (~95% HR reduction)
    }

    def __init__(self) -> None:
        super().__init__("Anesthesia", self.MODES)


class ElectrodeDevice(GenericStateDevice):
    """DC electric field electrode for galvanotaxis experiments.

    Applies a directional electric field to the culture dish.
    Epithelial and fibroblast cells undergo galvanotaxis — directed migration
    toward the cathode (negative electrode).  This device controls which
    direction is the cathode, and therefore in which direction cells migrate.

    State 0 = Off (no field).
    States 1-4 encode the cathode direction:
      1 = "+X"  → cathode on right side  → cells migrate right
      2 = "-X"  → cathode on left side   → cells migrate left
      3 = "+Y"  → cathode on bottom      → cells migrate down
      4 = "-Y"  → cathode on top         → cells migrate up

    Backends that support galvanotaxis read
    ``self.state_devices["Electrode"]["label"]`` during ``step()`` to add
    a directed migration bias proportional to ``galvanotaxis_strength``.
    """

    MODES = {
        0: "Off",   # no electric field
        1: "+X",    # cathode right  → cells migrate right (+X)
        2: "-X",    # cathode left   → cells migrate left  (-X)
        3: "+Y",    # cathode bottom → cells migrate down  (+Y)
        4: "-Y",    # cathode top    → cells migrate up    (-Y)
    }

    def __init__(self) -> None:
        super().__init__("Electrode", self.MODES)


class SLMModeDevice(GenericStateDevice):
    """SLM excitation mode for reaction-diffusion backend.

    States: 0=excite (increase activator), 1=inhibit (suppress activator).
    No-arg constructor so it can be loaded via #py pyDevice in .cfg files.
    """

    MODES = {0: "excite", 1: "inhibit"}

    def __init__(self) -> None:
        super().__init__("SLM-Mode", self.MODES)


class ResetDevice(GenericStateDevice):
    """Agent-facing 'reset to fresh sample' trigger.

    Used by backends with one-way irreversible dynamics (wound closes,
    drug kills cells, photoconversion etc.) to give the agent a way to
    start over without re-serving the challenge — analogous to moving
    the stage to a fresh region of the well plate.

    States:
        0 = "Idle"     (default, no-op)
        1 = "Trigger"  (call sim.reset_sample() and snap back to 0)

    Backends opt in by:
      1. Adding the device to their .cfg:
         #py pyDevice,Reset,vmteach.devices.state,ResetDevice
      2. Implementing ``reset_sample(seed=None)`` on the sim. The method
         should re-initialise per-cell / per-field state to the same
         distribution the constructor would produce, but does NOT need
         to re-bind devices or recreate the core.

    The device automatically returns to state 0 after triggering, so
    ``core.setState("Reset", 1)`` is a one-shot edge.
    """

    MODES = {0: "Idle", 1: "Trigger"}

    def __init__(self) -> None:
        super().__init__("Reset", self.MODES)

    def set_state(self, position: int | str) -> None:
        if isinstance(position, str):
            try:
                position = int(position)
            except ValueError:
                # Map label → state (case-insensitive)
                position = next(
                    (s for s, lbl in self._state_to_label.items()
                     if lbl.lower() == position.lower()),
                    0,
                )
        prev = self._current_state
        # Update internal state first so pymmcore_plus property machinery
        # doesn't recurse (the parent set_state path tries to confirm the
        # new state by re-reading it, which would call us again if we
        # overwrite it inside this method).
        self._current_state = position
        self._current_label = self._state_to_label.get(position)
        # Fire only on a 0 → 1 rising edge so toggling is meaningful.
        if prev != 1 and position == 1:
            bridge = bridge_module.GLOBAL_BRIDGE
            sim = getattr(bridge, "_sim", None) if bridge is not None else None
            if sim is not None and hasattr(sim, "reset_sample"):
                try:
                    sim.reset_sample()
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "ResetDevice: sim.reset_sample() raised %s; sample "
                        "may be in inconsistent state", exc,
                    )
        self.update_microscope_simulation()


class HemocytometerChannelDevice(GenericStateDevice):
    """Channel selector for hemocytometer backend (brightfield / trypan-blue)."""

    MODES = {0: "brightfield", 1: "trypan-blue"}

    def __init__(self) -> None:
        super().__init__("Channel", self.MODES)


class ColonyCounterChannelDevice(GenericStateDevice):
    """Channel selector for colony counter backend (transmitted / blue-filter / GFP)."""

    MODES = {0: "transmitted", 1: "blue-filter", 2: "GFP-excitation"}

    def __init__(self) -> None:
        super().__init__("Channel", self.MODES)


class FlowCytometerDetectorDevice(GenericStateDevice):
    """Detector channel for flow cytometry backend (FSC-SSC / FL1-FITC / FL2-PE)."""

    MODES = {0: "FSC-SSC", 1: "FL1-FITC", 2: "FL2-PE"}

    def __init__(self) -> None:
        super().__init__("Detector", self.MODES)


class PlateReaderChannelDevice(GenericStateDevice):
    """Channel selector for plate reader backend (wavelengths set dynamically)."""

    MODES = {0: "primary", 1: "reference"}

    def __init__(self) -> None:
        super().__init__("Channel", self.MODES)


class SpectrophotometerChannelDevice(GenericStateDevice):
    """Channel selector for spectrophotometer backend (rack visualisation
    or spectral heatmap).
    """

    MODES = {0: "rack", 1: "spectrum"}

    def __init__(self) -> None:
        super().__init__("Channel", self.MODES)


class ModalityDevice(GenericStateDevice):
    """Acquisition-modality state device — generalises ch594's
    objective-swap proxy into a real first-class config.

    Backends opt in by registering a profile dict via
    ``SimBase._register_modality_profiles({label: {...}})`` with keys:

        psf_sigma_mult       (float, default 1.0): multiplier on each
                              OpticalPipeline's PSF sigma at apply time.
                              0.25 = sub-diffraction (STED-like).
        exposure_ceiling_ms  (float, default inf): clamps the snap's
                              exposure_ms argument before noise pass.
        bleach_rate_mult     (float, default 1.0): multiplier on each
                              pipeline's photobleach_rate. STED bursts
                              are typically 5-10x more bleachy than
                              widefield.
        channels_allowed     (list[str] | None): if set, modes whose
                              FilterWheel label is NOT in this list
                              render an empty frame (modality lock-out
                              of unavailable channels).

    Default labels (subset; a backend can subclass and add):
        0: widefield_scout    — fast, low-detail, full channels
        1: confocal           — slower, sharper, full channels
        2: sted_burst         — sub-diffraction, GCaMP only, high bleach
        3: lattice_lightsheet — fast Z-volume, full channels

    Without a registered profile (or with the default ``widefield_scout``
    profile, which is identity), behavior is byte-equivalent to a sim
    that doesn't have a Modality device — backwards-compatible.

    Maps to the Alvelid 2022 / Shi 2024 / Conrad 2011 / Mahecic 2022
    family of adaptive-acquisition papers where the agent must alternate
    between a fast scout regime and a slow detail regime under a fixed
    sample-bleach budget.
    """

    MODES = {
        0: "widefield_scout",
        1: "confocal",
        2: "sted_burst",
        3: "lattice_lightsheet",
    }

    def __init__(self) -> None:
        super().__init__("Modality", self.MODES)


class FluidicsDevice(GenericStateDevice):
    """Programmable multi-reservoir fluidics controller.

    Generalises ``PerfusionPumpDevice`` from a single drug channel to
    a labelled-reservoir model with dead-volume + settle-time kinetics
    and a sticky live→fixed transition. Maps to the Almada 2019
    NanoJ-Fluidics + Alvelid 2022 + Conrad 2011 + Lugagne 2024 +
    Mahecic 2022 family of papers where the agent must alternate
    reagent exchanges (live drug → wash → fix → multi-round STORM
    probe rounds) under a realistic dead-volume budget.

    Distinct device from PerfusionPumpDevice — both can coexist.
    Backends that wear ``FluidicsDevice`` should:

      1. Read ``state_devices["Fluidics"]`` for the active label.
      2. On label change, snapshot a `_fluidics_arrived_at = sim_time
         + dead_volume_s` so the new reagent's biology kicks in only
         AFTER dead-volume has cleared.
      3. Use exponential relaxation (tau ≈ settle_time_s / 3) for
         smooth transitions.
      4. On label == "fix", set a sticky `_fluidics_fixed = True`
         flag and freeze step()-driven biology (cell motion, division,
         drug response). Bleach + chromatic continue (consistent with
         real fixed samples).

    Reservoir vocabulary (extensible by subclass):
      - "buffer"   : neutral baseline (default)
      - "drug_A"   : primary drug effect (alias for legacy "Drug")
      - "drug_B"   : secondary drug effect
      - "wash"     : flush; biology relaxes to baseline
      - "probe"    : multi-round imaging buffer (STORM/DNA-PAINT)
      - "fix"      : fixative; biology freezes (sticky)
    """

    MODES = {
        0: "buffer",
        1: "drug_A",
        2: "drug_B",
        3: "wash",
        4: "probe",
        5: "fix",
    }

    DEFAULT_DEAD_VOLUME_S: float = 5.0
    DEFAULT_SETTLE_TIME_S: float = 30.0

    def __init__(self) -> None:
        super().__init__("Fluidics", self.MODES)


class DeformableMirrorDevice(GenericStateDevice):
    """Adaptive-optics deformable mirror — Zernike-coefficient state device.

    Each labelled state encodes a 5-element Zernike-coefficient vector
    ``(defocus, astig_x, astig_y, coma_x, coma_y)`` in radians-rms
    wavefront. The ``flat`` state pushes ``(0, 0, 0, 0, 0)``; the
    other states push a single non-zero coefficient at ±0.5 rad rms,
    giving the renderer enough wavefront to bend a PSF noticeably
    without making the image unrecognisable.

    Maps to the Hu 2023 (DAOSM), Zhang 2023 (sensorless AO closed-loop),
    Royer 2016 (lattice light-sheet AO) family of papers where the
    agent must compensate sample-induced aberrations by sweeping a
    DM and minimising a sharpness metric.

    Bridge protocol:
        update_state({
            "DeformableMirror": {
                "state": "<n>", "label": "<label>",
                "zernike": (defocus, astig_x, astig_y, coma_x, coma_y),
            }
        })

    Backwards-compatible: a backend that doesn't honour the
    ``zernike`` key (most don't, today) sees an unchanged image —
    only backends that explicitly read ``state_devices["DeformableMirror"]``
    in their pipeline pass apply the wavefront.

    Subclassing recipe: override ``MODES`` if you want a denser
    coefficient grid (e.g. ±0.25, ±0.5, ±0.75 each), but stay within
    the same 5-element Zernike vocabulary so the renderer hook
    stays uniform across backends.
    """

    # 5-element coefficient vocabulary, in radians-rms wavefront.
    # 0..10 covers identity + each of the 5 Zernikes at ±0.5 rad rms.
    MODES = {
        0: "flat",
        1: "defocus_+0.5",
        2: "defocus_-0.5",
        3: "astig_x_+0.5",
        4: "astig_x_-0.5",
        5: "astig_y_+0.5",
        6: "astig_y_-0.5",
        7: "coma_x_+0.5",
        8: "coma_x_-0.5",
        9: "coma_y_+0.5",
        10: "coma_y_-0.5",
    }

    # Order of axes in the 5-tuple; index → axis name.
    ZERNIKE_AXES = ("defocus", "astig_x", "astig_y", "coma_x", "coma_y")

    def __init__(self) -> None:
        super().__init__("DeformableMirror", self.MODES)

    @classmethod
    def parse_label(cls, label: str | None) -> tuple[float, float, float, float, float]:
        """Translate a label like ``"defocus_+0.5"`` to a 5-tuple."""
        if not label or label == "flat":
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        # Format: "<axis>_<sign><magnitude>" e.g. "coma_y_-0.5".
        # Find the last '_' that separates the magnitude from the axis.
        # Axis names contain '_' (astig_x, coma_y) so split from the right.
        sep = label.rfind("_")
        if sep < 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        axis = label[:sep]
        mag_str = label[sep + 1:]
        try:
            mag = float(mag_str)
        except ValueError:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            idx = cls.ZERNIKE_AXES.index(axis)
        except ValueError:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        coeffs[idx] = mag
        return tuple(coeffs)  # type: ignore[return-value]

    def update_microscope_simulation(self) -> None:
        bridge = bridge_module.GLOBAL_BRIDGE
        if bridge is not None:
            bridge.update_state({
                self._name: {
                    "state": str(self._current_state),
                    "label": self._current_label,
                    "zernike": self.parse_label(self._current_label),
                }
            })


class OffAxisLEDDevice(GenericStateDevice):
    """Off-axis LED illumination source for single-shot autofocus.

    Maps to the Pinkard 2019 deep-learning single-shot autofocus
    paper. Geometry: when the LED is biased off the optical axis (not
    in standard Köhler position), an out-of-focus sample plane appears
    laterally translated in the image — the translation magnitude
    scales linearly with signed defocus, and the translation direction
    matches the LED's offset direction. A single image carries enough
    information to regress the *signed* defocus (above vs below focus)
    in one shot, without the Z-sweep required by gradient-magnitude
    autofocus.

    States:
      0 = "Off"   — standard on-axis Köhler illumination (no shift).
      1 = "On_+x" — LED biased on the +x side; defocus shifts image +x.
      2 = "On_-x" — LED biased on the -x side; defocus shifts image -x.
      3 = "On_+y" — defocus shifts image +y.
      4 = "On_-y" — defocus shifts image -y.

    The renderer hook in ``SimBase`` reads the unit-direction tuple
    pushed in the ``direction`` key alongside ``state``/``label``,
    multiplies by signed defocus and a per-magnification coefficient,
    and translates the viewport via ``cv2.warpAffine`` before the
    optical-pipeline pass. Backwards-compatible: state 0 (default)
    pushes ``(0.0, 0.0)`` so the renderer hook is a no-op.
    """

    MODES = {
        0: "Off",
        1: "On_+x",
        2: "On_-x",
        3: "On_+y",
        4: "On_-y",
    }

    # Label → (dx_unit, dy_unit) unit vector. ``Off`` returns the
    # zero vector so the renderer hook short-circuits.
    DIRECTIONS = {
        "Off":   (0.0, 0.0),
        "On_+x": (1.0, 0.0),
        "On_-x": (-1.0, 0.0),
        "On_+y": (0.0, 1.0),
        "On_-y": (0.0, -1.0),
    }

    def __init__(self) -> None:
        super().__init__("OffAxisLED", self.MODES)

    @classmethod
    def parse_label(cls, label: str | None) -> tuple[float, float]:
        """Translate a label to a 2-element unit-direction tuple."""
        if not label:
            return (0.0, 0.0)
        return cls.DIRECTIONS.get(label, (0.0, 0.0))

    def update_microscope_simulation(self) -> None:
        bridge = bridge_module.GLOBAL_BRIDGE
        if bridge is not None:
            bridge.update_state({
                self._name: {
                    "state": str(self._current_state),
                    "label": self._current_label,
                    "direction": self.parse_label(self._current_label),
                }
            })


class RotationStageDevice(GenericStateDevice):
    """Continuous-rotation specimen stage (8 discrete labelled angles).

    Maps to He 2020 (smart rotation light-sheet), Royer 2016
    (AutoPilot 5-axis alignment), McDole 2018 (adaptive imaging
    mouse embryo) — papers where a sample-rotation θ axis becomes
    a real action knob alongside XYZ. The 8 quantised angles
    (0/45/90/.../315°) match how every other state device exposes
    a finite, agent-discoverable action space; a continuous-angle
    subclass can layer on later if needed.

    Foundation only: this wakeup ships the device + a 2D affine
    placeholder rotation in the renderer (``cv2.warpAffine`` about
    the FOV centre). That's geometrically right for thin-layer
    specimens (voronoi tissue, yeast colonies) but obviously wrong
    for 3D Z-stack backends (volvox, zebrafish, organoid). Real
    sample-orientation rotation that re-projects per-depth content
    lands when those backends explicitly opt in. Until then the
    device is wired only on voronoi.

    Bridge protocol:
        update_state({
            "RotationStage": {
                "state": "<n>", "label": "<degrees>",
                "angle_rad": <float, radians>,
            }
        })

    Backwards-compatible: state 0 (default) pushes ``angle_rad=0.0``
    so the renderer hook short-circuits and all 36 backends without
    a RotationStage in their cfg see no behavior change.
    """

    MODES = {
        0: "0",
        1: "45",
        2: "90",
        3: "135",
        4: "180",
        5: "225",
        6: "270",
        7: "315",
    }

    def __init__(self) -> None:
        super().__init__("RotationStage", self.MODES)

    @classmethod
    def parse_label(cls, label: str | None) -> float:
        """Translate a degree-string label to radians."""
        if not label:
            return 0.0
        try:
            import math
            return math.radians(float(label))
        except (TypeError, ValueError):
            return 0.0

    def update_microscope_simulation(self) -> None:
        bridge = bridge_module.GLOBAL_BRIDGE
        if bridge is not None:
            bridge.update_state({
                self._name: {
                    "state": str(self._current_state),
                    "label": self._current_label,
                    "angle_rad": self.parse_label(self._current_label),
                }
            })


class SIMPatternDevice(GenericStateDevice):
    """Structured-illumination pattern selector (Jin 2020 DL-SIM).

    9 active states encoding the canonical 3-orientation × 3-phase
    raw-frame protocol that DL-SIM and SR-SIM reconstruction
    pipelines consume. Plus state 0 ``"off"`` for identity (no
    pattern, normal widefield illumination).

    Each labelled state pushes ``(orientation_rad, phase_rad,
    k_lp_per_px)`` to the bridge. The renderer hook
    (``SimBase._apply_sim_pattern``) multiplies the per-pixel
    fluorescence emission by ``0.5 * (1 + cos(2π·k·(x cos θ +
    y sin θ) + φ))`` — a sinusoidal grating with period
    ``1/k`` pixels along direction θ.

    Default ``k_lp_per_px = 1/6`` gives a 6-pixel fringe period —
    near the Abbe limit at 40×, distinguishable in the rendered
    image without aliasing.

    Maps to Jin 2020 (deep-learning SIM). The agent drives the 9
    active states sequentially to acquire the raw frame set, then
    applies their reconstruction (out of scope for the simulator —
    this is renderer-side only). Identity at state 0 ``"off"`` so
    backends without an SIM scenario see no behaviour change.
    """

    import math as _math

    ORIENTATIONS = (0.0, _math.pi / 3.0, 2.0 * _math.pi / 3.0)
    PHASES = (0.0, 2.0 * _math.pi / 3.0, 4.0 * _math.pi / 3.0)
    DEFAULT_K_LP_PER_PX = 1.0 / 6.0

    MODES = {
        0: "off",
        1: "o0_p0", 2: "o0_p1", 3: "o0_p2",
        4: "o1_p0", 5: "o1_p1", 6: "o1_p2",
        7: "o2_p0", 8: "o2_p1", 9: "o2_p2",
    }

    def __init__(self) -> None:
        super().__init__("SIMPattern", self.MODES)

    @classmethod
    def parse_label(cls, label):
        """Translate a label to (orientation_rad, phase_rad, k_lp_per_px).

        Returns ``(0.0, 0.0, 0.0)`` for the ``"off"`` state — the
        renderer hook treats k=0 as the identity short-circuit.
        """
        if not label or label == "off":
            return (0.0, 0.0, 0.0)
        # Label format: "o<i>_p<j>" with i, j ∈ {0, 1, 2}.
        try:
            o_part, p_part = label.split("_")
            i = int(o_part[1:])
            j = int(p_part[1:])
            return (cls.ORIENTATIONS[i], cls.PHASES[j],
                    cls.DEFAULT_K_LP_PER_PX)
        except (ValueError, IndexError):
            return (0.0, 0.0, 0.0)

    def update_microscope_simulation(self) -> None:
        bridge = bridge_module.GLOBAL_BRIDGE
        if bridge is not None:
            orient, phase, k = self.parse_label(self._current_label)
            bridge.update_state({
                self._name: {
                    "state": str(self._current_state),
                    "label": self._current_label,
                    "orientation_rad": orient,
                    "phase_rad": phase,
                    "k_lp_per_px": k,
                }
            })


class STEDDepletionDevice(GenericStateDevice):
    """STED depletion-beam state device — saturable PSF narrowing.

    Each labelled state encodes a saturation factor ``ζ ∈ [0, 0.95]``
    in the standard STED resolution model:
        FWHM_eff = FWHM_0 / sqrt(1 + ζ · SAT_GAIN)
    With SAT_GAIN ≈ 80 (matched to typical STED ~10× resolution gains
    at saturation), the labelled states span widefield → ~3× narrower.
    The ``off`` state pushes ζ=0.0 (identity, byte-equivalent to no
    device); ``saturated`` pushes ζ=0.95 → ~9× FWHM reduction.

    Maps to Hell 1994 (STED), Westphal 2008 (RESOLFT), Vicidomini 2018
    (STED multi-objective bleach scoring) where the agent must trade
    resolution gain against photobleaching cost.

    Bridge protocol:
        update_state({
            "STEDDepletion": {
                "state": "<n>", "label": "<label>",
                "saturation_factor": float,
            }
        })

    Backwards-compatible: a backend that doesn't honour the
    ``saturation_factor`` key sees an unchanged image — only the
    SimBase resolver hook applied in the snap path actually narrows
    the PSF.

    Composes cleanly with DeformableMirrorDevice: the SimBase snap
    path multiplies both sigma_x and sigma_y by the STED narrowing
    factor AFTER applying the DM Zernike-based PSF params, so a
    saturated STED + flat DM produces a narrow round PSF, while
    saturated STED + astigmatic DM produces a narrow elongated PSF.
    """

    SAT_GAIN = 80.0  # Calibrated so saturated (ζ=0.95) → ~9× FWHM gain

    MODES = {
        0: "off",
        1: "mild",
        2: "moderate",
        3: "strong",
        4: "saturated",
    }

    LABEL_TO_SATURATION = {
        "off": 0.0,
        "mild": 0.25,
        "moderate": 0.5,
        "strong": 0.75,
        "saturated": 0.95,
    }

    def __init__(self) -> None:
        super().__init__("STEDDepletion", self.MODES)

    @classmethod
    def parse_label(cls, label: str | None) -> float:
        """Translate a label like ``"saturated"`` to a saturation float."""
        if not label:
            return 0.0
        return cls.LABEL_TO_SATURATION.get(label, 0.0)

    def update_microscope_simulation(self) -> None:
        bridge = bridge_module.GLOBAL_BRIDGE
        if bridge is not None:
            bridge.update_state({
                self._name: {
                    "state": str(self._current_state),
                    "label": self._current_label,
                    "saturation_factor": self.parse_label(self._current_label),
                }
            })


class _LightSheetAxisDevice(GenericStateDevice):
    """Base class for the 3 light-sheet alignment axes (Y / TiltX / TiltY).

    Maps to Royer 2016 (AutoPilot 5-axis alignment) and McDole 2018
    (adaptive imaging mouse embryo) — papers where the light-sheet
    illumination plane needs continuous alignment with the detection
    focal plane to avoid stripe artefacts and brightness-gradient
    illumination.

    Each device exposes 5 quantised offsets (-2/-1/0/+1/+2) in
    half-unit steps (-1.0 / -0.5 / 0.0 / +0.5 / +1.0) of a normalised
    misalignment magnitude.  State 2 ("0.0") is the well-aligned
    default and pushes ``offset=0.0``, so backends without the device
    wired (or with the device at default) see byte-equivalent renders.

    The renderer hook (``SimBase._apply_lightsheet_envelope``, added
    in a later wakeup) multiplies the viewport by a 2D envelope:
      - LightSheetY     -> Gaussian peak shifted laterally by offset_y
      - LightSheetTiltX -> linear brightness gradient along Y
                            (sheet pitched so it hits y_top brighter
                            than y_bot or vice versa)
      - LightSheetTiltY -> linear brightness gradient along X

    Identity at all-zero state for all 3 axes.

    Bridge protocol:
        update_state({
            "<DeviceName>": {
                "state": "<n>", "label": "<offset string>",
                "offset": <float in [-1.0, +1.0]>,
            }
        })
    """

    LABEL_TO_OFFSET = {
        "-1.0": -1.0,
        "-0.5": -0.5,
        "0.0": 0.0,
        "+0.5": +0.5,
        "+1.0": +1.0,
    }
    MODES = {
        0: "-1.0",
        1: "-0.5",
        2: "0.0",
        3: "+0.5",
        4: "+1.0",
    }
    DEFAULT_STATE = 2

    @classmethod
    def parse_label(cls, label: str | None) -> float:
        if not label:
            return 0.0
        return cls.LABEL_TO_OFFSET.get(label, 0.0)

    def update_microscope_simulation(self) -> None:
        bridge = bridge_module.GLOBAL_BRIDGE
        if bridge is not None:
            bridge.update_state({
                self._name: {
                    "state": str(self._current_state),
                    "label": self._current_label,
                    "offset": self.parse_label(self._current_label),
                }
            })


class LightSheetYDevice(_LightSheetAxisDevice):
    """Lateral Y offset of the light-sheet illumination peak.

    State 2 ("0.0") is centred (well-aligned). Non-zero states shift
    the Gaussian illumination envelope's peak away from the FOV centre
    along the Y axis — the off-peak side of the FOV dims.

    Matches Royer 2016 AutoPilot's first alignment axis.
    """

    def __init__(self) -> None:
        super().__init__("LightSheetY", self.MODES)
        # Default to well-aligned (state 2 = "0.0")
        self.set_state(self.DEFAULT_STATE)


class LightSheetTiltXDevice(_LightSheetAxisDevice):
    """Tilt of the light sheet about the X axis.

    State 2 is well-aligned (sheet parallel to detection plane).
    Non-zero states pitch the sheet so it intersects the detection
    plane along a line, producing a linear brightness gradient along
    the Y axis of the camera.

    Maps to Royer 2016 / McDole 2018 tilt-X alignment axis.
    """

    def __init__(self) -> None:
        super().__init__("LightSheetTiltX", self.MODES)
        self.set_state(self.DEFAULT_STATE)


class LightSheetTiltYDevice(_LightSheetAxisDevice):
    """Tilt of the light sheet about the Y axis.

    Mirror image of LightSheetTiltX — non-zero states produce a linear
    brightness gradient along the X axis of the camera.
    """

    def __init__(self) -> None:
        super().__init__("LightSheetTiltY", self.MODES)
        self.set_state(self.DEFAULT_STATE)
