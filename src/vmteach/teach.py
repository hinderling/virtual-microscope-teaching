"""Teaching helpers: the entire course-facing API surface.

Design notes
------------
* ``load_microscope`` defaults to **stepped mode**: no background thread,
  simulated time advances only through :func:`advance`. Same seed + same
  loop = identical trajectories on every machine. That is what makes the
  course exercises reproducible.
* ``mode="realtime"`` starts the wall-clock RealtimeEngine (idle pause
  disabled so the sample never silently freezes while a learner reads
  instructions). Used for the GUI activity and the latency lesson.
* Stimulation is applied when a mask is set (``core.setSLMImage``) and on
  each snap while that mask is displayed. The impulse sets (not adds) the
  cell velocity toward the light, so within one loop iteration this is
  effectively one stimulus, so the feedback loop frequency is the
  stimulation frequency.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _make_opto_sim(**kwargs):
    from vmteach.sim import OptoCellSim
    return OptoCellSim(**kwargs)


#: Registry of simulated samples ("backends"). A backend is a factory that
#: returns a sim object implementing the bridge contract: ``snap_frame()``,
#: ``step(dt)``, ``reset(seed)``, ``set_focal_plane(z)``, the device-facing
#: attributes used by :mod:`vmteach.bridge` (``state_devices``, ``stage``,
#: ``sensor_size``, ...) and, for realtime mode, ``continuous = True``.
#: Register your own with :func:`register_backend`.
BACKENDS: dict = {"optogenetic": _make_opto_sim}


def register_backend(name: str, factory) -> None:
    """Add a simulated sample so ``load_microscope(name)`` can create it.

    ``factory(**kwargs)`` must return a sim object satisfying the bridge
    contract (see :data:`BACKENDS`). The ``optogenetic`` backend's
    :class:`vmteach.sim.OptoCellSim` is the reference implementation.
    """
    BACKENDS[str(name)] = factory


def load_microscope(backend: str = "optogenetic", *, n_cells: int = 20,
                    seed: int = 0, mode: str = "stepped", warmup: bool = True,
                    **kwargs):
    """Create a virtual microscope and return ``(core, sim)``.

    Args:
        backend: Which simulated sample to load, by registry name (see
            :data:`BACKENDS` and :func:`register_backend`). This package
            ships ``"optogenetic"``: light-responsive cells with a nuclear
            marker, a membrane-bound optogenetic receptor, and an ERK-KTR
            activity reporter.
        n_cells: Cell density, as cells per 10x field of view (512 x 512
            um). Each 2048 um well holds 16 fields, so the default
            population is 16x this number per well.
        seed: Random seed. Same seed, same experiment, on every machine.
        mode: ``"stepped"`` (default): simulated time advances only via
            :func:`advance`; fully deterministic.
            ``"realtime"``: the sample evolves in wall-clock time while
            your code runs, like on a real microscope.
        warmup: Pre-compile the physics (numba, cached on disk after the
            first ever run) so the first snap in the exercise is instant.
        **kwargs: Forwarded to the backend factory (for ``optogenetic``:
            ``base_radius``, ``well_size``, ``n_wells``, ...).

    Returns:
        core: ``UniMMCore``, the microscope control object. The identical
            pymmcore API controls real Micro-Manager hardware.
        sim: The simulation handle (for :func:`advance` and ``.reset()``).
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"backend {backend!r} not available; registered backends: "
            f"{sorted(BACKENDS)}. Add your own with "
            "vmteach.register_backend(name, factory).")
    if mode not in ("stepped", "realtime"):
        raise ValueError(f"mode must be 'stepped' or 'realtime', got {mode!r}")

    from vmteach.bridge import SimulationBridge, set_global_bridge

    sim = BACKENDS[backend](n_cells=n_cells, seed=seed, **kwargs)
    bridge = SimulationBridge(sim)
    set_global_bridge(bridge)

    global _VirtualMicroscopeCore
    if _VirtualMicroscopeCore is None:
        _VirtualMicroscopeCore = _make_core_class()
    # Always use the psygnal events backend, even when a Qt app is already
    # running (pymmcore-plus would then pick Qt signals, which lack the
    # psygnal API our setConfig workaround needs, and would make a core
    # created after launch_gui behave differently from the usual
    # core-first path).
    import os
    _prev = os.environ.get("PYMM_SIGNALS_BACKEND")
    os.environ["PYMM_SIGNALS_BACKEND"] = "psygnal"
    try:
        core = _VirtualMicroscopeCore()
    finally:
        if _prev is None:
            os.environ.pop("PYMM_SIGNALS_BACKEND", None)
        else:
            os.environ["PYMM_SIGNALS_BACKEND"] = _prev
    # devices, channels, per-objective pixel sizes and the startup state
    # (10x, phase-contrast, binning 1) all come from the config file, as
    # they would for real hardware
    core.loadSystemConfiguration(str(Path(__file__).parent / "optogenetic.cfg"))

    if warmup:
        print("Preparing virtual microscope ...", end=" ", flush=True)
        sim.step(0.0)        # numba JIT (disk-cached after first ever run)
        sim.snap_frame()
        print("done.")

    if mode == "realtime":
        from vmteach.engine import RealtimeEngine
        engine = RealtimeEngine(sim, time_scale=1.0, tick_hz=20,
                                idle_timeout=0.0, bridge=bridge)
        engine.patch_snap_frame()
        engine.start()
        bridge._engine = engine

    return core, sim


def _make_core_class():
    from pymmcore_plus.experimental.unicore import UniMMCore

    class VirtualMicroscopeCore(UniMMCore):
        """UniMMCore that resolves pixel-size presets against Python devices.

        The C++ core matches pixel-size configs (``ConfigPixelSize`` in the
        .cfg) only against its own devices, so with a pure-Python objective
        ``getCurrentPixelSizeConfig()`` is always empty. Resolve it here so
        ``core.getPixelSizeUm()`` and ``core.getPixelSizeAffine()`` report
        the objective's pixel size (times the camera binning), as they do on
        a real system.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # setConfig reentrancy guard (per thread) and cross-thread
            # serialization of the paused-emission window; see setConfig.
            import threading
            self._in_set_config = threading.local()
            self._set_config_lock = threading.Lock()
            # UniMMCore's Python state devices emit propertyChanged BEFORE
            # the core writes its property cache, so listeners that react by
            # reading getPropertyFromCache (e.g. the GUI channel widget,
            # which then shows "<no match>") see the previous value. Sync
            # the cache in a handler connected here, at construction:
            # psygnal runs callbacks in connection order, so this runs
            # before any widget's. Drop once fixed upstream.
            self.events.propertyChanged.connect(self._sync_pydevice_cache)

        def _sync_pydevice_cache(self, device: str, prop: str, value) -> None:
            if device in self._pydevices:
                self._state_cache[(device, prop)] = value

        def setConfig(self, groupName: str, configName: str) -> None:
            """Apply a preset atomically for propertyChanged listeners.

            UniMMCore applies a preset's Python device properties one by
            one, each emitting propertyChanged synchronously on the calling
            thread, so listeners observe half-applied presets. pymmcore-
            widgets' PresetsWidget re-matches presets on every such event;
            in the mid-config window where no preset matches it parks its
            combo on '<no match>', and on the next event removes that item
            OUTSIDE signals_blocked, which moves the Qt current item and
            re-enters setConfig with the OLD preset, silently reverting
            the switch (channel changes from the GUI 'do nothing').

            Queue the emissions and flush them after the preset is fully
            applied: every listener then evaluates against the final,
            consistent state. The property cache is written by setProperty
            during application regardless.

            A listener may itself call setConfig (the presets widget does
            on some paths). Pausing the signal again during its own flush
            recurses endlessly, so nested calls apply directly; their
            synchronous emissions converge because state devices skip
            no-op moves.
            """
            if getattr(self._in_set_config, "active", False):
                return super().setConfig(groupName, configName)
            if not hasattr(self.events.propertyChanged, "paused"):
                # non-psygnal events backend (load_microscope prevents
                # this, but a manually built core may use Qt signals)
                return super().setConfig(groupName, configName)
            with self._set_config_lock:
                self._in_set_config.active = True
                try:
                    with self.events.propertyChanged.paused(reducer=None):
                        super().setConfig(groupName, configName)
                finally:
                    self._in_set_config.active = False

        def getCurrentPixelSizeConfig(self, cached: bool = False) -> str:
            # Python devices are always read live: the C++ property cache is
            # only refreshed by getProperty, not by setProperty/setStateLabel
            # on a Python device, so a cached read after an objective change
            # resolves the previous preset (and MDA metadata reads cached).
            def get(dev, prop):
                if cached and dev not in self._pydevices:
                    return self.getPropertyFromCache(dev, prop)
                return self.getProperty(dev, prop)

            for res in self.getAvailablePixelSizeConfigs():
                data = self.getPixelSizeConfigData(res)
                try:
                    if all(str(get(dev, prop)) == str(val)
                           for dev, prop, val in data):
                        return res
                except Exception:
                    continue
            return ""

        def getPixelSizeUm(self, cached: bool = False) -> float:
            res = self.getCurrentPixelSizeConfig(cached)
            if not res:
                return 0.0
            binning = 1.0
            try:
                binning = float(self.getProperty(self.getCameraDevice(), "Binning"))
            except Exception:
                pass
            return self.getPixelSizeUmByID(res) * binning / self.getMagnificationFactor()

        def getPixelSizeAffine(self, cached: bool = False) -> tuple:
            """Pixel-size affine of the current preset, scaled by binning.

            ``UniMMCore.getPixelSizeAffine`` asks the C++ core for the affine
            at binning 1, but the C++ core never matched the preset (it only
            sees Python devices through us) and aborts the interpreter on
            Windows instead of returning it. Every MDA hits this through the
            summary metadata, so resolve it here like ``getPixelSizeUm``.
            """
            res = self.getCurrentPixelSizeConfig(cached)
            if not res:
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            binning = 1.0
            try:
                binning = float(self.getProperty(self.getCameraDevice(), "Binning"))
            except Exception:
                pass
            factor = binning / self.getMagnificationFactor()
            return tuple(v * factor for v in self.getPixelSizeAffineByID(res))

        def setShutterOpen(self, *args) -> None:
            """Resolve the current shutter through the Python device registry.

            ``CMMCorePlus.setShutterOpen(state)`` looks up the current
            shutter with ``super().getShutterDevice()``, i.e. on the C++
            core, which does not know Python devices and returns "". The
            MDA engine then emits a propertyChanged event for device ""
            after every frame, crashing GUI widgets. Present on pymmcore-plus
            main too; drop this once fixed upstream.
            """
            if len(args) == 1:
                args = (self.getShutterDevice(), args[0])
            super().setShutterOpen(*args)

    return VirtualMicroscopeCore


_VirtualMicroscopeCore = None


def advance(sim, seconds: float = 1.0, dt: float = 0.05) -> None:
    """Advance simulated time deterministically (stepped mode).

    Replaces ``time.sleep()`` from real experiments: on hardware you *wait*
    for the sample to respond, on the deterministic virtual microscope you
    *advance* it.
    """
    n = max(1, round(seconds / dt))
    for _ in range(n):
        sim.step(dt)


def detect_nuclei(img: np.ndarray, min_area: int = 20,
                  exclude_border: bool = True) -> list:
    """Reference detector: nuclei centroids from a nuclear-marker image.

    Snap the miRFP channel (H2B-miRFP labels the nuclei) and pass the
    frame here.

    Nuclei are bright, compact, and, unlike cell bodies, never touch
    (cells collide before their nuclei can), so a plain Otsu threshold
    stays reliable even in crowded fields. Use this as the robust
    detection for feedback loops and tracking; write your own detector
    in the activities to understand what it does.

    Nuclei cut off by the image border are dropped by default (standard
    practice: a clipped object has a biased centroid, and intensity
    measurements around it sample the background).

    Returns a list of ``(x, y)`` integer centroids.
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        if exclude_border:
            x0, y0, bw, bh = cv2.boundingRect(c)
            if x0 <= 0 or y0 <= 0 or x0 + bw >= w or y0 + bh >= h:
                continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    return out


def measure_activity(ktr_img: np.ndarray, centroids, radius: int = 5) -> list:
    """Per-cell pathway activity from an ERK-KTR (mScarlet channel) image.

    The kinase translocation reporter sits in the nucleus while the
    pathway is inactive and moves to the cytoplasm when it is active, so
    the *nuclear* KTR intensity encodes the state: bright nucleus =
    inactive, dark nucleus = active. This function samples the mean
    intensity in a small disc at each nucleus centroid (from
    :func:`detect_nuclei` on the miRFP channel) and rescales it to an
    activity estimate.

    Args:
        ktr_img: a frame from the mScarlet (ERK-KTR) channel.
        centroids: iterable of ``(x, y)`` nucleus positions, e.g. from
            :func:`detect_nuclei`.
        radius: sampling disc radius in pixels (keep it smaller than a
            nucleus so the disc never overlaps the cytoplasm).

    Returns:
        A list of floats in ``[0, 1]``, one per centroid: 0 = inactive
        (reporter fully nuclear), 1 = active (reporter fully exported).
        Threshold at 0.5 for a binary active/inactive call.
    """
    h, w = ktr_img.shape[:2]
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disc = (xx ** 2 + yy ** 2) <= radius ** 2
    # rendered nuclear levels: ~170 inactive, ~40 active (before optics);
    # rescale between conservative bounds so noise and exposure wiggle
    # do not push values outside [0, 1]
    hi, lo = 150.0, 60.0
    out = []
    for x, y in centroids:
        x, y = int(round(x)), int(round(y))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        d = disc[(y0 - y + radius):(y1 - y + radius),
                 (x0 - x + radius):(x1 - x + radius)]
        patch = ktr_img[y0:y1, x0:x1]
        nuc = float(patch[d].mean()) if d.any() else 0.0
        out.append(float(np.clip((hi - nuc) / (hi - lo), 0.0, 1.0)))
    return out


def link_tracks(detections, max_dist: float = 40.0,
                memory: int = 2) -> np.ndarray:
    """Link per-frame detections into tracks (Hungarian assignment).

    Globally optimal frame-to-frame matching (scipy
    ``linear_sum_assignment``) with distance gating: a detection farther
    than ``max_dist`` from every track starts a new track instead of
    producing a jumpy link. Tracks survive up to ``memory`` missed frames
    (detector dropouts, cells briefly merging).

    Args:
        detections: sequence over frames; each frame a sequence of (x, y).
        max_dist: gate, the maximum linking distance in pixels per frame.
        memory: frames a lost track is kept alive for re-linking.

    Returns:
        Array of rows ``(track_id, t, y, x)``, the napari Tracks format.
    """
    from scipy.optimize import linear_sum_assignment

    BIG = 1e9
    active: dict = {}          # tid -> (x, y, last_t)
    rows = []
    next_id = 0
    for t, pts in enumerate(detections):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        tids = list(active.keys())
        matched = set()
        if tids and len(pts):
            prev = np.array([active[i][:2] for i in tids])
            cost = np.linalg.norm(prev[:, None, :] - pts[None, :, :], axis=2)
            cost[cost > max_dist] = BIG
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] >= BIG:
                    continue
                tid = tids[r]
                active[tid] = (pts[c, 0], pts[c, 1], t)
                rows.append((tid, t, pts[c, 1], pts[c, 0]))
                matched.add(c)
        for c, (x, y) in enumerate(pts):
            if c in matched:
                continue
            active[next_id] = (x, y, t)
            rows.append((next_id, t, y, x))
            next_id += 1
        active = {i: v for i, v in active.items() if t - v[2] <= memory}
    return np.asarray(rows, dtype=float)


def overlay(img: np.ndarray, mask: np.ndarray,
            color: tuple = (80, 140, 255), alpha: float = 0.4) -> np.ndarray:
    """Overlay a stimulation mask on a grayscale image as a colored wash.

    Returns an RGB uint8 image (inverted grayscale, matplotlib ``gray_r``
    convention) with ``mask > 0`` pixels tinted in ``color``.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        norm = (img.astype(np.float32) - lo) / (hi - lo) * 255
    else:
        norm = img.astype(np.float32)
    inv = 255 - norm
    rgb = np.stack([inv, inv, inv], axis=-1)
    m = mask > 0
    for c in range(3):
        rgb[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * color[c]
    return rgb.astype(np.uint8)


def letter_mask(char: str, shape: tuple = (512, 512),
                fill: float = 0.6, thickness: int = 40) -> np.ndarray:
    """Binary target image of a letter, centered, for the assembly exercise.

    Args:
        char: A single character (e.g. ``"L"``).
        shape: Output image shape ``(height, width)``.
        fill: Approximate fraction of the smaller image dimension the
            letter should span.
        thickness: Stroke thickness in pixels. Keep it wider than a cell
            so cells fit on the stroke.

    Returns:
        uint8 array, 255 inside the letter, 0 elsewhere.
    """
    if len(char) != 1:
        raise ValueError("letter_mask takes a single character")

    target_px = int(min(shape) * fill)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    (w, h), _ = cv2.getTextSize(char, font, scale, thickness)
    scale = scale * target_px / max(h, 1)
    (w, h), _ = cv2.getTextSize(char, font, scale, thickness)

    mask = np.zeros(shape, dtype=np.uint8)
    org = ((shape[1] - w) // 2, (shape[0] + h) // 2)
    cv2.putText(mask, char, org, font, scale, 255, thickness, cv2.LINE_8)
    return mask
