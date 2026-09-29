"""Simulator control: load the virtual microscope, register sample
backends, advance simulated time. Each sample backend lives in its own
subpackage together with the analysis helpers for its readouts (the
shipped one: :mod:`vmteach.optogenetic`).

Design notes
------------
* ``load_microscope`` defaults to **realtime mode**: the wall-clock
  RealtimeEngine runs the sample in the background (idle pause disabled
  so the sample never silently freezes while a learner reads
  instructions), and scripts wait with ``time.sleep`` exactly as on real
  hardware. ``speed`` runs it faster than the wall clock.
* ``mode="stepped"``: no background thread, simulated time advances only
  through :func:`advance`. Same seed + same loop = identical trajectories
  on every machine, which is what tests and figure scripts need.
* Stimulation is applied when a mask is set (``core.setSLMImage``) and on
  each snap while that mask is displayed. The impulse sets (not adds) the
  cell velocity toward the light, so within one loop iteration this is
  effectively one stimulus, so the feedback loop frequency is the
  stimulation frequency.
"""

from __future__ import annotations

from pathlib import Path



def _make_opto_sim(**kwargs):
    from vmteach.optogenetic.sim import OptoCellSim
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
    :class:`vmteach.optogenetic.sim.OptoCellSim` is the reference implementation.
    """
    BACKENDS[str(name)] = factory


def load_microscope(backend: str = "optogenetic", *, n_cells: int = 20,
                    seed: int = 0, mode: str = "realtime", speed: float = 1.0,
                    warmup: bool = True, **kwargs):
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
        mode: ``"realtime"`` (default): the sample evolves in wall-clock
            time while your code runs, like on a real microscope; wait
            with ``time.sleep``.
            ``"stepped"``: simulated time advances only via
            :func:`advance`; fully deterministic (tests, figures).
        speed: Realtime mode only: how many times faster than the wall
            clock the sample evolves. ``speed=10`` makes one second of
            waiting cover ten seconds of cell behaviour, so slow biology
            can be tested quickly; shorten your waits by the same factor
            to keep the experiment's timing (a luxury real samples do not
            offer).
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
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed!r}")
    if speed != 1.0 and mode != "realtime":
        raise ValueError("speed only applies to mode='realtime'; in stepped "
                         "mode, advance() sets the pace")

    from vmteach.bridge import SimulationBridge, set_global_bridge

    sim = BACKENDS[backend](n_cells=n_cells, seed=seed, **kwargs)
    bridge = SimulationBridge(sim)
    # a previous microscope's realtime engine would keep stepping its
    # (now unreachable) sample in the background: stop it
    import vmteach.bridge as _bridge
    old = _bridge.GLOBAL_BRIDGE
    if old is not None and old._engine is not None:
        old._engine.stop()
    set_global_bridge(bridge)
    sim._vmteach_mode = mode

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
        engine = RealtimeEngine(sim, time_scale=float(speed), tick_hz=20,
                                idle_timeout=0.0, bridge=bridge)
        engine.patch_snap_frame()
        engine.start()
        bridge._engine = engine
        sim._engine = engine          # for sim.speed

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

        # UniMMCore.setProperty on a Python device changes the value but
        # emits no propertyChanged (real devices report changes through
        # the C++ callback relay). Only state devices emit, from their own
        # setters. GUI widgets such as the Device Property Browser listen
        # to propertyChanged, so they missed e.g. a scripted Binning
        # change. Emit it here, after the device lock is released, for
        # every property the device does not report itself. Drop once
        # fixed upstream.
        def setProperty(self, label, propName, propValue):
            super().setProperty(label, propName, propValue)
            self._emit_pydevice_change(label, propName)

        def setExposure(self, *args) -> None:
            super().setExposure(*args)
            label = args[0] if len(args) == 2 else self.getCameraDevice()
            self._emit_pydevice_change(label, "Exposure")

        def _emit_pydevice_change(self, label, propName) -> None:
            dev = self._pydevices[label] if label in self._pydevices else None
            if dev is None:
                return
            from pymmcore_plus.experimental.unicore import StateDevice
            if isinstance(dev, StateDevice) and propName in ("State", "Label"):
                return                      # state devices emit themselves
            self.events.propertyChanged.emit(
                label, propName, self.getProperty(label, propName))

        # Core role properties (Core-Camera, Core-Focus, ...): UniMMCore
        # routes setProperty("Core", ...) to its Python-device-aware
        # setters, but getProperty / getAllowedPropertyValues fall through
        # to the C++ core, which does not know Python devices and reports
        # "" (the property browser shows empty role dropdowns). Answer them
        # from the device registry. Drop once fixed upstream.
        _ROLES = {"Camera": ("getCameraDevice", "CameraDevice"),
                  "Focus": ("getFocusDevice", "StageDevice"),
                  "XYStage": ("getXYStageDevice", "XYStageDevice"),
                  "Shutter": ("getShutterDevice", "ShutterDevice"),
                  "SLM": ("getSLMDevice", "SLMDevice")}

        def getProperty(self, label, propName):
            if label == "Core" and propName in self._ROLES:
                return getattr(self, self._ROLES[propName][0])()
            return super().getProperty(label, propName)

        def getAllowedPropertyValues(self, label, propName):
            if label == "Core" and propName in self._ROLES:
                from pymmcore_plus import DeviceType
                dtype = DeviceType[self._ROLES[propName][1]]
                return ("",) + tuple(self.getLoadedDevicesOfType(dtype))
            return super().getAllowedPropertyValues(label, propName)

    return VirtualMicroscopeCore


_VirtualMicroscopeCore = None


def advance(sim, seconds: float = 1.0, dt: float = 0.05) -> None:
    """Advance simulated time deterministically (stepped mode).

    Replaces ``time.sleep()`` from real experiments: on hardware you *wait*
    for the sample to respond, on the deterministic virtual microscope you
    *advance* it. In realtime mode the sample advances on its own; wait
    with ``time.sleep`` there instead.
    """
    if getattr(sim, "_vmteach_mode", "stepped") == "realtime":
        raise RuntimeError(
            "advance() is for stepped mode; this microscope runs in "
            "realtime mode, where the sample evolves on its own. Wait with "
            "time.sleep(seconds) instead, or load the microscope with "
            "mode='stepped' for deterministic, advance()-driven time.")
    n = max(1, round(seconds / dt))
    for _ in range(n):
        sim.step(dt)
