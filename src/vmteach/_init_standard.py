"""Standard device stack initializer — shared by all programmatic setup_* functions.

The single entry point is ``load_cfg(sim, cfg_path)`` which:
  1. Wires the simulation into the global bridge
  2. Loads the backend .cfg (devices, channels, everything)
  3. Installs pixel-size helper
  4. Optionally auto-starts the RealtimeEngine for continuous sims

Usage:
    from vmteach._init_standard import load_cfg
    core = load_cfg(sim, Path(__file__).parent / "bacteria.cfg")
"""

import logging
from pathlib import Path

from pymmcore_plus.experimental.unicore import UniMMCore

import vmteach.engine.simulation_bridge as bridge_module
from vmteach.engine.simulation_bridge import SimulationBridge, set_global_bridge

logger = logging.getLogger(__name__)


def _install_pixel_size(core: UniMMCore, sim) -> None:
    """Install getPixelSizeUm() returning correct um/camera-px for current objective.

    Camera always outputs 512x512 px regardless of objective.
    FOV in world px: 10x=512, 20x=256, 40x=128, 100x=64.
    """
    world_px_um = getattr(sim, 'world_pixel_size_um', 1.0)
    _obj_factors = {0: 1, 1: 2, 2: 4, 3: 8}

    def _get_pixel_size_um(cached=False):
        try:
            obj_state = core.getState("Objective")
        except Exception:
            return world_px_um
        factor = _obj_factors.get(obj_state, 1)
        return world_px_um / factor

    core.getPixelSizeUm = _get_pixel_size_um


def load_cfg(sim, cfg_path: Path, *,
             factory=None,
             factory_kwargs: dict | None = None,
             realtime: bool | None = None,
             time_scale: float = 1.0,
             tick_hz: int = 10,
             idle_timeout: float = 30.0) -> UniMMCore:
    """Load a backend .cfg after pre-creating the simulation with custom params.

    This is the single entry point for programmatic setup_*() calls.
    The .cfg is the sole source of truth for devices and channel definitions.

    Args:
        sim: Simulation backend instance (already created with custom params).
        cfg_path: Path to the backend's .cfg file.
        factory: Optional callable(**kwargs) that creates a new sim instance.
            Used by ``NewExperimentDevice`` for hard-reset (recreate).
        factory_kwargs: Dict of keyword arguments used to create the current
            sim.  Stored so ``recreate_simulation()`` can reproduce it.
        realtime: Whether to auto-start the RealtimeEngine.  ``None`` (default)
            auto-detects from ``sim.continuous``.  ``True`` forces it on,
            ``False`` forces it off.
        time_scale: Simulation time multiplier (1.0 = real-time).
        tick_hz: Background tick rate (steps/second).
        idle_timeout: Seconds without snaps before the engine auto-pauses
            (0 = never pause).

    Returns:
        Configured UniMMCore instance.
    """
    bridge = SimulationBridge(sim, factory=factory)
    if factory_kwargs:
        bridge._factory_kwargs = factory_kwargs
    set_global_bridge(bridge)
    core = UniMMCore()
    core.loadSystemConfiguration(str(cfg_path))
    _install_pixel_size(core, sim)

    # Auto-start real-time engine for continuous sims
    should_start = realtime if realtime is not None else getattr(sim, 'continuous', False)
    if should_start and hasattr(sim, 'step'):
        from vmteach.engine.realtime import RealtimeEngine
        # Pre-warm the numba-JIT'd physics step BEFORE the RealtimeEngine
        # starts. The engine's background loop calls step() under a shared
        # lock on every tick, so it would JIT-compile step() within ~0.1s
        # of start() regardless -- but if that multi-second compile runs on
        # the engine thread it holds the lock, stalling the first batch of
        # snap_frame() calls (an MDA acquisition then sees its opening
        # frames blocked for several seconds, delivered in a burst).
        # Compiling here -- on the calling thread, before the engine exists
        # -- moves that unavoidable cost to setup so the lock is free when
        # the first snap arrives. Zero net cost: the engine compiles step()
        # anyway. The renderer JIT is deliberately NOT pre-warmed: it only
        # compiles when something snaps, so forcing it here would burn
        # compute on virtual microscopes that are created but never imaged.
        # The first real snap pays the renderer JIT (~1-2s for frame 0),
        # which is acceptable -- it's a slow first frame, not a stall.
        try:
            sim.step(0.0)
        except Exception:
            logger.debug("step() pre-warm failed (non-fatal)", exc_info=True)

        engine = RealtimeEngine(sim, time_scale=time_scale,
                                tick_hz=tick_hz, idle_timeout=idle_timeout,
                                bridge=bridge)
        engine.patch_snap_frame()
        engine.start()
        bridge._engine = engine
        logger.info("RealtimeEngine auto-started for %s (tick_hz=%d, idle_timeout=%.0fs)",
                    type(sim).__name__, tick_hz, idle_timeout)

    return core
