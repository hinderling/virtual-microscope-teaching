"""SimServer — GenericDevice that bootstraps the simulation from a .cfg file.

Lives first in every backend's .cfg. On initialize() it:
  1. Reads the Backend property → imports vmteach.backends.<backend>
  2. Reads SimConfig (optional path to .sim.json; defaults to backend package default)
  3. Calls mod.create_sim(**params) to build the simulation
  4. Wraps it in SimulationBridge and sets GLOBAL_BRIDGE
  5. Auto-starts RealtimeEngine for continuous sims

This allows `core.loadSystemConfiguration("bacteria/bacteria.cfg")` to work
with zero custom Python code — the standard napari-micromanager workflow.
"""

import importlib
import json
import logging
from pathlib import Path

from pymmcore_plus.experimental.unicore import GenericDevice
import vmteach.engine.simulation_bridge as bridge_module
from vmteach.engine.simulation_bridge import SimulationBridge, set_global_bridge

logger = logging.getLogger(__name__)


class SimServer(GenericDevice):
    """Bootstrap device that instantiates a simulation backend from .cfg properties.

    Pre-init properties (set before initialize() is called):
      Backend   — backend module name, e.g. "bacteria" or "voronoi"
      SimConfig — optional path to .sim.json override (empty = use backend default)
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_property("Backend", default_value="", is_pre_init=True)
        self.register_property("SimConfig", default_value="", is_pre_init=True)
        # Clear so state devices wait for this instance's initialize() to finish
        bridge_module.bridge_ready.clear()

    def initialize(self) -> None:
        # If the bridge is already wired (programmatic setup via load_cfg),
        # skip sim creation — the .cfg is only being loaded for devices/channels.
        if bridge_module.GLOBAL_BRIDGE is not None:
            bridge_module.bridge_ready.set()
            return

        backend = self.get_property_value("Backend")
        if not backend:
            raise RuntimeError("SimServer: Backend property must be set before initialize()")

        sim_config = self.get_property_value("SimConfig")

        # Import the backend module
        mod = importlib.import_module(f"vmteach.backends.{backend}")

        # Resolve sim config path
        if not sim_config:
            # Use the .sim.json bundled with the backend package
            pkg_dir = Path(mod.__file__).parent
            sim_config_path = pkg_dir / f"{backend}.sim.json"
        else:
            sim_config_path = Path(sim_config)

        # Load params from JSON
        if sim_config_path.exists():
            params = json.loads(sim_config_path.read_text())
        else:
            params = {}

        # Create simulation and install bridge
        sim = mod.create_sim(**params)
        bridge = SimulationBridge(sim)
        set_global_bridge(bridge)

        # Auto-start RealtimeEngine for continuous sims
        if getattr(sim, 'continuous', False) and hasattr(sim, 'step'):
            from vmteach.engine.realtime import RealtimeEngine
            engine = RealtimeEngine(sim, time_scale=1.0,
                                    tick_hz=10, idle_timeout=30.0,
                                    bridge=bridge)
            engine.patch_snap_frame()
            engine.start()
            bridge._engine = engine
            logger.info("RealtimeEngine auto-started for %s", type(sim).__name__)

    def shutdown(self) -> None:
        # Stop engine on device shutdown
        bridge = bridge_module.GLOBAL_BRIDGE
        if bridge is not None and hasattr(bridge, '_engine') and bridge._engine is not None:
            bridge._engine.stop()
            bridge._engine = None
