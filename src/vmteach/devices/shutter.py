"""Virtual shutter: holds the light on for a manual exposure.

Snaps and live frames open it implicitly (auto-shutter); opening it by
hand with ``core.setShutterOpen(True)`` keeps the current channel's light
on the sample, which in the CyanStim channel delivers the SLM pattern.
"""
from pymmcore_plus.experimental.unicore import ShutterDevice

import vmteach.bridge as bridge_module
from vmteach.devices._base import ReentrantLockMixin


class SimShutterDevice(ReentrantLockMixin, ShutterDevice):

    def __init__(self):
        super().__init__()
        self._shutter = False  # default closed

    def get_open(self) -> bool:
        return self._shutter

    def set_open(self, open_shutter: bool):
        self._shutter = bool(open_shutter)
        if bridge_module.GLOBAL_BRIDGE is not None:
            bridge_module.GLOBAL_BRIDGE.set_shutter(self._shutter)
