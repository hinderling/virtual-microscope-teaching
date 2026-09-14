import logging

from pymmcore_plus.experimental.unicore import XYStageDevice
from vmteach.devices._base import ReentrantLockMixin
import vmteach.bridge as bridge_module

logger = logging.getLogger(__name__)


class SimStageDevice(ReentrantLockMixin, XYStageDevice):


    def __init__(self) -> None:
        super().__init__()
        #self._x = 0.0
        #self._y = 0.0
        self.origin: tuple[float, float] = (0.0 , 0.0)
        self.position: tuple[float, float] = (0.0 , 0.0)
        # verify the microscope simulation exists
        #if microscope_sim is None:
        #    raise ValueError("microscope_sim must be provided.")
        self.bridge = bridge_module.GLOBAL_BRIDGE

    def home(self) -> None:
        """
        Move to its home position
        """
        #self._x = 0.0
        #self._y = 0.0
        self.position = (0.0 , 0.0)

    def stop(self) -> None:
        """
        Stop the movement of the stage
        """
        return

    def _get_bridge(self):
        """Always use the current global bridge (may be swapped at runtime)."""
        return bridge_module.GLOBAL_BRIDGE

    def set_position_um(self, x: float, y: float) -> None:
        """Move to (x, y) um. Like a stage with soft limits, a move beyond
        the travel range stops at the limit; the reported position is the
        one actually reached."""
        bridge = self._get_bridge()
        if bridge is not None:
            cx, cy = bridge.set_stage(x, y)
            if (cx, cy) != (x, y):
                logger.warning("XY stage: (%.0f, %.0f) is outside the travel "
                               "range, stopped at (%.0f, %.0f)", x, y, cx, cy)
            x, y = cx, cy
        self.position = (x, y)
        self.core.events.XYStagePositionChanged.emit(self.get_label(), x, y)

    def get_position_um(self) -> tuple[float, float]:
        """
        Return a float representing the current stage position
        """
        return self.position


    def set_origin_x(self) -> None:
        """
        Set the x coordinate of the stage origin
        """
        px, py = self.position
        self.origin = (px, self.origin[1])
        self.position = (0.0, py)
        #self._x = 0.0


    def set_origin_y(self) -> None:
        """
        Set the y coordinate of the stage origin
        """
        px, py = self.position
        self.origin = (self.origin[0], py)
        self.position = (px, 0.0)
        #self._y = 0.0

    #def update_camera_offset(self) -> None:
    #    """
    #    This method updates the camera offset of the virtual microscope
    #    """
        #self._microscope_sim.camera_offset = (self._x, self._y)
    #    self.bridge.set_stage(self._x, self._y)
