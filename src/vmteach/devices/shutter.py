from pymmcore_plus.experimental.unicore import ShutterDevice
from vmteach.devices._base import ReentrantLockMixin


class SimShutterDevice(ReentrantLockMixin, ShutterDevice):


    def __init__(self):
        super().__init__()
        self._shutter = False # default close


    def get_open(self) -> bool:
        """
        Returns True if the device is open, False otherwise.
        """
        if self._shutter:
            return True
        else:
            return False

    def set_open(self, open_shutter: bool):
        """
        Set True if the device is open, False otherwise.
        """
        self._shutter = open_shutter