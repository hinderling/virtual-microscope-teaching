from pymmcore_plus.experimental.unicore import StateDevice

import vmteach.bridge as bridge_module
from vmteach.devices._base import ReentrantLockMixin


class GenericStateDevice(ReentrantLockMixin, StateDevice):
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
        """Push current state to bridge once the bridge exists.

        initializeAllDevices() runs all devices in parallel threads, so we
        wait for the bridge_ready event before pushing state. Timeout of
        5s prevents hangs if the bridge is never created.
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
    """Objective turret (5 magnifications; pixel sizes live in the .cfg)."""

    def __init__(self) -> None:
        super().__init__("Objective", {
            0: "4x",
            1: "10x",
            2: "20x",
            3: "40x",
            4: "60x",
        })
