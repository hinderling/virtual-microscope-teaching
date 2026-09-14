"""Virtual camera: a fixed 512 x 512 sensor in front of the simulation.

The objective never changes the image size, only the pixel size (declared
as pixel-size configs in optogenetic.cfg). Binning is a preset (1x1, 2x2,
4x4): the frame shrinks to 512/b and each output pixel sums b^2 sensor
pixels, so the image gets b^2 brighter (lower the exposure, as you would
on a real camera).
"""

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Callable, Optional

import numpy as np
from numpy.typing import DTypeLike
from pymmcore_plus import PropertyType
from pymmcore_plus.experimental.unicore import CameraDevice, pymm_property

import vmteach.bridge as bridge_module
from vmteach.devices._base import ReentrantLockMixin

class _IntPresets(tuple):
    """Allowed values that accept ``2`` and ``"2"`` alike.

    unicore validates ``value in allowed_values`` without coercion, but
    config files and the GUI send strings while scripts send ints (real
    MMCore stringifies everything, so both must work).
    """

    def __contains__(self, value) -> bool:
        try:
            return tuple.__contains__(self, int(value))
        except (TypeError, ValueError):
            return False


BINNING_PRESETS = _IntPresets((1, 2, 4))


class SimCameraDevice(ReentrantLockMixin, CameraDevice):
    """pymmcore camera device rendering frames from the simulation."""

    _exposure: float = 50.0
    _brightness: float = 1.0
    _gain: float = 1.0
    _binning: int = 1
    # Region of interest as (x, y, width, height) in *output* pixels;
    # None == full frame.
    _roi: Optional[tuple[int, int, int, int]] = None

    # Minimum interval between frames (seconds) to avoid starving the Qt main
    # thread.  The rendering holds the GIL, so without a gap the UI freezes.
    _MIN_FRAME_INTERVAL: float = 0.033  # ~30 FPS max

    def __init__(self) -> None:
        super().__init__()
        # Binning is a choice of presets, not a slider
        self.set_property_limits("Binning", None)
        self.set_property_allowed_values("Binning", BINNING_PRESETS)

    def _get_bridge(self):
        """Always use the current global bridge (may be swapped at runtime)."""
        return bridge_module.GLOBAL_BRIDGE

    # ── exposure ────────────────────────────────────────────────────────

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = exposure
        self.core.events.exposureChanged.emit(self.get_label(), exposure)

    # ── geometry ────────────────────────────────────────────────────────

    def _sensor_size(self) -> int:
        bridge = self._get_bridge()
        if bridge is None:
            from vmteach.sim import SENSOR_SIZE
            return SENSOR_SIZE
        return bridge._sim.sensor_size

    def _frame_hw(self) -> tuple[int, int]:
        """Full (height, width) of the output frame at the current binning."""
        n = self._sensor_size() // self._binning
        return n, n

    def shape(self) -> tuple[int, ...]:
        """Frame shape, ROI-cropped (h, w) when an ROI is active."""
        if self._roi is not None:
            _, _, w, h = self._roi
            return (h, w)
        return self._frame_hw()

    def dtype(self) -> DTypeLike:
        return np.uint8

    # ── acquisition ─────────────────────────────────────────────────────

    def start_sequence(
        self,
        n: int | None,
        get_buffer: Callable[[Sequence[int], DTypeLike], np.ndarray],
    ) -> Iterator[Mapping]:
        count = 0
        while n is None or count < n:
            t0 = time.perf_counter()
            bridge = self._get_bridge()
            frame = bridge.snap(brightness=self._brightness,
                                exposure=self._exposure, gain=self._gain,
                                binning=self._binning)  # type: ignore

            # Digital ROI: the sim always renders the full frame; the ROI
            # is purely a readout crop.
            if self._roi is not None:
                x, y, w, h = self._roi
                frame = frame[y:y + h, x:x + w]

            buf = get_buffer(frame.shape, self.dtype())
            buf[:] = frame
            yield {"data": buf, "timestamp": time.time()}
            count += 1

            # Throttle: ensure minimum interval so the main thread gets GIL time
            remaining = self._MIN_FRAME_INTERVAL - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

    # ── properties ──────────────────────────────────────────────────────

    @pymm_property(
        limits=(0.0, 100.0),
        sequence_max_length=100,
        name="brightness",
        property_type=PropertyType.Float,
    )
    def brightness(self) -> float:
        """Illumination brightness multiplier of the virtual light source."""
        return self._brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        self._brightness = value

    @brightness.sequence_loader
    def _load_brightness_sequence(self, sequence: Sequence[float]) -> None:
        pass

    @brightness.sequence_starter
    def _start_brightness_sequence(self) -> None:
        pass

    @pymm_property(
        limits=(1.0, 32.0),
        sequence_max_length=100,
        name="Gain",
        property_type=PropertyType.Float,
    )
    def gain(self) -> float:
        """Analog gain multiplier (1.0 = none, 32.0 = max).

        Higher gain amplifies the signal but also the noise. Use it for
        dim samples when increasing exposure is not an option.
        """
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = max(1.0, min(32.0, value))

    def get_binning(self) -> int:
        """Current binning factor (1, 2 or 4)."""
        return self._binning

    def set_binning(self, binning: int) -> None:
        """Set binning; the frame shrinks to sensor/b and the ROI resets."""
        b = int(binning)
        if b not in BINNING_PRESETS:
            raise ValueError(
                f"Binning {binning!r} not supported; choose one of "
                f"{BINNING_PRESETS}")
        if b != self._binning:
            self._binning = b
            self._roi = None

    # ── ROI ─────────────────────────────────────────────────────────────
    #
    # Implementing set_roi also fixes a teardown bug: pymmcore-plus's
    # MDAEngine snapshots the ROI at sequence start and restores it via
    # setROI() in teardown_sequence(). The unicore CameraDevice base raises
    # NotImplementedError from set_roi, and that exception propagates out
    # of MDARunner._finish_run *before* it emits sequenceFinished, silently
    # stranding listeners (e.g. napari-micromanager's _mda_running flag
    # stays True, freezing the snap preview after a run).

    def get_roi(self) -> tuple[int, int, int, int]:
        """Return the current ROI as (x, y, width, height)."""
        if self._roi is not None:
            return self._roi
        h, w = self._frame_hw()
        return (0, 0, w, h)

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set a rectangular ROI; frames are cropped to it on readout."""
        x, y, width, height = int(x), int(y), int(width), int(height)
        sh, sw = self._frame_hw()
        if x == 0 and y == 0 and width == sw and height == sh:
            self._roi = None  # full frame -> no crop
            return
        if (width <= 0 or height <= 0 or x < 0 or y < 0
                or x + width > sw or y + height > sh):
            raise ValueError(
                f"ROI ({x}, {y}, {width}, {height}) is out of bounds for a "
                f"{sw}x{sh} frame.")
        self._roi = (x, y, width, height)

    def clear_roi(self) -> None:
        """Reset the ROI to the full frame."""
        self._roi = None
