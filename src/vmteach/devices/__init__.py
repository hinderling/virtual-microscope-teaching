"""Virtual microscope device adapters for pymmcore-plus.

All device classes can be loaded via core.loadPyDevice() or via .cfg files
using the #py pyDevice syntax (pymmcore-plus >= 0.17.0).
"""

from vmteach.devices.camera import SimCameraDevice
from vmteach.devices.stage import SimStageDevice
from vmteach.devices.z_stage import SimZStageDevice
from vmteach.devices.shutter import SimShutterDevice
from vmteach.devices.slm import SimSLMDevice
from vmteach.devices.state import (
    GenericStateDevice,
    LEDDevice,
    FilterWheelDevice,
    ObjectiveDevice,
    TemperatureControllerDevice,
    PerfusionPumpDevice,
    StretchDevice,
    AnesthesiaDevice,
    ElectrodeDevice,
    ResetDevice,
    ModalityDevice,
    FluidicsDevice,
    DeformableMirrorDevice,
    OffAxisLEDDevice,
    RotationStageDevice,
    SIMPatternDevice,
)
from vmteach.devices.sim_server import SimServer

__all__ = [
    "SimCameraDevice",
    "SimStageDevice",
    "SimZStageDevice",
    "SimShutterDevice",
    "SimSLMDevice",
    "GenericStateDevice",
    "LEDDevice",
    "FilterWheelDevice",
    "ObjectiveDevice",
    "TemperatureControllerDevice",
    "PerfusionPumpDevice",
    "StretchDevice",
    "AnesthesiaDevice",
    "ElectrodeDevice",
    "ResetDevice",
    "ModalityDevice",
    "FluidicsDevice",
    "DeformableMirrorDevice",
    "OffAxisLEDDevice",
    "RotationStageDevice",
    "SIMPatternDevice",
    "SimServer",
]
