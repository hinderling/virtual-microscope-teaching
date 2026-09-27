"""vmteach: a minimal virtual microscope for teaching smart microscopy.

A fully simulated, light-responsive specimen behind the pymmcore-plus
device API. Code written against this virtual microscope runs unchanged
on real hardware supported by Micro-Manager; only the configuration
changes.

The top-level namespace is the simulator itself:

    from vmteach import load_microscope

    core, sim = load_microscope("optogenetic", n_cells=20, seed=0)
    core.snapImage()                # acquire, identical call on real hardware
    img = core.getImage()
    core.setSLMImage("SLM", mask)   # upload pattern, identical on real hardware

Each sample backend is a subpackage with the analysis helpers for its
readouts (the shipped one: :mod:`vmteach.optogenetic`); the napari GUI
lives in :mod:`vmteach.gui`.
"""

from vmteach.runner import Run, run_experiment
from vmteach.teach import (
    load_microscope,
    register_backend,
    BACKENDS,
    advance,
)

__all__ = [
    "load_microscope",
    "register_backend",
    "BACKENDS",
    "advance",
    "run_experiment",
    "Run",
]

__version__ = "0.1.0"
