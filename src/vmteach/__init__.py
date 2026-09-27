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

Image-analysis reference helpers (hardware-agnostic) live in
:mod:`vmteach.analysis`; the napari GUI in :mod:`vmteach.gui`.
"""

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
]

__version__ = "0.1.0"
