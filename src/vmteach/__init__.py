"""vmteach: a minimal virtual microscope for teaching smart microscopy.

A fully simulated, light-responsive specimen behind the pymmcore-plus
device API. Code written against this virtual microscope runs unchanged
on real hardware supported by Micro-Manager; only the configuration
changes.

Teaching API (all you need for the course):

    from vmteach import load_microscope, advance, overlay, letter_mask

    core, sim = load_microscope("optogenetic", n_cells=20, seed=0)
    core.snapImage()                # acquire, identical call on real hardware
    img = core.getImage()
    core.setSLMImage("SLM", mask)   # upload pattern, identical on real hardware
    advance(sim, seconds=1.0)       # deterministically advance simulated time

This package is a teaching subset of
https://github.com/hinderling/virtual-microscope
(see PROVENANCE in the README for the source commit).
"""

from vmteach.teach import (
    load_microscope,
    advance,
    overlay,
    letter_mask,
    detect_nuclei,
    measure_activity,
    link_tracks,
)

__all__ = [
    "load_microscope",
    "advance",
    "overlay",
    "letter_mask",
    "detect_nuclei",
    "measure_activity",
    "link_tracks",
]

__version__ = "0.1.0"
