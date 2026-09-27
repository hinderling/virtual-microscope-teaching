"""The ``optogenetic`` sample backend and its analysis helpers.

Cells with a nuclear marker (H2B-miRFP), a membrane-bound optogenetic
receptor (optoFGFR-mVenus) and an ERK activity reporter (ERK-KTR-
mScarlet), plus reference image-analysis functions for exactly these
readouts:

    from vmteach.optogenetic import detect_nuclei, measure_activity

The helpers take plain images, so they also run on frames of a real
sample labelled the same way.
"""

from vmteach.optogenetic.analysis import (
    cn_ratio,
    detect_nuclei,
    letter_mask,
    link_tracks,
    measure_activity,
    overlay,
)
from vmteach.optogenetic.sim import OptoCellSim

__all__ = ["OptoCellSim", "detect_nuclei", "cn_ratio", "measure_activity",
           "link_tracks", "overlay", "letter_mask"]
