# %%
# Advanced: declare what you want, not what the microscope should do
#
# This is the pymmcore-plus implementation of the declarative pattern:
# acquisition steps are described as hardware-agnostic useq-schema events
# ("an image at time t in channel c with stimulation pattern p") and the
# MDA engine translates them into hardware commands. The same event
# descriptions could be executed by other engines on other systems.
#
# In pymmcore-plus an event is a useq.MDAEvent, and the engine entry point
# is `run_mda()`. The trick for feedback experiments: `run_mda()` accepts
# any *iterable* of events, including a Queue that is being filled WHILE
# the acquisition runs. Analysis decides, event by event, what happens
# next. This adapts the Analyzer/Controller pattern from the guide:
# https://pymmcore-plus.github.io/pymmcore-plus/guides/event_driven_acquisition/
#
#   Controller ──puts──▶ Queue ──iterated by──▶ MDA engine ──▶ microscope
#       ▲                                                          │
#       └────────── frameReady (image + event) ◀───────────────────┘

import time
from queue import Queue

import cv2
import numpy as np
from useq import MDAEvent

from vmteach import detect_nuclei, load_microscope

# Real-time mode: the MDA engine paces acquisition by wall clock
# (min_start_time), and the sample evolves in wall clock, like on
# real hardware.
core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
                            mode="realtime")

N_FRAMES = 30
INTERVAL = 0.4          # seconds between frames
NUCLEI = {"config": "miRFP", "group": "Channel"}   # H2B-miRFP
STIM = {"config": "CyanStim", "group": "Channel"}


# %%
class Analyzer:
    """ANALYZE + DECIDE: segment nuclei, build the steering mask."""

    def run(self, img: np.ndarray) -> dict:
        cells = detect_nuclei(img)
        mask = np.zeros((512, 512), np.uint8)
        for cx, cy in cells:
            cv2.circle(mask, (cx, max(cy - 15, 0)), 11, 255, -1)
        return {"n_cells": len(cells), "mask": mask}


class Controller:
    """Feeds the MDA engine one event at a time, driven by analysis."""

    STOP = object()

    def __init__(self, analyzer: Analyzer, core, queue: Queue,
                 n_frames: int = N_FRAMES):
        self._analyzer = analyzer
        self._core = core
        self._queue = queue
        self._n_frames = n_frames
        # the engine calls this for every acquired frame
        core.mda.events.frameReady.connect(self._on_frame_ready)

    def _on_frame_ready(self, img: np.ndarray, event: MDAEvent) -> None:
        t = event.index.get("t", 0)
        ch = event.channel.config if event.channel else None

        if ch == "miRFP":
            # Analysis frame: segment, then declare the whole stimulation
            # as ONE event. The event carries both the light path
            # (channel="CyanStim") and the pattern (slm_image=mask). The
            # engine switches the LED and filter, uploads the pattern to
            # the SLM, delivers the light, and records an image of the
            # projected pattern. In the manual loop all of that was our
            # own choreography of core calls; here it is a declaration.
            result = self._analyzer.run(img)
            self._queue.put(MDAEvent(index={"t": t}, channel=STIM,
                                     slm_image=result["mask"]))
            print(f"frame {t + 1}/{self._n_frames}: "
                  f"{result['n_cells']} cells", end="\r")
        else:
            # stimulation frame delivered (and imaged). Queue the next
            # analysis frame, or stop.
            if t + 1 >= self._n_frames:
                self._queue.put(self.STOP)      # sentinel ends the MDA
            else:
                self._queue.put(MDAEvent(
                    index={"t": t + 1},
                    channel=NUCLEI,
                    min_start_time=(t + 1) * INTERVAL,  # pace by wall clock
                ))

    def run(self):
        # a Queue is not iterable; iter(get, sentinel) makes it one.
        # run_mda is non-blocking and returns the acquisition thread.
        self.thread = self._core.run_mda(iter(self._queue.get, self.STOP))
        # seed the acquisition with the first event; analysis takes over
        self._queue.put(MDAEvent(index={"t": 0}, channel=NUCLEI,
                                 min_start_time=0.0))


# %%
# Run it. Note there is NO acquisition loop in our code anymore. The MDA
# engine drives the microscope; our code only reacts to frames and decides
# the next event.
# Only the cells in the field of view can be steered (each well is 4 x 4
# fields), so measure those.
off, fov = sim.view_origin, sim.fov_um
rel = sim.centers - off
in_view = (rel > 0).all(axis=1) & (rel < fov).all(axis=1)
y_before = sim.centers[:, 1].copy()

q = Queue()
controller = Controller(Analyzer(), core, q)
controller.run()

# wait for the acquisition thread to finish (polling core.mda.is_running()
# right after run_mda is racy: the thread may not have started yet)
controller.thread.join()

dy = sim.centers[:, 1] - y_before
print(f"\nmean displacement of the {in_view.sum()} cells in view after "
      f"{N_FRAMES} frames: {dy[in_view].mean():+.1f} um (negative = up)")

# %%
# Why bother, when the for-loop worked fine?
#  - the light choreography disappeared: one declared event carries the
#    channel AND the SLM pattern; the engine switches the hardware,
#    delivers the light, and logs an image of the projected pattern
#  - useq MDAEvents carry the full acquisition vocabulary (z-stacks,
#    positions, exposure, channels) in one declarative object
#  - the engine handles hardware timing/synchronization; events are logged
#    and reproducible
#  - the identical Controller runs on real hardware; just swap the core
#  - a GUI (napari-micromanager's MDA panel) and your feedback logic can
#    produce events for the same engine
#
# On real microscopes this is the recommended architecture for feedback
# experiments. See rtm-pymmcore or FARO for a full implementation.
