# %%
# Advanced: the same feedback loop as an event-driven MDA acquisition
#
# The for-loop in example 01 is the clearest way to LEARN feedback control,
# but production acquisitions in pymmcore-plus are built on useq-schema
# MDAEvents executed by the MDA engine: hardware-timed, logged, and
# GUI-compatible (the napari-micromanager MDA panel builds the same events).
#
# The trick for feedback experiments: `run_mda()` accepts any *iterable* of
# events — including a Queue that is being filled WHILE the acquisition
# runs. Analysis decides, event by event, what the microscope does next.
# This adapts the Analyzer/Controller pattern from the pymmcore-plus guide:
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
# (min_start_time), and the sample evolves in wall clock — like on
# real hardware.
core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
                            mode="realtime")

N_FRAMES = 30
INTERVAL = 0.4          # seconds between frames
DAPI = {"config": "DAPI", "group": "Channel"}
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

        if ch == "DAPI":
            # analysis frame: segment, upload the pattern, then declare the
            # stimulation as an EVENT. In the manual loop we had to
            # choreograph the light path ourselves (setConfig on, off, on);
            # here we just say "channel=CyanStim" and the engine switches
            # the hardware — and records an image of the projected light.
            result = self._analyzer.run(img)
            self._core.setSLMImage("SLM", result["mask"])
            self._queue.put(MDAEvent(index={"t": t}, channel=STIM))
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
                    channel=DAPI,
                    min_start_time=(t + 1) * INTERVAL,  # pace by wall clock
                ))

    def run(self):
        # a Queue is not iterable — iter(get, sentinel) makes it one
        self._core.run_mda(iter(self._queue.get, self.STOP))
        # seed the acquisition with the first event; analysis takes over
        self._queue.put(MDAEvent(index={"t": 0}, channel=DAPI,
                                 min_start_time=0.0))


# %%
# Run it. Note there is NO acquisition loop in our code anymore — the MDA
# engine drives the microscope; our code only reacts to frames and decides
# the next event.
y_before = sim.centers[:, 1].copy()

q = Queue()
controller = Controller(Analyzer(), core, q)
controller.run()

# run_mda is non-blocking; wait for the acquisition to finish
while core.mda.is_running():
    time.sleep(0.1)

dy = sim.centers[:, 1] - y_before
dy -= sim.height * np.round(dy / sim.height)      # periodic world wrap
print(f"\nmean displacement after {N_FRAMES} frames: {dy.mean():+.1f} px "
      "(negative = up)")

# %%
# Why bother, when the for-loop worked fine?
#  - the light-path choreography disappeared: stimulation is just an event
#    with channel="CyanStim" — the engine switches LED + filter, delivers
#    the pattern, and even logs an image of the projected light
#  - useq MDAEvents carry the full acquisition vocabulary (z-stacks,
#    positions, exposure, channels) in one declarative object
#  - the engine handles hardware timing/synchronization; events are logged
#    and reproducible
#  - the identical Controller runs on real hardware — swap the core
#  - a GUI (napari-micromanager's MDA panel) and your feedback logic can
#    produce events for the same engine
#
# On real microscopes this is the recommended architecture for feedback
# experiments — see rtm-pymmcore / FARO for a full implementation.
