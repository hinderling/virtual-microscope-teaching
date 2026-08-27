# virtual-microscope-teaching

A **minimal virtual microscope for learning smart microscopy**: a simulated,
light-responsive specimen behind the same device API
([pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus) /
Micro-Manager) that controls real microscopes.

> **The point:** the simulator models the *microscope–code interaction*, not
> the biology. It lets you develop, debug, and teach image-processing
> pipelines and microscope-control logic in minutes on any laptop — then swap
> in a real microscope by changing only the configuration.

![Swap virtual and real microscope](docs/images/placeholder_puzzle.svg)

Code written against the virtual microscope runs unchanged on real hardware:
`core.snapImage()`, `core.setConfig("Channel", ...)`, `core.setSLMImage(...)`
are the identical calls in both worlds, because both implement the same
Micro-Manager core API.

This package accompanies the NEUBIAS
[training-resources](https://neubias.github.io/training-resources/) module on
**feedback photomanipulation**. It began as a subset of the full
[virtual-microscope](https://github.com/hinderling/virtual-microscope)
research simulator and was rewritten for teaching: ~1,900 lines of Python,
one specimen (optogenetic cells), 4 ms per frame (250 fps) on a laptop.

## Install

```bash
pip install virtual-microscope-teaching
```

With the napari GUI (interactive microscope control, results exploration):

```bash
pip install "virtual-microscope-teaching[gui]"
```

Requires Python ≥ 3.10, runs locally on any laptop. No hardware, no
Micro-Manager device adapters, no C++ — everything is pure Python.

> The very first `load_microscope()` compiles the simulation physics
> (numba, ~5 s) and caches the result on disk — every later load takes
> under a second, including after restarting Python. A full 100-cycle
> feedback experiment runs in ~2 s.

## Quick start: a complete feedback experiment

```python
import cv2, numpy as np
from vmteach import load_microscope, advance

core, sim = load_microscope("optogenetic", n_cells=20, seed=0)

for cycle in range(100):
    # ACQUIRE — identical call on real hardware
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    img = core.getImage()

    # ANALYZE — segment cells (any method works; here Otsu + contours)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # DECIDE — place a light spot above each cell (cells move toward light)
    mask = np.zeros_like(img)
    for c in contours:
        m = cv2.moments(c)
        if m["m00"] > 100:
            cv2.circle(mask, (int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"]) - 15), 11, 255, -1)

    # ACTUATE — identical calls on real hardware: upload the pattern,
    # then engage the stimulation light path to deliver it
    core.setSLMImage("SLM", mask)
    core.setConfig("Channel", "CyanStim")

    # let the sample respond (deterministic simulated time)
    advance(sim, seconds=1.0)
```

The cell population migrates upward, steered by your loop.

## Timing model (read this before designing experiments)

1. **Stimulation is gated on the light path**: `setSLMImage` only *uploads*
   the pattern — the SLM modulates light that is not on yet. Switching to
   the `CyanStim` channel engages the stimulation LED and delivers the
   pattern (an impulse that *sets* cell velocity toward the light);
   switching to an imaging channel turns it off. One delivery per loop
   iteration — the feedback-loop frequency is the stimulation frequency,
   as in pulsed optogenetic protocols. Snapping in `CyanStim` images the
   projected light itself (mask–sample alignment check).
2. **Stepped mode (default)** — simulated time advances only via
   `advance(sim, seconds=...)`. Same seed + same loop = identical result on
   every machine. This is the course default.
3. **Real-time mode** — `load_microscope(..., mode="realtime")`: the sample
   evolves in wall-clock time *while your code runs*, like on a real
   microscope. Your analysis latency becomes part of the experiment.

## API

| Function | Purpose |
|---|---|
| `load_microscope(backend, n_cells, seed, mode)` | Create the microscope → `(core, sim)` |
| `advance(sim, seconds)` | Advance simulated time (stepped mode) |
| `overlay(img, mask)` | RGB visualization of a stimulation mask on an image |
| `letter_mask(char)` | Binary letter target for the assembly exercise |

`core` is a full `pymmcore-plus` core: stage (`setXYPosition`), objectives
(`setState("Objective", ...)`), four channels (phase-contrast, DAPI,
membrane, and **CyanStim** — which images the projected SLM light itself,
for verifying mask–sample alignment like on a real system), exposure,
SLM — explore with the GUI below.

## napari GUI

```python
from vmteach.gui import launch_gui
core, sim = load_microscope("optogenetic", mode="realtime")
viewer = launch_gui(core)   # napari + micro-manager widgets on the virtual scope
```

![napari-micromanager on the virtual microscope](docs/images/napari_gui.png)

Snap, Live (~10 fps of crawling simulated cells), channel and objective
dropdowns, exposure, MDA — every control issues the same core API calls your
scripts make. The GUI and the script are two faces of one microscope. See the
[GUI walkthrough](docs/gui_walkthrough.md) for the click-by-click tour, and
`vmteach.gui.show_results(...)` to explore finished experiments (images,
segmentation, stimulation masks, tracks) as napari layers:

![results explorer](docs/images/results_explorer.png)

## Provenance

The cell physics (numba vertex model, optogenetic response) is taken from
[hinderling/virtual-microscope](https://github.com/hinderling/virtual-microscope)
at commit `25ddf57` (branch `virtual-env`). Rendering, optics, the device
bridge, and the device set were rewritten for this package: teaching needs
speed, determinism, and a small readable codebase more than the full
simulator's 35 specimen backends.

## License

MIT — see [LICENSE](LICENSE). If you use this in teaching or research,
please cite the FARO paper (see [CITATION.cff](CITATION.cff)).
