# GUI walkthrough: controlling the virtual microscope in napari

> Screenshots are auto-captured from the actual GUI (`scripts/make_screenshots.py`);
> re-run that script after UI changes to refresh them.

## 1. Launch

```python
from vmteach import load_microscope
from vmteach.gui import launch_gui

core, sim = load_microscope("optogenetic", n_cells=20, seed=0, mode="realtime")
viewer = launch_gui(core)
```

![napari with micro-manager widgets](images/napari_gui.png)

The top rows are the **napari-micromanager** control toolbars — the same
plugin used on real Micro-Manager systems:

| Control | What it does | Script equivalent |
|---|---|---|
| **Snap** (camera icon) | acquire one image into the `preview` layer | `core.snapImage()` |
| **Live** (film icon) | continuous acquisition (~10 fps) — the simulated cells crawl in real time | `core.startContinuousSequenceAcquisition()` |
| **Channel** dropdown | phase-contrast / DAPI / membrane / **CyanStim** (images the projected SLM light — use it to verify mask–sample alignment) | `core.setConfig("Channel", ...)` |
| **Objectives** dropdown | 10x / 20x / 40x / 100x — field of view shrinks, pixel size updates | `core.setState("Objective", ...)` |
| **Exposure** | exposure time in ms | `core.setExposure(...)` |
| **MDA** | multi-dimensional acquisition editor (time-lapse, channels, positions) | `core.mda.run(...)` |

**The one idea to take away:** every control in this GUI issues a pymmcore
API call on the *same core object* your script holds. Set the objective from
code and the dropdown updates; press Snap in the GUI and your script could
read the same image. GUI and script are two faces of one microscope — and
swapping the virtual microscope for real hardware changes neither of them.

## 2. Watch a feedback script drive the microscope

Keep the GUI open and run the feedback loop from
[`examples/01_feedback_loop.py`](../examples/01_feedback_loop.py) in the same
session: each `core.snapImage()` of the loop refreshes the `preview` layer,
so you watch the smart-acquisition script "click through" the experiment.

## 3. Explore finished experiments as layers

```python
from vmteach.gui import show_results
show_results(images, masks=masks, centroids=centroids, segmentations=segs,
             name="split steering")
```

![results explorer](images/results_explorer.png)

Four layers, scrubbed with the time slider:

- **images** — the acquired time-lapse
- **segmentation** — what your analysis saw (label colors per cell)
- **stimulation** — where the light went (cyan spots; note: above the cells
  in the left half, below in the right half — the per-object decision)
- **tracks** — where the cells went (tails show the steering outcome)

Toggle layers on/off to debug: *"did my mask land where I thought?"* is a
one-click question here, and exactly the same layer stack you would use to
QC a real experiment.
