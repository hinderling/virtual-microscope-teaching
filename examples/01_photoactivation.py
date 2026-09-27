# %%
# Your first smart microscopy experiment: image -> mask -> stimulate -> image
#
# Install (once): pip install "virtual-microscope-teaching[gui]"
# Run it cell by cell in Jupyter (or VS Code), or as a plain script.
#
# The sample: cells expressing an optogenetic receptor (optoFGFR, tagged
# mVenus) and a kinase activity reporter (ERK-KTR, tagged mScarlet).
# Blue light activates the receptor; the reporter shows the response in a
# single snapshot: in resting cells it sits in the nucleus (bright
# nucleus), in activated cells it moves out into the cytoplasm (dark
# nucleus). No timelapse needed to see whether stimulation worked.
#
# Channels (named after the fluorophore, as on a real microscope):
#   phase-contrast  overview
#   miRFP           H2B nuclear marker: find the cells
#   mVenus          optoFGFR, the optogenetic tool (membrane)
#   mScarlet        ERK-KTR, the activity readout
#   CyanStim        the blue stimulation light path (images the pattern)

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vmteach import load_microscope, run_experiment
from vmteach.gui import launch_gui
from vmteach.optogenetic import detect_nuclei, measure_activity

# Real-time mode: the sample lives in wall-clock time, like on a real
# microscope, and napari-micromanager shows every snap as it happens.
core, sim = load_microscope("optogenetic", n_cells=20, seed=0, mode="realtime")
viewer = launch_gui(core)


def snap(channel: str) -> np.ndarray:
    """Switch channel, snap, return the image: three core calls."""
    core.setConfig("Channel", channel)
    core.snapImage()
    return core.getImage()


# %%
# The experiment, written exactly as for a real microscope. It runs in the
# background (run_experiment), so the viewer stays live: watch the channels
# switch, the light pattern appear, and the reporter respond.


def photoactivation(run):
    # 1. IMAGE: find the cells (nuclei in miRFP) and check that everyone
    #    is resting (bright nuclei in the mScarlet/ERK-KTR channel).
    #    measure_activity: per-cell cytoplasm-to-nucleus (C/N) reporter
    #    ratio, rescaled to 0 = resting, 1 = fully activated
    nuclei = snap("miRFP")
    cells = detect_nuclei(nuclei)
    ktr_before = snap("mScarlet")
    before = measure_activity(ktr_before, nuclei, cells)

    # 2. MASK: choose who gets light. Here: every cell in the left half of
    #    the field of view. The mask is a black-and-white image in camera
    #    pixels; white = light.
    mask = np.zeros((512, 512), np.uint8)
    for x, y in cells:
        if x < 256:
            cv2.circle(mask, (x, y), 25, 255, -1)

    # 3. STIMULATE: upload the mask to the SLM, then open the blue light
    #    path. The SLM only shapes the light; nothing reaches the sample
    #    until the stimulation channel turns the blue LED on. Snapping in
    #    CyanStim also records what the projected pattern looked like.
    core.setSLMImage("SLM", mask)
    projected = snap("CyanStim")     # delivers the pulse + images the light
    core.setSLMImage("SLM", np.zeros((512, 512), np.uint8))
    snap("phase-contrast")           # light off again

    # the pathway needs a moment to respond (full response after ~5 s)
    run.sleep(5)

    # 4. IMAGE again: stimulated cells now show a dark nucleus
    ktr_after = snap("mScarlet")
    after = measure_activity(ktr_after, snap("miRFP"), cells)

    # 5. The response is reversible: the reporter returns to the nucleus
    #    within ~20 s once the light stays off
    run.sleep(20)
    nuclei = snap("miRFP")
    later = measure_activity(snap("mScarlet"), nuclei, detect_nuclei(nuclei))
    snap("phase-contrast")

    return dict(cells=cells, before=before, after=after, later=later,
                ktr_before=ktr_before, projected=projected, ktr_after=ktr_after)


run = run_experiment(photoactivation)     # returns immediately

# %%
# Wait for the result (about 30 s; the viewer keeps updating meanwhile).
res = run.wait()
n = lambda acts: sum(a > 0.5 for a in acts)
print(f"{len(res['cells'])} cells, {n(res['before'])} active before stimulation")
print(f"{n(res['after'])} active 5 s after the pulse")
print(f"{n(res['later'])} active 20 s later")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, title in [
    (axes[0], res["ktr_before"], "ERK-KTR before"),
    (axes[1], res["projected"], "projected light (CyanStim)"),
    (axes[2], res["ktr_after"], "ERK-KTR 5 s after the pulse"),
]:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
for (x, y), a in zip(res["cells"], res["after"]):
    color = "cyan" if a > 0.5 else "gray"
    axes[2].add_patch(plt.Circle((x, y), 14, fill=False, color=color, lw=1.2))
plt.tight_layout()
plt.show()

# %%
# That is the whole idea of feedback photomanipulation, minus the loop:
# measure, decide, act, measure again. The next examples close the loop
# (respond to what the cells do, 03) and hand the choreography over to
# the microscope's acquisition engine (04).
