# %%
# Your first smart microscopy experiment: image -> mask -> stimulate -> image
#
# Install (once): pip install virtual-microscope-teaching
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

import matplotlib.pyplot as plt
import numpy as np

from vmteach import advance, detect_nuclei, load_microscope, measure_activity

core, sim = load_microscope("optogenetic", n_cells=20, seed=0)


def snap(channel: str) -> np.ndarray:
    """Switch channel, snap, return the image: three core calls."""
    core.setConfig("Channel", channel)
    core.snapImage()
    return core.getImage()


# %%
# 1. IMAGE: find the cells (nuclei in the miRFP channel) and check that
#    everyone is resting (bright nuclei in the mScarlet/ERK-KTR channel)
nuclei_img = snap("miRFP")
cells = detect_nuclei(nuclei_img)
ktr_before = snap("mScarlet")
activity_before = measure_activity(ktr_before, cells)
print(f"{len(cells)} cells, "
      f"{sum(a > 0.5 for a in activity_before)} active before stimulation")

# %%
# 2. MASK: choose who gets light. Here: every cell in the left half of
#    the field of view. The mask is just a black-and-white image, in
#    camera pixels; white = light.
mask = np.zeros((512, 512), np.uint8)
for x, y in cells:
    if x < 256:
        # a small disc of light on each chosen cell
        import cv2
        cv2.circle(mask, (x, y), 25, 255, -1)

# %%
# 3. STIMULATE: upload the mask to the SLM, then open the blue light
#    path. The SLM only shapes the light; nothing reaches the sample
#    until the stimulation channel turns the blue LED on. Snapping in
#    CyanStim also records what the projected pattern looked like.
core.setSLMImage(mask)
projected = snap("CyanStim")     # delivers the pulse + images the light

# the pathway needs a moment to respond (full response after ~5 s)
advance(sim, 5)

# %%
# 4. IMAGE again: same field, same channel. Stimulated cells now show a
#    dark nucleus (reporter exported to the cytoplasm).
ktr_after = snap("mScarlet")
activity_after = measure_activity(ktr_after, cells)
print(f"{sum(a > 0.5 for a in activity_after)} active after stimulation")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, title in [
    (axes[0], ktr_before, "ERK-KTR before"),
    (axes[1], projected, "projected light (CyanStim)"),
    (axes[2], ktr_after, "ERK-KTR after"),
]:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
for (x, y), a0, a1 in zip(cells, activity_before, activity_after):
    color = "cyan" if a1 > 0.5 else "gray"
    axes[2].add_patch(plt.Circle((x, y), 14, fill=False, color=color, lw=1.2))
plt.tight_layout()
plt.show()

# %%
# 5. The response is reversible: with the light off, the reporter
#    returns to the nucleus within ~20 seconds.
advance(sim, 20)
activity_later = measure_activity(snap("mScarlet"), detect_nuclei(snap("miRFP")))
print(f"{sum(a > 0.5 for a in activity_later)} active 20 s later")

# %%
# That is the whole idea of feedback photomanipulation, minus the loop:
# measure, decide, act, measure again. The next examples close the loop
# (respond to what the cells do) and hand the choreography over to the
# microscope (useq-schema).
