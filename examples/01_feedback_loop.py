# %%
# Closed-loop feedback photomanipulation on a virtual microscope
#
# Install (once): pip install virtual-microscope-teaching
# Runs locally, Python >= 3.10.

import numpy as np
import cv2
import matplotlib.pyplot as plt

from vmteach import load_microscope, advance, overlay

# %%
# Load the virtual microscope.
# `core` is a pymmcore-plus object; the same API controls real microscopes.
# `sim` is a handle to the simulation (used only to advance time and reset).
# The microscope starts in deterministic "stepped" mode: the sample only
# evolves when we call advance(), so everyone gets the same result.
core, sim = load_microscope("optogenetic", n_cells=20, seed=0)

print("Channels:", core.getAvailableConfigs("Channel"))

# %%
# ACQUIRE: snap a phase-contrast image, exactly like on real hardware
core.setConfig("Channel", "phase-contrast")
core.snapImage()
img = core.getImage()

plt.imshow(img, cmap="gray_r")
plt.title(f"{img.shape} {img.dtype}")

# %%
# ANALYZE: segment the cells with an Otsu threshold and find their centroids.
# Any segmentation works here. The loop does not care how the objects
# were found, only that it gets a list of positions.


def detect_cells(img, min_area=100):
    """Return a list of (cx, cy, area) for each detected cell."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cells.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]), area))
    return cells


cells = detect_cells(img)
print(f"Detected {len(cells)} cells")

plt.imshow(img, cmap="gray_r")
for cx, cy, _ in cells:
    plt.plot(cx, cy, "c+", ms=10)

# %%
# DECIDE: build a stimulation mask.
# Stimulated cells protrude toward the light, so a spot placed ABOVE a cell
# pulls it UPWARD. The mask is a plain uint8 image, white pixels = light on.


def build_steer_mask(cells, direction_px=-15, spot_radius=11, shape=(512, 512)):
    """One illumination spot per cell, offset by direction_px in y."""
    mask = np.zeros(shape, dtype=np.uint8)
    for cx, cy, _ in cells:
        ty = int(np.clip(cy + direction_px, 0, shape[0] - 1))
        cv2.circle(mask, (cx, ty), spot_radius, 255, -1)
    return mask


mask = build_steer_mask(cells)
plt.imshow(overlay(img, mask))
plt.title("Stimulation mask (blue): spots above each cell")

# %%
# ACTUATE, in two steps, exactly like on real hardware:
# 1. upload the pattern to the SLM. This alone does nothing, because the
#    SLM only *modulates* light that is not on yet
core.setSLMImage("SLM", mask)
# 2. engage the stimulation light path: switching to the CyanStim channel
#    turns on the stimulation LED. NOW the pattern is delivered and the
#    illuminated cells receive a protrusion + motility impulse
core.setConfig("Channel", "CyanStim")

# %%
# SEE THE LIGHT: on a real microscope the stimulation light is physically
# projected onto the sample, and you can image it. Switch to the CyanStim
# channel (stimulation LED + matching emission filter) and snap: the SLM
# pattern appears, with optics halo and noise, over a faint reflection of
# the cells. This is how you verify mask–sample alignment at the scope.
core.setConfig("Channel", "CyanStim")
core.snapImage()
stim_img = core.getImage()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(overlay(img, mask))
axes[0].set_title("planned (software overlay)")
axes[1].imshow(stim_img, cmap="gray_r")
axes[1].set_title("actual (CyanStim channel)")
for ax in axes:
    ax.axis("off")

core.setConfig("Channel", "phase-contrast")

# %%
# CLOSE THE LOOP: acquire → analyze → decide → actuate → let time pass → repeat.
# Note the light choreography each cycle: switching to phase-contrast for the
# acquisition turns the stimulation light OFF; after uploading the new mask,
# switching to CyanStim turns it back ON. Forget the switch and nothing
# happens, which is a classic debugging moment at a real microscope.
# advance(sim, seconds=1.0) advances the simulation deterministically;
# on real hardware this would simply be the interval between acquisitions.

n_cycles = 100
history = []

sim.reset()
for i in range(n_cycles):
    core.setConfig("Channel", "phase-contrast") # light off, imaging channel
    core.snapImage()
    img = core.getImage()                       # acquire
    cells = detect_cells(img)                   # analyze
    mask = build_steer_mask(cells)              # decide
    core.setSLMImage("SLM", mask)               # upload pattern
    core.setConfig("Channel", "CyanStim")       # light on: deliver
    advance(sim, seconds=1.0)                   # sample responds
    history.append(np.array([(cx, cy) for cx, cy, _ in cells]))

print("Mean y position: first frame "
      f"{history[0][:, 1].mean():.0f} -> last frame {history[-1][:, 1].mean():.0f}")
# Appreciate that the mean y decreased: the population moved up.

# %%
# PER-OBJECT DECISIONS: steer each cell differently based on a measurement.
# Cells in the left half of the field go UP, cells in the right half go DOWN.
# Only the DECIDE step changes; acquisition and actuation stay identical.


def build_split_mask(cells, offset_px=15, spot_radius=11, shape=(512, 512)):
    mask = np.zeros(shape, dtype=np.uint8)
    for cx, cy, _ in cells:
        dy = -offset_px if cx < shape[1] // 2 else +offset_px   # the decision
        ty = int(np.clip(cy + dy, 0, shape[0] - 1))
        cv2.circle(mask, (cx, ty), spot_radius, 255, -1)
    return mask


sim.reset()
start = None
for i in range(n_cycles):
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    img = core.getImage()
    cells = detect_cells(img)
    if start is None:
        start = {i: (cx, cy) for i, (cx, cy, _) in enumerate(cells)}
    core.setSLMImage("SLM", build_split_mask(cells))
    core.setConfig("Channel", "CyanStim")
    advance(sim, seconds=1.0)

plt.imshow(overlay(img, build_split_mask(cells)))
plt.axvline(256, color="w", ls="--")
plt.title("Left half steered up, right half steered down")

# %%
# Appreciate that a single scalar measurement (here: x position) is enough to
# give every cell its own treatment. Any measured feature works the same way:
# size, intensity, shape, biosensor activity. This is what makes feedback
# experiments selective in ways manual ROI drawing cannot be.

# %%
# EXPLORE TIMING (the most important parameter of any feedback experiment):
# Re-run the loop with advance(sim, seconds=5.0) instead of 1.0
#  - the same total simulated time now contains 5x fewer stimulations
#  - each mask acts on older information
# Appreciate that the steering becomes weaker or fails entirely.
#
# Optional (advanced): load the microscope in real-time mode, where the
# sample evolves in wall-clock time while your code runs:
#     core, sim = load_microscope("optogenetic", n_cells=20, seed=0,
#                                 mode="realtime")
# Now the duration of YOUR analysis code changes the experiment. Add
# time.sleep(2) inside the loop and watch the steering degrade. This is
# exactly the situation on a real microscope.
