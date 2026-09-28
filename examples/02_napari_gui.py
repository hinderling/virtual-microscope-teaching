# %%
# The GUI and the script are the same microscope
#
# Install (once): pip install "virtual-microscope-teaching[gui]"
# Run this locally (napari needs a display). In Jupyter, run `%gui qt` first.
#
# napari-micromanager gives you the familiar microscope control GUI:
# Snap and Live buttons, channel and objective dropdowns, exposure. The
# only difference is that the "microscope" behind it is simulated. Every
# widget issues the same pymmcore API calls your scripts make. That
# equivalence is the point of this example.

from vmteach import load_microscope
from vmteach.gui import launch_gui

# Real-time mode: the sample keeps moving in wall-clock time, so the
# Live view looks like a real microscope with live cells.
core, sim = load_microscope("optogenetic", n_cells=20, seed=0, mode="realtime")

viewer = launch_gui(core)

# %%
# Try in the GUI (see docs/gui_walkthrough.md for annotated screenshots):
#  1. Press the Snap button           -> a "preview" layer appears
#  2. Press Live                      -> cells jiggle and crawl in real time
#  3. Switch Objectives: 10x -> 40x   -> smaller field of view (same 512x512
#                                        image, smaller pixels), same sample
#  4. Change Channel: phase-contrast -> miRFP (nuclei) / mVenus (membrane)
#  5. Change Exposure                 -> brighter / noisier image
#  6. Camera > Binning (property browser): 1 / 2 / 4 -> smaller, brighter
#                                        frame (lower the exposure to match)

# %%
# Now do EXACTLY the same things from code, and watch the GUI react:
core.snapImage()                       # = pressing "Snap"  (preview updates!)
core.setStateLabel("Objective", "20x")  # = choosing "20x" in the dropdown
core.setConfig("Channel", "miRFP")     # = choosing "miRFP" in the dropdown
core.setExposure(50.0)                 # = typing 50 in the Exposure box
core.snapImage()

# The dropdowns in the GUI now show 20x / miRFP / 50 ms: the GUI is not
# "another program", it is a viewer onto the same core object your script
# controls. On a real microscope this is identical: napari-micromanager
# in front, your feedback script behind, one shared core.

# %%
# Back to defaults for the next example
core.setState("Objective", 1)          # 10x
core.setConfig("Channel", "phase-contrast")
core.setExposure(50.0)

# %%
# While a feedback loop runs, every core.snapImage() lands in the same
# preview layer, so you can watch your smart acquisition script "click"
# through the experiment live. The other examples open this GUI themselves
# and run their loops with vmteach.run_experiment, which keeps the viewer
# live while the loop runs in the background.
