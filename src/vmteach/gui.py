"""napari integration: interactive GUI control and results exploration.

Requires the ``gui`` extra:  pip install "virtual-microscope-teaching[gui]"

Two entry points:

* :func:`launch_gui` opens napari with the napari-micromanager widgets
  bound to the *virtual* microscope core. Every button in the GUI issues
  the same pymmcore API calls your scripts make: Snap ⇒ ``core.snapImage()``,
  the objective dropdown ⇒ ``core.setState("Objective", ...)``, and so on.
  Snaps executed by a running script appear live in the preview layer.

* :func:`show_results` loads a finished experiment (images, stimulation
  masks, tracks) into napari layers for interactive exploration.
"""

from __future__ import annotations

import numpy as np


def launch_gui(core, *, title: str = "Virtual Microscope"):
    """Open napari with Micro-Manager control widgets on the given core.

    Args:
        core: The core returned by :func:`vmteach.load_microscope` (or any
            ``CMMCorePlus``-compatible core, including a real microscope).

    Returns:
        The napari ``Viewer``. Call ``napari.run()`` afterwards when using
        this from a plain script (not needed in IPython/Jupyter with the
        Qt event loop active, e.g. after ``%gui qt``).
    """
    import napari
    from napari_micromanager.main_window import MainWindow

    viewer = napari.Viewer(title=title)
    widget = MainWindow(viewer, mmcore=core)
    viewer.window.add_dock_widget(widget, name="Micro-Manager", area="top")
    return viewer


def show_results(images, masks=None, centroids=None, segmentations=None,
                 viewer=None, name: str = "experiment"):
    """Display a finished feedback experiment as napari layers.

    Args:
        images: sequence of 2D acquired frames → time-lapse image layer.
        masks: optional sequence of 2D stimulation masks (uint8/bool)
            → labels layer, so learners see *where* the light went.
        centroids: optional per-frame lists of (x, y) cell positions
            → tracks layer (Hungarian-linked, see vmteach.link_tracks).
        segmentations: optional sequence of 2D label images → labels layer.
        viewer: existing napari Viewer to add to (default: create one).

    Returns:
        The napari ``Viewer``.
    """
    import napari

    if viewer is None:
        viewer = napari.Viewer(title=f"Results: {name}")

    viewer.add_image(np.stack(images), name=f"{name}: images",
                     colormap="gray_r")
    if segmentations is not None:
        viewer.add_labels(np.stack([s.astype(np.int32) for s in segmentations]),
                          name=f"{name}: segmentation", opacity=0.4)
    if masks is not None:
        stim = np.stack([(m > 0).astype(np.int32) for m in masks])
        layer = viewer.add_labels(stim, name=f"{name}: stimulation", opacity=0.6)
        try:  # tint the stimulation mask blue-ish
            layer.colormap = {1: "cyan"}
        except Exception:
            pass
    if centroids is not None:
        from vmteach.teach import link_tracks
        tr = link_tracks(centroids)
        if len(tr):
            layer = viewer.add_tracks(tr, name=f"{name}: tracks",
                                      tail_length=60)
            layer.blending = "translucent"  # additive reads poorly here
    return viewer
