"""napari integration: interactive GUI control and results exploration.

Requires the ``gui`` extra:  pip install "virtual-microscope-teaching[gui]"

Two entry points:

* :func:`launch_gui` — opens napari with the napari-micromanager widgets
  bound to the *virtual* microscope core. Every button in the GUI issues
  the same pymmcore API calls your scripts make: Snap ⇒ ``core.snapImage()``,
  the objective dropdown ⇒ ``core.setState("Objective", ...)``, and so on.
  Snaps executed by a running script appear live in the preview layer.

* :func:`show_results` — loads a finished experiment (images, stimulation
  masks, tracks) into napari layers for interactive exploration.
"""

from __future__ import annotations

import numpy as np


def launch_gui(core, *, title: str = "Virtual Microscope"):
    """Open napari with Micro-Manager control widgets on the given core.

    Args:
        core: The core returned by :func:`vmteach.load_microscope` (or any
            ``CMMCorePlus``-compatible core — including a real microscope).

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


def _link_tracks(centroids_per_frame, max_dist: float = 60.0) -> np.ndarray:
    """Greedy nearest-neighbour linking → napari tracks array (id, t, y, x).

    Simple by design (it is teaching code): each frame's detections are
    matched to the previous frame's track heads by distance.
    """
    tracks: list[list] = []          # rows: [track_id, t, y, x]
    heads: dict[int, tuple] = {}     # track_id -> (x, y)
    next_id = 0
    for t, pts in enumerate(centroids_per_frame):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)  # (x, y)
        used = set()
        new_heads = {}
        # match existing heads to nearest unused detection
        for tid, (hx, hy) in heads.items():
            if len(pts) == 0:
                continue
            d = np.hypot(pts[:, 0] - hx, pts[:, 1] - hy)
            for j in np.argsort(d):
                if j in used:
                    continue
                if d[j] > max_dist:
                    break
                used.add(j)
                new_heads[tid] = (pts[j, 0], pts[j, 1])
                tracks.append([tid, t, pts[j, 1], pts[j, 0]])
                break
        # unmatched detections start new tracks
        for j, (x, y) in enumerate(pts):
            if j in used:
                continue
            new_heads[next_id] = (x, y)
            tracks.append([next_id, t, y, x])
            next_id += 1
        heads = new_heads
    return np.asarray(tracks, dtype=float)


def show_results(images, masks=None, centroids=None, segmentations=None,
                 viewer=None, name: str = "experiment"):
    """Display a finished feedback experiment as napari layers.

    Args:
        images: sequence of 2D acquired frames → time-lapse image layer.
        masks: optional sequence of 2D stimulation masks (uint8/bool)
            → labels layer, so learners see *where* the light went.
        centroids: optional per-frame lists of (x, y) cell positions
            → tracks layer (nearest-neighbour linked).
        segmentations: optional sequence of 2D label images → labels layer.
        viewer: existing napari Viewer to add to (default: create one).

    Returns:
        The napari ``Viewer``.
    """
    import napari

    if viewer is None:
        viewer = napari.Viewer(title=f"Results — {name}")

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
        tr = _link_tracks(centroids)
        if len(tr):
            viewer.add_tracks(tr, name=f"{name}: tracks", tail_length=60)
    return viewer
