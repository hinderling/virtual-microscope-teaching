"""napari integration: interactive GUI control and results exploration.

Requires the ``gui`` extra:  pip install "virtual-microscope-teaching[gui]"

Two entry points:

* :func:`launch_gui` opens napari with the napari-micromanager widgets
  bound to the *virtual* microscope core. Every button in the GUI issues
  the same pymmcore API calls your scripts make: Snap ⇒ ``core.snapImage()``,
  the objective dropdown ⇒ ``core.setState("Objective", ...)``, and so on.
  Snaps executed by a running script appear live in the preview layer.

* :func:`show_mask` overlays a mask (a target shape, the stimulation
  pattern) as a labels layer, from any thread.

* :func:`show_results` loads a finished experiment (images, stimulation
  masks, tracks) into napari layers for interactive exploration.
"""

from __future__ import annotations

import numpy as np


def launch_gui(core, *, title: str = "Virtual Microscope",
               channel_layers: bool = False):
    """Open napari with Micro-Manager control widgets on the given core.

    Args:
        core: The core returned by :func:`vmteach.load_microscope` (or any
            ``CMMCorePlus``-compatible core, including a real microscope).
        channel_layers: Off (default): snaps go to napari-micromanager's
            single ``preview`` layer, as in the plain plugin. On: a
            convenience for scripts that snap several channels per cycle;
            every snap goes to a layer named after its channel
            (``miRFP``, ``mScarlet``, ...), so all of them stay visible, not
            only the last one. Live mode and MDAs always use
            napari-micromanager's own layers.

    Returns:
        The napari ``Viewer``. Call ``napari.run()`` afterwards when using
        this from a plain script (not needed in IPython/Jupyter with the
        Qt event loop active, e.g. after ``%gui qt``).
    """
    import napari
    from napari_micromanager.main_window import MainWindow

    viewer = napari.Viewer(title=title)
    _deliver_gui_slots_on_main_thread(core)
    with _lazy_console(viewer):
        widget = MainWindow(viewer, mmcore=core)
    viewer.window.add_dock_widget(widget, name="Micro-Manager", area="top")
    _CORES[id(viewer)] = core
    _keep_masks_on_top(viewer)
    if channel_layers:
        _snaps_to_channel_layers(viewer, core, widget)
    return viewer


_CORES: dict = {}          # id(viewer) -> core, for the pixel-size scale

# how each channel is shown; unknown channels get a grey layer
CHANNEL_COLORMAPS = {"phase-contrast": "gray", "miRFP": "magenta",
                     "mVenus": "yellow", "mScarlet": "red",
                     "CyanStim": "cyan"}


def _snaps_to_channel_layers(viewer, core, widget) -> None:
    """Show each snapped image in a layer named after its channel.

    The image is read in the thread that snapped it, right after the snap:
    napari-micromanager reads ``core.getImage()`` only once its queued
    event reaches the GUI thread, by which time a script may already have
    snapped the next channel, so only the last channel of a cycle ever
    appeared.
    """
    from superqt.utils import ensure_main_thread

    link = getattr(widget, "_core_link", None)
    if link is not None:            # replace the single preview for snaps
        core.events.imageSnapped.disconnect(link._image_snapped)

    @ensure_main_thread
    def show(img, channel):
        _set_layer(viewer, channel or "snap", img, kind="image",
                   colormap=CHANNEL_COLORMAPS.get(channel, "gray"),
                   blending=("translucent" if channel == "phase-contrast"
                             else "additive"))

    def on_snap(*_):
        if core.mda.is_running():   # MDA frames go to the MDA layers
            return
        try:
            channel = core.getCurrentConfig("Channel")
        except Exception:
            channel = ""
        show(core.getImage().copy(), channel)

    core.events.imageSnapped.connect(on_snap)


def _keep_masks_on_top(viewer) -> None:
    """Move mask layers (show_mask) above any image layer added later.

    Otherwise the ``preview`` layer, created on the first snap, or a new
    channel layer would cover a mask added before it.
    """
    from qtpy.QtCore import QTimer

    def lift(*_):
        masks = [l for l in viewer.layers if l.metadata.get("vmteach_mask")]
        top = len(viewer.layers) - len(masks)
        if all(viewer.layers.index(l) >= top for l in masks):
            return
        for l in masks:
            viewer.layers.move(viewer.layers.index(l), len(viewer.layers))

    # after the insertion has finished, not from inside its event
    viewer.layers.events.inserted.connect(lambda e: QTimer.singleShot(0, lift))


def show_mask(viewer, mask, name: str = "mask", *, color: str = "cyan",
              opacity: float = 0.35):
    """Overlay a binary mask (camera pixels) as a labels layer.

    Use it for a target shape, a region of interest or the current
    stimulation pattern. Calling it again with the same ``name`` updates the
    layer in place, and it may be called from any thread, e.g. from inside
    an experiment running with :func:`vmteach.run_experiment`.
    """
    from superqt.utils import ensure_main_thread

    labels = (np.asarray(mask) > 0).astype(np.uint8)
    ensure_main_thread(_set_layer)(viewer, name, labels, kind="labels",
                                   color=color, opacity=opacity)


def _set_layer(viewer, name, data, *, kind, **style):
    """Create or update a layer, scaled like napari-micromanager's layers."""
    core = _CORES.get(id(viewer))
    px = core.getPixelSizeUm() if core is not None else 0
    scale = (px, px) if px else (1.0, 1.0)
    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = data
        layer.scale = scale
        return layer
    if kind == "labels":
        from napari.utils.colormaps import DirectLabelColormap
        cmap = DirectLabelColormap(color_dict={None: "transparent",
                                               0: "transparent",
                                               1: style["color"]})
        layer = viewer.add_labels(data, name=name, scale=scale,
                                  opacity=style["opacity"], colormap=cmap)
    else:
        first = not any(l.metadata.get("vmteach") for l in viewer.layers)
        layer = viewer.add_image(data, name=name, scale=scale,
                                 colormap=style["colormap"],
                                 blending=style["blending"])
        layer.metadata["vmteach"] = True
        if first:
            viewer.reset_view()
        return layer
    layer.metadata["vmteach_mask"] = True
    return layer


_MAIN_THREAD_SIGNALS: set[int] = set()
_CONNECT_PATCHED = False


def _deliver_gui_slots_on_main_thread(core) -> None:
    """Deliver the core's events to Qt widgets on the GUI thread.

    vmteach uses pymmcore-plus's psygnal event backend, so events are
    delivered synchronously in the thread that emits them (analysis
    callbacks such as an MDA ``frameReady`` handler run on the acquisition
    thread, where they belong). napari-micromanager and pymmcore-widgets
    assume the Qt backend, which queues events to the GUI thread: with
    psygnal, a GUI-launched MDA ran their slots on the MDA worker thread
    ("QObject::startTimer: Timers cannot be started from another thread",
    "QObject::setParent: ... different thread").

    From now on, every slot that belongs to a Qt object and connects to
    this core's (or its MDA runner's) signals is delivered on the main
    thread: immediately when the event comes from the main thread (GUI
    clicks behave exactly as before), queued to the Qt event loop when it
    comes from another thread. Plain functions, such as your own analysis
    callbacks, keep running in the emitting thread.
    """
    from psygnal import SignalInstance
    from qtpy.QtCore import QObject

    global _CONNECT_PATCHED
    for group in (core.events, core.mda.events):
        for name in dir(group):
            sig = getattr(group, name, None)
            if isinstance(sig, SignalInstance):
                _MAIN_THREAD_SIGNALS.add(id(sig))
    _start_main_thread_queue()
    if _CONNECT_PATCHED:
        return
    original = SignalInstance.connect

    def connect(self, slot=None, *, thread=None, **kwargs):
        if (thread is None and slot is not None
                and id(self) in _MAIN_THREAD_SIGNALS
                and isinstance(getattr(slot, "__self__", None), QObject)):
            thread = "main"
        return original(self, slot, thread=thread, **kwargs)

    SignalInstance.connect = connect
    _CONNECT_PATCHED = True


_QUEUE_TIMER = None


def _start_main_thread_queue() -> None:
    """Deliver queued events on the Qt main thread (once per process).

    Like ``psygnal.qt.start_emitting_from_queue``, but an event queued for
    a widget that was deleted before delivery (napari-micromanager rebuilds
    its preset dropdowns, for example) is dropped instead of raising: a
    direct emission would silently skip the dead receiver too.
    """
    global _QUEUE_TIMER
    if _QUEUE_TIMER is not None:
        return
    from psygnal import EmitLoopError, emit_queued
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    def drain():
        while True:
            try:
                emit_queued()
                return
            except EmitLoopError as exc:
                if not isinstance(exc.__cause__, ReferenceError):
                    raise

    _QUEUE_TIMER = QTimer(QApplication.instance())
    _QUEUE_TIMER.timeout.connect(drain)
    _QUEUE_TIMER.start(0)


class _lazy_console:
    """Keep napari's console lazy while napari-micromanager starts up.

    napari-micromanager pushes its variables with
    ``getattr(qt_viewer, "console").push(...)``, which creates the napari
    console right away, i.e. an in-process IPython kernel. In a plain
    ``python`` prompt that kernel takes over ``__main__`` and the prompt
    (``In :``): the user's variables vanish and typed commands no longer
    reach their session. Route those pushes through napari's lazy
    ``viewer.update_console`` instead, so the console is only created when
    the user opens it. Drop once fixed upstream.
    """

    def __init__(self, viewer):
        self._viewer = viewer
        self._cls = type(viewer.window._qt_viewer)
        self._orig = self._cls.console

    def __enter__(self):
        viewer, orig = self._viewer, self._orig

        class _Queue:
            def push(self, variables):
                viewer.update_console(variables)

        def get(qt_viewer):
            if qt_viewer._console is None:
                return _Queue()
            return orig.fget(qt_viewer)

        self._cls.console = property(get, orig.fset)
        return self

    def __exit__(self, *exc):
        self._cls.console = self._orig
        return False


def show_results(images, masks=None, centroids=None, segmentations=None,
                 viewer=None, name: str = "experiment"):
    """Display a finished feedback experiment as napari layers.

    Args:
        images: sequence of 2D acquired frames → time-lapse image layer.
        masks: optional sequence of 2D stimulation masks (uint8/bool)
            → labels layer, so learners see *where* the light went.
        centroids: optional per-frame lists of (x, y) cell positions
            → tracks layer (Hungarian-linked, see vmteach.optogenetic.link_tracks).
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
        from vmteach.optogenetic import link_tracks
        tr = link_tracks(centroids)
        if len(tr):
            layer = viewer.add_tracks(tr, name=f"{name}: tracks",
                                      tail_length=60)
            layer.blending = "translucent"  # additive reads poorly here
    return viewer
