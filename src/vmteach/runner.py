"""Run an experiment without blocking the notebook or the GUI.

A feedback loop written as a plain ``for`` loop blocks whatever runs it:
in a notebook, the kernel (and with it the live napari viewer) freezes
until the loop ends. :func:`run_experiment` runs the loop on a background
thread instead and returns immediately, so napari-micromanager keeps
showing every snap, channel switch and stimulation pattern while the
experiment runs, and the notebook stays usable (check progress, stop
early).

Nothing here is specific to the simulator: it only calls your function,
so the same code drives a real microscope.

    def experiment(run):
        for cycle in range(60):
            if run.stop_requested:
                break
            ...                       # acquire, analyze, decide, actuate
            run.sleep(1.0)            # wait; returns early on run.stop()

    run = run_experiment(experiment)
    ...
    run.stop()                        # optional: end early
    run.wait()                        # block until done, re-raise errors
"""

from __future__ import annotations

import threading
import time
import traceback


class Run:
    """Handle of an experiment running in the background."""

    def __init__(self, name: str):
        self.name = name
        self.result = None
        self.error: BaseException | None = None
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """True while the experiment function has not returned."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def stop_requested(self) -> bool:
        """True once :meth:`stop` was called; check it in your loop."""
        return self._stop.is_set()

    def stop(self) -> None:
        """Ask the experiment to end (at its next check or sleep)."""
        self._stop.set()

    def sleep(self, seconds: float) -> bool:
        """Wait ``seconds``; return early (and True) if a stop is requested."""
        return self._stop.wait(seconds)

    def wait(self, timeout: float | None = None):
        """Block until the experiment ends; return its result.

        Re-raises an exception raised inside the experiment. With a Qt GUI
        (napari) open, the GUI stays live while waiting, also in a plain
        Python script where no other event loop runs.
        """
        if self._thread is not None:
            app = _qt_app()
            if app is None:
                self._thread.join(timeout)
            else:
                end = None if timeout is None else time.monotonic() + timeout
                while self._thread.is_alive() and (
                        end is None or time.monotonic() < end):
                    app.processEvents()
                    self._thread.join(0.02)
        if self.error is not None:
            raise self.error
        return self.result

    def __repr__(self) -> str:
        state = ("running" if self.running else
                 "failed" if self.error is not None else "finished")
        return (f"<Run {self.name!r}: {state}, "
                f"{time.monotonic() - self.started:.0f} s>")


def _qt_app():
    """The running Qt application, if called from its (main) thread."""
    if threading.current_thread() is not threading.main_thread():
        return None
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:
        return None
    return QApplication.instance()


def run_experiment(fn, *args, **kwargs) -> Run:
    """Start ``fn(run, *args, **kwargs)`` on a background thread.

    Returns a :class:`Run` handle immediately. ``fn`` receives the handle
    as its first argument: use ``run.sleep(s)`` for the waits in your loop
    and check ``run.stop_requested`` to support stopping early. Errors are
    printed as they happen and re-raised by ``run.wait()``.
    """
    run = Run(getattr(fn, "__name__", "experiment"))

    def target():
        try:
            run.result = fn(run, *args, **kwargs)
        except BaseException as exc:          # surface it, don't lose it
            run.error = exc
            traceback.print_exc()

    run._thread = threading.Thread(target=target, daemon=True,
                                   name=f"vmteach-{run.name}")
    run._thread.start()
    return run
