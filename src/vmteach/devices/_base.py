"""Shared device plumbing."""

import threading


class ReentrantLockMixin:
    """Give a unicore device a re-entrant lock.

    ``UniMMCore.setProperty`` holds the device lock while the device runs
    its setter, and unicore state devices emit ``propertyChanged`` from
    inside that setter. GUI listeners (e.g. the napari-micromanager
    channel presets widget) react to that signal by calling
    ``core.setConfig(...)`` on the very same device, on the same thread.
    With unicore's default non-reentrant ``threading.Lock`` that nested
    call deadlocks the Qt main thread (the "filter wheel hang").

    Real MMCore device locks are recursive, so a re-entrant lock restores
    the behaviour of hardware devices. Mix in *before* the unicore base.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()
