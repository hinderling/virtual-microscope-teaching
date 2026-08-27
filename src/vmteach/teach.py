"""Teaching helpers: the entire course-facing API surface.

Design notes
------------
* ``load_microscope`` defaults to **stepped mode**: no background thread,
  simulated time advances only through :func:`advance`. Same seed + same
  loop = identical trajectories on every machine. This is what makes the
  course exercises reproducible.
* ``mode="realtime"`` starts the wall-clock RealtimeEngine (idle pause
  disabled so the sample never silently freezes while a learner reads
  instructions). Used only for the latency lesson at the end of the module.
* Stimulation is applied once per ``core.setSLMImage()`` call — the loop
  frequency IS the stimulation frequency. This mirrors pulsed optogenetic
  stimulation protocols and is central to the timing exercises.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np


def load_microscope(backend: str = "optogenetic", *, n_cells: int = 20,
                    seed: int = 0, mode: str = "stepped", warmup: bool = True,
                    **kwargs):
    """Create a virtual microscope and return ``(core, sim)``.

    Args:
        backend: Simulation backend name (the course uses ``"optogenetic"``).
        n_cells: Number of simulated cells.
        seed: Random seed — same seed, same experiment, on every machine.
        mode: ``"stepped"`` (default) — simulated time advances only via
            :func:`advance`; fully deterministic.
            ``"realtime"`` — the sample evolves in wall-clock time while
            your code runs, like on a real microscope.
        warmup: Pre-compile the simulation (numba JIT) and render one frame
            so the first ``snapImage()`` in the exercise is instant.
        **kwargs: Forwarded to the backend's ``create_sim``.

    Returns:
        core: :class:`pymmcore_plus...UniMMCore` — the microscope control
            object. The identical API controls real hardware.
        sim: The simulation handle (used by :func:`advance` and ``.reset()``).
    """
    if mode not in ("stepped", "realtime"):
        raise ValueError(f"mode must be 'stepped' or 'realtime', got {mode!r}")

    mod = importlib.import_module(f"vmteach.backends.{backend}")
    sim = mod.create_sim(n_cells=n_cells, seed=seed, **kwargs)

    from vmteach._init_standard import load_cfg

    cfg = Path(mod.__file__).parent / f"{backend}.cfg"
    core = load_cfg(
        sim, cfg,
        realtime=(mode == "realtime"),
        idle_timeout=0.0,  # teaching: never silently pause the sample
    )

    if warmup:
        # Pay the one-time numba JIT cost here (with a visible message)
        # instead of during the learner's first snap/advance.
        print("Preparing virtual microscope (one-time compilation) ...",
              end=" ", flush=True)
        if mode == "stepped":
            sim.step(0.0)
        sim.snap_frame()  # compiles the renderer path
        print("done.")

    return core, sim


def advance(sim, seconds: float = 1.0, dt: float = 0.05) -> None:
    """Advance simulated time deterministically (stepped mode).

    Replaces ``time.sleep()`` from real experiments: on hardware you *wait*
    for the sample to respond, on the deterministic virtual microscope you
    *advance* it. Calls ``sim.step(dt)`` repeatedly for ``seconds`` of
    simulated time.
    """
    n = max(1, round(seconds / dt))
    for _ in range(n):
        sim.step(dt)


def overlay(img: np.ndarray, mask: np.ndarray,
            color: tuple = (80, 140, 255), alpha: float = 0.4) -> np.ndarray:
    """Overlay a stimulation mask on a grayscale image as a colored wash.

    Returns an RGB uint8 image (inverted grayscale, matplotlib ``gray_r``
    convention) with ``mask > 0`` pixels tinted in ``color``.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        norm = (img.astype(np.float32) - lo) / (hi - lo) * 255
    else:
        norm = img.astype(np.float32)
    inv = 255 - norm
    rgb = np.stack([inv, inv, inv], axis=-1)
    m = mask > 0
    for c in range(3):
        rgb[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * color[c]
    return rgb.astype(np.uint8)


def letter_mask(char: str, shape: tuple = (512, 512),
                fill: float = 0.6, thickness: int = 40) -> np.ndarray:
    """Binary target image of a letter, centered, for the assembly exercise.

    Args:
        char: A single character (e.g. ``"L"``).
        shape: Output image shape ``(height, width)``.
        fill: Approximate fraction of the smaller image dimension the
            letter should span.
        thickness: Stroke thickness in pixels — keep it at least as wide
            as a cell so cells can cover the stroke.

    Returns:
        uint8 array, 255 inside the letter, 0 elsewhere.
    """
    if len(char) != 1:
        raise ValueError("letter_mask takes a single character")

    target_px = int(min(shape) * fill)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Find the font scale whose glyph height matches target_px
    scale = 1.0
    (w, h), _ = cv2.getTextSize(char, font, scale, thickness)
    scale = scale * target_px / max(h, 1)
    (w, h), baseline = cv2.getTextSize(char, font, scale, thickness)

    mask = np.zeros(shape, dtype=np.uint8)
    org = ((shape[1] - w) // 2, (shape[0] + h) // 2)  # centered
    cv2.putText(mask, char, org, font, scale, 255, thickness, cv2.LINE_8)
    return mask
