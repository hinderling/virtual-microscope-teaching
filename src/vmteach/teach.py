"""Teaching helpers: the entire course-facing API surface.

Design notes
------------
* ``load_microscope`` defaults to **stepped mode**: no background thread,
  simulated time advances only through :func:`advance`. Same seed + same
  loop = identical trajectories on every machine — that is what makes the
  course exercises reproducible.
* ``mode="realtime"`` starts the wall-clock RealtimeEngine (idle pause
  disabled so the sample never silently freezes while a learner reads
  instructions). Used for the GUI activity and the latency lesson.
* Stimulation is applied when a mask is set (``core.setSLMImage``) and on
  each snap while that mask is displayed. The impulse sets (not adds) the
  cell velocity toward the light, so within one loop iteration this is
  effectively one stimulus — the feedback-loop frequency is the
  stimulation frequency.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_microscope(backend: str = "optogenetic", *, n_cells: int = 20,
                    seed: int = 0, mode: str = "stepped", warmup: bool = True,
                    **kwargs):
    """Create a virtual microscope and return ``(core, sim)``.

    Args:
        backend: Only ``"optogenetic"`` exists in this teaching package.
        n_cells: Number of simulated cells.
        seed: Random seed — same seed, same experiment, on every machine.
        mode: ``"stepped"`` (default) — simulated time advances only via
            :func:`advance`; fully deterministic.
            ``"realtime"`` — the sample evolves in wall-clock time while
            your code runs, like on a real microscope.
        warmup: Pre-compile the physics (numba, cached on disk after the
            first ever run) so the first snap in the exercise is instant.
        **kwargs: Forwarded to :class:`vmteach.sim.OptoCellSim`
            (``base_radius``, ``world_size``, ...).

    Returns:
        core: ``UniMMCore`` — the microscope control object. The identical
            pymmcore API controls real Micro-Manager hardware.
        sim: The simulation handle (for :func:`advance` and ``.reset()``).
    """
    if backend != "optogenetic":
        raise ValueError(
            f"backend {backend!r} not available — this teaching package "
            "ships only 'optogenetic' (see the full virtual-microscope "
            "repo for more)")
    if mode not in ("stepped", "realtime"):
        raise ValueError(f"mode must be 'stepped' or 'realtime', got {mode!r}")

    from pymmcore_plus.experimental.unicore import UniMMCore

    from vmteach.bridge import SimulationBridge, set_global_bridge
    from vmteach.sim import OptoCellSim

    sim = OptoCellSim(n_cells=n_cells, seed=seed, **kwargs)
    bridge = SimulationBridge(sim)
    set_global_bridge(bridge)

    core = UniMMCore()
    core.loadSystemConfiguration(str(Path(__file__).parent / "optogenetic.cfg"))
    _install_pixel_size(core, sim)

    if warmup:
        print("Preparing virtual microscope ...", end=" ", flush=True)
        sim.step(0.0)        # numba JIT (disk-cached after first ever run)
        sim.snap_frame()
        print("done.")

    if mode == "realtime":
        from vmteach.engine import RealtimeEngine
        engine = RealtimeEngine(sim, time_scale=1.0, tick_hz=20,
                                idle_timeout=0.0, bridge=bridge)
        engine.patch_snap_frame()
        engine.start()
        bridge._engine = engine

    return core, sim


def _install_pixel_size(core, sim) -> None:
    """getPixelSizeUm() reflecting the current objective (10x = 1 um/px)."""
    factors = {0: 1, 1: 2, 2: 4, 3: 8}

    def _pixel_size(cached: bool = False) -> float:
        try:
            state = core.getState("Objective")
        except Exception:
            return sim.world_pixel_size_um
        return sim.world_pixel_size_um / factors.get(state, 1)

    core.getPixelSizeUm = _pixel_size


def advance(sim, seconds: float = 1.0, dt: float = 0.05) -> None:
    """Advance simulated time deterministically (stepped mode).

    Replaces ``time.sleep()`` from real experiments: on hardware you *wait*
    for the sample to respond, on the deterministic virtual microscope you
    *advance* it.
    """
    n = max(1, round(seconds / dt))
    for _ in range(n):
        sim.step(dt)


def detect_nuclei(img: np.ndarray, min_area: int = 20) -> list:
    """Reference detector: nuclei centroids from a DAPI image.

    Nuclei are bright, compact, and — unlike cell bodies — never touch
    (cells collide before their nuclei can), so a plain Otsu threshold
    stays reliable even in crowded fields. Use this as the robust
    detection for feedback loops and tracking; write your own detector
    in the activities to understand what it does.

    Returns a list of ``(x, y)`` integer centroids.
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    return out


def link_tracks(detections, max_dist: float = 40.0,
                memory: int = 2) -> np.ndarray:
    """Link per-frame detections into tracks (Hungarian assignment).

    Globally optimal frame-to-frame matching (scipy
    ``linear_sum_assignment``) with distance gating: a detection farther
    than ``max_dist`` from every track starts a new track instead of
    producing a jumpy link. Tracks survive up to ``memory`` missed frames
    (detector dropouts, cells briefly merging).

    Args:
        detections: sequence over frames; each frame a sequence of (x, y).
        max_dist: gate — maximum linking distance in pixels per frame.
        memory: frames a lost track is kept alive for re-linking.

    Returns:
        Array of rows ``(track_id, t, y, x)`` — the napari Tracks format.
    """
    from scipy.optimize import linear_sum_assignment

    BIG = 1e9
    active: dict = {}          # tid -> (x, y, last_t)
    rows = []
    next_id = 0
    for t, pts in enumerate(detections):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        tids = list(active.keys())
        matched = set()
        if tids and len(pts):
            prev = np.array([active[i][:2] for i in tids])
            cost = np.linalg.norm(prev[:, None, :] - pts[None, :, :], axis=2)
            cost[cost > max_dist] = BIG
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] >= BIG:
                    continue
                tid = tids[r]
                active[tid] = (pts[c, 0], pts[c, 1], t)
                rows.append((tid, t, pts[c, 1], pts[c, 0]))
                matched.add(c)
        for c, (x, y) in enumerate(pts):
            if c in matched:
                continue
            active[next_id] = (x, y, t)
            rows.append((next_id, t, y, x))
            next_id += 1
        active = {i: v for i, v in active.items() if t - v[2] <= memory}
    return np.asarray(rows, dtype=float)


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
        thickness: Stroke thickness in pixels — keep it wider than a cell
            so cells fit on the stroke.

    Returns:
        uint8 array, 255 inside the letter, 0 elsewhere.
    """
    if len(char) != 1:
        raise ValueError("letter_mask takes a single character")

    target_px = int(min(shape) * fill)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    (w, h), _ = cv2.getTextSize(char, font, scale, thickness)
    scale = scale * target_px / max(h, 1)
    (w, h), _ = cv2.getTextSize(char, font, scale, thickness)

    mask = np.zeros(shape, dtype=np.uint8)
    org = ((shape[1] - w) // 2, (shape[0] + h) // 2)
    cv2.putText(mask, char, org, font, scale, 255, thickness, cv2.LINE_8)
    return mask
