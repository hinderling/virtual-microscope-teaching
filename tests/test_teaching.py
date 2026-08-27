"""Smoke tests: the course exercises must work, fast, and reproducibly."""

import cv2
import numpy as np
import pytest

from vmteach import load_microscope, advance, overlay, letter_mask


@pytest.fixture(scope="module")
def scope():
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0, warmup=False)
    return core, sim


def detect_cells(img, min_area=100):
    _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cts:
        if cv2.contourArea(c) < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    return out


def steer_mask(cells, dy=-15, r=11, shape=(512, 512)):
    mask = np.zeros(shape, np.uint8)
    for cx, cy in cells:
        cv2.circle(mask, (cx, int(np.clip(cy + dy, 0, shape[0] - 1))), r, 255, -1)
    return mask


def test_snap(scope):
    core, sim = scope
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    img = core.getImage()
    assert img.shape == (512, 512) and img.dtype == np.uint8


def test_device_surface(scope):
    """The devices the course activities rely on must exist."""
    core, sim = scope
    assert set(core.getAvailableConfigs("Channel")) == {
        "phase-contrast", "DAPI", "membrane"}
    assert core.getStateLabels("Objective") == ("10x", "20x", "40x", "100x")
    assert "SLM" in core.getLoadedDevices()
    core.setXYPosition(10.0, 20.0)  # stage moves without error


def test_stepped_mode_is_frozen_without_advance(scope):
    """In stepped mode, snapping must not advance simulated time."""
    core, sim = scope
    sim.reset()
    p0 = sim.centers.copy()
    for _ in range(5):
        core.snapImage()
    assert np.array_equal(p0, sim.centers)


def test_feedback_loop_steers_cells(scope):
    """The canonical course experiment: closed-loop steering moves the
    population upward; an identical rerun reproduces it bit-exactly."""
    core, sim = scope
    core.setConfig("Channel", "phase-contrast")

    def run(n=40):
        sim.reset()
        ys = []
        for _ in range(n):
            core.snapImage()
            img = core.getImage()
            cells = detect_cells(img)
            core.setSLMImage("SLM", steer_mask(cells))
            advance(sim, seconds=1.0)
            ys.append(np.mean([cy for _, cy in cells]))
        return ys, img

    ys1, img1 = run()
    assert ys1[-1] < ys1[0] - 30, "steering failed: population did not move up"

    ys2, img2 = run()
    assert ys1 == ys2, "rerun after reset() is not deterministic"
    assert np.array_equal(img1, img2), "images not bit-identical after reset()"


def test_reset_clears_leftover_slm_mask(scope):
    core, sim = scope
    core.setSLMImage("SLM", np.full((512, 512), 255, np.uint8))
    sim.reset()
    p0 = sim.centers.copy()
    core.snapImage()  # must not stimulate with the stale mask
    advance(sim, 0.05, dt=0.05)
    core.snapImage()
    # cells moved only by brownian noise; a full-field stimulus would
    # protrude every cell — check velocities are not uniformly boosted
    assert not getattr(sim._cells[0], "is_stimulated", False)
    assert np.allclose(p0, sim.centers, atol=5.0)


def test_letter_mask():
    for ch in "ANEUBIS":
        m = letter_mask(ch)
        assert m.shape == (512, 512) and m.dtype == np.uint8
        assert (m > 0).sum() > 1000
    with pytest.raises(ValueError):
        letter_mask("AB")


def test_overlay():
    img = np.zeros((64, 64), np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[10:20, 10:20] = 255
    rgb = overlay(img, mask)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert not np.array_equal(rgb[15, 15], rgb[40, 40])  # tinted vs untinted


def test_realtime_mode_advances_on_its_own():
    import time
    core, sim = load_microscope("optogenetic", n_cells=5, seed=1,
                                mode="realtime", warmup=False)
    core.snapImage()
    p0 = sim.centers.copy()
    time.sleep(1.0)
    core.snapImage()
    assert not np.array_equal(p0, sim.centers), \
        "realtime mode: sample should evolve in wall-clock time"
