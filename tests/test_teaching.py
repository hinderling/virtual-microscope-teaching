"""Smoke tests: the course exercises must work, fast, and reproducibly."""

import cv2
import numpy as np
import pytest

from vmteach import load_microscope, advance, overlay, letter_mask


@pytest.fixture(scope="module")
def scope():
    core, sim = load_microscope("optogenetic", n_cells=20, seed=0, warmup=False)
    return core, sim


@pytest.fixture(autouse=True)
def _restore_global_bridge():
    """Tests that load a second microscope replace the global bridge the
    module-scoped core's devices route through; put it back afterwards."""
    import vmteach.bridge as b
    saved = b.GLOBAL_BRIDGE
    yield
    if saved is not None:
        b.set_global_bridge(saved)


def in_view(sim, margin=0):
    """Indices of cells whose centre lies in the current field of view."""
    off, fov = sim.view_origin, sim.fov_um
    x = sim.centers[:, 0] - off[0]
    y = sim.centers[:, 1] - off[1]
    return np.where((x > margin) & (x < fov - margin)
                    & (y > margin) & (y < fov - margin))[0]


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
        "phase-contrast", "miRFP", "mVenus", "mScarlet", "CyanStim"}
    assert core.getStateLabels("Objective") == ("4x", "10x", "20x", "40x", "60x")
    assert "SLM" in core.getLoadedDevices()
    core.setXYPosition(10.0, 20.0)  # stage moves without error
    core.setXYPosition(0.0, 0.0)
    # startup state comes from the config file (System/Startup preset)
    assert core.getStateLabel("Objective") == "10x"
    assert core.getCurrentConfig("Channel") == "phase-contrast"
    assert int(core.getProperty("Camera", "Binning")) == 1


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
        watched = in_view(sim, margin=40)     # cells the loop can act on
        y_prev = sim.centers[:, 1].copy()
        total_dy = np.zeros(len(y_prev))
        for _ in range(n):
            core.setConfig("Channel", "miRFP")             # light OFF, acquire
            core.snapImage()
            img = core.getImage()
            cells = detect_cells(img, min_area=20)
            core.setSLMImage("SLM", steer_mask(cells))    # upload pattern
            core.setConfig("Channel", "CyanStim")         # light ON: deliver
            advance(sim, seconds=1.0)
            total_dy += sim.centers[:, 1] - y_prev
            y_prev = sim.centers[:, 1].copy()
        return total_dy[watched], img

    dy1, img1 = run()
    assert dy1.mean() < -25, \
        f"steering failed: mean dy {dy1.mean():.1f} (should be << 0)"

    dy2, img2 = run()
    assert np.array_equal(dy1, dy2), "rerun after reset() is not deterministic"
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
    # protrude every cell; check velocities are not uniformly boosted
    assert not getattr(sim._cells[0], "is_stimulated", False)
    assert np.allclose(p0, sim.centers, atol=5.0)


def test_stimulation_gated_on_light_path(scope):
    """Uploading a mask alone must NOT stimulate; engaging the CyanStim
    light path must; switching away must clear the stimulation."""
    core, sim = scope
    sim.reset()
    core.setConfig("Channel", "phase-contrast")
    mask = np.full((512, 512), 255, np.uint8)
    core.setSLMImage("SLM", mask)
    assert not any(c.is_stimulated for c in sim._cells), \
        "mask upload alone must not stimulate"
    core.setConfig("Channel", "CyanStim")
    # only cells inside the illuminated field of view can receive light
    off, fov = sim.view_origin, sim.fov_um
    in_view = [c for c in sim._cells
               if off[0] + 30 < c.center[0] < off[0] + fov - 30
               and off[1] + 30 < c.center[1] < off[1] + fov - 30]
    assert in_view and all(c.is_stimulated for c in in_view), \
        "engaging the light path must deliver the pattern to in-view cells"
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    assert not any(c.is_stimulated for c in sim._cells), \
        "light off must clear stimulation"


def test_cyanstim_images_projected_light(scope):
    """The CyanStim channel shows the SLM pattern, bright inside the spots."""
    core, sim = scope
    sim.reset()
    mask = np.zeros((512, 512), np.uint8)
    cv2.circle(mask, (256, 200), 30, 255, -1)
    core.setSLMImage("SLM", mask)
    core.setConfig("Channel", "CyanStim")
    core.snapImage()
    img = core.getImage()
    core.setConfig("Channel", "phase-contrast")
    assert img[mask > 0].mean() > img[mask == 0].mean() + 100, \
        "projected light not visible in CyanStim channel"


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
    moved = not np.array_equal(p0, sim.centers)
    # stop the background engine so no thread lingers at interpreter exit
    from vmteach.bridge import GLOBAL_BRIDGE
    if GLOBAL_BRIDGE is not None and GLOBAL_BRIDGE._engine is not None:
        GLOBAL_BRIDGE._engine.stop()
    assert moved, "realtime mode: sample should evolve in wall-clock time"


def test_event_driven_mda_queue():
    """The advanced activity: queue-fed MDA with analysis-driven events."""
    import time
    from queue import Queue
    from useq import MDAEvent

    core, sim = load_microscope("optogenetic", n_cells=10, seed=3,
                                mode="realtime", warmup=False)
    STOP = object()
    q = Queue()
    frames = []

    def on_frame(img, event):
        t = event.index.get("t", 0)
        ch = event.channel.config if event.channel else None
        frames.append((t, ch))
        if ch == "miRFP":
            cells = detect_cells(img, min_area=20)
            # stimulation = one declared event carrying the light path AND
            # the pattern; no manual setSLMImage or setConfig needed
            q.put(MDAEvent(index={"t": t},
                           channel={"config": "CyanStim", "group": "Channel"},
                           slm_image=steer_mask(cells)))
        elif t + 1 >= 3:
            q.put(STOP)
        else:
            q.put(MDAEvent(index={"t": t + 1},
                           channel={"config": "miRFP", "group": "Channel"},
                           min_start_time=(t + 1) * 0.2))

    core.mda.events.frameReady.connect(on_frame)
    core.run_mda(iter(q.get, STOP))
    q.put(MDAEvent(index={"t": 0},
                   channel={"config": "miRFP", "group": "Channel"}))
    deadline = time.time() + 15
    while core.mda.is_running() and time.time() < deadline:
        time.sleep(0.05)
    core.mda.events.frameReady.disconnect(on_frame)
    from vmteach.bridge import GLOBAL_BRIDGE
    if GLOBAL_BRIDGE is not None and GLOBAL_BRIDGE._engine is not None:
        GLOBAL_BRIDGE._engine.stop()
    assert not core.mda.is_running(), "MDA did not finish"
    expected = [(0, "miRFP"), (0, "CyanStim"), (1, "miRFP"), (1, "CyanStim"),
                (2, "miRFP"), (2, "CyanStim")]
    assert frames == expected, f"frames received: {frames}"


def test_cells_stay_inside_their_wells(scope):
    """The well wall is a hard boundary: after a long run no cell pokes
    through it, and no cell changes well."""
    core, sim = scope
    sim.reset()
    well0 = sim.cell_well.copy()
    advance(sim, seconds=20.0)
    for c, r, w in zip(sim.centers, sim.radii, sim.cell_well):
        assert sim.well_distance(c[0], c[1], w) + r.max() <= 1.0
    assert np.array_equal(well0, sim.cell_well)
    # both wells populated equally
    assert sim.n_wells == 2 and sim.cells_per_well * 2 == sim.n_cells


def test_well_wall_visible_in_phase_only(scope):
    """At the well edge, phase contrast shows plastic and a bright rim;
    fluorescence shows only background (the cell-free zone)."""
    core, sim = scope
    sim.reset()
    core.setStateLabel("Objective", "10x")
    core.setXYPosition(sim.well_half + 20, 0.0)     # wall runs mid-frame
    core.setConfig("Channel", "phase-contrast")
    core.snapImage()
    ph = core.getImage()
    core.setConfig("Channel", "miRFP")
    core.snapImage()
    fl = core.getImage()
    core.setXYPosition(0.0, 0.0)
    core.setConfig("Channel", "phase-contrast")
    inside, outside = ph[:, :200], ph[:, 300:]
    assert np.median(outside) < np.median(inside) - 20, "plastic not darker"
    profile = np.median(ph, axis=0)                  # column profile
    assert profile[200:300].max() > np.median(inside) + 12, "no bright well edge"
    assert fl[:, 300:].max() < 60, "wall visible in fluorescence"


# ── realism: camera, objectives, binning, SLM at magnification ───────────


def test_objective_changes_pixel_size_not_image_size(scope):
    """A real camera has a fixed sensor: objectives change the pixel size
    (from the .cfg pixel-size presets) and the field of view, never the
    image dimensions."""
    core, sim = scope
    expected = {"4x": 2.5, "10x": 1.0, "20x": 0.5, "40x": 0.25, "60x": 0.1667}
    for label, px in expected.items():
        core.setStateLabel("Objective", label)
        core.snapImage()
        assert core.getImage().shape == (512, 512)
        assert core.getPixelSizeUm() == pytest.approx(px, rel=1e-3)
        assert sim.fov_um == pytest.approx(512 * px, rel=1e-3)
    core.setStateLabel("Objective", "10x")


def test_binning_presets_shrink_and_brighten(scope):
    core, sim = scope
    core.setConfig("Channel", "miRFP")
    core.setExposure(5.0)                     # keep 4x4 out of saturation
    assert tuple(core.getAllowedPropertyValues("Camera", "Binning")) == ("1", "2", "4")
    means = {}
    for b in (1, 2, 4):
        core.setProperty("Camera", "Binning", b)
        core.snapImage()
        img = core.getImage()
        assert img.shape == (512 // b, 512 // b)
        assert core.getPixelSizeUm() == pytest.approx(1.0 * b)
        means[b] = img[img > 60].mean() if (img > 60).any() else 0.0
    core.setProperty("Camera", "Binning", 1)
    core.setExposure(50.0)
    core.setConfig("Channel", "phase-contrast")
    assert means[2] > 2 * means[1] and means[4] > 2 * means[2], means
    with pytest.raises(Exception):
        core.setProperty("Camera", "Binning", 3)


def test_phase_contrast_is_neutral_gray(scope):
    """Background mid-gray, cell bodies slightly darker, bright halo."""
    core, sim = scope
    sim.reset()
    core.setConfig("Channel", "phase-contrast")
    core.setStateLabel("Objective", "10x")
    core.snapImage()
    img = core.getImage()
    bg = np.median(img)
    assert 100 < bg < 150, f"background {bg} is not neutral gray"
    assert (img > bg + 25).mean() > 0.005, "no bright halos"
    assert (img < bg - 8).mean() > 0.02, "no darker cell bodies"


def test_slm_follows_objective_magnification(scope):
    """The SLM maps 1:1 onto the sensor at every objective, so a spot drawn
    on a cell's membrane in a 40x image stimulates that cell."""
    core, sim = scope
    sim.reset()
    core.setStateLabel("Objective", "40x")
    core.setConfig("Channel", "miRFP")
    core.snapImage()
    from vmteach import detect_nuclei
    # keep border-clipped nuclei: at 40x most of the few visible nuclei
    # touch the frame edge, and a clipped centroid still lands on the cell
    nuclei = detect_nuclei(core.getImage(), exclude_border=False)
    assert nuclei, "no nuclei in the 40x field"
    cx, cy = nuclei[0]
    r_px = 20 / core.getPixelSizeUm()               # ~cell radius in px
    mask = np.zeros((512, 512), np.uint8)
    cv2.circle(mask, (cx, int(cy - r_px)), int(r_px / 2), 255, -1)
    core.setSLMImage("SLM", mask)
    core.setConfig("Channel", "CyanStim")
    assert any(c.is_stimulated for c in sim._cells), \
        "spot on a 40x membrane did not stimulate"
    # the projected light is imaged where the mask is, at 40x too
    core.snapImage()
    img = core.getImage()
    assert img[mask > 0].mean() > img[mask == 0].mean() + 100
    core.setConfig("Channel", "phase-contrast")
    core.setStateLabel("Objective", "10x")


def test_multi_position_fields_are_distinct(scope):
    """The world holds 4 x 4 fields at 10x; stage moves reach new cells."""
    core, sim = scope
    sim.reset()
    assert sim.well_size >= 2048 and sim.cells_per_well >= 16 * 10
    core.setConfig("Channel", "miRFP")
    seen = []
    wx, wy = sim.well_positions[1]                 # second well
    for x, y in [(0, 0), (512, 0), (0, 512), (-512, -512), (wx, wy)]:
        core.setXYPosition(float(x), float(y))
        core.snapImage()
        seen.append(core.getImage())
    assert (seen[-1] > 100).sum() > 500, "no cells in the second well"
    core.setXYPosition(0.0, 0.0)
    core.setConfig("Channel", "phase-contrast")
    for a, b in zip(seen, seen[1:]):
        assert not np.array_equal(a, b)


def test_stage_travel_is_limited(scope):
    """Moves beyond the travel range stop at the limit (soft limits)."""
    core, sim = scope
    assert sim.stage_limits == ((-1016.0, 3320.0), (-1000.0, 1000.0))
    core.setXYPosition(5000.0, -3000.0)
    assert core.getXYPosition() == (3320.0, -1000.0)
    assert tuple(sim.stage) == (3320.0, -1000.0)
    core.setXYPosition(2304.0, 0.0)             # well B centre is reachable
    assert core.getXYPosition() == (2304.0, 0.0)
    core.setXYPosition(0.0, 0.0)


def test_property_change_listener_may_call_setconfig(scope):
    """Regression for the filter-wheel hang: a propertyChanged listener
    (e.g. the napari-micromanager channel presets widget) that calls
    setConfig on the same device must not deadlock on the device lock."""
    import threading
    core, sim = scope

    def snap_back(dev, prop, value):
        if dev == "Filter Wheel" and prop == "Label":
            core.setConfig("Channel", "phase-contrast")

    core.events.propertyChanged.connect(snap_back)
    done = threading.Event()

    def worker():
        core.setProperty("Filter Wheel", "Label", "mVenus(515/528)")
        done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    finished = done.wait(timeout=5.0)
    core.events.propertyChanged.disconnect(snap_back)
    assert finished, "setProperty deadlocked while a listener called setConfig"
    core.setConfig("Channel", "phase-contrast")


def test_pixel_size_affine_follows_objective_and_binning(scope):
    """getPixelSizeAffine is resolved in Python: the C++ core never matched
    the preset and aborts on Windows if asked (issue #1). Every MDA reads it
    through pymmcore-plus's summary metadata."""
    core, _ = scope
    cam = core.getCameraDevice()
    for obj, px in (("4x", 2.5), ("10x", 1.0), ("60x", 0.1667)):
        core.setStateLabel("Objective", obj)
        for b in (1, 2):
            core.setProperty(cam, "Binning", b)
            assert core.getPixelSizeAffine() == pytest.approx((px * b, 0, 0, 0, px * b, 0), rel=1e-3)
            assert core.getPixelSizeUm() == pytest.approx(px * b, rel=1e-3)
    core.setProperty(cam, "Binning", 1)
    core.setStateLabel("Objective", "10x")


def test_mda_summary_metadata_builds(scope):
    """The metadata pymmcore-plus attaches to every MDA sequence, which is
    where getPixelSizeAffine used to abort the process."""
    from pymmcore_plus.metadata.functions import summary_metadata

    core, _ = scope
    core.setStateLabel("Objective", "60x")   # cached reads must not see the old preset
    core.setStateLabel("Objective", "10x")
    core.setProperty(core.getCameraDevice(), "Binning", 1)
    meta = summary_metadata(core)
    info = meta["image_infos"][0]
    assert info["pixel_size_um"] == pytest.approx(1.0)
    assert info["pixel_size_config_name"] == "Res10x"


def test_ktr_reporter_shows_activation_and_reverses(scope):
    """The ERK-KTR channel (mScarlet) makes one stimulation pulse visible in
    a single snapshot: targeted cells flip to active (dark nucleus) within
    5 s and reverse within 20 s. No timelapse, no tracking."""
    import numpy as np

    from vmteach import advance, detect_nuclei, measure_activity

    core, sim = scope
    sim.reset()

    def snap(ch):
        core.setConfig("Channel", ch)
        core.snapImage()
        return core.getImage()

    cells = detect_nuclei(snap("miRFP"))
    assert len(cells) >= 5
    before = measure_activity(snap("mScarlet"), cells)
    assert all(a < 0.2 for a in before), before

    # stimulate only the left half of the field
    mask = np.zeros((512, 512), np.uint8)
    mask[:, :256] = 255
    core.setSLMImage(mask)
    snap("CyanStim")                    # gated delivery of one pulse
    core.setSLMImage(np.zeros((512, 512), np.uint8))
    advance(sim, 5)                     # rise time

    cells1 = detect_nuclei(snap("miRFP"))
    act = measure_activity(snap("mScarlet"), cells1)
    left = [a for (x, _), a in zip(cells1, act) if x < 236]
    right = [a for (x, _), a in zip(cells1, act) if x >= 276]
    assert left and all(a > 0.8 for a in left), left
    assert right and all(a < 0.2 for a in right), right

    advance(sim, 20)                    # full reversal
    cells2 = detect_nuclei(snap("miRFP"))
    after = measure_activity(snap("mScarlet"), cells2)
    assert all(a < 0.2 for a in after), after
    core.setConfig("Channel", "phase-contrast")


def test_well_grid_layouts():
    """n_wells accepts an (nx, ny) grid; geometry and limits follow."""
    from vmteach.sim import OptoCellSim

    sim = OptoCellSim(n_cells=2, n_wells=(2, 2), seed=0)
    assert sim.n_wells == 4 and len(sim.wells) == 4
    assert sim.width == sim.height              # square 2x2 plate
    (x0, x1), (y0, y1) = sim.stage_limits
    assert y1 > sim.well_size                   # travel reaches the second row
    # row-major: well 1 right of well 0, well 2 below well 0
    assert sim.wells[1, 0] > sim.wells[0, 0]
    assert sim.wells[1, 1] == sim.wells[0, 1]
    assert sim.wells[2, 1] > sim.wells[0, 1]

    row = OptoCellSim(n_cells=2, n_wells=2, seed=0)   # old int form
    assert row.n_wells == 2 and row.height < row.width
