"""
Debris Overlay — Floating artifacts, out-of-focus particles, dust spots.

Adds realistic imaging artifacts to rendered tissue images:
  - Bright debris particles (floating matter in media)
  - Dark spots (dust on sensor or optics)
  - Out-of-focus blobs (defocused particles above/below focal plane)

Applied as a post-rendering step, after tissue rendering but before
the optical pipeline (or after, depending on use case).

Usage:
    from debris_overlay import add_debris

    img = add_debris(img, density=0.001, rng_seed=42)

    # Or with more control:
    img = add_debris(
        img, density=0.002,
        bright_fraction=0.6,  # 60% bright, 40% dark
        size_range=(2, 15),   # particle radius 2-15px
        defocus_fraction=0.3, # 30% of particles are defocused
        rng_seed=42,
    )
"""

import numpy as np
import cv2


def add_debris(
    img: np.ndarray,
    density: float = 0.0002,
    bright_fraction: float = 0.6,
    size_range: tuple = (2, 12),
    defocus_fraction: float = 0.3,
    intensity_range: tuple = (30, 200),
    rng_seed: int | None = None,
) -> np.ndarray:
    """Add debris particles to an image.

    Args:
        img: Input image (HxWx3 or HxW, uint8)
        density: Particles per pixel (0.001 = ~260 particles in 512x512)
        bright_fraction: Fraction of particles that are bright (vs dark)
        size_range: (min_radius, max_radius) for particles
        defocus_fraction: Fraction of particles rendered as defocused blobs
        intensity_range: (min, max) intensity for bright particles
        rng_seed: Random seed for reproducibility

    Returns:
        Image with debris added (same shape as input)
    """
    rng = np.random.default_rng(rng_seed)
    h, w = img.shape[:2]
    is_color = img.ndim == 3

    # Number of particles
    n_particles = max(1, int(h * w * density))

    # Work on a copy
    result = img.copy()

    # Generate particle properties
    xs = rng.uniform(0, w, n_particles)
    ys = rng.uniform(0, h, n_particles)
    radii = rng.uniform(size_range[0], size_range[1], n_particles)
    is_bright = rng.random(n_particles) < bright_fraction
    is_defocused = rng.random(n_particles) < defocus_fraction

    for i in range(n_particles):
        x, y, r = int(xs[i]), int(ys[i]), radii[i]

        if is_bright[i]:
            # Bright particle
            val = int(rng.uniform(*intensity_range))

            if is_defocused[i]:
                # Defocused: large, dim, blurry blob
                r_def = int(r * rng.uniform(2, 4))
                val_dim = max(10, val // 3)
                _draw_defocused_blob(result, x, y, r_def, val_dim, rng, is_color)
            else:
                # Sharp particle
                r_int = max(1, int(r))
                color = (val, val, val) if is_color else val
                cv2.circle(result, (x, y), r_int, color, -1, cv2.LINE_AA)
                # Bright center
                if r_int >= 2:
                    core_val = min(255, int(val * 1.3))
                    core_color = (core_val, core_val, core_val) if is_color else core_val
                    cv2.circle(result, (x, y), max(1, r_int // 2), core_color, -1, cv2.LINE_AA)
        else:
            # Dark spot (dust)
            r_int = max(1, int(r * 0.7))

            if is_defocused[i]:
                # Defocused dark: large shadow
                r_def = int(r * rng.uniform(1.5, 3))
                _draw_dark_shadow(result, x, y, r_def, rng, is_color)
            else:
                # Sharp dark spot
                darkness = int(rng.uniform(10, 40))
                _subtract_circle(result, x, y, r_int, darkness, is_color)

    return result


def _draw_defocused_blob(img, cx, cy, radius, intensity, rng, is_color):
    """Draw a defocused (blurred ring) blob."""
    h, w = img.shape[:2]
    r = max(2, radius)

    # Draw ring (donut shape)
    outer = max(2, r)
    inner = max(1, int(r * 0.5))
    color = (intensity, intensity, intensity) if is_color else intensity

    # Draw filled outer
    cv2.circle(img, (cx, cy), outer, color, -1, cv2.LINE_AA)
    # Dim center (ring effect)
    dim_val = max(0, intensity // 2)
    dim_color = (dim_val, dim_val, dim_val) if is_color else dim_val

    # Blend center with original (partial transparency)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), inner, 255, -1)
    if is_color:
        for c in range(3):
            channel = img[:, :, c].astype(np.float32)
            channel[mask > 0] = channel[mask > 0] * 0.7 + dim_val * 0.3
            img[:, :, c] = channel.clip(0, 255).astype(np.uint8)
    else:
        channel = img.astype(np.float32)
        channel[mask > 0] = channel[mask > 0] * 0.7 + dim_val * 0.3
        img[:] = channel.clip(0, 255).astype(np.uint8)


def _draw_dark_shadow(img, cx, cy, radius, rng, is_color):
    """Draw a defocused dark shadow."""
    r = max(2, radius)
    darkness = int(rng.uniform(15, 50))
    _subtract_circle(img, cx, cy, r, darkness, is_color)


def _subtract_circle(img, cx, cy, radius, amount, is_color):
    """Subtract intensity in a circular region."""
    h, w = img.shape[:2]
    r = max(1, radius)

    y1 = max(0, cy - r)
    y2 = min(h, cy + r + 1)
    x1 = max(0, cx - r)
    x2 = min(w, cx + r + 1)

    if x2 <= x1 or y2 <= y1:
        return

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    mask = dist <= r

    # Gaussian falloff
    falloff = np.exp(-(dist**2) / (2 * (r * 0.6)**2))
    subtract = (falloff * amount).astype(np.uint8)

    if is_color:
        for c in range(img.shape[2]):
            patch = img[y1:y2, x1:x2, c].astype(np.int16)
            patch[mask] -= subtract[mask].astype(np.int16)
            img[y1:y2, x1:x2, c] = patch.clip(0, 255).astype(np.uint8)
    else:
        patch = img[y1:y2, x1:x2].astype(np.int16)
        patch[mask] -= subtract[mask].astype(np.int16)
        img[y1:y2, x1:x2] = patch.clip(0, 255).astype(np.uint8)
