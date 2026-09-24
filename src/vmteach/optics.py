"""Slim camera optics: PSF + widefield glow, uneven illumination, exposure,
binning gain, camera offset, hot pixels, sensor noise.

Fast (~2 ms per 512x512 frame) and deterministic: same seed -> bit-identical
noise sequence; ``reset()`` restores the sequence for reproducible reruns.
"""

from __future__ import annotations

import cv2
import numpy as np


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a float32 image; cheap for any ``sigma``.

    Large kernels are computed on a downsampled copy and interpolated back
    (a large blur has no fine detail to lose), so the cost stays ~constant
    instead of growing with the kernel size.
    """
    if sigma <= 0.3:
        return img
    if sigma < 4.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape
    # power-of-two factor: integer-ratio INTER_AREA is the fast resize path
    k = 1 << int(np.log2(sigma / 2.0))
    while k > 1 and (w % k or h % k):
        k >>= 1
    small = cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / k)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# hot pixels are a property of the camera, shared by every channel
_HOT_SEED = 20260924
_HOT_FRACTION = 1.5e-4
_hot_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _hot_pixels(shape: tuple[int, int]):
    """(flat indices, amplitude at 50 ms) of the camera's hot pixels."""
    if shape not in _hot_cache:
        rng = np.random.default_rng(_HOT_SEED + shape[0])
        n = max(1, int(_HOT_FRACTION * shape[0] * shape[1]))
        idx = rng.choice(shape[0] * shape[1], n, replace=False)
        amp = rng.uniform(25.0, 110.0, n).astype(np.float32)
        _hot_cache[shape] = (idx, amp)
    return _hot_cache[shape]


class Optics:
    """Per-channel optical post-processing applied to a rendered camera frame."""

    def __init__(self, seed: int, psf_sigma: float = 0.8,
                 read_std: float = 2.0, photon_k: float = 0.35,
                 vignette: float = 0.08, vignette_center=(0.0, 0.0),
                 glow: float = 0.0, glow_um: float = 7.0,
                 offset: float = 3.0):
        """
        Args:
            psf_sigma: in-focus PSF, sensor px at 10x (scale 1).
            read_std, photon_k: read noise and shot-noise gain.
            vignette: illumination fall-off at the field corner.
            vignette_center: illumination hotspot offset, as a fraction of
                the half field (x, y); real epi-illumination is rarely
                perfectly centred.
            glow: widefield out-of-focus haze, as a fraction of the
                in-focus signal spread over ``glow_um``.
            offset: camera baseline (ADU), independent of exposure.
        """
        self.psf_sigma = psf_sigma
        self.read_std = read_std
        self.photon_k = photon_k
        self.vignette = vignette
        self.vignette_center = tuple(vignette_center)
        self.glow = glow
        self.glow_um = glow_um
        self.offset = offset
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._vignette_cache: dict[tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        """Restore the noise sequence (for deterministic experiment reruns)."""
        self.rng = np.random.default_rng(self._seed)

    def _vignette_map(self, shape: tuple[int, int]) -> np.ndarray:
        if shape not in self._vignette_cache:
            h, w = shape
            cx, cy = self.vignette_center
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = np.hypot((xx - w / 2) / (w / 2) - cx,
                         (yy - h / 2) / (h / 2) - cy)
            self._vignette_cache[shape] = (
                1.0 - self.vignette * r ** 2).astype(np.float32)
        return self._vignette_cache[shape]

    def apply(self, img: np.ndarray, exposure: float = 50.0,
              intensity: float = 1.0, defocus: float = 0.0,
              scale: float = 1.0, binning: int = 1) -> np.ndarray:
        """Rendered uint8 frame -> uint8 frame with optics + noise applied.

        Args:
            exposure: ms; the signal scales linearly (50 ms = nominal).
            intensity: illumination multiplier.
            defocus: extra blur (px) from the Z stage.
            scale: output px per world um; the PSF grows with magnification.
            binning: each output pixel sums ``binning**2`` sensor pixels, so
                the signal (and its shot noise) grows by that factor while
                the read noise is paid once per output pixel. Exposure
                usually has to come down when binning up, as on a real
                8-bit camera.
        """
        f = img.astype(np.float32)

        # PSF blur (+ defocus from the Z stage), in output pixels
        sigma = (self.psf_sigma * scale ** 0.5 + abs(defocus)) / binning
        if sigma > 0.05:
            f = cv2.GaussianBlur(f, (0, 0), sigma)
        # widefield: out-of-focus light from above/below the focal plane
        # spreads as a broad, faint haze around bright structures
        if self.glow > 0.0:
            f += self.glow * blur(f, self.glow_um * scale)

        # exposure / illumination / binning gain and uneven illumination
        f *= (exposure / 50.0) * intensity * (binning * binning)
        f *= self._vignette_map(f.shape)

        # camera: hot pixels (dark current grows with exposure), offset
        idx, amp = _hot_pixels(f.shape)
        f.reshape(-1)[idx] += amp * (exposure / 50.0)
        f += self.offset

        # sensor noise: read noise + photon (shot) noise
        noise = self.rng.standard_normal(f.shape, dtype=np.float32)
        f += noise * (self.read_std + self.photon_k * np.sqrt(f))

        return np.clip(f, 0, 255).astype(np.uint8)
