"""Slim camera optics: blur, vignette, exposure, binning gain, sensor noise.

Fast (~2 ms per 512x512 frame) and deterministic: same seed -> bit-identical
noise sequence; ``reset()`` restores the sequence for reproducible reruns.
"""

from __future__ import annotations

import cv2
import numpy as np


class Optics:
    """Per-channel optical post-processing applied to a rendered camera frame."""

    def __init__(self, seed: int, psf_sigma: float = 0.8,
                 read_std: float = 2.0, photon_k: float = 0.35,
                 vignette: float = 0.08):
        self.psf_sigma = psf_sigma        # in sensor px at 10x (scale 1)
        self.read_std = read_std
        self.photon_k = photon_k
        self.vignette = vignette
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._vignette_cache: dict[tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        """Restore the noise sequence (for deterministic experiment reruns)."""
        self.rng = np.random.default_rng(self._seed)

    def _vignette_map(self, shape: tuple[int, int]) -> np.ndarray:
        if shape not in self._vignette_cache:
            h, w = shape
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
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

        # exposure / illumination / binning gain and vignetting
        f *= (exposure / 50.0) * intensity * (binning * binning)
        f *= self._vignette_map(f.shape)

        # sensor noise: read noise + photon (shot) noise
        noise = self.rng.standard_normal(f.shape, dtype=np.float32)
        f += noise * (self.read_std + self.photon_k * np.sqrt(f))

        return np.clip(f, 0, 255).astype(np.uint8)
