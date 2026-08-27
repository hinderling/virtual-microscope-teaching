"""Slim camera optics: blur, vignette, exposure, sensor noise.

Replaces the original OpticalPipeline (~29 ms/frame) with a fast (~2 ms)
seeded equivalent. Deterministic: same seed → bit-identical noise sequence;
``reset()`` restores the sequence for reproducible experiment reruns.
"""

from __future__ import annotations

import cv2
import numpy as np


class Optics:
    """Per-channel optical post-processing applied to the cropped viewport."""

    def __init__(self, seed: int, psf_sigma: float = 0.8,
                 read_std: float = 2.0, photon_k: float = 0.35,
                 vignette: float = 0.08, size: int = 512):
        self.psf_sigma = psf_sigma
        self.read_std = read_std
        self.photon_k = photon_k
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        # precomputed radial vignette map
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        r = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
        self._vignette = (1.0 - vignette * r ** 2).astype(np.float32)

    def reset(self) -> None:
        """Restore the noise sequence (for deterministic experiment reruns)."""
        self.rng = np.random.default_rng(self._seed)

    def apply(self, img: np.ndarray, exposure: float = 50.0,
              intensity: float = 1.0, defocus: float = 0.0) -> np.ndarray:
        """img: uint8 viewport frame → uint8 with optics + noise applied."""
        f = img.astype(np.float32)

        # PSF blur (+ defocus from the Z stage)
        sigma = self.psf_sigma + abs(defocus)
        if sigma > 0.05:
            f = cv2.GaussianBlur(f, (0, 0), sigma)

        # exposure/illumination scaling and vignetting
        f *= (exposure / 50.0) * intensity
        if f.shape == self._vignette.shape:
            f *= self._vignette

        # sensor noise: read noise + photon (shot) noise
        noise = self.rng.standard_normal(f.shape, dtype=np.float32)
        f += noise * (self.read_std + self.photon_k * np.sqrt(f))

        return np.clip(f, 0, 255).astype(np.uint8)
