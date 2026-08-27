"""SLMProcessor — Centralized SLM mask mapping + stimulation field with decay.

Replaces the duplicated ``_map_slm_to_world()`` method found in bacteria,
dictyostelium, neuron, and tissue_dynamics sims.  Adds realistic exponential
decay of the stimulation field so that past illumination gradually fades.

Usage::

    proc = SLMProcessor(world_width=512, world_height=512, decay_rate=0.1)

    # On each SLM update (called by SimulationBridge.set_slm_mask):
    proc.update_mask(mask, camera_offset, objective_mag, intensity=1.0)

    # On each step/tick (RealtimeEngine or snap-driven):
    proc.tick(dt)

    # Query stimulation at cell positions:
    stim = proc.get_stimulation_vectorized(xs, ys)  # shape (N,)
"""

import cv2
import numpy as np


class SLMProcessor:
    """Centralized SLM mask mapping + stimulation field with decay."""

    # FOV in world pixels per objective magnification
    _FOV_MAP = {100: 64, 40: 128, 20: 256}

    def __init__(self, world_width: int, world_height: int,
                 decay_rate: float = 0.1,
                 illumination_multiply: bool = False):
        """
        Args:
            world_width: Simulation world width in pixels.
            world_height: Simulation world height in pixels.
            decay_rate: Exponential decay rate (per second). 0 = no decay.
            illumination_multiply: If True, ``apply_illumination()`` multiplies
                the SLM pattern onto the rendered image (spatial filter mode).
        """
        self.world_width = world_width
        self.world_height = world_height
        self.decay_rate = decay_rate
        self.illumination_multiply = illumination_multiply

        # Stimulation field: float32 in [0, 1], decays over time
        self._field = np.zeros((world_height, world_width), dtype=np.float32)

        # Latest raw mask (bool, world coords) — for backward compatibility
        self._raw_mask: np.ndarray | None = None

    def map_to_world(self, mask: np.ndarray, camera_offset: np.ndarray,
                     objective_mag: int, viewport_width: int = 512,
                     viewport_height: int = 512) -> np.ndarray:
        """Map a viewport-space SLM mask to a world-coordinate bool array.

        Replaces the duplicated ``_map_slm_to_world()`` found across backends.

        Args:
            mask: Bool array, typically (viewport_height, viewport_width).
            camera_offset: (x, y) camera offset in world pixels.
            objective_mag: Current objective magnification (10, 20, 40, 100).
            viewport_width: Viewport width in pixels.
            viewport_height: Viewport height in pixels.

        Returns:
            Bool array of shape (world_height, world_width).
        """
        fov_world = self._FOV_MAP.get(objective_mag,
                                       min(512, self.world_width))

        # Center of viewport in world coordinates
        cx = int(camera_offset[0]) + viewport_width // 2
        cy = int(camera_offset[1]) + viewport_height // 2

        # Resize mask from viewport to FOV size
        mask_fov = cv2.resize(
            mask.astype(np.uint8), (fov_world, fov_world),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # Place in world
        world_mask = np.zeros((self.world_height, self.world_width), dtype=bool)
        half = fov_world // 2
        x0 = max(0, min(cx - half, self.world_width - fov_world))
        y0 = max(0, min(cy - half, self.world_height - fov_world))

        wx1 = min(self.world_width, x0 + fov_world)
        wy1 = min(self.world_height, y0 + fov_world)
        mw, mh = wx1 - x0, wy1 - y0
        world_mask[y0:wy1, x0:wx1] = mask_fov[:mh, :mw]

        return world_mask

    def update_mask(self, mask: np.ndarray, camera_offset: np.ndarray,
                    objective_mag: int, intensity: float = 1.0,
                    viewport_width: int = 512,
                    viewport_height: int = 512) -> None:
        """Merge new SLM mask into the stimulation field.

        New illumination is added with ``max(existing, new * intensity)``,
        so overlapping patterns don't stack beyond 1.0.

        Args:
            mask: Bool array in viewport space.
            camera_offset: (x, y) camera offset.
            objective_mag: Current objective magnification.
            intensity: Illumination intensity [0, 1].
            viewport_width: Viewport width.
            viewport_height: Viewport height.
        """
        world_mask = self.map_to_world(mask, camera_offset, objective_mag,
                                       viewport_width, viewport_height)
        self._raw_mask = world_mask

        # Merge: max(existing, new * intensity)
        new_field = world_mask.astype(np.float32) * intensity
        np.maximum(self._field, new_field, out=self._field)

    def tick(self, dt: float) -> None:
        """Decay the stimulation field by ``exp(-decay_rate * dt)``.

        Call this once per simulation step or once per snap.

        Args:
            dt: Time step in seconds.
        """
        if self.decay_rate > 0 and dt > 0:
            factor = np.exp(-self.decay_rate * dt)
            self._field *= factor
            # Zero out negligible values to save computation
            self._field[self._field < 1e-4] = 0.0

    def get_stimulation(self, x: float, y: float) -> float:
        """Query stimulation intensity at a single world coordinate.

        Args:
            x: World x coordinate.
            y: World y coordinate.

        Returns:
            Stimulation intensity in [0, 1].
        """
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < self.world_width and 0 <= iy < self.world_height:
            return float(self._field[iy, ix])
        return 0.0

    def get_stimulation_vectorized(self, xs: np.ndarray,
                                    ys: np.ndarray) -> np.ndarray:
        """Query stimulation intensity at multiple world coordinates.

        Args:
            xs: Array of world x coordinates, shape (N,).
            ys: Array of world y coordinates, shape (N,).

        Returns:
            Stimulation intensities, shape (N,), in [0, 1].
        """
        ixs = np.clip(np.round(xs).astype(int), 0, self.world_width - 1)
        iys = np.clip(np.round(ys).astype(int), 0, self.world_height - 1)
        return self._field[iys, ixs]

    def apply_illumination(self, image: np.ndarray,
                           camera_offset: np.ndarray,
                           objective_mag: int) -> np.ndarray:
        """Multiply rendered image with the SLM stimulation field.

        Used for spatial-filter mode where the SLM controls which parts
        of the image are illuminated.

        Args:
            image: Rendered image (H, W) or (H, W, C).
            camera_offset: Camera offset in world pixels.
            objective_mag: Current objective magnification.

        Returns:
            Modulated image, same shape and dtype as input.
        """
        fov_world = self._FOV_MAP.get(objective_mag,
                                       min(512, self.world_width))
        h, w = image.shape[:2]
        cx = int(camera_offset[0]) + w // 2
        cy = int(camera_offset[1]) + h // 2
        half = fov_world // 2
        x0 = max(0, min(cx - half, self.world_width - fov_world))
        y0 = max(0, min(cy - half, self.world_height - fov_world))

        # Extract the relevant region of the stimulation field
        field_crop = self._field[y0:y0 + fov_world, x0:x0 + fov_world]

        # Resize to image dimensions
        if field_crop.shape[:2] != (h, w):
            field_resized = cv2.resize(field_crop, (w, h),
                                       interpolation=cv2.INTER_LINEAR)
        else:
            field_resized = field_crop

        # Apply as multiplicative mask
        if image.ndim == 3:
            field_resized = field_resized[:, :, np.newaxis]
        result = (image.astype(np.float32) * field_resized).clip(0, 255)
        return result.astype(np.uint8)

    @property
    def field(self) -> np.ndarray:
        """The current stimulation field (read-only view)."""
        return self._field

    @property
    def raw_mask(self) -> np.ndarray | None:
        """Latest raw boolean mask in world coordinates (backward compat)."""
        return self._raw_mask

    def reset(self) -> None:
        """Clear the stimulation field and raw mask."""
        self._field[:] = 0.0
        self._raw_mask = None
