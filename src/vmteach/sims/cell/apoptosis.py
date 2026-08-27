"""Apoptosis phase rendering functions.

Standalone functions for drawing cells in various apoptotic phases.
Functions that need _draw_smooth_cell receive it as a callback parameter.
"""
import numpy as np
import cv2
import random
from typing import Tuple, Callable

from .chromatin import draw_condensed_chromatin


def get_apoptosis_opacity(cell) -> float:
    """Calculate opacity for current apoptosis phase."""
    if cell.apoptosis_death_phase == 'Phagocytosis':
        phagocytosis_progress = (
            (cell.death_timer - cell.time_table_apoptosis['Apoptotic bodies'])
            / (cell.max_death_timer - cell.time_table_apoptosis['Apoptotic bodies'])
        )
        return 1.0 - phagocytosis_progress
    else:
        return 1.0


def generate_bleb_points(center_screen: np.ndarray,
                         vertices: np.ndarray,
                         num_blebs: int = 8) -> list:
    """Generate bleb protrusion positions on membrane.

    Returns list of (position, radius) tuples for each bleb.
    """
    blebs = []
    radii_orig = np.linalg.norm(vertices - center_screen, axis=1)
    max_radius = np.max(radii_orig)

    for i in range(num_blebs):
        angle = random.uniform(0, 2 * np.pi)
        base_radius = max_radius * 0.9
        bleb_extension = max_radius * 0.4
        bleb_radius = int(max_radius * 0.15)

        bleb_distance = base_radius + bleb_extension * random.uniform(0.3, 1.0)

        bleb_x = center_screen[0] + bleb_distance * np.cos(angle)
        bleb_y = center_screen[1] + bleb_distance * np.sin(angle)
        bleb_pos = np.array([bleb_x, bleb_y])

        blebs.append((bleb_pos, bleb_radius))

    return blebs


def draw_shrinkage_apoptosis(img: np.ndarray, cell,
                             center_screen: np.ndarray,
                             vertices: np.ndarray,
                             chromatin_pts_screen: list,
                             camera_offset: Tuple[float, float],
                             draw_smooth_cell_fn: Callable,
                             master_shape: np.ndarray) -> None:
    """Draw shrinking cell with condensed chromatin visible inside."""
    shrinkage_factor = cell._get_current_shrinkage_factor()
    vertices_shrunk = center_screen + (vertices - center_screen) * shrinkage_factor

    layers = 10
    for i in range(layers, 0, -1):
        s = i / layers * shrinkage_factor
        shade = 80 + int(100 * s / shrinkage_factor) if shrinkage_factor > 0 else 80
        color = (shade, shade, 255)
        scaled_verts = center_screen + (vertices_shrunk - center_screen) * (i / layers)
        draw_smooth_cell_fn(img, center_screen, scaled_verts, color, thickness=-1)

    draw_smooth_cell_fn(img, center_screen, vertices_shrunk, (0, 0, 0), thickness=2)

    nucleus_radius = int(0.4 * cell.base_r * shrinkage_factor)
    nucleus_pos = tuple(center_screen.astype(int))
    cv2.circle(img, nucleus_pos, nucleus_radius, (150, 60, 60), -1,
               lineType=cv2.LINE_AA)

    draw_condensed_chromatin(img, np.array(cell.center), cell.base_r,
                            camera_offset, master_shape, num_chromosomes=10)


def draw_blebbing_apoptosis(img: np.ndarray, cell,
                            center_screen: np.ndarray,
                            vertices: np.ndarray,
                            chromatin_pts_screen: list,
                            draw_smooth_cell_fn: Callable,
                            master_shape: np.ndarray) -> None:
    """Draw cell with membrane blebs (8-10 protrusions) forming."""
    blebbing_progress = (
        (cell.death_timer - cell.time_table_apoptosis['Shrinkage'])
        / (cell.time_table_apoptosis['Blebbing'] - cell.time_table_apoptosis['Shrinkage'])
    )

    shrinkage_factor = (
        cell.base_r / (cell.original_base_r_for_apoptosis
                       if hasattr(cell, 'original_base_r_for_apoptosis')
                       else cell.base_r)
    )
    vertices_blebbed = center_screen + (vertices - center_screen) * shrinkage_factor

    layers = 10
    for i in range(layers, 0, -1):
        s = i / layers * shrinkage_factor
        shade = 80 + int(100 * s / shrinkage_factor) if shrinkage_factor > 0 else 80
        color = (shade, shade, 255)
        scaled_verts = center_screen + (vertices_blebbed - center_screen) * (i / layers)
        draw_smooth_cell_fn(img, center_screen, scaled_verts, color, thickness=-1)

    bleb_positions = generate_bleb_points(
        center_screen, vertices_blebbed,
        num_blebs=int(8 + 2 * blebbing_progress))

    bleb_color = (130, 130, 255)
    for bleb_pos, bleb_radius in bleb_positions:
        cv2.circle(img, tuple(bleb_pos.astype(int)), bleb_radius, bleb_color,
                   -1, lineType=cv2.LINE_AA)

    draw_smooth_cell_fn(img, center_screen, vertices_blebbed, (0, 0, 0), thickness=1)

    nucleus_radius = int(0.35 * cell.base_r * 0.7)
    nucleus_pos = tuple(center_screen.astype(int))
    cv2.circle(img, nucleus_pos, nucleus_radius, (150, 60, 60), -1,
               lineType=cv2.LINE_AA)


def draw_apoptotic_bodies_phase(img: np.ndarray, cell,
                                center_screen: np.ndarray,
                                camera_offset: Tuple[float, float]) -> None:
    """Draw scattered apoptotic bodies with fragmented nucleus inside."""
    if not hasattr(cell, 'apoptotic_body_positions'):
        return

    body_radius = int(cell.base_r * 0.25)
    body_color = (130, 130, 255)

    for body_pos in cell.apoptotic_body_positions:
        body_screen = np.array([body_pos[0] - camera_offset[0],
                                body_pos[1] - camera_offset[1]])
        cv2.circle(img, tuple(body_screen.astype(int)), body_radius, body_color,
                   -1, lineType=cv2.LINE_AA)
        cv2.circle(img, tuple(body_screen.astype(int)), body_radius, (0, 0, 0),
                   1, lineType=cv2.LINE_AA)


def draw_phagocytosis_apoptosis(img: np.ndarray, cell,
                                center_screen: np.ndarray) -> None:
    """Draw fading apoptotic bodies with decreasing opacity."""
    if not hasattr(cell, 'apoptotic_body_positions'):
        return

    phagocytosis_progress = (
        (cell.death_timer - cell.time_table_apoptosis['Apoptotic bodies'])
        / (cell.max_death_timer - cell.time_table_apoptosis['Apoptotic bodies'])
    )

    body_radius = int(cell.base_r * 0.25)
    fade_factor = 1.0 - phagocytosis_progress
    fade_b = int(130 * fade_factor)
    fade_g = int(130 * fade_factor)
    fade_r = int(255 * fade_factor)

    for body_pos in cell.apoptotic_body_positions:
        cv2.circle(img, tuple(np.array(body_pos).astype(int)), body_radius,
                   (fade_b, fade_g, fade_r), -1, lineType=cv2.LINE_AA)


def draw_apoptosis_phase(img: np.ndarray, cell,
                         center_screen: np.ndarray,
                         vertices: np.ndarray,
                         chromatin_pts_screen: list,
                         camera_offset: Tuple[float, float],
                         opacity: float, kernel_size: int,
                         height: int, width: int,
                         draw_smooth_cell_fn: Callable,
                         master_shape: np.ndarray) -> None:
    """Main dispatcher for apoptotic cell rendering based on phase."""
    cell_img = np.full((height, width, 3), 0, dtype=np.uint8)

    if cell.apoptosis_death_phase == 'Shrinkage':
        draw_shrinkage_apoptosis(cell_img, cell, center_screen, vertices,
                                 chromatin_pts_screen, camera_offset,
                                 draw_smooth_cell_fn, master_shape)
    elif cell.apoptosis_death_phase == 'Blebbing':
        draw_blebbing_apoptosis(cell_img, cell, center_screen, vertices,
                                chromatin_pts_screen, draw_smooth_cell_fn,
                                master_shape)
    elif cell.apoptosis_death_phase == 'Apoptotic bodies':
        draw_apoptotic_bodies_phase(cell_img, cell, center_screen,
                                    camera_offset)
    elif cell.apoptosis_death_phase == 'Phagocytosis':
        draw_phagocytosis_apoptosis(cell_img, cell, center_screen)

    if kernel_size > 0:
        cell_img = cv2.GaussianBlur(cell_img, (kernel_size, kernel_size),
                                    kernel_size / 3.0)

    apoptosis_opacity = get_apoptosis_opacity(cell) * opacity
    img[:] = cv2.addWeighted(img, 1.0, cell_img, apoptosis_opacity, 0)
