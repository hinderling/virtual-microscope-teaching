"""Chromatin and chromosome drawing functions.

Pure drawing functions — they take coordinates and draw on an image.
No renderer state needed; master_shape is passed as a parameter.
"""
import numpy as np
import cv2
from typing import Tuple


def draw_smooth_chromatin(img: np.ndarray,
                          control_points: list[tuple[float, float]],
                          num_strands: int = 46) -> None:
    """Draw uncondensed chromatin as smooth strands using Bezier curves."""
    if not control_points or len(control_points) < 2:
        return

    num_points_per_strand = 30

    for strand_idx in range(num_strands):
        offset_angle = (strand_idx / num_strands) * 2 * np.pi
        curve_pts = []

        for t in np.linspace(0, 1, num_points_per_strand):
            p0 = np.array(control_points[0])
            p1 = np.array(control_points[1 % len(control_points)])
            p2 = np.array(control_points[2 % len(control_points)])
            res = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
            curve_pts.append(res.astype(np.int32))

        curve_pts = np.array(curve_pts).reshape((-1, 1, 2))
        cv2.polylines(img, [curve_pts], isClosed=False,
                      color=(63, 0, 0), thickness=1,
                      lineType=cv2.LINE_AA)


def transform_points(points: np.ndarray, center: np.ndarray,
                     angle: float, scale: float) -> np.ndarray:
    """Transform points: scale and rotate around center, then translate."""
    scaled = points * scale

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a],
                           [sin_a,  cos_a]])

    rotated = scaled @ rot_matrix.T
    return rotated + center


def draw_chromosome_x(img: np.ndarray, center: np.ndarray,
                      rotation: float, scale: float,
                      master_shape: np.ndarray) -> None:
    """Draw a chromosome at given position and rotation."""
    transformed_points = transform_points(master_shape, center, rotation, scale)
    pts = transformed_points.astype(np.int32)

    cv2.fillPoly(img, [pts], (200, 0, 200), lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (150, 0, 150), 1, lineType=cv2.LINE_AA)


def draw_condensed_chromatin(img: np.ndarray, cell_center: np.ndarray,
                             base_r: float, camera_offset: Tuple[float, float],
                             master_shape: np.ndarray,
                             num_chromosomes: int = 46) -> None:
    """Draw condensed chromatin (X-shaped chromosomes) inside nucleus."""
    center = cell_center - np.array(camera_offset)
    nucleus_radius = int(0.4 * base_r)

    for i in range(num_chromosomes):
        angle = (i / num_chromosomes) * 2 * np.pi + np.random.uniform(-0.3, 0.3)
        radius = np.random.uniform(0, nucleus_radius * 0.8)

        chrom_x = center[0] + radius * np.cos(angle)
        chrom_y = center[1] + radius * np.sin(angle)
        chrom_center = np.array([chrom_x, chrom_y])

        rotation = np.random.uniform(0, 2 * np.pi)
        draw_chromosome_x(img, chrom_center, rotation, scale=1.0,
                          master_shape=master_shape)


def draw_condensed_chromatin_equator(img: np.ndarray, cell_center: np.ndarray,
                                     base_r: float,
                                     master_shape: np.ndarray,
                                     num_chromosomes: int = 46) -> None:
    """Draw condensed chromatin at the equator of the cell."""
    cell_radius = base_r
    plate_thickness = int(cell_radius * 0.2)

    chromosome_per_row = num_chromosomes // 2
    spacing = (cell_radius * 1.6) / (chromosome_per_row + 1)

    # Top row
    for i in range(chromosome_per_row):
        x_pos = cell_center[0] - cell_radius * 0.8 + (i + 1) * spacing
        y_pos = cell_center[1] - plate_thickness // 2
        chrom_center = np.array([x_pos, y_pos])
        rotation = np.pi / 2
        draw_chromosome_x(img, chrom_center, rotation, scale=1.0,
                          master_shape=master_shape)

    # Bottom row
    for i in range(num_chromosomes - chromosome_per_row):
        x_pos = cell_center[0] - cell_radius * 0.8 + (i + 1) * spacing
        y_pos = cell_center[1] - plate_thickness // 2
        chrom_center = np.array([x_pos, y_pos])
        rotation = np.pi / 2
        draw_chromosome_x(img, chrom_center, rotation, scale=1.0,
                          master_shape=master_shape)


def draw_condensed_chromatin_polar(img: np.ndarray, cell_center: np.ndarray,
                                   base_r: float,
                                   master_shape: np.ndarray,
                                   num_chromosomes: int = 46) -> None:
    """Draw condensed chromatin at the polar positions of the cell."""
    cell_radius = base_r
    pole_distance = cell_radius * 0.5

    chromosomes_per_pole = num_chromosomes // 2

    # North pole (top)
    pole_north = np.array([cell_center[0], cell_center[1] - pole_distance])
    for i in range(chromosomes_per_pole):
        angle = (i / max(1, chromosomes_per_pole)) * 2 * np.pi
        radius = np.random.uniform(0, cell_radius * 0.3)

        chrom_x = pole_north[0] + radius * np.cos(angle)
        chrom_y = pole_north[1] + radius * np.sin(angle)
        chrom_center = np.array([chrom_x, chrom_y])

        rotation = np.random.uniform(0, 2 * np.pi)
        draw_chromosome_x(img, chrom_center, rotation, scale=0.9,
                          master_shape=master_shape)

    # South pole (bottom)
    pole_south = np.array([cell_center[0], cell_center[1] + pole_distance])
    for i in range(num_chromosomes - chromosomes_per_pole):
        angle = (i / max(1, num_chromosomes - chromosomes_per_pole)) * 2 * np.pi
        radius = np.random.uniform(0, cell_radius * 0.3)

        chrom_x = pole_south[0] + radius * np.cos(angle)
        chrom_y = pole_south[1] + radius * np.sin(angle)
        chrom_center = np.array([chrom_x, chrom_y])

        rotation = np.random.uniform(0, 2 * np.pi)
        draw_chromosome_x(img, chrom_center, rotation, scale=0.9,
                          master_shape=master_shape)
