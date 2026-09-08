"""geometry — 바닥 평면 기하, 스케일 측정, 오차 전파. DESIGN.md §3.2 / §7 / §10."""

from .homography import (
    floor_grid_lines,
    floor_to_image,
    image_to_floor,
    invert,
    local_pixel_scale,
    perspective_scale_factor,
    polygon_area,
    solve_floor_homography,
    transform_points,
)
from .metrology import (
    FloorCalibration,
    rectangle_world_corners,
    scale_from_known_segment,
)
from .place import anchor_in_sprite, base_to_floor, estimate_footprint_m
from .uncertainty import Estimate, monte_carlo, monte_carlo_polygon_area

__all__ = [
    "solve_floor_homography",
    "invert",
    "transform_points",
    "image_to_floor",
    "floor_to_image",
    "local_pixel_scale",
    "perspective_scale_factor",
    "floor_grid_lines",
    "polygon_area",
    "FloorCalibration",
    "rectangle_world_corners",
    "scale_from_known_segment",
    "base_to_floor",
    "estimate_footprint_m",
    "anchor_in_sprite",
    "Estimate",
    "monte_carlo",
    "monte_carlo_polygon_area",
]
