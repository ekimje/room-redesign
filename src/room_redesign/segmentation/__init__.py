"""segmentation — 물체 분리. DESIGN.md §3.3.

경량 1순위: GrabCut (박스/스크리블). 이후 옵션: backends/ 의 SAM·YOLO.
"""

from .cutout import ObjectCutout, load_cutout_rgba, make_cutout, save_cutout_png
from .grabcut import grabcut_rect, grabcut_refine
from .mask_ops import (
    base_point,
    clean_mask,
    feather_alpha,
    largest_component,
    mask_bbox,
)

__all__ = [
    "grabcut_rect",
    "grabcut_refine",
    "clean_mask",
    "largest_component",
    "feather_alpha",
    "mask_bbox",
    "base_point",
    "ObjectCutout",
    "make_cutout",
    "save_cutout_png",
    "load_cutout_rgba",
]
