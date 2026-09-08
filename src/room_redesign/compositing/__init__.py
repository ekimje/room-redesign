"""compositing — 배경 인페인팅, 그림자, 2.5D 렌더러. DESIGN.md §3.6.

M2: inpaint. M3: shadow, renderer.
"""

from .inpaint import inpaint, inpaint_lama
from .renderer import (
    PlacedSprite,
    render_from_scene,
    render_scene,
    sprites_from_scene,
)
from .shadow import floor_shadow

__all__ = [
    "inpaint",
    "inpaint_lama",
    "floor_shadow",
    "PlacedSprite",
    "render_scene",
    "render_from_scene",
    "sprites_from_scene",
]
