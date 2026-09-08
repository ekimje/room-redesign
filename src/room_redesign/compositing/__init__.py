"""compositing — 배경 인페인팅, 그림자, 2.5D 렌더러. DESIGN.md §3.6.

M2: inpaint. M3 이후: shadow, renderer.
"""

from .inpaint import inpaint, inpaint_lama

__all__ = ["inpaint", "inpaint_lama"]
