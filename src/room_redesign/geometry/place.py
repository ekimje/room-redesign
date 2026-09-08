"""물체의 바닥(3D) 위치·발자국 추정. DESIGN.md §3.4.

가정: 물체 밑면이 바닥에 닿아 있다. 단일 이미지에서 전후 깊이는 알 수 없으므로
발자국(footprint)은 폭과 같은 정사각으로 근사한다 (외부 가구는 스펙 사용).
"""

from __future__ import annotations

import numpy as np

from .homography import image_to_floor


def base_to_floor(h_i2w: np.ndarray, base_point_px) -> tuple[float, float]:
    """바닥 접촉점 픽셀 → world 미터 좌표."""
    w = image_to_floor(h_i2w, [base_point_px])[0]
    return float(w[0]), float(w[1])


def estimate_footprint_m(
    h_i2w: np.ndarray, bbox_px: tuple[int, int, int, int]
) -> tuple[float, float]:
    """마스크 bbox 하단 좌우 끝을 바닥으로 역투영해 물체 폭(m)을 구하고,
    깊이도 같다고 보고 (폭, 폭) 을 반환한다."""
    x, y, w, h = bbox_px
    left = (x, y + h)
    right = (x + w, y + h)
    wl = image_to_floor(h_i2w, [left, right])
    width_m = float(np.linalg.norm(wl[0] - wl[1]))
    return width_m, width_m


def anchor_in_sprite(
    base_point_px, bbox_px: tuple[int, int, int, int]
) -> tuple[float, float]:
    """full-image 바닥 접촉점 → 스프라이트(크롭) 내부 좌표."""
    bx, by = base_point_px
    x, y, _, _ = bbox_px
    return float(bx - x), float(by - y)
