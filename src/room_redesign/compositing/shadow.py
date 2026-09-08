"""바닥 소프트 그림자. DESIGN.md §3.6 / §7.5.

물체 발자국을 바닥 평면에서 타원으로 근사 → 호모그래피로 이미지에 워프 →
가우시안 블러 → 곱하기 합성.
"""

from __future__ import annotations

import numpy as np

from ..geometry.homography import floor_to_image


def _ellipse_world(
    center_xy: tuple[float, float],
    footprint_m: tuple[float, float],
    offset_m: tuple[float, float],
    n: int = 48,
) -> np.ndarray:
    cx, cy = center_xy
    ax, ay = footprint_m[0] / 2.0, footprint_m[1] / 2.0
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xs = cx + offset_m[0] + ax * np.cos(t)
    ys = cy + offset_m[1] + ay * np.sin(t)
    return np.column_stack([xs, ys])


def floor_shadow(
    background_rgb: np.ndarray,
    h_w2i: np.ndarray,
    floor_xy: tuple[float, float],
    footprint_m: tuple[float, float],
    *,
    light_dir: tuple[float, float] = (0.15, 0.25),
    opacity: float = 0.35,
    blur_px: float = 21.0,
    inplace: bool = False,
) -> np.ndarray:
    """`floor_xy` 바닥 위치에 그림자를 드리운 이미지를 반환한다.

    light_dir : 그림자가 밀리는 방향(미터). 광원 반대쪽으로 offset.
    """
    import cv2

    img = background_rgb if inplace else background_rgb.copy()
    h, w = img.shape[:2]

    poly_w = _ellipse_world(floor_xy, footprint_m, light_dir)
    poly_i = floor_to_image(h_w2i, poly_w)
    poly_i = np.round(poly_i).astype(np.int32)

    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [poly_i], 1.0, lineType=cv2.LINE_AA)
    if blur_px and blur_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), float(blur_px))
    mask = np.clip(mask, 0.0, 1.0) * float(opacity)

    img[:] = (img.astype(np.float32) * (1.0 - mask[..., None])).round().astype(np.uint8)
    return img
