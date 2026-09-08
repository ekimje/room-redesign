"""GrabCut 기반 대화형 물체 분리. DESIGN.md §3.3 (경량 1순위).

이미지 배열은 RGB uint8 (H, W, 3) 을 기대한다. 내부에서 cv2(BGR)로 변환한다.
반환 마스크는 이진(uint8 {0,1}) 이다.
"""

from __future__ import annotations

import numpy as np

GC_ITER_RECT = 5
GC_ITER_MASK = 3

# cv2.GC_* 상수 (cv2 를 지연 import 하므로 값을 명시)
_GC_BGD, _GC_FGD, _GC_PR_BGD, _GC_PR_FGD = 0, 1, 2, 3


def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image_rgb)[:, :, ::-1])


def _binarize(gc_mask: np.ndarray) -> np.ndarray:
    """GrabCut 라벨 마스크 → 전경 이진 마스크 (FGD | PR_FGD)."""
    return np.where(
        (gc_mask == _GC_FGD) | (gc_mask == _GC_PR_FGD), 1, 0
    ).astype(np.uint8)


def grabcut_rect(
    image_rgb: np.ndarray,
    rect: tuple[int, int, int, int],
    iterations: int = GC_ITER_RECT,
) -> np.ndarray:
    """바운딩 박스(x, y, w, h)로 초기화한 GrabCut. 이진 마스크 반환."""
    import cv2

    img = _to_bgr(image_rgb)
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        raise ValueError("rect 의 폭/높이는 양수여야 함.")
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, (x, y, w, h), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    return _binarize(mask)


def grabcut_refine(
    image_rgb: np.ndarray,
    bin_mask: np.ndarray,
    fg_points: list[tuple[float, float]] | None = None,
    bg_points: list[tuple[float, float]] | None = None,
    brush: int = 6,
    iterations: int = GC_ITER_MASK,
) -> np.ndarray:
    """전경/배경 스크리블 힌트로 마스크를 보정한다 (GC_INIT_WITH_MASK)."""
    import cv2

    img = _to_bgr(image_rgb)
    gc = np.where(np.asarray(bin_mask).astype(bool), _GC_PR_FGD, _GC_PR_BGD).astype(np.uint8)
    for px, py in fg_points or []:
        cv2.circle(gc, (int(px), int(py)), brush, _GC_FGD, -1)
    for px, py in bg_points or []:
        cv2.circle(gc, (int(px), int(py)), brush, _GC_BGD, -1)
    if not np.any((gc == _GC_FGD) | (gc == _GC_PR_FGD)):
        raise ValueError("전경으로 표시된 픽셀이 없음.")
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    return _binarize(gc)
