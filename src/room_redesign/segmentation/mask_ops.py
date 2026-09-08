"""마스크 후처리 유틸. DESIGN.md §3.3.

모폴로지·최대 연결요소·알파 페더링·bbox·바닥 접촉점.
"""

from __future__ import annotations

import numpy as np


def largest_component(bin_mask: np.ndarray) -> np.ndarray:
    """가장 넓은 연결요소만 남긴다 (8-이웃)."""
    import cv2

    m = (np.asarray(bin_mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:  # 배경 + 최대 1개
        return m
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def clean_mask(
    bin_mask: np.ndarray,
    open_ksize: int = 3,
    close_ksize: int = 7,
    keep_largest: bool = True,
) -> np.ndarray:
    """열기(잡음 제거) → 닫기(구멍 메움) → 최대 연결요소."""
    import cv2

    m = (np.asarray(bin_mask) > 0).astype(np.uint8)
    if open_ksize and open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_ksize and close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if keep_largest:
        m = largest_component(m)
    return m


def feather_alpha(bin_mask: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """가장자리를 부드럽게 한 [0, 1] float 알파."""
    import cv2

    m = (np.asarray(bin_mask) > 0).astype(np.float32)
    if radius and radius > 0:
        m = cv2.GaussianBlur(m, (0, 0), float(radius))
    return np.clip(m, 0.0, 1.0)


def mask_bbox(bin_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """전경 픽셀의 (x, y, w, h). 비어 있으면 None."""
    ys, xs = np.where(np.asarray(bin_mask) > 0)
    if xs.size == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def base_point(bin_mask: np.ndarray, band: int = 2) -> tuple[float, float] | None:
    """바닥 접촉점: 최하단 몇 행의 전경 픽셀 x 평균, y 최대값 (full-image px).

    M3 에서 이 점을 바닥 평면으로 역투영해 물체의 world 좌표를 얻는다.
    """
    ys, xs = np.where(np.asarray(bin_mask) > 0)
    if xs.size == 0:
        return None
    ymax = int(ys.max())
    sel = ys >= ymax - max(0, band - 1)
    return float(xs[sel].mean()), float(ymax)
