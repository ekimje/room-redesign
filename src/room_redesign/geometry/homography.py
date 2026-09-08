"""바닥 평면 호모그래피 — 이미지 픽셀 ↔ 탑다운 미터 좌표.

DESIGN.md §3.2 / §7.1~7.3.

좌표계
------
- world: 바닥 평면 위 미터 좌표 (X = 오른쪽, Y = 방 안쪽/카메라에서 멀어지는 방향)
- image: 픽셀 좌표 (u = 오른쪽, v = 아래)

핵심 함수는 numpy 만 사용한다 (OpenCV 의존 없음). 4점 대응은 DLT 로 정확해를 구한다.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Point = Sequence[float]


def _as_pts(pts) -> np.ndarray:
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    return arr


def transform_points(matrix: np.ndarray, pts) -> np.ndarray:
    """3x3 사영행렬을 점들에 적용하고 동차좌표를 정규화한다.

    Parameters
    ----------
    matrix : (3, 3) ndarray
    pts : (N, 2) 좌표

    Returns
    -------
    (N, 2) ndarray
    """
    matrix = np.asarray(matrix, dtype=float)
    p = _as_pts(pts)
    homo = np.hstack([p, np.ones((len(p), 1))])          # N x 3
    out = homo @ matrix.T                                 # N x 3
    w = out[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("점이 사영 후 무한대로 매핑됨 (지평선 위 좌표일 수 있음).")
    return out[:, :2] / w


def solve_floor_homography(world_pts, image_pts) -> np.ndarray:
    """world → image 호모그래피 H_w2i 를 4개 이상 대응점에서 DLT 로 구한다.

    Parameters
    ----------
    world_pts : (N>=4, 2)  바닥 미터 좌표
    image_pts : (N>=4, 2)  대응하는 픽셀 좌표

    Returns
    -------
    (3, 3) ndarray, H[2, 2] == 1 로 정규화됨.
    """
    w = _as_pts(world_pts)
    im = _as_pts(image_pts)
    if len(w) != len(im):
        raise ValueError("world_pts 와 image_pts 개수가 다름.")
    if len(w) < 4:
        raise ValueError("최소 4개의 대응점이 필요함.")

    rows = []
    for (x, y), (u, v) in zip(w, im):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a = np.asarray(rows, dtype=float)

    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("퇴화된 대응 (동일 직선상 점 등) — 호모그래피를 구할 수 없음.")
    return h / h[2, 2]


def invert(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(matrix, dtype=float))


def image_to_floor(h_i2w: np.ndarray, image_pts) -> np.ndarray:
    """픽셀 좌표를 바닥 미터 좌표로 역투영한다 (바닥 위 점만 유효)."""
    return transform_points(h_i2w, image_pts)


def floor_to_image(h_w2i: np.ndarray, world_pts) -> np.ndarray:
    """바닥 미터 좌표를 픽셀 좌표로 투영한다."""
    return transform_points(h_w2i, world_pts)


def local_pixel_scale(h_w2i: np.ndarray, world_xy: Point, eps: float = 1e-3) -> float:
    """world 점 부근에서 '미터당 픽셀' 등방 스케일 근사값.

    호모그래피의 국소 야코비안 행렬식(|det J|, px²/m²)의 제곱근.
    바닥에 서 있는 스프라이트의 원근 크기 보정에 사용한다 (DESIGN.md §7.3).
    """
    p = np.asarray(world_xy, dtype=float)
    o = transform_points(h_w2i, p[None, :])[0]
    px = transform_points(h_w2i, (p + [eps, 0.0])[None, :])[0]
    py = transform_points(h_w2i, (p + [0.0, eps])[None, :])[0]
    jx = (px - o) / eps
    jy = (py - o) / eps
    det = abs(jx[0] * jy[1] - jx[1] * jy[0])
    return math.sqrt(det)


def perspective_scale_factor(
    h_w2i: np.ndarray, world_from: Point, world_to: Point, eps: float = 1e-3
) -> float:
    """원래 위치 대비 새 위치의 스프라이트 리사이즈 배율 s(P') / s(P0)."""
    s0 = local_pixel_scale(h_w2i, world_from, eps)
    s1 = local_pixel_scale(h_w2i, world_to, eps)
    if s0 < 1e-9:
        raise ValueError("원래 위치의 스케일이 0에 가까움.")
    return s1 / s0


def floor_grid_lines(
    world_bounds: tuple[float, float, float, float], spacing: float = 1.0
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """축 정렬 바닥 영역에 대한 그리드 선분 목록 (world 좌표).

    world_bounds = (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = world_bounds
    if spacing <= 0:
        raise ValueError("spacing 은 양수여야 함.")
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []

    x = math.ceil(xmin / spacing) * spacing
    while x <= xmax + 1e-9:
        lines.append(((x, ymin), (x, ymax)))
        x += spacing
    y = math.ceil(ymin / spacing) * spacing
    while y <= ymax + 1e-9:
        lines.append(((xmin, y), (xmax, y)))
        y += spacing
    return lines


def polygon_area(world_polygon) -> float:
    """shoelace 공식으로 다각형 넓이(m²)를 구한다 (world 좌표)."""
    p = _as_pts(world_polygon)
    if len(p) < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
