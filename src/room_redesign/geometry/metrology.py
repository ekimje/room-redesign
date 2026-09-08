"""스케일 기준 → 실제 치수 측정. DESIGN.md §3.4b / §10.

사진 추정 모드:
- 사용자가 바닥 직사각형 코너 4점을 클릭하고 그 실측 폭(W)·길이(L)를 입력한다.
- world 코너 = [(0,0), (W,0), (W,L), (0,L)] 로 두고 호모그래피를 만든다.
- 이후 이미지에서 클릭한 임의의 바닥 다각형/선분의 실제 크기를 측정할 수 있다.

W·L 을 모를 때는 `scale_from_known_segment` 로 기준 선분 하나에서 배율을 얻는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .homography import (
    image_to_floor,
    invert,
    polygon_area,
    solve_floor_homography,
)

Point = Sequence[float]


def rectangle_world_corners(width_m: float, length_m: float) -> np.ndarray:
    """near-left, near-right, far-right, far-left 순서의 world 코너."""
    if width_m <= 0 or length_m <= 0:
        raise ValueError("width_m, length_m 은 양수여야 함.")
    return np.array(
        [[0.0, 0.0], [width_m, 0.0], [width_m, length_m], [0.0, length_m]], dtype=float
    )


@dataclass
class FloorCalibration:
    """확정된 바닥 스케일. 이미지↔미터 변환의 단일 진입점."""

    image_corners: np.ndarray          # (4, 2) 클릭한 픽셀 좌표
    world_corners: np.ndarray          # (4, 2) 대응 미터 좌표
    h_w2i: np.ndarray                  # world → image
    h_i2w: np.ndarray                  # image → world

    @classmethod
    def from_rectangle(
        cls, image_corners, width_m: float, length_m: float
    ) -> "FloorCalibration":
        world = rectangle_world_corners(width_m, length_m)
        image = np.asarray(image_corners, dtype=float).reshape(4, 2)
        h_w2i = solve_floor_homography(world, image)
        return cls(image, world, h_w2i, invert(h_w2i))

    # -- 측정 ---------------------------------------------------------------
    def measure_area_m2(self, image_polygon) -> float:
        """이미지에서 클릭한 바닥 다각형의 실제 넓이(m²)."""
        world_poly = image_to_floor(self.h_i2w, image_polygon)
        return polygon_area(world_poly)

    def measure_length_m(self, image_p1: Point, image_p2: Point) -> float:
        """이미지에서 클릭한 두 바닥 점 사이 실제 거리(m)."""
        w = image_to_floor(self.h_i2w, [image_p1, image_p2])
        return float(np.linalg.norm(w[0] - w[1]))

    def floor_area_m2(self) -> float:
        return polygon_area(self.world_corners)

    def floor_bounds(self) -> tuple[float, float, float, float]:
        w = self.world_corners
        return (
            float(w[:, 0].min()),
            float(w[:, 0].max()),
            float(w[:, 1].min()),
            float(w[:, 1].max()),
        )


def scale_from_known_segment(
    image_calib_unit: FloorCalibration, image_p1: Point, image_p2: Point, real_length_m: float
) -> float:
    """단위 스케일로 만든 보정값에서, 기준 선분의 실측 길이로 배율(k)을 구한다.

    최종 미터값 = 측정값 * k, 최종 넓이 = 측정값 * k**2.
    """
    measured = image_calib_unit.measure_length_m(image_p1, image_p2)
    if measured < 1e-9:
        raise ValueError("기준 선분의 측정 길이가 0에 가까움.")
    return real_length_m / measured
