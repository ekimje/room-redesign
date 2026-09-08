"""geometry.homography 단위 테스트 (numpy 만 필요)."""

from __future__ import annotations

import numpy as np
import pytest

from room_redesign.geometry import (
    floor_grid_lines,
    image_to_floor,
    invert,
    local_pixel_scale,
    perspective_scale_factor,
    polygon_area,
    solve_floor_homography,
    transform_points,
)
from room_redesign.geometry.metrology import rectangle_world_corners


def test_solve_recovers_ground_truth(h_gt, room_wl):
    w, length = room_wl
    world = rectangle_world_corners(w, length)
    image = transform_points(h_gt, world)

    h_est = solve_floor_homography(world, image)

    # 재투영 오차가 미미해야 함 (의미 있는 검증)
    grid = np.array(
        [[x, y] for x in np.linspace(0, w, 5) for y in np.linspace(0, length, 5)]
    )
    assert np.allclose(
        transform_points(h_est, grid), transform_points(h_gt, grid), atol=1e-6
    )
    # h22 == 1 정규화되어 있으므로 행렬도 근사 일치 (재투영이 1차 검증, 이건 보조)
    assert np.allclose(h_est, h_gt, rtol=1e-3, atol=1e-9)


def test_image_floor_roundtrip(h_gt, room_wl):
    world = rectangle_world_corners(*room_wl)
    image = transform_points(h_gt, world)
    h_est = solve_floor_homography(world, image)
    h_i2w = invert(h_est)

    probe_world = np.array([[1.0, 0.5], [2.2, 2.7], [3.9, 0.1]])
    probe_image = transform_points(h_est, probe_world)
    back = image_to_floor(h_i2w, probe_image)

    assert np.allclose(back, probe_world, atol=1e-6)


def test_polygon_area_shoelace():
    square = [[0, 0], [2, 0], [2, 3], [0, 3]]
    assert polygon_area(square) == pytest.approx(6.0)
    # 반대 방향(시계)도 절대값
    assert polygon_area(square[::-1]) == pytest.approx(6.0)


def test_local_scale_decreases_with_depth(h_gt):
    """원근 분모가 커지는 먼 쪽(y 큰 곳)일수록 미터당 픽셀이 작아진다."""
    near = local_pixel_scale(h_gt, (2.0, 0.0))
    far = local_pixel_scale(h_gt, (2.0, 3.0))
    assert far < near
    # 배율은 두 스케일의 비
    ratio = perspective_scale_factor(h_gt, (2.0, 0.0), (2.0, 3.0))
    assert ratio == pytest.approx(far / near, rel=1e-6)


def test_floor_grid_lines_counts():
    lines = floor_grid_lines((0.0, 4.0, 0.0, 3.0), spacing=1.0)
    # x = 0,1,2,3,4 (5) + y = 0,1,2,3 (4) = 9
    assert len(lines) == 9
    for a, b in lines:
        assert len(a) == 2 and len(b) == 2


def test_solve_requires_four_points():
    with pytest.raises(ValueError):
        solve_floor_homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])
