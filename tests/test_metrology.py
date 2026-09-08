"""geometry.metrology 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from room_redesign.geometry import transform_points
from room_redesign.geometry.metrology import (
    FloorCalibration,
    rectangle_world_corners,
    scale_from_known_segment,
)


def _image_corners(h_gt, w, length):
    return transform_points(h_gt, rectangle_world_corners(w, length))


def test_floor_area_matches_input(h_gt, room_wl):
    w, length = room_wl
    calib = FloorCalibration.from_rectangle(_image_corners(h_gt, w, length), w, length)
    assert calib.floor_area_m2() == pytest.approx(w * length, rel=1e-9)


def test_measure_subrectangle_area(h_gt, room_wl):
    w, length = room_wl
    calib = FloorCalibration.from_rectangle(_image_corners(h_gt, w, length), w, length)

    sub_world = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]])  # 2 x 1 = 2 m²
    sub_image = transform_points(h_gt, sub_world)

    assert calib.measure_area_m2(sub_image) == pytest.approx(2.0, rel=1e-6)


def test_measure_length(h_gt, room_wl):
    w, length = room_wl
    calib = FloorCalibration.from_rectangle(_image_corners(h_gt, w, length), w, length)
    p_img = transform_points(h_gt, np.array([[0.5, 0.5], [3.5, 0.5]]))  # 3 m
    assert calib.measure_length_m(p_img[0], p_img[1]) == pytest.approx(3.0, rel=1e-6)


def test_scale_from_known_segment(h_gt):
    """올바른 종횡비 + 미지의 전체 스케일에서, 실측 선분으로 배율 k 를 복원한다.

    (단일 기준 선분은 등방 스케일만 준다 → 종횡비는 이미 맞아야 함)
    """
    true_w, true_l = 4.0, 3.0
    image_corners = _image_corners(h_gt, true_w, true_l)

    # 4:3 비율은 알지만 스케일은 모른다고 가정 (0.8 x 0.6)
    guess = FloorCalibration.from_rectangle(image_corners, 0.8, 0.6)
    p1, p2 = image_corners[0], image_corners[1]  # near 변, 실제 4 m
    k = scale_from_known_segment(guess, p1, p2, real_length_m=true_w)

    assert guess.measure_length_m(p1, p2) * k == pytest.approx(true_w, rel=1e-6)
    assert guess.floor_area_m2() * k**2 == pytest.approx(true_w * true_l, rel=1e-6)


def test_rectangle_corners_validation():
    with pytest.raises(ValueError):
        rectangle_world_corners(0.0, 3.0)
