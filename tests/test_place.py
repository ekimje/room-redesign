"""geometry.place 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from room_redesign.geometry import transform_points
from room_redesign.geometry.homography import image_to_floor, invert
from room_redesign.geometry.place import (
    anchor_in_sprite,
    base_to_floor,
    estimate_footprint_m,
)


def test_base_to_floor_roundtrip(h_gt):
    p_world = (1.7, 2.3)
    p_img = transform_points(h_gt, [p_world])[0]
    back = base_to_floor(invert(h_gt), p_img)
    assert back == pytest.approx(p_world, abs=1e-6)


def test_estimate_footprint_measures_bottom_edge(h_gt):
    """bbox 하단 좌우 끝의 world 거리를 폭으로, 깊이도 같게(정사각) 반환."""
    h_i2w = invert(h_gt)
    bbox = (400, 300, 250, 180)
    x, y, w, h = bbox
    left = image_to_floor(h_i2w, [(x, y + h)])[0]
    right = image_to_floor(h_i2w, [(x + w, y + h)])[0]
    expect = float(np.linalg.norm(left - right))

    fp = estimate_footprint_m(h_i2w, bbox)
    assert fp[0] == pytest.approx(expect, rel=1e-9)
    assert fp[0] == fp[1]


def test_anchor_in_sprite():
    assert anchor_in_sprite((150.0, 240.0), (100, 80, 60, 200)) == (50.0, 160.0)
