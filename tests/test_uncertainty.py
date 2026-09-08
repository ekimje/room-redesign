"""geometry.uncertainty 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from room_redesign.geometry import transform_points
from room_redesign.geometry.metrology import rectangle_world_corners
from room_redesign.geometry.uncertainty import (
    Estimate,
    monte_carlo,
    monte_carlo_polygon_area,
)


def _setup(h_gt, w=4.0, length=3.0):
    image_corners = transform_points(h_gt, rectangle_world_corners(w, length))
    sub_world = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]])  # 2 m²
    sub_image = transform_points(h_gt, sub_world)
    return image_corners, sub_image


def test_zero_noise_is_exact(h_gt):
    corners, sub = _setup(h_gt)
    est = monte_carlo_polygon_area(
        corners, 4.0, 3.0, sub, sigma_px=0.0, sigma_wl_rel=0.0, n=64
    )
    assert est.mean == pytest.approx(2.0, rel=1e-6)
    assert est.std == pytest.approx(0.0, abs=1e-9)


def test_noise_produces_band_around_truth(h_gt):
    corners, sub = _setup(h_gt)
    est = monte_carlo_polygon_area(
        corners, 4.0, 3.0, sub, sigma_px=2.0, sigma_wl_rel=0.03, n=400, seed=1
    )
    assert est.std > 0.0
    assert est.p5 < est.mean < est.p95
    # 참값(2.0)이 90% 구간 안에 들어야 함
    assert est.p5 <= 2.0 <= est.p95
    # 상대오차는 상식적인 범위
    assert 0.0 < est.rel < 0.5


def test_reproducible_with_seed(h_gt):
    corners, sub = _setup(h_gt)
    kw = dict(sigma_px=2.0, sigma_wl_rel=0.03, n=128, seed=42)
    a = monte_carlo_polygon_area(corners, 4.0, 3.0, sub, **kw)
    b = monte_carlo_polygon_area(corners, 4.0, 3.0, sub, **kw)
    assert a.mean == b.mean and a.std == b.std


def test_estimate_str_and_rel():
    est = Estimate(mean=10.0, std=1.3, p5=8.0, p95=12.2, unit="m²")
    assert est.rel == pytest.approx(0.13)
    assert "±" in str(est)


def test_generic_monte_carlo():
    est = monte_carlo(lambda rng: rng.normal(5.0, 1.0), n=2000, seed=0)
    assert est.mean == pytest.approx(5.0, abs=0.15)
