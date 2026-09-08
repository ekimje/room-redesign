"""몬테카를로 오차 전파. DESIGN.md §3.4b / §10.5.

코너 클릭 노이즈(σ_px)·기준 실측 노이즈(σ_m) 등을 정규분포로 반복 샘플링하여
면적·길이·점유율 분포의 평균/표준편차/5·95 퍼센타일을 산출한다.
결과는 항상 `값 ± σ` 로 보고한다 (단일 숫자 단독 표기 금지).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from .homography import image_to_floor, invert, polygon_area, solve_floor_homography
from .metrology import rectangle_world_corners


@dataclass
class Estimate:
    """오차범위를 동반한 스칼라 추정값."""

    mean: float
    std: float
    p5: float
    p95: float
    unit: str = ""

    @property
    def rel(self) -> float:
        """상대 표준편차 (std / |mean|)."""
        return self.std / abs(self.mean) if self.mean else float("nan")

    def __str__(self) -> str:  # 예: "10.08 ± 1.31 m² (±13.0%)"
        u = f" {self.unit}" if self.unit else ""
        return f"{self.mean:.2f} ± {self.std:.2f}{u} (±{self.rel * 100:.1f}%)"

    @classmethod
    def from_samples(cls, samples, unit: str = "") -> "Estimate":
        s = np.asarray(samples, dtype=float)
        return cls(
            mean=float(s.mean()),
            std=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            p5=float(np.percentile(s, 5)),
            p95=float(np.percentile(s, 95)),
            unit=unit,
        )


def monte_carlo_polygon_area(
    image_corners,
    width_m: float,
    length_m: float,
    image_polygon,
    *,
    sigma_px: float = config.DEFAULT_CORNER_CLICK_PX,
    sigma_wl_rel: float = config.DEFAULT_RECT_ASSUMPTION_REL,
    n: int = config.DEFAULT_MC_SAMPLES,
    seed: int | None = 0,
) -> Estimate:
    """이미지에서 클릭한 바닥 다각형의 실제 넓이 추정 + 오차범위.

    Parameters
    ----------
    image_corners : (4, 2) 클릭한 바닥 코너 픽셀
    width_m, length_m : 입력한 방 실측 폭/길이
    image_polygon : (M, 2) 넓이를 재고 싶은 바닥 다각형 픽셀
    sigma_px : 코너 클릭 표준편차(픽셀)
    sigma_wl_rel : 방 치수/직사각 가정의 상대 오차
    """
    rng = np.random.default_rng(seed)
    corners = np.asarray(image_corners, dtype=float).reshape(4, 2)
    poly = np.asarray(image_polygon, dtype=float).reshape(-1, 2)

    areas = np.empty(n, dtype=float)
    for i in range(n):
        noisy_corners = corners + rng.normal(0.0, sigma_px, size=(4, 2))
        w = width_m * (1.0 + rng.normal(0.0, sigma_wl_rel))
        length_val = length_m * (1.0 + rng.normal(0.0, sigma_wl_rel))
        world = rectangle_world_corners(max(w, 1e-3), max(length_val, 1e-3))
        try:
            h_w2i = solve_floor_homography(world, noisy_corners)
            h_i2w = invert(h_w2i)
            world_poly = image_to_floor(h_i2w, poly)
            areas[i] = polygon_area(world_poly)
        except (ValueError, np.linalg.LinAlgError):
            areas[i] = np.nan

    areas = areas[~np.isnan(areas)]
    if areas.size == 0:
        raise RuntimeError("몬테카를로 표본이 모두 실패함 (입력 좌표 확인 필요).")
    return Estimate.from_samples(areas, unit="m²")


def monte_carlo(fn, n: int = config.DEFAULT_MC_SAMPLES, seed: int | None = 0, unit: str = "") -> Estimate:
    """범용 몬테카를로: fn(rng) -> float 를 n 회 호출해 분포를 요약한다."""
    rng = np.random.default_rng(seed)
    samples = np.array([fn(rng) for _ in range(n)], dtype=float)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        raise RuntimeError("몬테카를로 표본이 모두 실패함.")
    return Estimate.from_samples(samples, unit=unit)
