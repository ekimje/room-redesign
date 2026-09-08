"""장면 그래프 pydantic 스키마. DESIGN.md §3.5.

사진 추정 모드 기준. 절대 치수가 없어도 동작하며, 있으면 fit_report 를 채운다.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Camera(BaseModel):
    image_size: tuple[int, int]                      # (width, height) px
    K: Optional[list[list[float]]] = None            # 3x3 intrinsics (있으면)
    focal_px: Optional[float] = None
    height_m: Optional[float] = None                 # 바닥 → 렌즈


class ScaleReference(BaseModel):
    """스케일 기준. DESIGN.md §10.1."""

    mode: Literal["room_rectangle", "known_length", "camera_height", "wall_length"]
    image_pts: Optional[list[list[float]]] = None    # 클릭한 픽셀 점들
    real_m: Optional[float] = None                   # 실측 길이/치수
    sigma_m: float = 0.02
    # mode == "room_rectangle" 일 때
    width_m: Optional[float] = None
    length_m: Optional[float] = None


class Uncertainty(BaseModel):
    corner_click_px: float = 3.0
    focal_rel: float = 0.05
    lens_distortion_rel: float = 0.04
    rect_assumption_rel: float = 0.06
    mc_samples: int = 400


class Room(BaseModel):
    image_corners: list[list[float]]                 # 4x2 px (near-left ... far-left)
    floor_homography: Optional[list[list[float]]] = None   # world(m) -> image(px)
    floor_polygon_m: Optional[list[list[float]]] = None
    walls: list[dict] = Field(default_factory=list)


class Placement(BaseModel):
    floor_xy: tuple[float, float]
    yaw_deg: float = 0.0
    scale: float = 1.0


class SceneObject(BaseModel):
    id: str
    source: Literal["cutout", "external"]
    asset: str
    placement: Placement
    original_floor_xy: Optional[tuple[float, float]] = None
    anchor_px: Optional[tuple[float, float]] = None      # 스프라이트 내부 바닥 접촉점
    footprint_m: Optional[tuple[float, float]] = None
    footprint_source: Optional[Literal["spec", "photo_estimate"]] = None
    z_order_hint: Optional[int] = None


class FitReport(BaseModel):
    floor_area_m2: Optional[tuple[float, float]] = None       # (값, σ)
    used_area_m2: Optional[tuple[float, float]] = None
    occupancy_ratio: Optional[tuple[float, float]] = None
    wall_clearances_cm: list[float] = Field(default_factory=list)
    verdict: Optional[Literal["fits_with_margin", "tight", "overflows"]] = None
    confidence: Optional[float] = None


class Scene(BaseModel):
    version: int = 1
    image: str
    camera: Camera
    scale_reference: Optional[ScaleReference] = None
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    room: Room
    objects: list[SceneObject] = Field(default_factory=list)
    background: Optional[str] = None
    fit_report: Optional[FitReport] = None
