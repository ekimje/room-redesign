"""2.5D 스프라이트 렌더러. DESIGN.md §3.6 / §7.3~7.4.

바닥에 서 있는 물체를 RGBA 스프라이트로 보고:
- 새 바닥 위치의 국소 픽셀 스케일 / 원래 위치의 스케일 비로 크기 조정
- 바닥 접촉점(anchor)이 새 위치의 투영점에 오도록 배치
- 카메라에 가까운(이미지 v 가 큰) 물체를 나중에 그려 가림 처리
- 물체 아래에 소프트 그림자
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..geometry.homography import floor_to_image, local_pixel_scale
from .shadow import floor_shadow


@dataclass
class PlacedSprite:
    rgba: np.ndarray                          # (h0, w0, 4) uint8 — 원본 컷아웃
    anchor_xy: tuple[float, float]            # 스프라이트 내부 바닥 접촉점 (원본 px)
    origin_floor_xy: tuple[float, float]      # 원래 바닥 위치 (m)
    floor_xy: tuple[float, float]             # 현재 바닥 위치 (m)
    footprint_m: tuple[float, float] = (0.6, 0.6)
    scale: float = 1.0                        # 사용자 배율
    yaw_deg: float = 0.0                      # 이미지 평면 내 회전(근사)
    obj_id: str = ""


def _resize_rgba(rgba: np.ndarray, fx: float) -> np.ndarray:
    import cv2

    h, w = rgba.shape[:2]
    nw, nh = max(1, round(w * fx)), max(1, round(h * fx))
    interp = cv2.INTER_AREA if fx < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(rgba, (nw, nh), interpolation=interp)


def _rotate_rgba(rgba: np.ndarray, deg: float, anchor: tuple[float, float]):
    import cv2

    if abs(deg) < 1e-3:
        return rgba, anchor
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D((float(anchor[0]), float(anchor[1])), deg, 1.0)
    out = cv2.warpAffine(rgba, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderValue=(0, 0, 0, 0))
    return out, anchor


def _alpha_paste(dst_rgb: np.ndarray, sprite_rgba: np.ndarray, top_left: tuple[int, int]) -> None:
    """dst_rgb 위에 sprite_rgba 를 알파 합성 (경계 클리핑)."""
    H, W = dst_rgb.shape[:2]
    sh, sw = sprite_rgba.shape[:2]
    x0, y0 = int(round(top_left[0])), int(round(top_left[1]))
    x1, y1 = x0 + sw, y0 + sh

    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x1), min(H, y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    sx0, sy0 = dx0 - x0, dy0 - y0
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)

    patch = sprite_rgba[sy0:sy1, sx0:sx1].astype(np.float32)
    a = (patch[..., 3:4] / 255.0)
    dst = dst_rgb[dy0:dy1, dx0:dx1].astype(np.float32)
    dst_rgb[dy0:dy1, dx0:dx1] = (patch[..., :3] * a + dst * (1.0 - a)).round().astype(np.uint8)


def render_scene(
    background_rgb: np.ndarray,
    h_w2i: np.ndarray,
    sprites: list[PlacedSprite],
    *,
    shadows: bool = True,
    shadow_opacity: float = 0.33,
) -> np.ndarray:
    """배경 + 배치된 스프라이트 → 합성 RGB uint8."""
    out = np.ascontiguousarray(background_rgb[:, :, :3]).copy()
    h_w2i = np.asarray(h_w2i, dtype=float)

    def base_v(sp: PlacedSprite) -> float:
        return float(floor_to_image(h_w2i, [sp.floor_xy])[0][1])

    for sp in sorted(sprites, key=base_v):  # 먼 것(v 작음)부터
        s0 = local_pixel_scale(h_w2i, sp.origin_floor_xy)
        s1 = local_pixel_scale(h_w2i, sp.floor_xy)
        fx = (s1 / s0 if s0 > 1e-9 else 1.0) * float(sp.scale)
        fx = float(np.clip(fx, 0.05, 20.0))

        if shadows:
            floor_shadow(
                out, h_w2i, sp.floor_xy,
                (sp.footprint_m[0] * sp.scale, sp.footprint_m[1] * sp.scale),
                opacity=shadow_opacity, inplace=True,
            )

        spr = _resize_rgba(sp.rgba, fx)
        anchor = (sp.anchor_xy[0] * fx, sp.anchor_xy[1] * fx)
        spr, anchor = _rotate_rgba(spr, sp.yaw_deg, anchor)

        b = floor_to_image(h_w2i, [sp.floor_xy])[0]
        top_left = (b[0] - anchor[0], b[1] - anchor[1])
        _alpha_paste(out, spr, top_left)

    return out


# --- Scene(pydantic) 연동 ------------------------------------------------

def sprites_from_scene(scene, base_dir: str | Path = ".") -> list[PlacedSprite]:
    """scene.objects → PlacedSprite 목록 (에셋 로드 포함)."""
    from ..segmentation.cutout import load_cutout_rgba

    base = Path(base_dir)
    out: list[PlacedSprite] = []
    for obj in scene.objects:
        rgba = load_cutout_rgba(base / obj.asset if not Path(obj.asset).is_absolute() else obj.asset)
        h0, w0 = rgba.shape[:2]
        anchor = tuple(obj.anchor_px) if obj.anchor_px else (w0 / 2.0, float(h0))
        origin = tuple(obj.original_floor_xy) if obj.original_floor_xy else tuple(obj.placement.floor_xy)
        fp = tuple(obj.footprint_m) if obj.footprint_m else (0.6, 0.6)
        out.append(
            PlacedSprite(
                rgba=rgba,
                anchor_xy=anchor,
                origin_floor_xy=origin,
                floor_xy=tuple(obj.placement.floor_xy),
                footprint_m=fp,
                scale=obj.placement.scale,
                yaw_deg=obj.placement.yaw_deg,
                obj_id=obj.id,
            )
        )
    return out


def render_from_scene(scene, background_rgb: np.ndarray, base_dir: str | Path = ".", **kw) -> np.ndarray:
    if scene.room.floor_homography is None:
        raise ValueError("scene.room.floor_homography 가 없음 — 먼저 01_calibrate 로 보정하세요.")
    h_w2i = np.asarray(scene.room.floor_homography, dtype=float)
    return render_scene(background_rgb, h_w2i, sprites_from_scene(scene, base_dir), **kw)
