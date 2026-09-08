"""분리 결과(ObjectCutout) 생성과 PNG 저장/불러오기. DESIGN.md §3.3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mask_ops import base_point, feather_alpha, mask_bbox


@dataclass
class ObjectCutout:
    rgba: np.ndarray                         # (h, w, 4) uint8, bbox 로 크롭됨
    bbox: tuple[int, int, int, int]          # full-image 기준 x, y, w, h
    base_point: tuple[float, float]          # 바닥 접촉점 (full-image px)
    mask: np.ndarray                         # (H, W) uint8 {0,1} 전체 이진 마스크

    @property
    def size(self) -> tuple[int, int]:
        return self.bbox[2], self.bbox[3]


def make_cutout(
    image_rgb: np.ndarray, bin_mask: np.ndarray, feather: float = 2.0
) -> ObjectCutout:
    """이미지 + 이진 마스크 → RGBA 컷아웃 (bbox 크롭 + 페더링된 알파)."""
    image_rgb = np.asarray(image_rgb)
    bbox = mask_bbox(bin_mask)
    if bbox is None:
        raise ValueError("빈 마스크로는 컷아웃을 만들 수 없음.")
    x, y, w, h = bbox
    alpha = feather_alpha(bin_mask, feather)[y : y + h, x : x + w]
    rgb = image_rgb[y : y + h, x : x + w, :3].astype(np.uint8)
    rgba = np.dstack([rgb, (alpha * 255.0).round().astype(np.uint8)])
    bp = base_point(bin_mask) or (x + w / 2.0, y + h)
    return ObjectCutout(
        rgba=rgba,
        bbox=bbox,
        base_point=bp,
        mask=(np.asarray(bin_mask) > 0).astype(np.uint8),
    )


def save_cutout_png(cutout: ObjectCutout, path: str | Path) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cutout.rgba, mode="RGBA").save(path)
    return path


def load_cutout_rgba(path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))
