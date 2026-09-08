"""이미지 로딩과 EXIF 읽기. DESIGN.md §3.1."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_image_rgb(path: str | Path) -> np.ndarray:
    """이미지를 RGB uint8 (H, W, 3) 배열로 읽는다. Pillow 사용 (OpenCV BGR 회피)."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def exif_focal_px(path: str | Path, image_width_px: int) -> Optional[float]:
    """EXIF 의 35mm 환산 초점거리에서 픽셀 단위 초점거리를 근사한다.

    focal_px ≈ focal_35mm / 36mm * image_width_px
    없으면 None.
    """
    from PIL import ExifTags, Image

    try:
        with Image.open(path) as im:
            exif = im.getexif()
    except Exception:
        return None
    if not exif:
        return None

    tag_map = {v: k for k, v in ExifTags.TAGS.items()}
    f35 = exif.get(tag_map.get("FocalLengthIn35mmFilm"))
    if f35:
        return float(f35) / 36.0 * image_width_px
    return None
