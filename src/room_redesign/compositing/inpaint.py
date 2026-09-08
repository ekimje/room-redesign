"""배경 인페인팅 — 물체를 들어낸 자리를 메운다. DESIGN.md §3.6.

기본: cv2.inpaint (Telea / NS). 옵션: simple-lama-inpainting (torch 필요).
"""

from __future__ import annotations

import numpy as np


def inpaint(
    image_rgb: np.ndarray,
    bin_mask: np.ndarray,
    method: str = "telea",
    radius: float = 3.0,
    dilate: int = 3,
) -> np.ndarray:
    """마스크 영역을 주변 픽셀로 채운 RGB uint8 이미지를 반환한다.

    dilate > 0 이면 마스크를 조금 부풀려 물체 경계의 잔여 픽셀까지 덮는다.
    """
    import cv2

    img = np.asarray(image_rgb)
    m = (np.asarray(bin_mask) > 0).astype(np.uint8)
    if dilate and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate(m, k)
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    bgr = np.ascontiguousarray(img[:, :, ::-1])
    out = cv2.inpaint(bgr, m * 255, float(radius), flag)
    return np.ascontiguousarray(out[:, :, ::-1])


def inpaint_lama(image_rgb: np.ndarray, bin_mask: np.ndarray) -> np.ndarray:
    """고품질 인페인팅 (LaMa). `pip install -r requirements-ml.txt` 필요."""
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:  # pragma: no cover - 선택 의존성
        raise RuntimeError(
            "simple-lama-inpainting 가 설치되지 않음. `pip install -r requirements-ml.txt`"
        ) from exc

    from PIL import Image

    lama = SimpleLama()
    img = Image.fromarray(np.asarray(image_rgb).astype(np.uint8))
    m = Image.fromarray(((np.asarray(bin_mask) > 0).astype(np.uint8) * 255))
    result = lama(img, m)
    return np.asarray(result.convert("RGB"))
