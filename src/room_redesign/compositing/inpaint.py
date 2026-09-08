"""배경 인페인팅 — 물체를 들어낸 자리를 메운다. DESIGN.md §3.6.

- 기본(권장): LaMa (`simple-lama-inpainting`, torch 필요) — `method="lama"`
- 폴백:       cv2.inpaint (Telea / NS) — 의존성 없음

`method="auto"` 는 LaMa 가 설치돼 있으면 LaMa, 아니면 Telea 로 폴백한다.
"""

from __future__ import annotations

import numpy as np

_LAMA = None  # SimpleLama 인스턴스 캐시 (모델 로드가 무겁다)


def _dilate(mask_u8: np.ndarray, size: int) -> np.ndarray:
    import cv2

    if not size or size <= 0:
        return mask_u8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask_u8, k)


def _lama_available() -> bool:
    try:
        import simple_lama_inpainting  # noqa: F401
    except Exception:
        return False
    return True


def inpaint(
    image_rgb: np.ndarray,
    bin_mask: np.ndarray,
    method: str = "auto",
    radius: float = 3.0,
    dilate: int = 3,
) -> np.ndarray:
    """마스크 영역을 채운 RGB uint8 이미지를 반환한다.

    method : "auto" | "lama" | "telea" | "ns"
    dilate : 마스크를 부풀려 물체 경계의 잔여 픽셀까지 덮는다.
    """
    img = np.asarray(image_rgb)
    m = (np.asarray(bin_mask) > 0).astype(np.uint8)
    m = _dilate(m, dilate)

    if method == "auto":
        method = "lama" if _lama_available() else "telea"
    if method == "lama":
        return _inpaint_lama(img, m)
    return _inpaint_cv(img, m, radius, method)


def _inpaint_cv(img: np.ndarray, m: np.ndarray, radius: float, method: str) -> np.ndarray:
    import cv2

    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    bgr = np.ascontiguousarray(img[:, :, ::-1])
    out = cv2.inpaint(bgr, m * 255, float(radius), flag)
    return np.ascontiguousarray(out[:, :, ::-1])


def _inpaint_lama(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    global _LAMA
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:  # pragma: no cover - 선택 의존성
        raise RuntimeError(
            "simple-lama-inpainting 미설치. `pip install -r requirements-ml.txt`"
        ) from exc

    from PIL import Image

    if _LAMA is None:
        _LAMA = SimpleLama()

    pil_img = Image.fromarray(img[:, :, :3].astype(np.uint8))
    pil_mask = Image.fromarray((m > 0).astype(np.uint8) * 255).convert("L")
    result = _LAMA(pil_img, pil_mask)
    out = np.asarray(result.convert("RGB"))
    # LaMa 는 8의 배수로 패딩 후 되돌리지만, 크기가 어긋나면 잘라 맞춘다
    h, w = img.shape[:2]
    return np.ascontiguousarray(out[:h, :w])


def inpaint_lama(image_rgb: np.ndarray, bin_mask: np.ndarray, dilate: int = 3) -> np.ndarray:
    """LaMa 직접 호출 (설치 안 됐으면 RuntimeError)."""
    m = _dilate((np.asarray(bin_mask) > 0).astype(np.uint8), dilate)
    return _inpaint_lama(np.asarray(image_rgb), m)
