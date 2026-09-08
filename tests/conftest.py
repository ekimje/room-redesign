"""테스트 공통 픽스처."""

from __future__ import annotations

import numpy as np
import pytest

# 임의의(하지만 타당한) 바닥 호모그래피: world 미터 → image 픽셀.
# 완만한 원근을 가지도록 h20, h21 을 작게 둔다.
H_GT = np.array(
    [
        [900.0, 40.0, 300.0],
        [10.0, 850.0, 200.0],
        [5.0e-4, 1.6e-3, 1.0],
    ]
)


@pytest.fixture
def h_gt() -> np.ndarray:
    return H_GT.copy()


@pytest.fixture
def room_wl() -> tuple[float, float]:
    """테스트용 방 폭/길이 (m)."""
    return 4.0, 3.0


@pytest.fixture
def synthetic_object():
    """단색 배경 위 뚜렷한 물체. (rgb, true_mask, rect) 반환.

    GrabCut / 인페인팅 검증용. cv2 로 도형을 그린다.
    """
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    h, w = 300, 400
    bg = np.array([60, 90, 180], dtype=np.float32)          # 파란 배경
    fg = np.array([230, 140, 40], dtype=np.float32)         # 주황 물체
    img = np.tile(bg, (h, w, 1))
    img += rng.normal(0.0, 4.0, img.shape).astype(np.float32)

    true_mask = np.zeros((h, w), np.uint8)
    # 사각형 몸통 + 아래로 뻗은 다리 (바닥 접촉점 테스트)
    cv2.rectangle(true_mask, (150, 80), (270, 200), 1, -1)
    cv2.rectangle(true_mask, (200, 200), (220, 240), 1, -1)

    m3 = true_mask.astype(bool)
    obj = np.tile(fg, (h, w, 1)) + rng.normal(0.0, 4.0, img.shape).astype(np.float32)
    img[m3] = obj[m3]
    img = np.clip(img, 0, 255).astype(np.uint8)

    rect = (135, 65, 155, 190)  # 물체를 여유 있게 감싸는 박스
    return img, true_mask, rect
