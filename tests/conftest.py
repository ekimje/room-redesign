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


@pytest.fixture
def room_homography():
    """800x600 이미지에 4x3 m 방을 담는 바닥 호모그래피. (h_w2i, (W,H)px, (W,L)m)."""
    from room_redesign.geometry import rectangle_world_corners, solve_floor_homography

    room_w, room_l = 4.0, 3.0
    image_corners = np.array(
        [[120, 520], [680, 520], [560, 240], [240, 240]], dtype=float
    )  # near 가 넓고 아래, far 가 좁고 위
    h_w2i = solve_floor_homography(rectangle_world_corners(room_w, room_l), image_corners)
    return h_w2i, (800, 600), (room_w, room_l)


@pytest.fixture
def red_sprite():
    """40x60 불투명 빨강 RGBA, 바닥 접촉점 = 하단 중앙."""
    spr = np.zeros((60, 40, 4), np.uint8)
    spr[..., 0] = 220
    spr[..., 3] = 255
    return spr, (20.0, 60.0)  # rgba, anchor_xy
