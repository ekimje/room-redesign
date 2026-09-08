"""compositing.shadow 단위 테스트 (cv2 필요)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from room_redesign.compositing.shadow import floor_shadow  # noqa: E402
from room_redesign.geometry.homography import floor_to_image  # noqa: E402


def test_shadow_darkens_near_and_spares_far(room_homography):
    h_w2i, size, _ = room_homography
    w, h = size
    bg = np.full((h, w, 3), 240, np.uint8)

    floor_xy = (2.0, 1.2)
    out = floor_shadow(bg, h_w2i, floor_xy, (0.8, 0.8), opacity=0.4, blur_px=9)

    assert out.shape == bg.shape and out.dtype == np.uint8

    cx, cy = floor_to_image(h_w2i, [floor_xy])[0]
    near = out[int(cy) - 5 : int(cy) + 5, int(cx) - 5 : int(cx) + 5].mean()
    far = out[5:20, 5:20].mean()
    assert near < 235          # 어두워짐
    assert far == pytest.approx(240, abs=1)  # 멀리는 그대로


def test_shadow_not_inplace_by_default(room_homography):
    h_w2i, size, _ = room_homography
    w, h = size
    bg = np.full((h, w, 3), 240, np.uint8)
    _ = floor_shadow(bg, h_w2i, (2.0, 1.2), (0.8, 0.8))
    assert (bg == 240).all()  # 원본 보존
