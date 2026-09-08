"""segmentation (GrabCut + 마스크 후처리 + 컷아웃) 단위 테스트.

cv2 가 필요하므로 없으면 스킵된다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from room_redesign.segmentation import (  # noqa: E402
    base_point,
    clean_mask,
    grabcut_rect,
    grabcut_refine,
    make_cutout,
    mask_bbox,
    save_cutout_png,
)
from room_redesign.segmentation.cutout import load_cutout_rgba  # noqa: E402


def _iou(a, b) -> float:
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def test_grabcut_rect_recovers_object(synthetic_object):
    img, true_mask, rect = synthetic_object
    mask = clean_mask(grabcut_rect(img, rect))
    assert _iou(mask, true_mask) > 0.85
    # 박스 바깥은 전경이 되면 안 됨
    x, y, w, h = rect
    outside = mask.copy()
    outside[y : y + h, x : x + w] = 0
    assert outside.sum() < 0.02 * true_mask.sum()


def test_refine_removes_false_foreground(synthetic_object):
    img, true_mask, rect = synthetic_object
    mask = clean_mask(grabcut_rect(img, rect))
    # 배경 한 점을 배경으로 강제해도 여전히 물체를 잘 잡아야 함
    refined = clean_mask(grabcut_refine(img, mask, bg_points=[(10, 10)]))
    assert _iou(refined, true_mask) > 0.8


def test_mask_bbox_and_base_point(synthetic_object):
    _, true_mask, _ = synthetic_object
    x, y, w, h = mask_bbox(true_mask)
    assert (x, y) == pytest.approx((150, 80), abs=2)
    assert w == pytest.approx(120, abs=2)
    bx, by = base_point(true_mask)
    # 다리 끝(대략 x=210, y=239)
    assert bx == pytest.approx(210, abs=6)
    assert by == pytest.approx(239, abs=2)


def test_make_and_save_cutout(synthetic_object, tmp_path):
    img, true_mask, _ = synthetic_object
    cut = make_cutout(img, true_mask, feather=1.5)
    bw, bh = cut.size
    assert cut.rgba.shape == (bh, bw, 4)
    # 중심부는 불투명, 바깥은 투명
    assert cut.rgba[bh // 2, bw // 2, 3] > 200

    p = save_cutout_png(cut, tmp_path / "obj.png")
    back = load_cutout_rgba(p)
    assert back.shape == cut.rgba.shape
    assert back[..., 3].max() == 255


def test_grabcut_rect_validates():
    with pytest.raises(ValueError):
        grabcut_rect(np.zeros((10, 10, 3), np.uint8), (0, 0, 0, 5))
