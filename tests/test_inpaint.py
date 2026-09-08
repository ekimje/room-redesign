"""compositing.inpaint 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from room_redesign.compositing import inpaint  # noqa: E402
from room_redesign.compositing.inpaint import _lama_available  # noqa: E402


def test_inpaint_fills_with_background(synthetic_object):
    img, true_mask, _ = synthetic_object
    out = inpaint(img, true_mask, method="telea", radius=3, dilate=3)

    assert out.shape == img.shape
    assert out.dtype == np.uint8

    bg_color = np.array([60, 90, 180])
    fg_color = np.array([230, 140, 40])
    core = np.zeros_like(true_mask)
    core[90:190, 170:250] = 1
    core &= true_mask
    mean_after = out[core > 0].mean(axis=0)
    assert np.linalg.norm(mean_after - bg_color) < np.linalg.norm(mean_after - fg_color)
    assert np.linalg.norm(mean_after - fg_color) > 60


def test_inpaint_leaves_unmasked_untouched(synthetic_object):
    img, true_mask, _ = synthetic_object
    out = inpaint(img, true_mask, method="telea", dilate=0)
    far = (slice(0, 20), slice(0, 20))
    assert np.array_equal(out[far], img[far])


def test_inpaint_ns_method(synthetic_object):
    img, true_mask, _ = synthetic_object
    out = inpaint(img, true_mask, method="ns")
    assert out.shape == img.shape


def test_auto_method_runs(synthetic_object):
    img, true_mask, _ = synthetic_object
    out = inpaint(img, true_mask, method="auto")
    assert out.shape == img.shape and out.dtype == np.uint8


@pytest.mark.skipif(not _lama_available(), reason="simple-lama-inpainting 미설치")
def test_lama_fills_region(synthetic_object):
    img, true_mask, _ = synthetic_object
    out = inpaint(img, true_mask, method="lama", dilate=3)
    assert out.shape == img.shape and out.dtype == np.uint8

    fg_color = np.array([230, 140, 40])
    core = np.zeros_like(true_mask)
    core[90:190, 170:250] = 1
    core &= true_mask
    mean_after = out[core > 0].mean(axis=0)
    # 주황 물체가 사라졌는지
    assert np.linalg.norm(mean_after - fg_color) > 60
