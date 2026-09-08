"""compositing.renderer 단위 테스트 (cv2 필요)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from room_redesign.compositing.renderer import (  # noqa: E402
    PlacedSprite,
    render_from_scene,
    render_scene,
)
from room_redesign.geometry.homography import floor_to_image  # noqa: E402


def _white_bg(size):
    w, h = size
    return np.full((h, w, 3), 255, np.uint8)


def _red_count(img):
    return int(((img[..., 0] > 150) & (img[..., 1] < 100) & (img[..., 2] < 100)).sum())


def test_perspective_scale_shrinks_when_moved_far(room_homography, red_sprite):
    h_w2i, size, (rw, rl) = room_homography
    rgba, anchor = red_sprite
    bg = _white_bg(size)

    near = PlacedSprite(rgba, anchor, origin_floor_xy=(2.0, 0.3), floor_xy=(2.0, 0.3),
                        footprint_m=(0.4, 0.4))
    far = PlacedSprite(rgba, anchor, origin_floor_xy=(2.0, 0.3), floor_xy=(2.0, 2.7),
                       footprint_m=(0.4, 0.4))

    n_near = _red_count(render_scene(bg, h_w2i, [near], shadows=False))
    n_far = _red_count(render_scene(bg, h_w2i, [far], shadows=False))
    assert n_near > 0 and n_far > 0
    assert n_far < 0.7 * n_near  # 뒤로 갈수록 확실히 작아짐


def test_sprite_anchored_at_projected_base(room_homography, red_sprite):
    h_w2i, size, _ = room_homography
    rgba, anchor = red_sprite
    bg = _white_bg(size)
    floor_xy = (2.0, 1.4)
    sp = PlacedSprite(rgba, anchor, origin_floor_xy=floor_xy, floor_xy=floor_xy)

    out = render_scene(bg, h_w2i, [sp], shadows=False)
    b = floor_to_image(h_w2i, [floor_xy])[0]

    ys, xs = np.where((out[..., 0] > 150) & (out[..., 2] < 100))
    assert xs.size > 0
    # 스프라이트 최하단이 투영된 바닥 접촉점 근처
    assert ys.max() == pytest.approx(b[1], abs=3)
    assert np.median(xs) == pytest.approx(b[0], abs=6)


def test_depth_order_nearer_on_top(room_homography):
    h_w2i, size, _ = room_homography
    bg = _white_bg(size)

    red = np.zeros((80, 80, 4), np.uint8); red[..., 0] = 220; red[..., 3] = 255
    blue = np.zeros((80, 80, 4), np.uint8); blue[..., 2] = 220; blue[..., 3] = 255
    anchor = (40.0, 80.0)

    far = PlacedSprite(red, anchor, (2.0, 1.2), (2.0, 1.2), obj_id="red_far")
    near = PlacedSprite(blue, anchor, (2.0, 1.0), (2.0, 1.0), obj_id="blue_near")

    out = render_scene(bg, h_w2i, [far, near], shadows=False)
    # 두 스프라이트가 겹치는 영역 위쪽(near 의 몸통)에는 파랑이 보여야
    b_near = floor_to_image(h_w2i, [(2.0, 1.0)])[0]
    patch = out[int(b_near[1]) - 30 : int(b_near[1]) - 10,
                int(b_near[0]) - 10 : int(b_near[0]) + 10]
    assert (patch[..., 2] > 150).mean() > 0.5


def test_shadow_darkens_output(room_homography, red_sprite):
    h_w2i, size, _ = room_homography
    rgba, anchor = red_sprite
    bg = _white_bg(size)
    sp = PlacedSprite(rgba, anchor, (2.0, 1.5), (2.0, 1.5), footprint_m=(0.6, 0.6))

    no_sh = render_scene(bg, h_w2i, [sp], shadows=False).astype(int)
    with_sh = render_scene(bg, h_w2i, [sp], shadows=True).astype(int)
    assert with_sh.sum() < no_sh.sum()  # 그림자로 전체 밝기 감소


def test_render_from_scene(room_homography, red_sprite, tmp_path):
    from PIL import Image

    from room_redesign.scene import Camera, Placement, Room, Scene, SceneObject

    h_w2i, size, _ = room_homography
    rgba, anchor = red_sprite
    asset = tmp_path / "obj.png"
    Image.fromarray(rgba, "RGBA").save(asset)

    scene = Scene(
        image="bg.png",
        camera=Camera(image_size=size),
        room=Room(image_corners=[[0, 0]] * 4, floor_homography=h_w2i.tolist()),
        objects=[
            SceneObject(
                id="o1", source="cutout", asset=str(asset),
                placement=Placement(floor_xy=(2.0, 1.5)),
                original_floor_xy=(2.0, 1.5), anchor_px=list(anchor),
                footprint_m=(0.4, 0.4),
            )
        ],
    )
    out = render_from_scene(scene, _white_bg(size), base_dir=tmp_path, shadows=False)
    assert out.shape == (size[1], size[0], 3)
    assert _red_count(out) > 0
