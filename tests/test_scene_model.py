"""scene.model / scene.graph 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from room_redesign.geometry import transform_points
from room_redesign.geometry.metrology import FloorCalibration, rectangle_world_corners
from room_redesign.scene import (
    Camera,
    Placement,
    Room,
    ScaleReference,
    Scene,
    SceneObject,
    load_scene,
    save_scene,
)


def _scene(h_gt) -> Scene:
    w, length = 4.0, 3.0
    image_corners = transform_points(h_gt, rectangle_world_corners(w, length))
    calib = FloorCalibration.from_rectangle(image_corners, w, length)
    return Scene(
        image="data/input/room01.jpg",
        camera=Camera(image_size=(1920, 1080), focal_px=1000.0),
        scale_reference=ScaleReference(
            mode="room_rectangle",
            image_pts=image_corners.tolist(),
            width_m=w,
            length_m=length,
        ),
        room=Room(
            image_corners=image_corners.tolist(),
            floor_homography=calib.h_w2i.tolist(),
            floor_polygon_m=calib.world_corners.tolist(),
        ),
        objects=[
            SceneObject(
                id="sofa_1",
                source="cutout",
                asset="data/output/cutouts/sofa_1.png",
                placement=Placement(floor_xy=(1.0, 1.5), yaw_deg=15.0),
            )
        ],
    )


def test_scene_roundtrip(tmp_path, h_gt):
    scene = _scene(h_gt)
    path = tmp_path / "room01.scene.json"
    save_scene(scene, path)
    assert path.exists()

    loaded = load_scene(path)
    assert loaded.image == scene.image
    assert loaded.camera.image_size == (1920, 1080)
    assert loaded.scale_reference.mode == "room_rectangle"
    assert loaded.objects[0].id == "sofa_1"
    assert loaded.objects[0].placement.yaw_deg == pytest.approx(15.0)

    h_back = np.asarray(loaded.room.floor_homography)
    assert np.allclose(h_back, h_gt, atol=1e-6)


def test_defaults_present(h_gt):
    scene = _scene(h_gt)
    assert scene.uncertainty.mc_samples == 400
    assert scene.version == 1
    assert scene.fit_report is None


def test_invalid_scale_mode_rejected():
    with pytest.raises(Exception):
        ScaleReference(mode="bogus")
