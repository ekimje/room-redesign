"""임포트 스모크 테스트 (M0/M1). cv2 없이 동작하는 부분만."""


def test_package_imports():
    import room_redesign

    assert room_redesign.__version__


def test_core_modules_import():
    from room_redesign import geometry, scene  # noqa: F401
    from room_redesign.geometry import solve_floor_homography  # noqa: F401
    from room_redesign.scene import Scene  # noqa: F401
