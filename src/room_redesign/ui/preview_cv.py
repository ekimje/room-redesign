"""1단계 OpenCV 프리뷰 — 바닥 코너 클릭 → 호모그래피 → 1m 그리드 오버레이.

DESIGN.md §4.1 / M1 · M1.5.

조작
----
- 좌클릭 x4 : 바닥 직사각형 코너 (near-left → near-right → far-right → far-left)
- r         : 코너 초기화
- m         : 측정 모드 토글 (바닥 다각형 좌클릭, Enter 로 넓이+오차범위 계산)
- s         : scene.json 저장
- q / ESC   : 종료

`width_m`, `length_m` 는 클릭한 near/far 변의 실측값. 인자로 주지 않으면 종료 시 stdin 으로 입력받는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config
from ..geometry import (
    FloorCalibration,
    floor_grid_lines,
    floor_to_image,
    transform_points,
)
from ..geometry.uncertainty import monte_carlo_polygon_area
from ..io_utils import exif_focal_px, load_image_rgb
from ..scene import Camera, Room, ScaleReference, Scene, Uncertainty, save_scene

_MAX_DISPLAY = 1400  # 표시 최대 변 길이(px)


@dataclass
class CalibratorState:
    image_path: Path
    image_rgb: np.ndarray
    disp_scale: float                                   # 원본 px → 표시 px
    corners: list[tuple[float, float]] = field(default_factory=list)   # 원본 px
    measure_mode: bool = False
    measure_pts: list[tuple[float, float]] = field(default_factory=list)
    width_m: Optional[float] = None
    length_m: Optional[float] = None
    calib: Optional[FloorCalibration] = None
    last_area_text: str = ""


def _fit_scale(h: int, w: int) -> float:
    longest = max(h, w)
    return min(1.0, _MAX_DISPLAY / longest)


def _draw(state: CalibratorState) -> np.ndarray:
    import cv2

    disp = cv2.cvtColor(state.image_rgb, cv2.COLOR_RGB2BGR)
    disp = cv2.resize(
        disp, None, fx=state.disp_scale, fy=state.disp_scale, interpolation=cv2.INTER_AREA
    )

    def to_disp(pt) -> tuple[int, int]:
        return int(round(pt[0] * state.disp_scale)), int(round(pt[1] * state.disp_scale))

    # 클릭한 코너
    for i, c in enumerate(state.corners):
        p = to_disp(c)
        cv2.circle(disp, p, 6, (0, 165, 255), -1)
        cv2.putText(
            disp, f"{i + 1}:{config.CORNER_ORDER[i]}", (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA,
        )
    if len(state.corners) >= 2:
        pts = np.array([to_disp(c) for c in state.corners], dtype=np.int32)
        cv2.polylines(disp, [pts], len(state.corners) == 4, (0, 165, 255), 1, cv2.LINE_AA)

    # 그리드
    if state.calib is not None:
        bounds = state.calib.floor_bounds()
        for a, b in floor_grid_lines(bounds, config.GRID_SPACING_M):
            seg = floor_to_image(state.calib.h_w2i, [a, b]) * state.disp_scale
            p0 = tuple(np.round(seg[0]).astype(int))
            p1 = tuple(np.round(seg[1]).astype(int))
            cv2.line(disp, p0, p1, (80, 220, 80), 1, cv2.LINE_AA)
        area = state.calib.floor_area_m2()
        cv2.putText(
            disp, f"floor ~ {area:.2f} m^2  ({bounds[1]:.2f} x {bounds[3]:.2f} m)",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2, cv2.LINE_AA,
        )

    # 측정 모드
    if state.measure_mode:
        cv2.putText(
            disp, "MEASURE: click floor polygon, Enter=area, Esc=cancel",
            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA,
        )
        for mp in state.measure_pts:
            cv2.circle(disp, to_disp(mp), 4, (255, 255, 0), -1)
        if len(state.measure_pts) >= 2:
            pts = np.array([to_disp(c) for c in state.measure_pts], dtype=np.int32)
            cv2.polylines(disp, [pts], False, (255, 255, 0), 1, cv2.LINE_AA)
    if state.last_area_text:
        cv2.putText(
            disp, state.last_area_text, (10, disp.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
        )
    return disp


def _try_build_calib(state: CalibratorState) -> None:
    if len(state.corners) == 4 and state.width_m and state.length_m:
        state.calib = FloorCalibration.from_rectangle(
            state.corners, state.width_m, state.length_m
        )


def _make_scene(state: CalibratorState) -> Scene:
    h, w = state.image_rgb.shape[:2]
    calib = state.calib
    return Scene(
        image=str(state.image_path),
        camera=Camera(
            image_size=(w, h),
            focal_px=exif_focal_px(state.image_path, w),
        ),
        scale_reference=ScaleReference(
            mode="room_rectangle",
            image_pts=[list(c) for c in state.corners],
            width_m=state.width_m,
            length_m=state.length_m,
        ),
        uncertainty=Uncertainty(
            corner_click_px=config.DEFAULT_CORNER_CLICK_PX,
            focal_rel=config.DEFAULT_FOCAL_REL,
            lens_distortion_rel=config.DEFAULT_LENS_DISTORTION_REL,
            rect_assumption_rel=config.DEFAULT_RECT_ASSUMPTION_REL,
            mc_samples=config.DEFAULT_MC_SAMPLES,
        ),
        room=Room(
            image_corners=[list(c) for c in state.corners],
            floor_homography=calib.h_w2i.tolist() if calib else None,
            floor_polygon_m=calib.world_corners.tolist() if calib else None,
        ),
    )


def run_calibrator(
    image_path: str | Path,
    width_m: Optional[float] = None,
    length_m: Optional[float] = None,
    scene_out: Optional[str | Path] = None,
) -> Optional[Scene]:
    """대화형 보정 창을 띄운다. 저장했으면 Scene 을 반환, 아니면 None."""
    import cv2

    image_path = Path(image_path)
    rgb = load_image_rgb(image_path)
    h, w = rgb.shape[:2]
    state = CalibratorState(
        image_path=image_path,
        image_rgb=rgb,
        disp_scale=_fit_scale(h, w),
        width_m=width_m,
        length_m=length_m,
    )

    win = "room-redesign · calibrate (M1)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = x / state.disp_scale, y / state.disp_scale
        if state.measure_mode:
            state.measure_pts.append((ox, oy))
        elif len(state.corners) < 4:
            state.corners.append((ox, oy))
            _try_build_calib(state)

    cv2.setMouseCallback(win, on_mouse)
    saved: Optional[Scene] = None

    while True:
        cv2.imshow(win, _draw(state))
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):  # q / ESC
            if state.measure_mode:
                state.measure_mode = False
                state.measure_pts.clear()
                continue
            break
        if key == ord("r"):
            state.corners.clear()
            state.calib = None
            state.last_area_text = ""
        elif key == ord("m"):
            state.measure_mode = not state.measure_mode
            state.measure_pts.clear()
        elif key in (13, 10):  # Enter
            if state.measure_mode and len(state.measure_pts) >= 3 and state.calib:
                est = monte_carlo_polygon_area(
                    state.corners, state.width_m, state.length_m, state.measure_pts,
                    sigma_px=config.DEFAULT_CORNER_CLICK_PX,
                    sigma_wl_rel=config.DEFAULT_RECT_ASSUMPTION_REL,
                    n=config.DEFAULT_MC_SAMPLES,
                )
                state.last_area_text = f"measured area = {est}"
                print(state.last_area_text)
                state.measure_mode = False
                state.measure_pts.clear()
        elif key == ord("s"):
            if len(state.corners) != 4:
                print("[s] 코너 4점을 먼저 찍으세요.")
                continue
            if not (state.width_m and state.length_m):
                try:
                    state.width_m = float(input("near 변 실측 폭 (m): "))
                    state.length_m = float(input("side 변 실측 길이 (m): "))
                except ValueError:
                    print("숫자를 입력하세요.")
                    continue
                _try_build_calib(state)
            out = Path(scene_out) if scene_out else image_path.with_suffix(".scene.json")
            saved = _make_scene(state)
            save_scene(saved, out)
            print(f"저장됨: {out}")

    cv2.destroyWindow(win)
    return saved
