"""1단계 OpenCV 배치 편집기 — 컷아웃을 바닥에서 드래그·스케일·회전. M3.

DESIGN.md §4.1.

조작
----
- 좌클릭+드래그 : 물체를 바닥 위에서 이동 (베이스 근처 클릭으로 선택)
- [ / ]        : 선택 물체 축소 / 확대
- , / .        : 선택 물체 회전(이미지 평면 근사)
- g            : 그림자 토글
- s            : scene.json 저장 + 렌더 PNG 저장
- r            : 배치를 원래 위치로
- q / ESC      : 종료
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..compositing.renderer import PlacedSprite, render_scene, sprites_from_scene
from ..geometry.homography import floor_to_image, image_to_floor, invert
from ..io_utils import load_image_rgb
from ..scene import load_scene, save_scene

_MAX_DISPLAY = 1400
_PICK_RADIUS_PX = 60  # full-image 기준 선택 반경


@dataclass
class EditState:
    scene_path: Path
    base_dir: Path
    background: np.ndarray
    h_w2i: np.ndarray
    sprites: list[PlacedSprite]
    disp_scale: float
    selected: Optional[int] = None
    dragging: bool = False
    grab_offset_m: tuple[float, float] = (0.0, 0.0)
    shadows: bool = True
    _origins: list[tuple[float, float]] = field(default_factory=list)

    @property
    def h_i2w(self) -> np.ndarray:
        return invert(self.h_w2i)


def _fit_scale(h: int, w: int) -> float:
    return min(1.0, _MAX_DISPLAY / max(h, w))


def _pick(state: EditState, ox: float, oy: float) -> Optional[int]:
    best, best_d = None, _PICK_RADIUS_PX
    for i, sp in enumerate(state.sprites):
        b = floor_to_image(state.h_w2i, [sp.floor_xy])[0]
        d = float(np.hypot(b[0] - ox, b[1] - oy))
        if d < best_d:
            best, best_d = i, d
    return best


def _draw(state: EditState):
    import cv2

    frame = render_scene(state.background, state.h_w2i, state.sprites,
                         shadows=state.shadows)
    disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    disp = cv2.resize(disp, None, fx=state.disp_scale, fy=state.disp_scale,
                      interpolation=cv2.INTER_AREA)
    s = state.disp_scale

    for i, sp in enumerate(state.sprites):
        b = floor_to_image(state.h_w2i, [sp.floor_xy])[0] * s
        p = (int(round(b[0])), int(round(b[1])))
        color = (0, 215, 255) if i == state.selected else (200, 200, 200)
        cv2.circle(disp, p, 5, color, -1)
        cv2.putText(disp, sp.obj_id or f"obj{i}", (p[0] + 8, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    hud = "drag=move  [ ] scale  , . rotate  g shadow  s save  r reset  q quit"
    if state.selected is not None:
        sp = state.sprites[state.selected]
        hud = (f"{sp.obj_id}: xy=({sp.floor_xy[0]:.2f},{sp.floor_xy[1]:.2f})m "
               f"scale={sp.scale:.2f} yaw={sp.yaw_deg:.0f}  |  " + hud)
    cv2.putText(disp, hud, (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return disp


def _save(state: EditState) -> None:
    from PIL import Image

    scene = load_scene(state.scene_path)
    by_id = {sp.obj_id: sp for sp in state.sprites}
    for obj in scene.objects:
        sp = by_id.get(obj.id)
        if sp is None:
            continue
        obj.placement.floor_xy = (float(sp.floor_xy[0]), float(sp.floor_xy[1]))
        obj.placement.scale = float(sp.scale)
        obj.placement.yaw_deg = float(sp.yaw_deg)
    save_scene(scene, state.scene_path)

    frame = render_scene(state.background, state.h_w2i, state.sprites,
                         shadows=state.shadows)
    out_png = state.scene_path.with_name(state.scene_path.stem + ".render.png")
    Image.fromarray(frame).save(out_png)
    print(f"저장: {state.scene_path}  +  {out_png}")


def run_editor(scene_path: str | Path, base_dir: Optional[str | Path] = None) -> int:
    import cv2

    scene_path = Path(scene_path)
    scene = load_scene(scene_path)
    if scene.room.floor_homography is None:
        raise ValueError("floor_homography 없음 — 먼저 scripts/01_calibrate.py 로 보정하세요.")

    base = Path(base_dir) if base_dir else Path(".")
    bg_path = scene.background if scene.background and Path(scene.background).exists() else scene.image
    background = load_image_rgb(bg_path)
    sprites = sprites_from_scene(scene, base)
    if not sprites:
        print("배치할 물체가 없습니다 — 먼저 scripts/02_segment.py 로 컷아웃을 추가하세요.")
        return 0

    h, w = background.shape[:2]
    state = EditState(
        scene_path=scene_path,
        base_dir=base,
        background=background,
        h_w2i=np.asarray(scene.room.floor_homography, dtype=float),
        sprites=sprites,
        disp_scale=_fit_scale(h, w),
        _origins=[tuple(sp.floor_xy) for sp in sprites],
    )

    win = "room-redesign · arrange (M3)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, _param):
        ox, oy = x / state.disp_scale, y / state.disp_scale
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = _pick(state, ox, oy)
            state.selected = idx
            if idx is not None:
                state.dragging = True
                fx, fy = image_to_floor(state.h_i2w, [(ox, oy)])[0]
                sx, sy = state.sprites[idx].floor_xy
                state.grab_offset_m = (sx - float(fx), sy - float(fy))
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging and state.selected is not None:
            fx, fy = image_to_floor(state.h_i2w, [(ox, oy)])[0]
            gx, gy = state.grab_offset_m
            state.sprites[state.selected].floor_xy = (float(fx) + gx, float(fy) + gy)
        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False

    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, _draw(state))
        key = cv2.waitKey(20) & 0xFF
        sel = None if state.selected is None else state.sprites[state.selected]
        if key in (ord("q"), 27):
            break
        elif key == ord("g"):
            state.shadows = not state.shadows
        elif key == ord("s"):
            _save(state)
        elif key == ord("r"):
            for sp, o in zip(state.sprites, state._origins):
                sp.floor_xy = tuple(o)
                sp.scale, sp.yaw_deg = 1.0, 0.0
        elif sel is not None and key == ord("["):
            sel.scale = max(0.1, sel.scale * 0.91)
        elif sel is not None and key == ord("]"):
            sel.scale = min(10.0, sel.scale * 1.1)
        elif sel is not None and key == ord(","):
            sel.yaw_deg -= 5.0
        elif sel is not None and key == ord("."):
            sel.yaw_deg += 5.0

    cv2.destroyWindow(win)
    return len(state.sprites)
