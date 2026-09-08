"""1단계 OpenCV 물체 분리 — GrabCut 박스/스크리블 + 배경 인페인팅 미리보기.

DESIGN.md §4.1 / M2.

조작
----
- 박스 드래그      : 물체를 감싸는 사각형 → GrabCut 실행
- f / b           : 보정 브러시를 전경/배경으로 전환
- 좌클릭(보정 중)  : 브러시 스크리블 추가 후 GrabCut 재실행
- i               : 배경 인페인팅 미리보기 토글
- s               : 컷아웃 PNG + 인페인팅 배경 PNG 저장 (scene.json 있으면 물체 추가)
- r               : 초기화
- q / ESC         : 종료
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..compositing import inpaint
from ..geometry import estimate_footprint_m, image_to_floor, invert
from ..io_utils import load_image_rgb
from ..scene import Placement, SceneObject
from ..scene import load_scene, save_scene
from ..segmentation import (
    clean_mask,
    grabcut_rect,
    grabcut_refine,
    make_cutout,
    save_cutout_png,
)

_MAX_DISPLAY = 1400


@dataclass
class SegState:
    image_path: Path
    image_rgb: np.ndarray
    disp_scale: float
    out_dir: Path
    scene_path: Optional[Path] = None
    rect: Optional[tuple[int, int, int, int]] = None      # full-image px
    drag_start: Optional[tuple[float, float]] = None
    drag_now: Optional[tuple[float, float]] = None
    mask: Optional[np.ndarray] = None                     # 이진, full-image
    brush_fg: bool = True
    fg_pts: list[tuple[float, float]] = field(default_factory=list)
    bg_pts: list[tuple[float, float]] = field(default_factory=list)
    show_inpaint: bool = False
    inpaint_cache: Optional[np.ndarray] = None            # 마스크 바뀌면 무효화
    saved_count: int = 0

    def get_inpaint(self) -> Optional[np.ndarray]:
        if self.mask is None:
            return None
        if self.inpaint_cache is None:
            self.inpaint_cache = inpaint(self.image_rgb, self.mask)
        return self.inpaint_cache


def _fit_scale(h: int, w: int) -> float:
    return min(1.0, _MAX_DISPLAY / max(h, w))


def _run_grabcut(state: SegState) -> None:
    if state.rect is None:
        return
    if not state.fg_pts and not state.bg_pts:
        m = grabcut_rect(state.image_rgb, state.rect)
    else:
        base = state.mask
        if base is None:
            base = grabcut_rect(state.image_rgb, state.rect)
        m = grabcut_refine(state.image_rgb, base, state.fg_pts, state.bg_pts)
    state.mask = clean_mask(m)
    state.inpaint_cache = None


def _draw(state: SegState):
    import cv2

    disp = cv2.cvtColor(state.image_rgb, cv2.COLOR_RGB2BGR)
    disp = cv2.resize(disp, None, fx=state.disp_scale, fy=state.disp_scale,
                      interpolation=cv2.INTER_AREA)
    s = state.disp_scale

    if state.show_inpaint and state.mask is not None:
        bg = state.get_inpaint()
        disp = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        disp = cv2.resize(disp, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.putText(disp, "inpaint preview (i)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 2, cv2.LINE_AA)
        return disp

    if state.mask is not None:
        m = cv2.resize(state.mask, (disp.shape[1], disp.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = disp.copy()
        overlay[m] = (0, 200, 0)
        disp = cv2.addWeighted(overlay, 0.35, disp, 0.65, 0)
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(disp, cont, -1, (0, 255, 0), 1, cv2.LINE_AA)

    # 드래그 중 사각형
    if state.drag_start and state.drag_now:
        p0 = tuple(int(round(v * s)) for v in state.drag_start)
        p1 = tuple(int(round(v * s)) for v in state.drag_now)
        cv2.rectangle(disp, p0, p1, (0, 165, 255), 1, cv2.LINE_AA)
    elif state.rect:
        x, y, w, h = state.rect
        cv2.rectangle(disp, (int(x * s), int(y * s)),
                      (int((x + w) * s), int((y + h) * s)), (0, 165, 255), 1)

    for px, py in state.fg_pts:
        cv2.circle(disp, (int(px * s), int(py * s)), 3, (0, 255, 0), -1)
    for px, py in state.bg_pts:
        cv2.circle(disp, (int(px * s), int(py * s)), 3, (0, 0, 255), -1)

    brush = "FG" if state.brush_fg else "BG"
    cv2.putText(disp, f"brush={brush} (f/b)  i=inpaint  s=save  r=reset  q=quit",
                (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return disp


def _save(state: SegState) -> None:
    if state.mask is None or not state.mask.any():
        print("[s] 먼저 물체를 분리하세요.")
        return
    cutout = make_cutout(state.image_rgb, state.mask)
    stem = state.image_path.stem
    oid = f"{stem}_obj{state.saved_count + 1}"
    cutout_path = state.out_dir / "cutouts" / f"{oid}.png"
    save_cutout_png(cutout, cutout_path)

    bg = state.get_inpaint()
    from PIL import Image

    bg_path = state.out_dir / f"{stem}_inpainted.png"
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bg).save(bg_path)

    print(f"컷아웃: {cutout_path}")
    print(f"배경  : {bg_path}")

    if state.scene_path and state.scene_path.exists():
        scene = load_scene(state.scene_path)
        floor_xy = (0.0, 0.0)
        footprint = None
        if scene.room.floor_homography is not None:
            h_i2w = invert(np.asarray(scene.room.floor_homography))
            floor_xy = tuple(image_to_floor(h_i2w, [cutout.base_point])[0].tolist())
            footprint = estimate_footprint_m(h_i2w, cutout.bbox)
        bx, by = cutout.base_point
        x, y, _, _ = cutout.bbox
        scene.objects.append(
            SceneObject(
                id=oid,
                source="cutout",
                asset=str(cutout_path),
                placement=Placement(floor_xy=floor_xy),
                original_floor_xy=floor_xy,
                anchor_px=(bx - x, by - y),
                footprint_m=footprint,
                footprint_source="photo_estimate" if footprint else None,
            )
        )
        scene.background = str(bg_path)
        save_scene(scene, state.scene_path)
        print(f"scene 갱신: {state.scene_path}  (+{oid} @ {floor_xy})")

    state.saved_count += 1


def run_segmenter(
    image_path: str | Path,
    scene_path: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
) -> int:
    import cv2

    image_path = Path(image_path)
    rgb = load_image_rgb(image_path)
    h, w = rgb.shape[:2]
    state = SegState(
        image_path=image_path,
        image_rgb=rgb,
        disp_scale=_fit_scale(h, w),
        out_dir=Path(out_dir) if out_dir else image_path.parent.parent / "output",
        scene_path=Path(scene_path) if scene_path else None,
    )

    win = "room-redesign · segment (M2)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, _param):
        ox, oy = x / state.disp_scale, y / state.disp_scale
        if event == cv2.EVENT_LBUTTONDOWN:
            if state.mask is None:
                state.drag_start = (ox, oy)
                state.drag_now = (ox, oy)
            else:
                (state.fg_pts if state.brush_fg else state.bg_pts).append((ox, oy))
                _run_grabcut(state)
        elif event == cv2.EVENT_MOUSEMOVE and state.drag_start:
            state.drag_now = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and state.drag_start:
            x0, y0 = state.drag_start
            x1, y1 = (ox, oy)
            rx, ry = int(min(x0, x1)), int(min(y0, y1))
            rw, rh = int(abs(x1 - x0)), int(abs(y1 - y0))
            state.drag_start = state.drag_now = None
            if rw > 5 and rh > 5:
                state.rect = (rx, ry, rw, rh)
                _run_grabcut(state)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, _draw(state))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("f"):
            state.brush_fg = True
        elif key == ord("b"):
            state.brush_fg = False
        elif key == ord("i"):
            state.show_inpaint = not state.show_inpaint
        elif key == ord("r"):
            state.rect = state.mask = None
            state.fg_pts.clear()
            state.bg_pts.clear()
            state.show_inpaint = False
        elif key == ord("s"):
            _save(state)

    cv2.destroyWindow(win)
    return state.saved_count
