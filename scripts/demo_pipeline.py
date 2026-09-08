"""M1→M2→M3 헤드리스 엔드투엔드 데모 (GUI 없이).

합성 방에 가짜 가구를 올려놓고: 보정 → GrabCut 분리 → 인페인팅 →
원위치/이동 위치로 렌더해 원근 스케일과 그림자를 눈으로 확인한다.

    python scripts/demo_pipeline.py
    -> data/output/demo_pipeline.png  (input | cutout+bg | 재배치 결과)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2  # noqa: E402

from room_redesign.compositing import inpaint, render_scene  # noqa: E402
from room_redesign.compositing.renderer import PlacedSprite  # noqa: E402
from room_redesign.geometry import estimate_footprint_m  # noqa: E402
from room_redesign.geometry.homography import image_to_floor  # noqa: E402
from room_redesign.geometry.metrology import FloorCalibration  # noqa: E402
from room_redesign.io_utils import load_image_rgb  # noqa: E402
from room_redesign.segmentation import clean_mask, grabcut_rect, make_cutout  # noqa: E402


def main() -> int:
    root = _SRC.parent
    gt = json.loads((root / "data/input/synthetic_room.gt.json").read_text(encoding="utf-8"))
    room = load_image_rgb(root / "data/input/synthetic_room.png").copy()

    # 1) 보정: 정답 코너 + 실측 치수 -> 바닥 호모그래피
    calib = FloorCalibration.from_rectangle(
        gt["image_corners"], gt["room_width_m"], gt["room_length_m"]
    )
    h_w2i, h_i2w = calib.h_w2i, calib.h_i2w

    # 2) 가짜 '수납장'을 바닥에 올린다
    cv2.rectangle(room, (620, 545), (900, 760), (150, 92, 60), -1)
    rng = np.random.default_rng(2)
    sl = (slice(540, 765), slice(615, 905))
    room[sl] = np.clip(room[sl] + rng.normal(0, 5, room[sl].shape), 0, 255).astype(np.uint8)

    # 3) GrabCut 분리 + LaMa(auto) 인페인팅
    mask = clean_mask(grabcut_rect(room, (600, 525, 320, 260)))
    cut = make_cutout(room, mask)
    bg = inpaint(room, mask, method="auto", dilate=5)

    anchor = (cut.base_point[0] - cut.bbox[0], cut.base_point[1] - cut.bbox[1])
    origin_xy = tuple(image_to_floor(h_i2w, [cut.base_point])[0].tolist())
    footprint = estimate_footprint_m(h_i2w, cut.bbox)
    print(f"물체 바닥 위치 {tuple(round(v,2) for v in origin_xy)} m, 발자국 ~{footprint[0]:.2f} m")

    def sprite_at(xy):
        return PlacedSprite(cut.rgba, anchor, origin_xy, xy, footprint_m=footprint)

    same = render_scene(bg, h_w2i, [sprite_at(origin_xy)])
    moved_xy = (origin_xy[0] - 1.0, origin_xy[1] + 1.2)   # 왼쪽·뒤로
    moved = render_scene(bg, h_w2i, [sprite_at(moved_xy)])

    def lab(im, t):
        im = im.copy()
        cv2.putText(im, t, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        return im

    montage = np.hstack([lab(room, "1. input"), lab(same, "2. cut+inpaint, same spot"),
                         lab(moved, "3. moved back-left (smaller + shadow)")])
    out = root / "data/output/demo_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(montage, None, fx=0.5, fy=0.5)[:, :, ::-1])
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
