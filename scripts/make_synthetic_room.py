"""GUI/보정 스모크용 합성 방 이미지 생성.

탑다운 체커보드 바닥(1 타일 = 0.5 m)을 원근 호모그래피로 워프해서 저장하고,
정답(코너 픽셀, 방 실측 폭/길이)을 사이드카 JSON 으로 남긴다.

    python scripts/make_synthetic_room.py
    -> data/input/synthetic_room.png
    -> data/input/synthetic_room.gt.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from room_redesign.geometry import floor_to_image, solve_floor_homography  # noqa: E402
from room_redesign.geometry.metrology import rectangle_world_corners  # noqa: E402

IMG_W, IMG_H = 1600, 1000
ROOM_W, ROOM_L = 3.6, 2.8          # m
TILE = 0.5                          # m


def main() -> int:
    # 바닥 코너를 이미지 안 사다리꼴로 직접 배치 (near 가 넓고 far 가 좁게)
    image_corners = np.array(
        [
            [230, 900],   # near-left
            [1370, 900],  # near-right
            [1080, 430],  # far-right
            [520, 430],   # far-left
        ],
        dtype=float,
    )
    world_corners = rectangle_world_corners(ROOM_W, ROOM_L)
    h_w2i = solve_floor_homography(world_corners, image_corners)

    img = np.full((IMG_H, IMG_W, 3), 235, dtype=np.uint8)  # 밝은 배경(벽)

    # 체커보드 타일을 폴리곤으로 채운다
    try:
        import cv2
    except ImportError:
        print("cv2 가 필요합니다: pip install -r requirements.txt")
        return 1

    nx = int(round(ROOM_W / TILE))
    ny = int(round(ROOM_L / TILE))
    for i in range(nx):
        for j in range(ny):
            shade = 170 if (i + j) % 2 == 0 else 120
            quad_w = np.array(
                [
                    [i * TILE, j * TILE],
                    [(i + 1) * TILE, j * TILE],
                    [(i + 1) * TILE, (j + 1) * TILE],
                    [i * TILE, (j + 1) * TILE],
                ]
            )
            quad_i = floor_to_image(h_w2i, quad_w).astype(np.int32)
            cv2.fillConvexPoly(img, quad_i, (shade, shade, shade), lineType=cv2.LINE_AA)

    # 코너 표시
    for (x, y) in image_corners.astype(int):
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

    out_img = _SRC.parents[0] / "data" / "input" / "synthetic_room.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), img)

    gt = {
        "image": str(out_img),
        "image_corners": image_corners.tolist(),
        "room_width_m": ROOM_W,
        "room_length_m": ROOM_L,
        "floor_area_m2": ROOM_W * ROOM_L,
        "corner_order": ["near-left", "near-right", "far-right", "far-left"],
    }
    out_gt = out_img.with_suffix(".gt.json")
    out_gt.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    print(f"이미지: {out_img}")
    print(f"정답:   {out_gt}  (면적 {ROOM_W * ROOM_L:.2f} m²)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
