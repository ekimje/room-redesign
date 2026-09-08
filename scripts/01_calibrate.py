"""M1 · M1.5 — 방 사진에서 바닥을 보정하고 그리드/면적을 확인한다.

사용법:
    python scripts/01_calibrate.py data/input/room01.jpg
    python scripts/01_calibrate.py data/input/room01.jpg --width 3.6 --length 2.8
    python scripts/01_calibrate.py data/input/room01.jpg -o data/output/room01.scene.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from room_redesign.ui import run_calibrator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="바닥 보정 + 1m 그리드 (M1)")
    ap.add_argument("image", type=Path, help="방 사진 경로")
    ap.add_argument("--width", type=float, default=None, help="near 변 실측 폭 (m)")
    ap.add_argument("--length", type=float, default=None, help="side 변 실측 길이 (m)")
    ap.add_argument("-o", "--out", type=Path, default=None, help="scene.json 출력 경로")
    args = ap.parse_args()

    if not args.image.exists():
        ap.error(f"이미지를 찾을 수 없음: {args.image}")

    scene = run_calibrator(args.image, args.width, args.length, args.out)
    if scene is None:
        print("저장 없이 종료됨.")
        return 1
    print("보정 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
