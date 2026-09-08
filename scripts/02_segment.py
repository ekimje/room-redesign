"""M2 — 방 사진에서 물체를 GrabCut 으로 분리하고 배경을 인페인팅한다.

사용법:
    python scripts/02_segment.py data/input/room01.jpg
    python scripts/02_segment.py data/input/room01.jpg --scene data/output/room01.scene.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from room_redesign.ui import run_segmenter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="GrabCut 물체 분리 + 배경 인페인팅 (M2)")
    ap.add_argument("image", type=Path, help="방 사진 경로")
    ap.add_argument("--scene", type=Path, default=None, help="물체를 추가할 scene.json")
    ap.add_argument("-o", "--out", type=Path, default=None, help="출력 디렉터리 (기본: data/output)")
    args = ap.parse_args()

    if not args.image.exists():
        ap.error(f"이미지를 찾을 수 없음: {args.image}")

    n = run_segmenter(args.image, args.scene, args.out)
    print(f"저장한 컷아웃: {n} 개")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
