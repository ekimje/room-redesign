"""M3 — 컷아웃을 바닥에서 드래그·스케일·회전하며 방을 재배치한다.

사용법:
    python scripts/03_arrange.py data/output/room01.scene.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from room_redesign.ui import run_editor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="바닥 위 물체 배치 편집 (M3)")
    ap.add_argument("scene", type=Path, help="scene.json 경로")
    ap.add_argument("--base", type=Path, default=Path("."), help="에셋 상대경로 기준 디렉터리")
    args = ap.parse_args()

    if not args.scene.exists():
        ap.error(f"scene 파일을 찾을 수 없음: {args.scene}")

    n = run_editor(args.scene, args.base)
    print(f"물체 {n} 개 편집 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
