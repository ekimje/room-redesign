"""전역 기본값. DESIGN.md §3.5 / §10 참고."""

from __future__ import annotations

# 바닥 그리드 간격 (미터)
GRID_SPACING_M: float = 1.0

# 몬테카를로 오차 전파 기본값 (DESIGN.md §3.5 uncertainty 블록)
DEFAULT_CORNER_CLICK_PX: float = 3.0
DEFAULT_FOCAL_REL: float = 0.05
DEFAULT_LENS_DISTORTION_REL: float = 0.04
DEFAULT_RECT_ASSUMPTION_REL: float = 0.06
DEFAULT_MC_SAMPLES: int = 400

# 바닥 코너 클릭 순서 (UI 안내에 사용)
CORNER_ORDER = ("near-left", "near-right", "far-right", "far-left")

# 사진 추정 모드에서 면적/판정에 항상 오차범위를 함께 표기
ALWAYS_REPORT_UNCERTAINTY: bool = True
