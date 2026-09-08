"""ui — 대화형 편집/보정. DESIGN.md §4.

- preview_cv : 1단계 OpenCV 프리뷰 (M1 바닥 보정 + 그리드)
- app / canvas : 2단계 PySide6 편집기 (이후)
"""

from .preview_cv import run_calibrator

__all__ = ["run_calibrator"]
