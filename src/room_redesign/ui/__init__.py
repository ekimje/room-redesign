"""ui — 대화형 편집/보정. DESIGN.md §4.

- preview_cv : 1단계 바닥 보정 + 그리드 (M1)
- segment_cv : 물체 분리 + 인페인팅 (M2)
- edit_cv    : 바닥 위 배치/드래그/스케일 + 그림자 (M3)
- app / canvas : 2단계 PySide6 편집기 (이후)
"""

from .edit_cv import run_editor
from .preview_cv import run_calibrator
from .segment_cv import run_segmenter

__all__ = ["run_calibrator", "run_segmenter", "run_editor"]
