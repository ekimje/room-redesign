"""테스트 공통 픽스처."""

from __future__ import annotations

import numpy as np
import pytest

# 임의의(하지만 타당한) 바닥 호모그래피: world 미터 → image 픽셀.
# 완만한 원근을 가지도록 h20, h21 을 작게 둔다.
H_GT = np.array(
    [
        [900.0, 40.0, 300.0],
        [10.0, 850.0, 200.0],
        [5.0e-4, 1.6e-3, 1.0],
    ]
)


@pytest.fixture
def h_gt() -> np.ndarray:
    return H_GT.copy()


@pytest.fixture
def room_wl() -> tuple[float, float]:
    """테스트용 방 폭/길이 (m)."""
    return 4.0, 3.0
