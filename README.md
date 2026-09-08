# room-redesign

방 사진 한 장으로 가구·소품을 분리하고, 실제 방 구조 위에서 인테리어 게임처럼
재배치해 보는 도구입니다. 가구/인형 등을 **구매하기 전에** 우리 집에 어떻게 놓일지 미리 확인하는 것이 목표입니다.

- 언어: Python 3.10+
- 핵심: OpenCV, NumPy (바닥 평면 호모그래피 기반 2.5D 재배치)
- 접근: **경량 우선** — 딥러닝 의존 없이 먼저 동작, SAM/YOLO는 이후 단계 옵션

## 문서
설계도(아키텍처, 파이프라인, 마일스톤)는 [docs/DESIGN.md](docs/DESIGN.md) 참고.

## 상태
초기 스캐폴드 (M0). 구현 시작 전 설계 검토 단계.

## 설치 (예정)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 구조
```
src/room_redesign/   패키지 (calibration / geometry / segmentation / scene / compositing / ui)
scripts/             단계별 실행 스크립트
data/                입력 사진·에셋·출력 (git 제외)
tests/               단위 테스트
```
