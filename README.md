# room-redesign

방 사진 한 장으로 가구·소품을 분리하고, 실제 방 구조 위에서 인테리어 게임처럼
재배치해 보는 도구입니다. 가구/인형 등을 **구매하기 전에** 우리 집에 어떻게 놓일지 미리 확인하는 것이 목표입니다.

- 언어: Python 3.10+
- 핵심: OpenCV, NumPy (바닥 평면 호모그래피 기반 2.5D 재배치)
- 접근: **경량 우선** — 딥러닝 의존 없이 먼저 동작, SAM/YOLO는 이후 단계 옵션
- 측정 모드: **사진 추정 모드** (스케일 기준 1개 + 오차범위 동반 표기)

## 문서
설계도(아키텍처, 파이프라인, 마일스톤, 치수/오차)는 [docs/DESIGN.md](docs/DESIGN.md) 참고.

## 진행 상태
- **M0** 스캐폴드 + 설계도 ✔
- **M1 / M1.5** 바닥 호모그래피 · 1m 그리드 오버레이 · 면적 측정 + 몬테카를로 오차범위 ✔
- **M2** GrabCut 물체 분리(박스/스크리블) + RGBA 컷아웃 + 배경 인페인팅 ✔
- 다음: M3 (컷아웃을 바닥에 배치 · 드래그 · 원근 크기 보정 · 그림자)

## 설치
Python 3.12 가 PATH 에 없으면 전체 경로를 쓰세요:
`C:\Users\<사용자>\AppData\Local\Programs\Python\Python312\python.exe`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## 실행 (M1)
```bash
# 스모크용 합성 방 이미지 생성 (선택)
python scripts/make_synthetic_room.py

# 바닥 보정 + 그리드 오버레이 + 면적
python scripts/01_calibrate.py data/input/synthetic_room.png --width 3.6 --length 2.8
```
창 조작: 좌클릭 x4 코너(near-left→near-right→far-right→far-left) · `m` 측정 모드 · `s` 저장 · `q` 종료

## 실행 (M2)
```bash
python scripts/02_segment.py data/input/room01.jpg --scene data/output/room01.scene.json
```
창 조작: 물체를 박스로 드래그 → GrabCut · `f`/`b` 브러시 전환 후 좌클릭으로 보정 · `i` 인페인팅 미리보기 · `s` 저장 · `q` 종료
저장물: `data/output/cutouts/<이름>_objN.png` (RGBA), `data/output/<이름>_inpainted.png`, scene.json 에 물체 추가.
큰 평면 배경은 Telea 자국이 남을 수 있음 → `pip install -r requirements-ml.txt` 후 `inpaint_lama` 사용.

## 구조
```
src/room_redesign/
  geometry/    호모그래피·스케일 측정·오차 전파 (numpy 만, cv2 불필요)
  scene/       장면 그래프 pydantic 모델 + JSON 저장/불러오기
  ui/          preview_cv (1단계 OpenCV), app/canvas (2단계 PySide6, 예정)
  segmentation/ compositing/ viewer3d/   (M2 이후)
scripts/       단계별 실행 스크립트
data/          입력 사진·에셋·출력 (git 제외)
tests/         단위 테스트 (21개 통과)
```
