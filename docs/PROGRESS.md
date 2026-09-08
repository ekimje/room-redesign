# 진행 상황 (Progress Log)

room-redesign 구현 진행 기록. 설계는 [DESIGN.md](DESIGN.md), 마일스톤 표는 DESIGN.md §8.

| 마일스톤 | 상태 | 커밋 |
|---|---|---|
| M0 스캐폴드 + 설계도 | ✅ | `b2a1f21` |
| §10 치수/오차 설계 추가 | ✅ | `8669b2d` |
| M1 / M1.5 바닥 호모그래피·그리드·면적+오차 | ✅ | `0fa5981` |
| M2 GrabCut 분리 + 인페인팅 | ✅ | `3b17dd4` |
| M3 바닥 배치·드래그·원근 스케일·그림자 | ✅ | (이번 커밋) |
| M4 다물체 장면·가림 정렬 정식화 | ⬜ 다음 | |

측정 모드: **사진 추정 모드** 고정 (단순 인테리어 변경 목적). 정확 모드는 미구현.
인페인팅: **LaMa 기본**(`method="auto"`), 미설치 시 Telea 폴백.

---

## M1 · M1.5 — 바닥 보정 & 면적 측정

### 구현
- `geometry/homography.py` — numpy 전용(cv2 불필요)
  - `solve_floor_homography` : 4점 이상 대응에서 DLT 로 world(m)→image(px) 호모그래피
  - `image_to_floor` / `floor_to_image` / `invert` / `transform_points`
  - `local_pixel_scale` : 국소 야코비안 |det J|^0.5 = 미터당 픽셀 (원근 크기 보정용)
  - `perspective_scale_factor` : 두 바닥 위치의 스케일 비
  - `floor_grid_lines`, `polygon_area`(shoelace)
- `geometry/metrology.py`
  - `FloorCalibration.from_rectangle(image_corners, W, L)` — 사진 추정 모드 진입점
  - `measure_area_m2` / `measure_length_m` — 이미지에서 클릭한 바닥 도형의 실측
  - `scale_from_known_segment` — 종횡비만 알 때 기준 선분으로 등방 배율
- `geometry/uncertainty.py`
  - `monte_carlo_polygon_area` — 코너 클릭 σ + 방치수 σ 를 정규분포로 N회 샘플 → 넓이 분포
  - `Estimate` — `mean ± std`, `p5`/`p95`, 상대오차. **단일 숫자 단독 표기 금지**
- `scene/model.py` + `graph.py` — pydantic 스키마(DESIGN §3.5 전체) + JSON 저장/불러오기
- `io_utils.py` — Pillow 로 RGB 로드, EXIF 35mm 환산 초점거리 → focal_px 근사
- `ui/preview_cv.py` + `scripts/01_calibrate.py` — 코너 4클릭 → 그리드 오버레이 → 측정 모드
- `scripts/make_synthetic_room.py` — 스모크용 합성 방(3.6×2.8 m, 0.5 m 타일) + 정답 JSON

### 주요 결정
- **좌표계**: world = 바닥 미터(X 오른쪽, Y 카메라에서 멀어짐), image = 픽셀(u,v).
- **호모그래피는 numpy DLT** 로 직접 풀어 핵심 기하에서 OpenCV 의존 제거 → 테스트가 가볍다.
- 원근 크기 보정은 forward 벡터 없이 **야코비안 행렬식**으로 등방 근사 (§7.3 대안).
- 절대 크기는 스케일 기준 1개 필수 — 없으면 상대비율만.

### 검증
- 합성 방: 면적 정답 10.08 m² → 보정 복원 **10.08 m²**.
- 코너 3px + 치수 6% 노이즈 → **10.08 ± 0.83 m² (±8.2%)** (설계도 "좋은 조건 ±10~15%" 부합).
- 테스트: `test_homography` `test_metrology` `test_uncertainty` `test_scene_model`.

---

## M2 — 물체 분리 & 배경 인페인팅

### 구현
- `segmentation/grabcut.py`
  - `grabcut_rect` : 바운딩 박스 초기화(`GC_INIT_WITH_RECT`)
  - `grabcut_refine` : 전경/배경 스크리블(`GC_INIT_WITH_MASK`)
- `segmentation/mask_ops.py` : 열기·닫기 모폴로지, 최대 연결요소, 알파 페더링,
  `mask_bbox`, `base_point`(바닥 접촉점 — M3 로 이어짐)
- `segmentation/cutout.py` : `ObjectCutout(rgba, bbox, base_point, mask)` + RGBA PNG 저장/불러오기
- `compositing/inpaint.py`
  - `inpaint(method="auto"|"lama"|"telea"|"ns", dilate=...)` — auto 는 LaMa 설치 시 LaMa
  - `_inpaint_lama` : `SimpleLama` 인스턴스 모듈 캐시(모델 로드가 무겁다)
- `ui/segment_cv.py` + `scripts/02_segment.py` — 박스 드래그 → GrabCut, f/b 브러시 보정,
  i 인페인팅 미리보기(결과 캐시), s 저장. `--scene` 시 scene.json 에 물체 추가하며
  `base_point` 를 바닥으로 역투영해 `original_floor_xy`·`anchor_px`·`footprint_m` 기록.

### 주요 결정
- **경량 1순위 GrabCut** 유지. SAM/YOLO 는 `segmentation/backends/` 로 미룸(M7+).
- 인페인팅 기본을 **LaMa 로 전환**(사용자 요청). torch 는 `requirements-ml.txt` 에만 두고
  런타임은 없으면 Telea 로 자동 폴백 → 경량 설치도 계속 동작.
- 큰 균일 평면(체커보드 등)에서 Telea 는 별표 자국 → LaMa 는 거의 이음매 없음(데모로 확인).

### 검증
- 합성 물체 GrabCut IoU **> 0.85**, 박스 바깥 오염 < 2%.
- 인페인팅 후 마스크 코어 평균색이 배경색으로 수렴(주황→파랑).
- LaMa 경로는 설치 시에만 도는 테스트(`test_lama_fills_region`).
- 테스트: `test_grabcut` `test_inpaint`.

### 의존성 변화 (LaMa 설치 여파)
- torch 생태계가 **numpy<2** 를 요구 → `requirements.txt` 를 `numpy>=1.26,<2`,
  `opencv-python>=4.9,<5` 로 고정. 미사용이던 `scipy` 는 제거.
- 설치된 버전: numpy 1.26.4 · opencv 4.11.0 · torch 2.14.0+cpu · Pillow 9.5.0.

---

## M3 — 바닥 배치 · 드래그 · 원근 스케일 · 그림자

### 구현
- `geometry/place.py`
  - `base_to_floor` : 바닥 접촉점 픽셀 → world(m)
  - `estimate_footprint_m` : bbox 하단 좌우를 역투영해 폭(m), 깊이도 같다고 보고 정사각
  - `anchor_in_sprite` : full-image 접촉점 → 스프라이트 내부 좌표
- `compositing/shadow.py` — `floor_shadow` : 발자국 타원을 world 에서 만들어
  호모그래피로 워프 → `fillPoly` → GaussianBlur → 곱하기 합성. `light_dir` 로 오프셋.
- `compositing/renderer.py`
  - `PlacedSprite(rgba, anchor_xy, origin_floor_xy, floor_xy, footprint_m, scale, yaw_deg)`
  - `render_scene` : 이미지 v(깊이) 오름차순 정렬 → 먼 것부터, 각 물체마다
    ① `local_pixel_scale(new)/local_pixel_scale(origin) * scale` 로 리사이즈
    ② anchor 가 `floor_to_image(floor_xy)` 에 오도록 배치 ③ 알파 합성 + 그림자
  - `sprites_from_scene` / `render_from_scene` : pydantic Scene → 스프라이트(에셋 로드) → 합성
- `scene/model.py` : `SceneObject.anchor_px` 추가
- `ui/edit_cv.py` + `scripts/03_arrange.py` — 베이스 근처 클릭으로 선택, 드래그로 바닥 이동,
  `[`/`]` 스케일, `,`/`.` 회전(이미지 평면 근사), `g` 그림자, `s` scene.json + render PNG 저장
- `scripts/demo_pipeline.py` — M1→M2→M3 헤드리스 엔드투엔드(입력 | cut+inpaint | 이동 결과)

### 주요 결정
- **가림 처리**: 카메라 원점을 모르므로 바닥 접촉점의 이미지 v 를 깊이 프록시로 사용,
  v 오름차순(먼 것 먼저) 합성. 벽/배경은 자연히 맨 뒤(배경 이미지).
- **yaw**: 진짜 회전은 원근 워프가 필요 → M3 에서는 이미지 평면 내 회전으로 근사(기본 0).
- **footprint 깊이**: 단일 이미지에서 알 수 없어 폭과 동일 가정. 외부 가구는 스펙 W×D 사용 예정(M6).
- 그림자는 물체보다 **먼저** 그려 스프라이트가 자기 그림자를 덮게.

### 검증
- 뒤로 옮기면 빨강 픽셀 수가 근접 대비 **< 70%** 로 감소(원근 스케일).
- 스프라이트 최하단 y ≈ 투영된 바닥 접촉점(±3 px), x 중앙 ≈ 접촉점(±6 px).
- 겹친 두 물체 중 가까운(파랑) 것이 위에 보임.
- 그림자 On 시 전체 밝기 합 감소, 먼 구석은 불변, 원본 배열 보존(inplace=False).
- `render_from_scene` : Scene → 합성 이미지 shape/내용 확인.
- 데모: 물체 바닥 위치 (1.65, 0.47) m, 발자국 ~1.08 m — 이동 시 축소 + 그림자 육안 확인.
- 테스트: `test_place` `test_renderer` `test_shadow`.

---

## 현재 상태 요약
- 테스트 **41개 통과** (`pytest -q`), LaMa 경로 포함 시 ~27s.
- 패키지: `geometry`(numpy) · `segmentation`(cv2) · `compositing`(cv2[+torch]) · `scene`(pydantic) · `ui`(cv2).
- 스크립트: `01_calibrate` → `02_segment` → `03_arrange`, `make_synthetic_room`, `demo_pipeline`.

## 알려진 한계
- 2.5D: 물체를 옆에서 본 왜곡·정확한 yaw 회전 불가(빌보드 근사).
- footprint 깊이 추정 없음(정사각 가정) → 긴 소파류는 M6 에서 스펙 입력 권장.
- 바닥 코너가 가구에 가리면 클릭 정밀도 저하 → 오차 증가(§10.4).
- 그림자는 단일 소프트 블롭(방향·세기 파라미터만). 접촉 그림자/AO 없음.
- LaMa 는 CPU 에서 수 초 소요(결과 캐시로 UI 프레임마다 재계산은 방지).

## 다음 (M4~)
- M4: scene.json 다물체 왕복·`z_order_hint`·가림 규칙 정식 테스트.
- M5: PySide6 편집기(속성 패널·물체 목록·내보내기).
- M6: 외부 가구 PNG + 스펙 footprint 배치, §10 fit-check 패널.
