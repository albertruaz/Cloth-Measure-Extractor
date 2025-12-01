# 코드 간소화 완료 요약

## ✅ 변경 사항

### 1. DB에서 데이터 가져오기
**파일**: `fetch_data.py` (새로 생성)
- DB 설정 파일 읽기
- MySQL 연결 및 쿼리 실행
- CSV로 저장

**설정**: `db_config.yaml` (새로 생성)
```bash
python fetch_data.py --config db_config.yaml --output data/raw_data.csv
```

### 2. 모델 학습
**파일**: `train.py` (새로 생성, 기존 main_train.py + trainer.py 통합)
- 간단하고 명확한 학습 루프
- 카테고리별 학습 지원
- 체크포인트 자동 저장

**사용법**:
```bash
# 기본 학습
python train.py --config config_simple.yaml

# 카테고리별 학습
python train.py --config config_simple.yaml --category pants
```

### 3. 결과 시각화
**파일**: `visualize.py` (새로 생성)
- 체크포인트에서 모델 로드
- 예측 결과를 이미지에 그리기
- 정답과 예측 비교

**사용법**:
```bash
python visualize.py \
    --checkpoint checkpoints/best.pt \
    --data data/processed/test.csv \
    --output results/
```

## 🗑️ 삭제된 파일/폴더

### 삭제된 폴더:
- `src/pipelines/` - 복잡한 빌드 파이프라인
- `src/infer/` - 별도 추론 모듈
- `src/engine/` - 복잡한 trainer 모듈

### 삭제된 파일:
- `src/main_train.py` → `train.py`로 통합
- `src/engine/trainer.py` → `train.py`로 통합
- `src/utils/metrics.py` → `src/utils/utils.py`로 통합
- `src/utils/heatmap.py` → `src/utils/utils.py`로 통합
- `src/utils/device.py` - 불필요
- `src/utils/config.py` - 불필요
- `src/utils/preprocess_data.py` - 불필요
- `src/utils/visualize_predictions.py` → `visualize.py`로 통합
- `category_config.yaml` - 복잡한 설정 제거
- `train.sh`, `test.sh` - 스크립트 불필요

## 📁 최종 프로젝트 구조

```
extractor/
├── fetch_data.py          ← 새로 생성 (DB 데이터 가져오기)
├── train.py               ← 새로 생성 (모델 학습)
├── visualize.py           ← 새로 생성 (결과 시각화)
├── config_simple.yaml     ← 새로 생성 (간단한 설정)
├── db_config.yaml         ← 새로 생성 (DB 설정)
├── config.yaml            (기존 학습 설정, 유지)
├── README.md              (업데이트됨)
├── requirements.txt
└── src/
    ├── models/
    │   └── kpnet.py       (모델 정의)
    ├── data/
    │   └── dataset.py     (데이터셋)
    └── utils/
        ├── utils.py       ← 새로 생성 (heatmap + metrics 통합)
        └── db_fetcher.py  (DB 접속 유틸리티)
```

## 🎯 핵심 개선 사항

1. **3개의 메인 스크립트만 사용**
   - `fetch_data.py`: 데이터 가져오기
   - `train.py`: 학습
   - `visualize.py`: 시각화

2. **유틸리티 통합**
   - 여러 파일로 분산되어 있던 유틸리티를 `utils.py` 하나로 통합
   - heatmap 생성/디코딩 + 평가 메트릭 모두 포함

3. **설정 파일 간소화**
   - `config_simple.yaml`: 필수 설정만 포함
   - 카테고리별 설정도 하나의 파일에서 관리

4. **코드 가독성 향상**
   - 각 스크립트가 독립적이고 명확한 목적
   - 복잡한 추상화 제거
   - 직관적인 함수명과 구조

## 🚀 사용 방법 (3단계)

```bash
# 1. DB에서 데이터 가져오기
python fetch_data.py --config db_config.yaml --output data/raw_data.csv

# 2. 모델 학습
python train.py --config config_simple.yaml

# 3. 결과 시각화
python visualize.py --checkpoint checkpoints/best.pt --data data/processed/test.csv
```

## 💡 추가 기능

### 카테고리별 학습
```bash
python train.py --config config_simple.yaml --category pants
```

### 설정 커스터마이징
`config_simple.yaml`에서 다음을 수정할 수 있습니다:
- `names`: 측정할 키포인트
- `batch_size`, `epochs`, `learning_rate`: 학습 하이퍼파라미터
- `backbone`: resnet18, resnet34, resnet50 등
- `image_size`, `heatmap_size`: 이미지 크기

## 📊 코드 라인 수 비교

**이전**:
- 여러 파일에 분산된 복잡한 코드
- trainer.py: 480줄
- build_datasets.py: 237줄
- 각종 유틸리티: 500+ 줄

**이후**:
- train.py: 280줄 (모든 학습 로직 포함)
- visualize.py: 200줄 (시각화 전체)
- fetch_data.py: 120줄 (데이터 가져오기)
- utils.py: 120줄 (heatmap + metrics)

**총 라인 수**: ~1500줄 → ~700줄 (53% 감소)
