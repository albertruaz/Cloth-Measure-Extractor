# Pants Measurement Keypoint Detection

팬츠 측정 항목의 키포인트를 검출하는 딥러닝 모델입니다.

## 설정 (Conda 등 환경)

```bash
# Conda 환경 생성 (선택사항)
conda create -n measure-extractor python=3.8
conda activate measure-extractor

# 패키지 설치
pip install -r requirements.txt
```

## 시작 명령어

### 1. 데이터 전처리 및 분할 (카테고리별 8:1:1)

```bash
python3 src/utils/preprocess_data.py
```

이 스크립트는:

- 원본 CSV에서 팬츠 카테고리만 필터링
- 측정 항목을 파싱하여 학습 가능한 형태로 변환
- 카테고리별로 train/val/test를 8:1:1로 분할
- `data/processed/` 폴더에 저장

### 2. 학습 실행

```bash
# 방법 1: 스크립트 사용 (권장)
./train.sh

# 방법 2: 직접 실행
python3 src/main_train.py --config config.yaml
```

학습 중 생성되는 파일:

- `checkpoints/best.pt`: 최고 성능 모델
- `checkpoints/config.yaml`: 학습 설정 백업

## 파일 구조

```
measure-extractor/
├── data/
│   ├── csv_data/                          # 원본 CSV 데이터
│   │   └── size_measurement_data_include_image_url.csv
│   └── processed/                         # 전처리된 데이터
│       ├── processed_pants_data.csv       # 팬츠 카테고리만 필터링된 데이터
│       ├── processed_pants_data.json      # JSON 형식
│       ├── data_statistics.json           # 데이터 통계
│       ├── train.csv                      # 학습 데이터 (8:1:1 분할)
│       ├── val.csv                        # 검증 데이터
│       └── test.csv                       # 테스트 데이터
├── src/
│   ├── data/
│   │   └── dataset.py                     # 데이터셋 클래스
│   ├── models/
│   │   └── kpnet.py                       # Heatmap 기반 키포인트 검출 모델
│   ├── engine/
│   │   └── trainer.py                     # 학습 엔진
│   ├── infer/
│   │   └── inference.py                   # 추론 스크립트
│   ├── utils/
│   │   ├── preprocess_data.py             # 데이터 전처리 및 분할
│   │   ├── heatmap.py                     # Heatmap 생성/디코딩
│   │   ├── metrics.py                     # 평가 메트릭
│   │   ├── config.py                      # Config 로더
│   │   └── device.py                      # Device 설정
│   └── main_train.py                      # 학습 메인 스크립트
├── checkpoints/                           # 학습된 모델 저장
├── config.yaml                            # 학습 설정
├── requirements.txt                       # 패키지 의존성
└── train.sh                               # 학습 실행 스크립트
```
# Cloth-Measure-Extractor
