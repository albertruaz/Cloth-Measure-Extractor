# 의류 측정 키포인트 추출기 (Cloth Measure Extractor)

의류 이미지에서 측정 키포인트를 추출하는 딥러닝 모델

## 🎯 주요 기능

1. **DB에서 데이터 가져오기** - MySQL DB에서 측정 데이터 추출
2. **모델 학습** - 카테고리별 키포인트 검출 모델 학습
3. **결과 시각화** - 학습된 모델의 예측 결과 시각화

## 📁 프로젝트 구조

```
extractor/
├── fetch_data.py          # 1. DB 데이터 가져오기
├── train.py               # 2. 모델 학습
├── visualize.py           # 3. 결과 시각화
├── .env                   # DB 연결 정보 (민감 정보)
├── db_config.yaml         # 쿼리 설정
├── config.yaml            # 학습 설정
└── src/                   # 소스 코드
```

## ⚙️ 설치

```bash
# 패키지 설치
pip install -r requirements.txt

# .env 파일 설정
cp .env.example .env
# .env 파일을 열어서 실제 DB 정보 입력
```

## 🚀 사용 방법

### 0. 환경 설정

**.env 파일 설정** (DB 연결 정보):
```bash
# SSH 터널 설정
SSH_ENABLED=false
SSH_HOST=13.125.49.0
SSH_USER=ubuntu
SSH_KEY_FILE=vingle.pem
SSH_LOCAL_PORT=3307

# 데이터베이스 연결
DB_HOST=localhost
DB_PORT=3307
DB_USER=vingle_ai_read
DB_PASSWORD=password1234
DB_NAME=vingle_ai
```

**db_config.yaml** (쿼리 설정):
```yaml
query:
  categories:
    - "데님 팬츠"
    - "코튼 팬츠"
  date_from: "2025-10-10"
  member_ids: [2, 3]
```

### 1. SSH 터널 설정

```bash
# 터미널 1 (SSH 터널 - 백그라운드 실행)
ssh -i vingle.pem ubuntu@13.125.49.0 \
    -L 3307:vingle-ai-rds-instance-1.c5gyzd5rkihd.ap-northeast-2.rds.amazonaws.com:3306 \
    -N &
```

### 2. DB에서 데이터 가져오기

```bash
# 기본 실행 (db_config.yaml 사용)
python fetch_data.py

# 출력 경로 지정
python fetch_data.py --output data/my_data.csv
```

### 3. 모델 학습

```bash
# 기본 실행 (config.yaml 사용)
python train.py

# 카테고리별 학습
python train.py --category pants
```

### 4. 결과 시각화

```bash
# 기본 실행
python visualize.py

# 옵션 지정
python visualize.py --num_samples 20
```

## 📊 모델 아키텍처

- **Backbone**: ResNet (resnet18/34/50/101)
- **Head**: Deconvolution layers (3층)
- **Output**: Gaussian heatmap
- **Loss**: MSE Loss (visibility mask 적용)

## 🎨 측정 키포인트

### 팬츠 (Pants)
- TOTAL_LENGTH, WAIST, CROTCH, HIP, THIGH, HEM

### 상의 (Tops)
- FRONT_LENGTH, SHOULDER, CHEST, SLEEVE

## 📝 데이터 형식

```csv
id,image_uri,category,measurements
1,http://example.com/img.jpg,데님 팬츠,"{""WAIST"": [100, 50, 200, 50]}"
```

## 📈 성능 평가

- **PCK**: Percentage of Correct Keypoints
- **Mean Distance**: 평균 픽셀 거리
- **MSE**: 평균 제곱 오차

## 🔒 보안

- `.env` 파일은 **절대 커밋하지 마세요** (이미 .gitignore에 추가됨)
- `.pem` 키 파일도 커밋하지 마세요
- 민감한 정보는 모두 `.env`에서 관리합니다
