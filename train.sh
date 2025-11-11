#!/bin/bash

# Pants Measurement Keypoint Detection Training Script

echo "=========================================="
echo "Pants Measurement Keypoint Detection"
echo "=========================================="
echo ""

# 데이터 전처리 및 분할 확인 (이미 완료되어 있을 수 있음)
if [ ! -f "data/processed/train.csv" ] || [ ! -f "data/processed/val.csv" ] || [ ! -f "data/processed/test.csv" ]; then
    echo "[1/2] 데이터 전처리 및 분할 중..."
    python3 src/utils/preprocess_data.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to preprocess data"
        exit 1
    fi
    echo ""
fi

echo "[2/2] Starting training..."
python3 src/main_train.py --config config.yaml

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training completed successfully!"
echo "=========================================="



