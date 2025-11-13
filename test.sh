#!/bin/bash

# Pants Measurement Inference + Visualization Script

echo "=========================================="
echo "Pants Measurement Inference & Visualization"
echo "=========================================="
echo ""

export PYTHONPATH=$(pwd)


# 사용할 모델 파일명 (checkpoints 아래)
MODEL_NAME="best.pt"

# config / checkpoint / 결과 디렉토리
CONFIG_PATH="config.yaml"
CHECKPOINT_PATH="checkpoints/${MODEL_NAME}"
OUTPUT_DIR="results"

# 시각화할 image_uri 리스트 (test.csv 안에 있는 uri 기준으로 넣으면 됨)
# 필요 없으면 배열 비워두고, 전체 중 일부만 보고 싶으면 몇 개만 넣어도 됨
IMAGE_URIS=(
    "https://images.ai.v~"
    "https://images.ai.v~"
)

# Python 스크립트 실행
# IMAGE_URIS가 비어있으면 --images 없이 호출 → PNG는 안 그리고
# predictions.json만 생성됨. (원하면 max_highlight로 상한만 걸어도 됨)
if [ ${#IMAGE_URIS[@]} -eq 0 ]; then
    echo "[1/1] Running inference on test.csv (no visualization targets specified)..."
    python3 src/utils/visualize_predictions.py \
        --config "${CONFIG_PATH}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --output_dir "${OUTPUT_DIR}"
else
    echo "[1/1] Running inference + visualization for selected images..."
    python3 src/utils/visualize_predictions.py \
        --config "${CONFIG_PATH}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --images "${IMAGE_URIS[@]}"
fi

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "Error: Inference/visualization failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Inference & visualization completed!"
echo "Results saved under: ${OUTPUT_DIR}/${MODEL_NAME%.*}/"
echo "=========================================="
