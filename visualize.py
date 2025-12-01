#!/usr/bin/env python3
"""
학습된 모델의 예측 결과를 시각화하는 스크립트

사용법:
    python visualize.py --category pants
    python visualize.py --category denim_pants --num-samples 20
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import io
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.models.kpnet import KPNet
    from src.data.dataset import PantsMeasurementDataset
    from src.utils.utils import decode_heatmaps_batch, resize_coords
except ImportError:
    from models.kpnet import KPNet
    from data.dataset import PantsMeasurementDataset
    from utils.utils import decode_heatmaps_batch, resize_coords

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device):
    """체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 모델 생성
    names = config['names']
    k_points = 2 * len(names)
    
    model = KPNet(
        k_points=k_points,
        backbone=config.get('backbone', 'resnet18'),
        pretrained=False,
        num_deconv_layers=config.get('num_deconv_layers', 3),
        num_deconv_filters=config.get('num_deconv_filters', [256, 256, 256]),
        num_deconv_kernels=config.get('num_deconv_kernels', [4, 4, 4])
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"모델 로드 완료: {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model, config


def load_image(image_path: str) -> Image.Image:
    """이미지 로드 (URL 또는 로컬 경로)"""
    try:
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"이미지 로드 실패 {image_path}: {e}")
        return None


def draw_keypoints(image: Image.Image, keypoints: np.ndarray, names: list, 
                   gt_keypoints: np.ndarray = None) -> Image.Image:
    """
    이미지에 키포인트 그리기
    
    Args:
        image: PIL 이미지
        keypoints: (K, 2) 예측 키포인트 좌표
        names: 측정 이름 리스트
        gt_keypoints: (K, 2) 정답 키포인트 (optional)
    """
    draw = ImageDraw.Draw(image)
    
    # 색상 (예측: 빨강, 정답: 초록)
    pred_color = (255, 0, 0)
    gt_color = (0, 255, 0)
    line_color = (255, 255, 0)
    
    # 각 측정치에 대해 start, end 키포인트 그리기
    for i, name in enumerate(names):
        start_idx = 2 * i
        end_idx = 2 * i + 1
        
        # 예측 키포인트
        start_x, start_y = keypoints[start_idx]
        end_x, end_y = keypoints[end_idx]
        
        # 선 그리기
        draw.line([(start_x, start_y), (end_x, end_y)], fill=line_color, width=2)
        
        # 점 그리기
        r = 5
        draw.ellipse([start_x-r, start_y-r, start_x+r, start_y+r], fill=pred_color)
        draw.ellipse([end_x-r, end_y-r, end_x+r, end_y+r], fill=pred_color)
        
        # 정답 키포인트 (있으면)
        if gt_keypoints is not None:
            gt_start_x, gt_start_y = gt_keypoints[start_idx]
            gt_end_x, gt_end_y = gt_keypoints[end_idx]
            
            # 0이 아닌 경우만 그리기 (missing keypoint 제외)
            if not (gt_start_x == 0 and gt_start_y == 0):
                draw.ellipse([gt_start_x-r, gt_start_y-r, gt_start_x+r, gt_start_y+r], 
                           outline=gt_color, width=2)
            if not (gt_end_x == 0 and gt_end_y == 0):
                draw.ellipse([gt_end_x-r, gt_end_y-r, gt_end_x+r, gt_end_y+r], 
                           outline=gt_color, width=2)
    
    return image


def visualize_dataset(model, data_csv: str, config: dict, output_dir: Path, 
                      num_samples: int = 10):
    """데이터셋에서 샘플 선택해 시각화"""
    device = next(model.parameters()).device
    
    # 데이터 로드 (탭 구분자)
    df = pd.read_csv(data_csv, sep='\t')
    logger.info(f"데이터 로드: {len(df)} 샘플")
    
    # 랜덤 샘플 선택
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 샘플 시각화
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='시각화 중'):
        try:
            # 이미지 로드
            image = load_image(row['image_uri'])
            if image is None:
                continue
            
            # 원본 크기 저장
            orig_w, orig_h = image.size
            
            # 리사이즈
            image_size = tuple(config['image_size'])
            image_resized = image.resize((image_size[1], image_size[0]))
            
            # 텐서로 변환
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image_resized).unsqueeze(0).to(device)
            
            # 예측
            with torch.no_grad():
                pred_heatmaps = model(image_tensor)
            
            # 좌표 디코딩
            pred_coords = decode_heatmaps_batch(pred_heatmaps.cpu())[0]  # (K, 2)
            
            # 좌표를 원본 이미지 크기로 변환
            pred_coords = resize_coords(
                pred_coords,
                from_size=tuple(config['heatmap_size'][::-1]),
                to_size=(orig_w, orig_h)
            )
            
            # 정답 좌표 (request_body에서 파싱)
            import json
            request_body = row['request_body']
            data = json.loads(request_body)
            
            # items 배열에서 measurements 추출
            measurements = {}
            if 'items' in data:
                for item in data['items']:
                    name = item.get('name')
                    if name:
                        measurements[name] = item
            elif 'measurements' in data:
                for item in data['measurements']:
                    name = item.get('name')
                    if name:
                        measurements[name] = item
            
            gt_coords = []
            for name in config['names']:
                if name in measurements:
                    item = measurements[name]
                    x1 = item.get('x1', 0)
                    y1 = item.get('y1', 0)
                    x2 = item.get('x2', 0)
                    y2 = item.get('y2', 0)
                    gt_coords.extend([[x1, y1], [x2, y2]])
                else:
                    gt_coords.extend([[0, 0], [0, 0]])
            gt_coords = np.array(gt_coords)
            
            # 시각화
            vis_image = draw_keypoints(image, pred_coords, config['names'], gt_coords)
            
            # 저장
            output_path = output_dir / f"sample_{idx}.jpg"
            vis_image.save(output_path)
            
        except Exception as e:
            logger.error(f"샘플 {idx} 처리 실패: {e}")
            continue
    
    logger.info(f"시각화 완료: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='모델 예측 시각화')
    parser.add_argument('--category', type=str, required=True,
                        help='카테고리 이름 (e.g., pants, denim_pants)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='시각화할 샘플 수 (기본: 10)')
    
    args = parser.parse_args()
    
    # 카테고리 이름으로 경로 자동 생성
    category_name = args.category
    checkpoint_path = Path(f'checkpoints/{category_name}/best.pt')
    data_csv = Path(f'data/processed/{category_name}_test.csv')
    output_dir = Path(f'results/{category_name}')
    
    # 파일 존재 확인
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not data_csv.exists():
        logger.error(f"Test CSV not found: {data_csv}")
        sys.exit(1)
    
    # 디바이스 설정 (CPU 사용)
    device = torch.device('cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Category: {category_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Test data: {data_csv}")
    logger.info(f"Output: {output_dir}")
    
    # 모델 로드
    model, config = load_checkpoint(str(checkpoint_path), device)
    
    # 시각화
    visualize_dataset(model, str(data_csv), config, output_dir, args.num_samples)


if __name__ == '__main__':
    main()
