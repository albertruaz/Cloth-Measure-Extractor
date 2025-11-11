"""
Inference script for pants measurement keypoint detection
"""

import torch
import numpy as np
from PIL import Image
import io
import requests
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple
import logging

from ..models.kpnet import create_model
from ..utils.config import load_config
from ..utils.heatmap import decode_heatmaps_batch, resize_coords

logger = logging.getLogger(__name__)


class PantsMeasurementPredictor:
    """팬츠 측정 키포인트 예측기"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            config_path: config.yaml 경로
            checkpoint_path: 학습된 모델 체크포인트 경로
            device: 'cuda' 또는 'cpu'
        """
        self.device = device
        
        # Config 로드
        self.config = load_config(config_path)
        self.names = self.config['names']
        self.image_size = tuple(self.config['image_size'])
        self.heatmap_size = tuple(self.config['heatmap_size'])
        
        # 모델 로드
        self.model = create_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Transform 설정
        self.transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        logger.info(f"Predictor initialized with {len(self.names)} measurements")
    
    def load_image(self, image_source: str) -> Image.Image:
        """
        이미지 로드 (URL 또는 로컬 경로)
        
        Args:
            image_source: 이미지 URL 또는 로컬 파일 경로
            
        Returns:
            PIL Image
        """
        try:
            if image_source.startswith('http://') or image_source.startswith('https://'):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_source).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    def predict(self, image_source: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        이미지에서 측정 키포인트 예측
        
        Args:
            image_source: 이미지 URL 또는 로컬 파일 경로
            
        Returns:
            {
                'TOTAL_LENGTH': {'start': (x1, y1), 'end': (x2, y2)},
                'WAIST': {'start': (x1, y1), 'end': (x2, y2)},
                ...
            }
            좌표는 원본 이미지 크기 기준
        """
        # 이미지 로드
        image = self.load_image(image_source)
        original_size = image.size  # (width, height)
        
        # 전처리
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        # 추론
        with torch.no_grad():
            heatmaps = self.model(image_tensor)  # (1, K, H_hm, W_hm)
        
        # Heatmap을 좌표로 변환
        keypoints_heatmap = decode_heatmaps_batch(heatmaps.cpu())  # (1, K, 2)
        keypoints_heatmap = keypoints_heatmap[0]  # (K, 2)
        
        # 좌표를 원본 이미지 크기로 변환
        keypoints_original = resize_coords(
            keypoints_heatmap,
            from_size=(self.heatmap_size[1], self.heatmap_size[0]),  # (W, H)
            to_size=original_size  # (W, H)
        )
        
        # 결과 포맷팅
        results = {}
        for i, name in enumerate(self.names):
            start_idx = 2 * i
            end_idx = 2 * i + 1
            
            start_point = tuple(keypoints_original[start_idx].tolist())
            end_point = tuple(keypoints_original[end_idx].tolist())
            
            results[name] = {
                'start': start_point,
                'end': end_point
            }
        
        return results
    
    def predict_batch(self, image_sources: List[str]) -> List[Dict[str, Dict[str, Tuple[float, float]]]]:
        """
        여러 이미지에 대해 배치 예측
        
        Args:
            image_sources: 이미지 URL 또는 로컬 파일 경로 리스트
            
        Returns:
            각 이미지에 대한 예측 결과 리스트
        """
        results = []
        for image_source in image_sources:
            try:
                result = self.predict(image_source)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict {image_source}: {e}")
                results.append(None)
        
        return results


def main():
    """예제 실행"""
    import sys
    from pathlib import Path
    
    # 프로젝트 루트
    project_root = Path(__file__).parent.parent.parent
    
    # Predictor 초기화
    config_path = project_root / 'config.yaml'
    checkpoint_path = project_root / 'checkpoints' / 'best.pt'
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: ./train.sh")
        sys.exit(1)
    
    predictor = PantsMeasurementPredictor(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 예제 이미지로 테스트
    test_csv = project_root / 'data' / 'processed' / 'val.csv'
    
    if test_csv.exists():
        import pandas as pd
        df = pd.read_csv(test_csv)
        
        # 첫 번째 샘플로 테스트
        image_uri = df.iloc[0]['image_uri']
        print(f"Testing with image: {image_uri}")
        
        results = predictor.predict(image_uri)
        
        print("\n=== Prediction Results ===")
        for name, coords in results.items():
            print(f"{name}:")
            print(f"  Start: ({coords['start'][0]:.1f}, {coords['start'][1]:.1f})")
            print(f"  End: ({coords['end'][0]:.1f}, {coords['end'][1]:.1f})")
    else:
        print(f"Error: Validation data not found at {test_csv}")
        print("Please run data preprocessing first")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()





