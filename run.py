import warnings
# torchvision 및 torch 관련 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import requests
import io
import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# src 디렉토리를 path에 추가하여 내부 모듈 로드 가능하게 함
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models.kpnet import KPNet
from src.utils.utils import decode_heatmaps_batch, resize_coords

# 전역 캐시: 동일 카테고리 모델을 반복 로드하지 않음
MODELS_CACHE = {}

def get_model(category: str, device: torch.device):
    """카테고리에 맞는 모델 로드 및 캐싱"""
    # 폴더명 규칙이 'tops', 'pants' 등일 수 있으므로 소문자 변환
    cat = category.lower()
    if not cat.endswith('s') and cat in ['top', 'pant', 'skirt']: # 간단한 복수형 보정
        cat = cat + 's'
        
    if cat in MODELS_CACHE:
        return MODELS_CACHE[cat]
    
    checkpoint_path = Path(f'checkpoints/{cat}/best.pt')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # 모델 정의
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
    
    MODELS_CACHE[cat] = (model, config)
    return model, config

def load_image(url: str) -> Image.Image:
    """URL에서 이미지 로드"""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert('RGB')

def process_images(image_tasks: list):
    """
    [{'url': '...', 'category': '...'}, ...] 형태의 리스트를 받아 결과 반환
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    for task in image_tasks:
        url = task['url']
        category = task['category']
        
        try:
            # 1. 모델/설정 로드
            model, config = get_model(category, device)
            image_size = tuple(config['image_size'])
            heatmap_size = tuple(config['heatmap_size'])
            names = config['names']
            
            # 2. 이미지 로드 및 전처리
            img = load_image(url)
            orig_w, orig_h = img.size
            
            img_resized = img.resize((image_size[1], image_size[0])) # (W, H)
            img_tensor = transform(img_resized).unsqueeze(0).to(device)
            
            # 3. 예측 실행
            with torch.no_grad():
                heatmaps = model(img_tensor)
            
            # 4. 좌표 복원
            coords = decode_heatmaps_batch(heatmaps.cpu())[0]
            coords_orig = resize_coords(
                coords,
                from_size=(heatmap_size[1], heatmap_size[0]),
                to_size=(orig_w, orig_h)
            )
            
            # 5. 결과 저장
            prediction = {}
            for i, name in enumerate(names):
                p1 = coords_orig[2*i]
                p2 = coords_orig[2*i+1]
                prediction[name] = {"start": p1.tolist(), "end": p2.tolist()}
            
            results.append({
                "url": url,
                "category": category,
                "predictions": prediction
            })
            
        except Exception as e:
            results.append({"url": url, "error": str(e)})
            
    return results

def main():
    # 입력 데이터 정의 (URL과 카테고리 쌍)
    image_urls = [
        {
            "url": "https://images.ai.vingle.kr/folder/462/019bea32-5487-180a-a78a-e32c3a8cea96.webp", 
            "category": "tops"
        },
        {
            "url": "https://images.ai.vingle.kr/folder/413/019bb124-03f8-f023-e093-9bc6443f8b46.webp", 
            "category": "tops"
        },
        {
            "url": "https://images.ai.vingle.kr/folder/442/019bd5be-aad6-5a83-85b7-4d25e14d3135.webp", 
            "category": "tops"
        }
    ]
    
    # 함수를 호출하여 결과만 가져옴
    final_results = process_images(image_urls)

    print(final_results)
    
    # 결과 출력
    for res in final_results:
        print(f"\nTarget: {res['url']}")
        if "error" in res:
            print(f"  Error: {res['error']}")
            continue
            
        print(f"  Category: {res['category']}")
        for name, pts in res['predictions'].items():
            s, e = pts['start'], pts['end']
            print(f"  - {name}: ({s[0]:.1f}, {s[1]:.1f}) -> ({e[0]:.1f}, {e[1]:.1f})")

if __name__ == '__main__':
    main()