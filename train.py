#!/usr/bin/env python3
"""
간단한 학습 스크립트

사용법:
    python train.py --category pants
    python train.py --category denim_pants
"""
import argparse
import logging
import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.models.kpnet import KPNet
    from src.data.dataset import PantsMeasurementDataset
    from src.utils.utils import decode_heatmaps_batch, resize_coords, compute_pck, compute_mse
except ImportError:
    # 직접 실행시
    from models.kpnet import KPNet
    from data.dataset import PantsMeasurementDataset
    from utils.utils import decode_heatmaps_batch, resize_coords, compute_pck, compute_mse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_measurements_from_csv(csv_path: str) -> list:
    """CSV 파일명으로부터 measurement names JSON 로드
    
    예: data/processed/pants_train.csv
        -> data/raw/pants_measurements.json
    """
    csv_path = Path(csv_path)
    # _train, _val, _test 제거
    category_name = csv_path.stem.replace('_train', '').replace('_val', '').replace('_test', '')
    
    # raw 폴더에서 measurements.json 찾기
    measurements_file = Path('data/raw') / f"{category_name}_measurements.json"
    
    if measurements_file.exists():
        with open(measurements_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 새 포맷: {"measurement_names": [...], "raw": {...}, "train": {...}, ...}
        if isinstance(data, dict) and 'measurement_names' in data:
            measurements = data['measurement_names']
            logger.info(f"Measurements loaded from {measurements_file}")
            logger.info(f"  Names: {measurements}")
            
            # 통계 정보 출력
            if 'raw' in data:
                logger.info(f"  Raw samples: {data['raw'].get('total_samples', 'N/A')}")
                logger.info(f"  Categories: {data['raw'].get('categories', {})}")
            if 'train' in data:
                logger.info(f"  Train samples: {data['train'].get('total_samples', 'N/A')}")
            if 'val' in data:
                logger.info(f"  Val samples: {data['val'].get('total_samples', 'N/A')}")
            
            return measurements
        # 구 포맷: [...] 직접 리스트
        elif isinstance(data, list):
            logger.info(f"Measurements loaded from {measurements_file}: {data}")
            return data
        else:
            logger.warning(f"Unknown format in {measurements_file}")
            return []
    else:
        logger.warning(f"Measurements file not found: {measurements_file}")
        return []


def create_model(config: dict) -> KPNet:
    """모델 생성"""
    names = config['names']
    k_points = 2 * len(names)  # 각 측정치마다 start, end 2개
    
    model = KPNet(
        k_points=k_points,
        backbone=config.get('backbone', 'resnet18'),
        pretrained=config.get('pretrained', True),
        num_deconv_layers=config.get('num_deconv_layers', 3),
        num_deconv_filters=config.get('num_deconv_filters', [256, 256, 256]),
        num_deconv_kernels=config.get('num_deconv_kernels', [4, 4, 4])
    )
    
    # 가중치 초기화
    for m in model.deconv_layers.modules():
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    nn.init.normal_(model.final_layer.weight, std=0.001)
    nn.init.constant_(model.final_layer.bias, 0)
    
    return model


def create_dataloaders(config: dict) -> tuple:
    """데이터로더 생성"""
    train_dataset = PantsMeasurementDataset(
        csv_path=config['train_csv'],
        names=config['names'],
        image_size=tuple(config['image_size']),
        heatmap_size=tuple(config['heatmap_size']),
        sigma=config['sigma'],
        is_train=True
    )
    
    val_dataset = PantsMeasurementDataset(
        csv_path=config['val_csv'],
        names=config['names'],
        image_size=tuple(config['image_size']),
        heatmap_size=tuple(config['heatmap_size']),
        sigma=config['sigma'],
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['image'].to(device)
        heatmaps = batch['heatmaps'].to(device)
        visibility = batch['visibility'].to(device)
        
        # Forward
        pred_heatmaps = model(images)
        
        # Loss (visibility mask 적용)
        loss_per_point = criterion(pred_heatmaps, heatmaps)
        
        # visibility에 따라 loss 마스킹
        if len(visibility.shape) == 2:  # (B, K)
            vis_mask = visibility.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        else:
            vis_mask = visibility
        
        loss = (loss_per_point * vis_mask).sum() / (vis_mask.sum() + 1e-8)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, config):
    """검증"""
    model.eval()
    total_loss = 0
    all_pred_coords = []
    all_gt_coords = []
    all_visibility = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            visibility = batch['visibility'].to(device)
            gt_coords = batch['keypoints'].cpu().numpy()
            
            # Forward
            pred_heatmaps = model(images)
            
            # Loss
            loss_per_point = criterion(pred_heatmaps, heatmaps)
            
            if len(visibility.shape) == 2:
                vis_mask = visibility.unsqueeze(-1).unsqueeze(-1)
            else:
                vis_mask = visibility
            
            loss = (loss_per_point * vis_mask).sum() / (vis_mask.sum() + 1e-8)
            total_loss += loss.item()
            
            # 좌표 디코딩
            pred_coords = decode_heatmaps_batch(pred_heatmaps.cpu())
            
            # 해상도 변환 (heatmap -> image)
            pred_coords = resize_coords(
                pred_coords,
                from_size=tuple(config['heatmap_size'][::-1]),  # (W, H)
                to_size=tuple(config['image_size'][::-1])
            )
            
            all_pred_coords.append(pred_coords)
            all_gt_coords.append(gt_coords)
            all_visibility.append(visibility.cpu().numpy())
    
    # 메트릭 계산
    all_pred_coords = np.concatenate(all_pred_coords, axis=0)
    all_gt_coords = np.concatenate(all_gt_coords, axis=0)
    all_visibility = np.concatenate(all_visibility, axis=0)
    
    pck_metrics = compute_pck(all_pred_coords, all_gt_coords, all_visibility)
    mse = compute_mse(all_pred_coords, all_gt_coords, all_visibility)
    
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, pck_metrics, mse


def main():
    parser = argparse.ArgumentParser(description='학습 스크립트')
    parser.add_argument('--category', type=str, required=True,
                        help='카테고리 이름 (e.g., pants, denim_pants)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='설정 파일 경로 (기본: config.yaml)')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 카테고리 이름으로 CSV 경로 자동 생성
    category_name = args.category
    logger.info(f"카테고리: {category_name}")
    
    train_csv_path = Path(f'data/processed/{category_name}_train.csv')
    val_csv_path = Path(f'data/processed/{category_name}_val.csv')
    
    if not train_csv_path.exists():
        logger.error(f"Train CSV not found: {train_csv_path}")
        sys.exit(1)
    
    if not val_csv_path.exists():
        logger.error(f"Val CSV not found: {val_csv_path}")
        sys.exit(1)
    
    config['train_csv'] = str(train_csv_path)
    config['val_csv'] = str(val_csv_path)
    
    # Measurement names 로드
    measurements = load_measurements_from_csv(str(train_csv_path))
    if not measurements:
        logger.error(f"No measurements found. Please check {train_csv_path.parent}/{category_name}_measurements.json")
        sys.exit(1)
    
    config['names'] = measurements
    logger.info(f"Measurements: {measurements}")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # 모델 생성
    model = create_model(config)
    model = model.to(device)
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 데이터로더
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Loss, optimizer, scheduler
    criterion = nn.MSELoss(reduction='none')
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config.get('weight_decay', 1e-4))
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # 체크포인트 디렉토리 (카테고리별로 생성)
    checkpoint_dir = Path('checkpoints') / category_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    
    # 학습
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config['epochs']}")
        logger.info(f"{'='*60}")
        
        # 학습
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # 검증
        val_loss, pck_metrics, mse = validate(model, val_loader, criterion, device, config)
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val PCK: {pck_metrics['pck']:.4f}")
        logger.info(f"Val Mean Distance: {pck_metrics['mean_distance']:.2f}")
        logger.info(f"Val MSE: {mse:.4f}")
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 베스트 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'category': category_name
            }, checkpoint_path)
            logger.info(f"✓ Best model saved: {checkpoint_path}")
    
    logger.info("\n학습 완료!")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
