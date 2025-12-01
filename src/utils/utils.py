"""통합 유틸리티 함수들"""
import numpy as np
import torch
from typing import Tuple, Dict


# ============ Heatmap 생성 및 디코딩 ============

def generate_gaussian_heatmap(
    heatmap_size: Tuple[int, int],
    center: Tuple[float, float],
    sigma: float
) -> np.ndarray:
    """2D Gaussian heatmap 생성"""
    h, w = heatmap_size
    x, y = center
    
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    heatmap = np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2))
    
    return heatmap


def decode_heatmap(heatmap: torch.Tensor) -> Tuple[float, float]:
    """Heatmap에서 (x, y) 좌표 추출"""
    h, w = heatmap.shape
    flat_idx = torch.argmax(heatmap.flatten())
    y = flat_idx // w
    x = flat_idx % w
    return x.item(), y.item()


def decode_heatmaps_batch(heatmaps: torch.Tensor) -> np.ndarray:
    """배치 heatmap들을 좌표로 변환
    
    Args:
        heatmaps: (B, K, H, W) 형태의 텐서
    Returns:
        (B, K, 2) 형태의 numpy 배열
    """
    batch_size, num_keypoints, h, w = heatmaps.shape
    coords = np.zeros((batch_size, num_keypoints, 2))
    
    for b in range(batch_size):
        for k in range(num_keypoints):
            x, y = decode_heatmap(heatmaps[b, k])
            coords[b, k] = [x, y]
    
    return coords


def resize_coords(
    coords: np.ndarray,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int]
) -> np.ndarray:
    """좌표를 다른 해상도로 변환"""
    from_w, from_h = from_size
    to_w, to_h = to_size
    
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    
    resized = coords.copy()
    resized[..., 0] *= scale_x
    resized[..., 1] *= scale_y
    
    return resized


# ============ 평가 메트릭 ============

def compute_pck(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    visibility: np.ndarray,
    threshold: float = 10.0
) -> Dict[str, float]:
    """PCK (Percentage of Correct Keypoints) 계산
    
    Args:
        pred_coords: (N, K, 2) 예측 좌표
        gt_coords: (N, K, 2) 정답 좌표
        visibility: (N, K) 가시성 마스크
        threshold: 픽셀 거리 임계값
    """
    distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=-1))
    valid_mask = visibility > 0
    
    if valid_mask.sum() == 0:
        return {'pck': 0.0, 'mean_distance': 0.0, 'num_valid': 0}
    
    correct = (distances < threshold) & valid_mask
    pck = correct.sum() / valid_mask.sum()
    mean_distance = distances[valid_mask].mean()
    
    return {
        'pck': float(pck),
        'mean_distance': float(mean_distance),
        'num_valid': int(valid_mask.sum())
    }


def compute_mse(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    visibility: np.ndarray
) -> float:
    """MSE 계산 (가시성 있는 키포인트에 대해서만)"""
    valid_mask = visibility > 0
    
    if valid_mask.sum() == 0:
        return 0.0
    
    squared_errors = np.sum((pred_coords - gt_coords) ** 2, axis=-1)
    mse = squared_errors[valid_mask].mean()
    
    return float(mse)
