"""Evaluation metrics for keypoint detection."""
import numpy as np
from typing import Dict


def compute_pck(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    visibility: np.ndarray,
    threshold: float = 10.0
) -> Dict[str, float]:
    """Compute Percentage of Correct Keypoints (PCK).
    
    Args:
        pred_coords: Predicted coordinates of shape (N, K, 2)
        gt_coords: Ground truth coordinates of shape (N, K, 2)
        visibility: Visibility mask of shape (N, K) - 1 if visible, 0 if not
        threshold: Distance threshold in pixels
        
    Returns:
        Dictionary with PCK metrics
    """
    # Compute Euclidean distances
    distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=-1))  # (N, K)
    
    # Apply visibility mask
    valid_mask = visibility > 0
    
    if valid_mask.sum() == 0:
        return {
            'pck': 0.0,
            'mean_distance': 0.0,
            'num_valid': 0
        }
    
    # Compute PCK
    correct = (distances < threshold) & valid_mask
    pck = correct.sum() / valid_mask.sum()
    
    # Mean distance for valid keypoints
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
    """Compute Mean Squared Error for visible keypoints.
    
    Args:
        pred_coords: Predicted coordinates of shape (N, K, 2)
        gt_coords: Ground truth coordinates of shape (N, K, 2)
        visibility: Visibility mask of shape (N, K)
        
    Returns:
        MSE value
    """
    valid_mask = visibility > 0
    
    if valid_mask.sum() == 0:
        return 0.0
    
    squared_errors = np.sum((pred_coords - gt_coords) ** 2, axis=-1)  # (N, K)
    mse = squared_errors[valid_mask].mean()
    
    return float(mse)

