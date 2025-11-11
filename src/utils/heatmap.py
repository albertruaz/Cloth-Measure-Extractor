"""Heatmap generation and decoding utilities."""
import numpy as np
import torch
from typing import Tuple


def generate_gaussian_heatmap(
    heatmap_size: Tuple[int, int],
    center: Tuple[float, float],
    sigma: float
) -> np.ndarray:
    """Generate a 2D Gaussian heatmap.
    
    Args:
        heatmap_size: (height, width) of the heatmap
        center: (x, y) center position in heatmap coordinates
        sigma: Standard deviation of the Gaussian
        
    Returns:
        2D numpy array of shape (height, width)
    """
    h, w = heatmap_size
    x, y = center
    
    # Create coordinate grids
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    
    # Compute Gaussian
    heatmap = np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2))
    
    return heatmap


def decode_heatmap(heatmap: torch.Tensor) -> Tuple[float, float]:
    """Decode heatmap to (x, y) coordinate using argmax.
    
    Args:
        heatmap: 2D tensor of shape (H, W)
        
    Returns:
        (x, y) coordinate
    """
    h, w = heatmap.shape
    
    # Flatten and find argmax
    flat_idx = torch.argmax(heatmap.flatten())
    y = flat_idx // w
    x = flat_idx % w
    
    return x.item(), y.item()


def decode_heatmaps_batch(heatmaps: torch.Tensor) -> np.ndarray:
    """Decode batch of heatmaps to coordinates.
    
    Args:
        heatmaps: Tensor of shape (B, K, H, W) where K is number of keypoints
        
    Returns:
        Numpy array of shape (B, K, 2) with (x, y) coordinates
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
    """Resize coordinates from one resolution to another.
    
    Args:
        coords: Array of shape (..., 2) with (x, y) coordinates
        from_size: (width, height) of source resolution
        to_size: (width, height) of target resolution
        
    Returns:
        Resized coordinates
    """
    from_w, from_h = from_size
    to_w, to_h = to_size
    
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    
    resized = coords.copy()
    resized[..., 0] *= scale_x
    resized[..., 1] *= scale_y
    
    return resized

